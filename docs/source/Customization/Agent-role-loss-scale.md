# Agent Role Loss Scale 技术实现与使用

## 1. 功能概述

Agent Role Loss Scale 为标准 Agent JSONL 提供一套共享配置的连续权重能力：

1. **数据注入**：按原始消息 role 向每条 message 写入基础 `loss_scale`，用于固化和审计实验数据。
2. **训练展开**：读取 message 权重，在 tokenization 前保留消息来源，进一步拆分 assistant 的 think 标签、思考内容和回答，并生成逐 token `loss_scale`。当前可选modifier只作用于response片段。

已验证范围是：`template_backend=swift`、因果语言模型 SFT、Qwen3.5 的 `qwen3_5` Agent 模板。其他 Agent 模板、Jinja 训练后端和 RLHF 不在当前验证范围内。

职责边界如下：

- `swift/loss_scale/role.py` 和模板链路是 role loss 的唯一语义实现，负责配置校验、来源追踪、assistant 分段和逐 token 权重生成。
- `scripts/utils/inject_role_loss_scale.py` 只调用上述实现提供的注入函数，将五种 role 的基础权重物化到 JSONL message。
- `scripts/utils/build_cached_dataset/distributed_cached_dataset.py` 只负责数据分片、多机调度、本地权重物化、调用当前仓库的 `swift export`、复制和合并，不重复实现任何 loss 计算规则。

权重配置使用以下五个语义类别：

- `system`
- `user`
- `assistant`
- `tool_call`
- `tool_response`

原始 Agent JSONL 的工具返回 role 可以是 `tool_response` 或官方等价别名 `tool`；两者都使用
`role_weights.tool_response`，注入时保留原始 role 不变。

空 think、模板控制 token、padding、多模态输入 token 和 assistant EOS 继续遵循 ms-swift 原有 SFT 规则。

## 2. 配置格式

```json
{
  "role_weights": {
    "system": 0.0,
    "user": 0.2,
    "assistant": 1.0,
    "tool_call": 1.0,
    "tool_response": 0.2
  },
  "assistant_weights": {
    "think_tag": 1.0,
    "think_content": 0.5,
    "answer": 1.0
  }
}
```

要求：

- 两组字段必须完整，不接受未知字段。
- 权重必须是非负有限数值。
- `role_weights` 是 message 缺少 `loss_scale` 时的默认基础权重。
- JSONL 中显式的 `message.loss_scale` 优先于 `role_weights`；因此注入后再修改 `role_weights` 不会覆盖已物化的 message 权重。
- `assistant_weights` 是相对 assistant 基础权重的乘数，训练时始终从配置读取。

assistant 最终权重示例：

```text
<think>      assistant × think_tag
思考内容      assistant × think_content
</think>     assistant × think_tag
回答正文      assistant × answer
```

模板自动补充的空 `<think>\n\n</think>\n\n` 固定为 0，不接受自定义覆盖。

## 3. 数据注入

### 3.1 命令

```bash
export MS_SWIFT_REPO=/data_train/vectorchen/github/ms-swift
export SWIFT_ENV=/data_train/vectorchen/env/swift-202609/swift-train-202609
export PYTHONPATH=$MS_SWIFT_REPO:${PYTHONPATH:-}

$SWIFT_ENV/bin/python \
  $MS_SWIFT_REPO/scripts/utils/inject_role_loss_scale.py \
  --input input.jsonl \
  --config role_loss_config.json \
  --output output.jsonl
```

原数据已有 `loss_scale` 时默认报错；确认需要覆盖时增加：

```bash
--overwrite
```

### 3.2 输入输出

输入：

```json
{
  "role": "tool_response",
  "content": "tool result",
  "reasoning_content": "",
  "tool_call_id": ""
}
```

输出：

```json
{
  "role": "tool_response",
  "content": "tool result",
  "reasoning_content": "",
  "tool_call_id": "",
  "loss_scale": 0.2
}
```

注入器逐行处理超大 JSONL，只新增或覆盖 `message.loss_scale`，不修改其他字段值、消息顺序和轨迹内容。JSON 会被重新序列化为单行紧凑格式，因此空白和缩进不保证与输入字节一致。输入中已有的 `reasoning_content`、`tool_call_id` 等字段会保留；缺失字段不会被自动创建。所有记录成功后通过原子替换生成目标文件，并继承源文件权限。

训练代码也允许直接读取未注入的数据，并在 `message.loss_scale` 缺失时回退到 `role_weights`。正式实验仍推荐先注入，以便数据自身记录基础权重。

### 3.3 分布式 cached dataset 一体化构建

大规模数据不需要先在共享存储中生成一份完整的 weighted JSONL。仓库提供：

```text
scripts/utils/build_cached_dataset/
├── distributed_cached_dataset.py
├── run_distributed_build.sh
├── cached_dataset_config.example.toml
└── role_loss_config.example.json
```

复制示例配置并填写数据、模型、集群和输出路径；关键配置为：

```toml
[runtime]
ms_swift_repo = "/data_train/vectorchen/github/ms-swift"

[export]
loss_scale = "role"
role_loss_config = "./role_loss_config.json"
materialize_role_weights = true
```

启动方式：

```bash
CONFIG=/path/to/cached_dataset_config.toml \
  scripts/utils/build_cached_dataset/run_distributed_build.sh
```

单个 shard 的处理流程为：

```text
原始raw shard
→ worker本地调用inject_role_loss_scale_jsonl
→ 生成临时weighted shard
→ 调用当前ms-swift源码分支执行swift export
→ 保存cached shard
→ 复制到共享目录
→ 合并最终cached dataset
```

该设计有以下约束：

- 不修改原始 JSONL，也不在共享存储中生成一份完整的 weighted JSONL。
- 本地临时 weighted shard 在导出结束后删除。
- cached dataset 保存已物化的 `message.loss_scale` 和 `lengths`，不保存最终逐 token `loss_scale`；逐 token 权重仍在训练取样时由同一个 `RoleLossScale` 生成。
- `role_loss_config` 在 cached export 和训练阶段都必须提供：message 中保存的是五种 role 的基础权重，`assistant_weights` 和 role 解析规则仍来自配置。
- worker 会设置 `PYTHONPATH` 并校验 `swift.__file__`，确保注入和 export 使用 `ms_swift_repo` 指定的源码，而不是环境中另一份已安装的 ms-swift。
- cached shard 和最终数据集记录 Git commit、关键源码哈希、role 配置哈希、模型配置哈希、模板及截断参数。已有 `_SUCCESS` 但指纹缺失或不一致时拒绝复用。

权重注入位于 raw shard 生成之后，避免改变 JSONL 行长度、破坏字节级并行分片的预计算偏移。

## 4. 训练使用

在原 SFT 命令中增加：

```bash
--loss_scale role \
--role_loss_config /path/to/role_loss_config.json
```

示例：

```bash
export MS_SWIFT_REPO=/data_train/vectorchen/github/ms-swift
export SWIFT_ENV=/data_train/vectorchen/env/swift-202609/swift-train-202609
export PYTHONPATH=$MS_SWIFT_REPO:${PYTHONPATH:-}

$SWIFT_ENV/bin/swift sft \
  --model /data_train/train/qwen/models/Qwen3.5-35B-A3B \
  --dataset /path/to/output.jsonl \
  --loss_scale role \
  --role_loss_config /path/to/role_loss_config.json \
  --output_dir /path/to/output_dir
```

使用分布式构建产物训练时，将 `--dataset` 替换为 `--cached_dataset`，并继续传入构建时相同的 role 配置：

```bash
export PYTHONPATH=/data_train/vectorchen/github/ms-swift:${PYTHONPATH:-}

$SWIFT_ENV/bin/swift sft \
  --model /data_train/train/qwen/models/Qwen3.5-35B-A3B \
  --cached_dataset /path/to/cached_dataset/train \
  --loss_scale role \
  --role_loss_config /path/to/role_loss_config.json \
  --packing true \
  --output_dir /path/to/output_dir
```

训练前应核对 cached dataset 根目录的 `manifest.json`：其 `ms_swift_commit`、`role_loss_config_sha256`、模板、模型和截断参数必须与本次训练一致。当前 ms-swift 的 `--cached_dataset` 加载器不会自动读取这个外部 manifest，因此启动脚本仍需负责该项校验。

role 策略固定使用连续权重路径。`--is_binary_loss_scale` 可以省略或显式设为 `false`，不能设为 `true`。

当前支持的role策略组合为：

```text
role
role+ignore_empty_think
role+hermes
role+hermes+ignore_empty_think
```

Qwen3.5 SFT 参数层会自动追加`ignore_empty_think`；`RoleLossScale`自身已固定屏蔽空think，因此工厂会过滤该重复modifier。其他modifier当前会直接报错。

如需在role权重基础上继续应用官方Hermes工具调用两倍权重，可使用：

```bash
--loss_scale role+hermes
```

此时最终tool_call权重为：

```text
message.loss_scale（缺失时取role_weights.tool_call）× 2
```

## 5. 训练时实现流程

```text
JSONL messages
→ 保存原始role和message loss_scale
→ Qwen3.5规范化assistant think格式
→ Agent模板格式化tool_call/tool_response
→ 合并消息并同步保留来源与权重
→ 标记SYSTEM、QUERY、RESPONSE、RESPONSE_SUFFIX
→ RoleLossScale拆分assistant think/answer
→ 应用可选modifier
→ tokenization
→ 展开逐token labels和loss_scale
→ trainer同步shift并计算加权交叉熵
```

### 5.1 模板边界

模板占位符会拆分为真实消息内容和固定模板字符：

```text
<|im_start|>user\n → 固定模板，权重0
用户正文               → user权重
<|im_end|>...        → 固定模板，按框架规则
```

因此正权重 `user` 或 `tool_response` 不会让 role header 参与训练。

### 5.2 Agent 消息合并

原始轨迹：

```text
assistant(weight=A)
tool_call(weight=C)
tool_response(weight=R)
```

格式化和合并后仍分别保留来源权重：

- assistant 内容使用 `A`，并进一步应用 `assistant_weights`。
- 完整 `<tool_call>...</tool_call>` 使用 `C`；Qwen3.5 在已有 assistant 正文后自动加入的 `\n\n`，以及并行调用之间的换行，也属于格式化后的 tool_call 片段并使用 `C`。
- 完整 `<tool_response>...</tool_response>` 使用 `R`；多个返回块之间的模板换行保持 0。
- 工具块外围的 ChatML role header 等模板字符保持 0。

并行 tool_call 或 tool_response 中，代码会检查所有显式提供的同名权重是否一致；标准注入流程会为每条消息写入权重，因此不一致时会直接报错，避免静默保留首个值。

### 5.3 Assistant 分段细节

Qwen3.5先对assistant字符串执行`strip`，并将已有think块规范化为：

```text
<think>\n{reasoning}\n</think>\n\n{answer}
```

随后`RoleLossScale`只接受以下三类输入：

- 以完整`<think>...</think>`开头：首个闭合标签前为think内容，闭合标签后的全部文本为answer；规范化产生的标签内换行使用`think_content`，标签后的换行使用`answer`。
- 不包含think标签：整段使用`answer`。
- 起始空think：空think及其尾随空白固定为0，后续文本使用`answer`。

字符串包含think标签但不是以上合法形式时直接报错。已有message的`loss=false`仍会把对应assistant或tool_call response片段基础权重置为0。

### 5.4 固定 SFT 规则

以下内容不开放自定义：

| 内容 | 权重规则 |
|---|---|
| 自动空 think | 0 |
| `<|im_start|>` 和 role header | 0 |
| system/user/tool-response 的 `<|im_end|>` | 0 |
| 历史 assistant `<|im_end|>` | 前一 response 含正权重 token 时为 1，否则为 0 |
| 最终 assistant `<|im_end|>` | 只要模板添加最终 suffix 就固定为 1，与 message 连续权重无关 |
| padding | 0 |
| 序列首 token | 0 |
| image/video/audio 输入 token | 0，且 label 为 `-100` |

顶层`tools`会由Agent模板格式化后与system内容合并，整体使用system message的显式`loss_scale`；system未显式设置时回退到`role_weights.system`。

## 6. Loss 计算

role 策略始终输出与输入等长的连续权重：

```text
len(input_ids) == len(labels) == len(loss_scale)
```

非序列并行路径中，Trainer 对 `labels` 和 `loss_scale` 同步左移，使原始位置 `t` 的权重作用于预测 `input_ids[t]` 的损失；序列并行路径会先按并行位置收集逐 token 权重，再进行同形状相乘：

```text
loss = sum(token_ce × loss_scale) / num_items_in_batch
```

`num_items_in_batch` 通常是当前批次中 `labels != -100` 的有效目标 token 数，并由 Trainer 兼容梯度累积和分布式统计。保持 ms-swift 现有归一化语义，不按 `sum(loss_scale)` 归一化。

## 7. Packing、Padding 与截断

现有框架对以下数组同步处理：

```text
input_ids
labels
loss_scale
```

- Packing：模板基类按相同顺序拼接三组数组。
- Left/right padding：`labels=-100`、`loss_scale=0`。
- Left/right truncation：使用同一索引裁剪，并将新序列首 token 设为 `labels=-100`、`loss_scale=0`。

因此基础数组处理不会发生权重错位；但截断仍可能删除完整角色内容。正式 Agent SFT 建议优先使用 `delete/raise` 过滤超长样本。

当前已手工验证模板基类的 packing、左右 padding 和左右 truncation 数组对齐；尚未执行 Qwen3.5 模型 forward 或端到端训练，也未完成 Qwen3.5 模型专属 position_ids/packing 路径的独立集成验证。

## 8. 代码实现位置

核心代码：

- `swift/loss_scale/role.py`：配置校验、JSONL 注入、role 权重和 assistant 分段。
- `swift/loss_scale/mapping.py`：注册 `role` 及 modifier 组合。
- `swift/template/template_inputs.py`：保留 system 权重和原始 role。
- `swift/template/base.py`：模板边界、Agent 合并和历史 EOS 处理。
- `swift/template/utils.py`：新增上下文类型。
- `swift/arguments/base_args/template_args.py`、`swift/template/register.py`：参数通路。
- `scripts/utils/inject_role_loss_scale.py`：独立注入命令入口。
- `scripts/utils/build_cached_dataset/distributed_cached_dataset.py`：分布式分片、worker本地权重物化、`swift export`调用、缓存指纹和最终合并。
- `scripts/utils/build_cached_dataset/cached_dataset_config.example.toml`：一体化构建配置示例。
- `tests/loss_scale/test_role_loss_scale.py`：单元和边界测试。

分布式构建器复用 `swift.loss_scale.role.inject_role_loss_scale_jsonl` 和 `swift export`，不包含第二套 role loss 算法。模型 forward、trainer、packing 和多模态 token 展开逻辑保持不变。

## 9. 当前验证结果

开发分支：`feat/agent-role-loss-scale-v20260908`。

核心 role loss 提交：`c02603690c700c2b24c5ae0aecdfde74ac3a3e71`。

已验证：

- `tests/loss_scale`：11 passed。
- 使用固定随机种子 `20260908`，按源文件随机字节偏移定位后续完整记录，抽取10条真实Agent轨迹；该方法可复现，但不是按JSONL行号等概率抽样。
- 10条轨迹全部完成权重注入和Qwen3.5 template encode，共1,002,340 tokens，三组数组严格等长。
- 四种assistant数据形态均覆盖，空think、tool_call、tool_response和EOS权重符合预期。
- 模板基类的packing、左右padding、left/right truncation数组对齐通过。
- 已使用一条标准Agent轨迹完成单分片集成验证：raw shard → 本地权重物化 → 新分支cached export → shared cached shard → merged cached dataset；产物保留五种role的`message.loss_scale`和`lengths`，分片与最终manifest指纹一致。
- 已将该 cached dataset 重新送入新分支模板编码，得到782个token，`input_ids`、`labels`和`loss_scale`严格等长，正权重集合为`0.2/0.5/1.0`。
- Ruff、IDE诊断、Python编译、Shell语法和`git diff --check`通过。

尚未验证：

- 多节点SSH环境的完整256-shard生产运行。
- Qwen3.5模型forward和完整训练。
- 训练入口对distributed builder外部`manifest.json`的自动一致性校验；当前需由训练启动脚本核对。
