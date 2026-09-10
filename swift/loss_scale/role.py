# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import math
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from swift.template import ContextType, Messages
from .base import LossScale

# 配置只开放有明确训练语义的五类role和三类assistant片段。
ROLE_NAMES = ('system', 'user', 'assistant', 'tool_call', 'tool_response')
ASSISTANT_WEIGHT_NAMES = ('think_tag', 'think_content', 'answer')
_EMPTY_THINK_RE = re.compile(r'^<think>\s*</think>\s*', re.DOTALL)
_TOOL_RESPONSE_RE = re.compile(r'<tool_response>.*?</tool_response>', re.DOTALL)


def _validate_weights(values: dict, required_keys: Sequence[str], field_name: str) -> Dict[str, float]:
    """校验权重字段完整性，并统一转换为非负有限浮点数。"""
    if not isinstance(values, dict):
        raise TypeError(f'{field_name} 必须是JSON对象。')
    missing = set(required_keys) - values.keys()
    unknown = values.keys() - set(required_keys)
    if missing or unknown:
        raise ValueError(f'{field_name}配置无效：缺少={sorted(missing)}，未知={sorted(unknown)}')
    result = {}
    for key in required_keys:
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f'{field_name}.{key}必须是数值，实际为{type(value).__name__}。')
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'{field_name}.{key}必须是非负有限数值，实际为{value}。')
        result[key] = value
    return result


@dataclass(frozen=True)
class RoleLossConfig:
    """保存五类消息基础权重及assistant内部语义权重。"""

    role_weights: Dict[str, float]
    assistant_weights: Dict[str, float]

    @classmethod
    def from_dict(cls, data: dict) -> 'RoleLossConfig':
        """从字典构造配置，并拒绝缺失或未知字段。"""
        if not isinstance(data, dict):
            raise TypeError('角色权重配置必须是JSON对象。')
        allowed_keys = {'role_weights', 'assistant_weights'}
        unknown = data.keys() - allowed_keys
        if unknown:
            raise ValueError(f'角色权重配置包含未知字段：{sorted(unknown)}')
        role_weights = _validate_weights(data.get('role_weights'), ROLE_NAMES, 'role_weights')
        assistant_weights = _validate_weights(
            data.get('assistant_weights'), ASSISTANT_WEIGHT_NAMES, 'assistant_weights')
        return cls(role_weights=role_weights, assistant_weights=assistant_weights)


def load_role_loss_config(path: str) -> RoleLossConfig:
    """从JSON文件加载并校验角色权重配置。"""
    if not path:
        raise ValueError('loss_scale以role开头时必须提供role_loss_config。')
    path = os.path.abspath(os.path.expanduser(path))
    with open(path, 'r', encoding='utf-8') as f:
        return RoleLossConfig.from_dict(json.load(f))


def inject_role_loss_scale_jsonl(input_path: str, config_path: str, output_path: str, *, overwrite: bool = False) -> dict:
    """逐行注入message权重，并通过临时文件原子生成目标JSONL。"""
    config = load_role_loss_config(config_path)
    input_path = os.path.abspath(os.path.expanduser(input_path))
    output_path = os.path.abspath(os.path.expanduser(output_path))
    if input_path == output_path:
        raise ValueError('输入和输出路径不能相同。')

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    role_counts = {role: 0 for role in ROLE_NAMES}
    line_count = 0
    tmp_path = None
    try:
        # 全部记录成功后再原子替换目标文件，避免产生半成品。
        with open(input_path, 'r', encoding='utf-8') as src, tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', dir=output_dir, prefix='.role-loss-scale-', suffix='.jsonl',
                delete=False) as dst:
            tmp_path = dst.name
            for line_no, line in enumerate(src, start=1):
                if not line.strip():
                    raise ValueError(f'JSONL第{line_no}行为空。')
                try:
                    row = json.loads(line)
                    messages = row['messages']
                    if not isinstance(messages, list):
                        raise TypeError('messages必须是列表。')
                    tools = row.get('tools')
                    if isinstance(tools, str) and not isinstance(json.loads(tools), list):
                        raise TypeError('tools字符串必须编码为JSON数组。')
                    for message in messages:
                        if not isinstance(message, dict):
                            raise TypeError('messages中的每一项必须是JSON对象。')
                        source_role = message.get('role')
                        weight_role = 'tool_response' if source_role == 'tool' else source_role
                        if weight_role not in config.role_weights:
                            raise ValueError(f'不支持的消息role：{source_role!r}。')
                        if 'loss_scale' in message and not overwrite:
                            raise ValueError('message已包含loss_scale；如需替换请启用overwrite。')
                        message['loss_scale'] = config.role_weights[weight_role]
                        role_counts[weight_role] += 1
                except Exception as e:
                    raise ValueError(f'处理JSONL第{line_no}行失败：{e}') from e
                dst.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
                line_count += 1
            dst.flush()
            os.fsync(dst.fileno())
        # 原子替换前继承源文件权限，避免临时文件的0600权限影响共享训练。
        os.chmod(tmp_path, stat.S_IMODE(os.stat(input_path).st_mode))
        os.replace(tmp_path, output_path)
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return {'lines': line_count, 'roles': role_counts, 'output': output_path}


class RoleLossScale(LossScale):
    """在分词前按消息role和assistant语义片段计算连续loss权重。"""

    is_binary = False

    def __init__(self, config: RoleLossConfig, modifiers: Optional[List[LossScale]] = None):
        """保存基础配置和后续乘法modifier，并声明role策略语义。"""
        super().__init__('default')
        self.base_strategy = 'role'
        self.config = config
        self.modifiers = modifiers or []

    @staticmethod
    def _as_list(context):
        """将单个文本或token序列包装为片段列表，保留已有片段列表。"""
        if not isinstance(context, list) or (context and isinstance(context[0], int)):
            return [context]
        return context

    @staticmethod
    def _item(value, index: int):
        """统一读取标量配置或与content对齐的列表配置。"""
        return value[index] if isinstance(value, list) else value

    def _message_weight(self, message: dict, source_role: str, index: int = 0) -> float:
        """优先读取message权重，缺失时回退到对应role的配置权重。"""
        value = self._item(message.get('loss_scale'), index)
        if value is None:
            value = self.config.role_weights[source_role]
        return float(value)

    def _apply_modifiers(self, contexts: List, weights: List[float], *, query=None) -> Tuple[List, List[float]]:
        """依次应用官方modifier，并将其权重与当前片段权重相乘。"""
        for modifier in self.modifiers:
            new_contexts, new_weights = [], []
            for context, weight in zip(contexts, weights):
                sub_contexts, sub_weights = modifier.get_loss_scale(context, query=query)
                new_contexts.extend(sub_contexts)
                new_weights.extend(weight * sub_weight for sub_weight in sub_weights)
            contexts, weights = new_contexts, new_weights
        return contexts, weights

    def _split_assistant(self, context: str, base_weight: float) -> Tuple[List[str], List[float]]:
        """拆分assistant的think标签、思考内容和回答，并固定屏蔽空think。"""
        # 空think属于框架固定mask，不受用户配置影响。
        empty_match = _EMPTY_THINK_RE.match(context)
        if empty_match:
            contexts = [empty_match.group(0)]
            weights = [0.0]
            if empty_match.end() < len(context):
                contexts.append(context[empty_match.end():])
                weights.append(base_weight * self.config.assistant_weights['answer'])
            return contexts, weights

        has_open = '<think>' in context
        has_close = '</think>' in context
        if not has_open and not has_close:
            return [context], [base_weight * self.config.assistant_weights['answer']]
        if not context.startswith('<think>') or not has_close:
            raise ValueError('assistant思考块格式错误，必须以完整的<think>...</think>开头。')

        close_start = context.find('</think>')
        close_end = close_start + len('</think>')
        contexts = ['<think>', context[len('<think>'):close_start], '</think>']
        weights = [
            base_weight * self.config.assistant_weights['think_tag'],
            base_weight * self.config.assistant_weights['think_content'],
            base_weight * self.config.assistant_weights['think_tag'],
        ]
        if close_end < len(context):
            contexts.append(context[close_end:])
            weights.append(base_weight * self.config.assistant_weights['answer'])
        return contexts, weights

    def _response_is_trainable(self, message: dict) -> bool:
        """判断历史response是否含正权重token，用于复现原框架EOS监督。"""
        contexts = self._as_list(message['content'])
        source_roles = message.get('_source_role', 'assistant')
        losses = message.get('loss')
        for index, context in enumerate(contexts):
            if self._item(losses, index) is False:
                continue
            source_role = self._item(source_roles, index)
            base_weight = self._message_weight(message, source_role, index)
            if base_weight <= 0:
                continue
            if source_role != 'assistant' or not isinstance(context, str):
                return True
            _, weights = self._split_assistant(context, base_weight)
            if any(weight > 0 for weight in weights):
                return True
        return False

    @staticmethod
    def _split_tool_response(context: str, base_weight: float) -> Tuple[List[str], List[float]]:
        """保留tool_response完整块权重，同时屏蔽其外围模板字符。"""
        contexts, weights = [], []
        start = 0
        # 只给真实返回块赋权，ChatML前后缀继续保持0。
        for match in _TOOL_RESPONSE_RE.finditer(context):
            if match.start() > start:
                contexts.append(context[start:match.start()])
                weights.append(0.0)
            contexts.append(match.group(0))
            weights.append(base_weight)
            start = match.end()
        if start < len(context):
            contexts.append(context[start:])
            weights.append(0.0)
        return (contexts, weights) if contexts else ([context], [0.0])

    def __call__(self, context_list: List[str], context_types: List[ContextType], messages: Messages,
                 **kwargs) -> Tuple[List[str], List[float]]:
        """按上下文来源计算基础权重，再展开语义片段并应用modifier。"""
        result_contexts, result_weights = [], []
        round_index = 0
        system_weight = kwargs.get('system_loss_scale')

        # 按模板标注的来源类型取基础权重，固定模板内容始终落入OTHER。
        for context, context_type in zip(context_list, context_types):
            if context_type == ContextType.SYSTEM:
                weight = self.config.role_weights['system'] if system_weight is None else float(system_weight)
                contexts, weights = [context], [weight]
            elif context_type == ContextType.QUERY:
                message = messages[2 * round_index]
                source_role = message.get('_source_role', message['role'])
                source_role = 'tool_response' if source_role == 'tool' else source_role
                weight = self._message_weight(message, source_role)
                if source_role == 'tool_response' and isinstance(context, str):
                    contexts, weights = self._split_tool_response(context, weight)
                else:
                    contexts, weights = [context], [weight]
            elif context_type == ContextType.RESPONSE:
                query = messages[2 * round_index]['content']
                message = messages[2 * round_index + 1]
                source_roles = message.get('_source_role', 'assistant')
                loss = message.get('loss')
                message_contexts = self._as_list(context)
                contexts, weights = [], []
                for index, item in enumerate(message_contexts):
                    source_role = self._item(source_roles, index)
                    base_weight = self._message_weight(message, source_role, index)
                    if self._item(loss, index) is False:
                        base_weight = 0.0
                    if isinstance(item, str) and source_role != 'assistant':
                        empty_match = _EMPTY_THINK_RE.match(item)
                    else:
                        empty_match = None
                    if source_role == 'assistant' and isinstance(item, str):
                        sub_contexts, sub_weights = self._split_assistant(item, base_weight)
                    elif empty_match:
                        sub_contexts, sub_weights = [empty_match.group(0)], [0.0]
                        if empty_match.end() < len(item):
                            sub_contexts.append(item[empty_match.end():])
                            sub_weights.append(base_weight)
                    else:
                        sub_contexts, sub_weights = [item], [base_weight]
                    sub_contexts, sub_weights = self._apply_modifiers(
                        sub_contexts, sub_weights, query=query)
                    contexts.extend(sub_contexts)
                    weights.extend(sub_weights)
                round_index += 1
            elif context_type == ContextType.RESPONSE_SUFFIX:
                # 历史EOS仅在前一response含正权重token时恢复为框架默认权重1。
                previous = messages[2 * (round_index - 1) + 1] if round_index > 0 else None
                weight = 1.0 if previous is not None and self._response_is_trainable(previous) else 0.0
                contexts, weights = [context], [weight]
            elif context_type == ContextType.SUFFIX:
                contexts, weights = [context], [1.0]
            else:
                contexts, weights = [context], [0.0]
            result_contexts.extend(contexts)
            result_weights.extend(weights)
        return result_contexts, result_weights
