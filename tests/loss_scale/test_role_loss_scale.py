# Copyright (c) ModelScope Contributors. All rights reserved.
import json

import pytest

from swift.loss_scale import get_loss_scale
from swift.loss_scale.role import inject_role_loss_scale_jsonl, load_role_loss_config
from swift.template import ContextType
from swift.template.base import Template


CONFIG = {
    'role_weights': {
        'system': 0.0,
        'user': 0.2,
        'assistant': 1.0,
        'tool_call': 1.5,
        'tool_response': 0.3,
    },
    'assistant_weights': {
        'think_tag': 0.8,
        'think_content': 0.5,
        'answer': 1.0,
    },
}


def _write_config(tmp_path):
    """写入测试所需的统一角色权重配置。"""
    path = tmp_path / 'role_loss.json'
    path.write_text(json.dumps(CONFIG), encoding='utf-8')
    return str(path)


def test_role_loss_config_validation(tmp_path):
    """验证配置加载及非法连续权重拒绝逻辑。"""
    config = load_role_loss_config(_write_config(tmp_path))
    assert config.role_weights['tool_response'] == 0.3
    assert config.assistant_weights['think_content'] == 0.5

    invalid = dict(CONFIG)
    invalid['role_weights'] = dict(CONFIG['role_weights'], assistant=-1)
    path = tmp_path / 'invalid.json'
    path.write_text(json.dumps(invalid), encoding='utf-8')
    with pytest.raises(ValueError):
        load_role_loss_config(str(path))


def test_inject_role_loss_scale_jsonl(tmp_path):
    """验证JSONL流式注入及原始消息字段保留。"""
    config_path = _write_config(tmp_path)
    source = tmp_path / 'input.jsonl'
    output = tmp_path / 'output.jsonl'
    row = {
        'tools': '[]',
        'messages': [{
            'role': role,
            'content': '',
            'reasoning_content': '',
            'tool_call_id': '',
        } for role in CONFIG['role_weights']]
    }
    source.write_text(json.dumps(row) + '\n', encoding='utf-8')

    summary = inject_role_loss_scale_jsonl(str(source), config_path, str(output))
    result = json.loads(output.read_text(encoding='utf-8'))

    assert summary['lines'] == 1
    assert [message['loss_scale'] for message in result['messages']] == list(CONFIG['role_weights'].values())
    assert all(message['reasoning_content'] == '' and message['tool_call_id'] == ''
               for message in result['messages'])


def test_assistant_semantic_weights_and_empty_think(tmp_path):
    """验证assistant语义分段及空think固定mask。"""
    loss_scale = get_loss_scale('role+ignore_empty_think', _write_config(tmp_path))
    messages = [{
        'role': 'user',
        'content': 'question',
        '_source_role': 'user',
        'loss_scale': 0.2,
    }, {
        'role': 'assistant',
        'content': '<think>\nreasoning\n</think>\n\nanswer',
        '_source_role': 'assistant',
        'loss_scale': 1.0,
    }]
    contexts, weights = loss_scale(
        ['system', 'question', messages[1]['content'], '<|im_end|>'],
        [ContextType.SYSTEM, ContextType.QUERY, ContextType.RESPONSE, ContextType.SUFFIX],
        messages,
        system_loss_scale=0.0)

    assert ''.join(contexts) == 'systemquestion<think>\nreasoning\n</think>\n\nanswer<|im_end|>'
    assert weights == [0.0, 0.2, 0.8, 0.5, 0.8, 1.0, 1.0]

    messages[1]['content'] = '<think>\n\n</think>\n\nanswer'
    contexts, weights = loss_scale([messages[1]['content']], [ContextType.RESPONSE], messages)
    assert ''.join(contexts) == messages[1]['content']
    assert weights == [0.0, 1.0]


def test_template_placeholder_boundaries():
    """验证模板固定字符与system/query内容边界不会混淆。"""
    contexts, context_types = [], []
    Template._concat_context_list(
        ['<s>{{SYSTEM}}</s><u>{{QUERY}}</u><a>'],
        contexts,
        context_types,
        system='system',
        query='query')

    assert contexts == ['<s>', 'system', '</s><u>', 'query', '</u><a>']
    assert context_types == [
        ContextType.OTHER,
        ContextType.SYSTEM,
        ContextType.OTHER,
        ContextType.QUERY,
        ContextType.OTHER,
    ]


def test_tool_response_masks_template_wrapper(tmp_path):
    """验证tool_response按role加权且外围模板字符保持mask。"""
    loss_scale = get_loss_scale('role', _write_config(tmp_path))
    messages = [{
        'role': 'tool',
        'content': [],
        '_source_role': 'tool_response',
        'loss_scale': 0.3,
    }, {
        'role': 'assistant',
        'content': 'done',
        '_source_role': 'assistant',
        'loss_scale': 1.0,
    }]
    context = '<|im_start|>user\n<tool_response>\nresult\n</tool_response><|im_end|>\n<|im_start|>assistant\n'
    contexts, weights = loss_scale([context], [ContextType.QUERY], messages)

    assert ''.join(contexts) == context
    assert weights == [0.0, 0.3, 0.0]


def test_direct_tool_call_keeps_injected_empty_think_masked(tmp_path):
    """验证直接tool_call前自动插入的空think不会参与训练。"""
    loss_scale = get_loss_scale('role', _write_config(tmp_path))
    content = '<think>\n\n</think>\n\n<tool_call>call</tool_call>'
    messages = [{
        'role': 'user',
        'content': 'question',
        '_source_role': 'user',
        'loss_scale': 0.2,
    }, {
        'role': 'assistant',
        'content': content,
        '_source_role': 'tool_call',
        'loss_scale': 1.5,
    }]
    contexts, weights = loss_scale([content], [ContextType.RESPONSE], messages)

    assert ''.join(contexts) == content
    assert weights == [0.0, 1.5]


def test_role_composes_with_hermes(tmp_path):
    """验证role基础权重与Hermes工具调用权重按乘法组合。"""
    loss_scale = get_loss_scale('role+hermes', _write_config(tmp_path))
    tool_call = '<tool_call>\n{"name":"search","arguments":{}}\n</tool_call>'
    messages = [{
        'role': 'user',
        'content': 'question',
        '_source_role': 'user',
        'loss_scale': 0.2,
    }, {
        'role': 'assistant',
        'content': tool_call,
        '_source_role': 'tool_call',
        'loss_scale': 1.5,
    }]
    contexts, weights = loss_scale([tool_call], [ContextType.RESPONSE], messages)

    assert contexts == [tool_call]
    assert weights == [3.0]


def test_historical_and_final_response_suffix(tmp_path):
    """验证历史和最终assistant结束符保持框架固定监督语义。"""
    loss_scale = get_loss_scale('role', _write_config(tmp_path))
    messages = [{
        'role': 'user',
        'content': 'q1',
        '_source_role': 'user',
        'loss_scale': 0.2,
    }, {
        'role': 'assistant',
        'content': 'a1',
        '_source_role': 'assistant',
        'loss_scale': 0.4,
    }, {
        'role': 'user',
        'content': 'q2',
        '_source_role': 'user',
        'loss_scale': 0.2,
    }, {
        'role': 'assistant',
        'content': 'a2',
        '_source_role': 'assistant',
        'loss_scale': 0.6,
    }]
    _, weights = loss_scale(
        ['a1', '<|im_end|>', 'q2', 'a2', '<|im_end|>'],
        [ContextType.RESPONSE, ContextType.RESPONSE_SUFFIX, ContextType.QUERY, ContextType.RESPONSE,
         ContextType.SUFFIX], messages)

    assert weights == [0.4, 1.0, 0.2, 0.6, 1.0]
