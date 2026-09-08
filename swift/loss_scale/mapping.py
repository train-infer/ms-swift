# Copyright (c) ModelScope Contributors. All rights reserved.
from .agent import AgentFlanLossScale, AlphaUmiLossScale, HermesLossScale, QwenLossScale, REACTLossScale
from .base import ALL_BASE_STRATEGY, ConcatLossScale, LossScale
from .other import IgnoreEmptyThinkLossScale, IgnoreThinkPrefixLossScale
from .role import RoleLossScale, load_role_loss_config

# Add your loss scale here, use --loss_scale xxx to train
loss_scale_map = {
    'base': LossScale,
    'ignore_empty_think': IgnoreEmptyThinkLossScale,
    'ignore_think_prefix': IgnoreThinkPrefixLossScale,
    # agent
    'react': REACTLossScale,
    'hermes': HermesLossScale,
    'qwen': QwenLossScale,
    'agentflan': AgentFlanLossScale,
    'alpha_umi': AlphaUmiLossScale,
}


def get_loss_scale(loss_scale: str, role_loss_config: str = None) -> LossScale:
    """Factory function to create a loss scale object from a string specification.

    The loss_scale string supports the following formats (segments separated by '+'):
    1. A strategy name alone (e.g., 'default', 'last_round', 'all') - uses base LossScale
    2. A loss scale type alone (e.g., 'hermes', 'react') - uses 'default' strategy
    3. A strategy name followed by a loss scale type (e.g., 'default+react', 'last_round+qwen')
    4. Multiple loss scale types chained together, optionally led by a base strategy
       (e.g., 'hermes+ignore_empty_think', 'last_round+hermes+ignore_empty_think').
       The chained loss scales are applied sequentially: each loss scale processes the
       output of the previous one and the corresponding weights are multiplied together.

    Args:
        loss_scale: String specifying the loss scale configuration.
        role_loss_config: role策略使用的JSON配置文件路径。

    Returns:
        LossScale: An instance of the appropriate LossScale subclass. When multiple loss
            scale types are specified, a ``ConcatLossScale`` wrapping them is returned.

    Examples:
        >>> get_loss_scale('default')  # Uses default strategy with base LossScale
        >>> get_loss_scale('react')  # Uses default strategy with REACTLossScale
        >>> get_loss_scale('last_round+hermes')  # last_round strategy with HermesLossScale
        >>> get_loss_scale('last_round+hermes+ignore_empty_think')  # chain hermes then ignore_empty_think
    """
    parts = loss_scale.split('+')
    if parts[0] == 'role':
        config = load_role_loss_config(role_loss_config)
        # RoleLossScale在语义拆分前固定屏蔽空think；若对拆分后的片段再次应用正则，
        # 非空思考块的</think>会被错误屏蔽。
        modifier_names = [name for name in parts[1:] if name != 'ignore_empty_think']
        unsupported = set(modifier_names) - {'hermes'}
        if unsupported:
            raise ValueError(f'role loss_scale不支持以下modifier：{sorted(unsupported)}')
        modifiers = [loss_scale_map[name]('default') for name in modifier_names]
        return RoleLossScale(config, modifiers)
    if role_loss_config is not None:
        raise ValueError('仅当loss_scale以role开头时才能设置role_loss_config。')
    if parts[0] in ALL_BASE_STRATEGY:
        base_strategy = parts[0]
        ls_names = parts[1:] or ['base']
    else:
        base_strategy = 'default'
        ls_names = parts
    if len(ls_names) == 1:
        return loss_scale_map[ls_names[0]](base_strategy)
    # The base_strategy is owned by the outer ConcatLossScale; sub loss scales only
    # contribute their `get_loss_scale` (which does not reference base_strategy), so
    # any valid placeholder ('default') is fine here.
    sub_loss_scales = [loss_scale_map[name]('default') for name in ls_names]
    return ConcatLossScale(sub_loss_scales, base_strategy)
