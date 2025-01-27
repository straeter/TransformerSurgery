from typing import Callable

import torch
from jaxtyping import Float
from transformer_lens.hook_points import HookPoint
from transformer_lens.utils import get_act_name


def take_action(activation: torch.Tensor, action: str):
    if action == "zero":
        activation = 0.
    elif action == "double":
        activation = activation * 2
    elif action == "flip":
        activation = -activation
    return activation


def get_ablation_hook(
        act_type: str = "v",
        layer_type: str = None,
        layer_idx: int = 0,
        head_idx: int = 0,
        position_list: list = None,
        ablation_type: str = "zero"
) -> tuple[str, Callable]:
    act_name: str = get_act_name(act_type, layer_idx, layer_type)

    def head_ablation_hook(
            value: Float[torch.Tensor, "batch pos head_index d_head"],
            hook: HookPoint
    ) -> Float[torch.Tensor, "batch pos head_index d_head"]:
        if position_list:
            for position in position_list:
                value[:, position, head_idx, :] = take_action(value[:, position, head_idx, :], ablation_type)
        else:
            value[:, :, head_idx, :] = take_action(value[:, :, head_idx, :], ablation_type)

        return value

    def ablation_hook(
            value: Float[torch.Tensor, "batch pos d_model"],
            hook: HookPoint
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        if position_list:
            for position in position_list:
                value[:, position, :] = take_action(value[:, position, :], ablation_type)
        else:
            value[:, :, :] = take_action(value[:, :, :], ablation_type)

        return value

    if act_type in ["key", "query", "value"]:
        return act_name, head_ablation_hook
    else:
        return act_name, ablation_hook


layer_type_alias = [
    "attention",
    "mlp",
    "blocks"
]

act_name_alias = {
    "key": "k",
    "query": "q",
    "value": "v",
    "attn": "pattern",
    "attn_logits": "attn_scores",
    "mlp_pre": "pre",
    "mlp_mid": "mid",
    "mlp_post": "post",
    "resid_pre": "resid_pre",
    "resid_post": "resid_post",
}


def get_activation_types(hook_dict: dict, layer_idx: int) -> list[str]:
    act_types = [key for key in act_name_alias.keys() if get_act_name(key, layer_idx) in hook_dict]

    return act_types
