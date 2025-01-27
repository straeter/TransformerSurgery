from typing import Callable

import torch
from jaxtyping import Float
from transformer_lens.hook_points import HookPoint


def take_action(activation: torch.Tensor, action: str):
    if action == "zero":
        activation = 0.
    elif action == "double":
        activation = activation * 2
    elif action == "flip":
        activation = -activation
    return activation


def get_ablation_hook(
        act_type: str,
        act_name: str,
        head_idx: int = None,
        position_list: list = None,
        ablation_type: str = "zero"
) -> Callable:

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

    if act_type == "attention" and act_name != "attention output":
        return head_ablation_hook
    else:
        return ablation_hook


layer_type_alias = [
    "attention",
    "mlp",
    "blocks"
]

act_aliases = {
    "attention": {
        "attention input": "hook_attn_in",
        "query input": "hook_q_input",
        "key input": "hook_k_input",
        "value input": "hook_v_input",
        "query": "attn.hook_q",
        "key": "attn.hook_k",
        "value": "attn.hook_v",
        "z": "attn.hook_z",
        "attn_scores": "attn.hook_attn_scores",
        "attn_pattern": "attn.hook_pattern",
        "attn_result": "attn.hook_attn_result",
        "attention output": "hook_attn_out",
    },
    "mlp": {
        "mlp_input": "hook_mlp_in",
        "mlp_output": "hook_mlp_out",
        "mlp_pre": "mlp.hook_pre",
        "mlp_mid": "mlp.hook_mid",
        "mlp_post": "mlp.hook_post",
    },
    "residual stream":
        {
            "resid_pre": "hook_resid_pre",
            "resid_mid": "hook_resid_mid",
            "resid_post": "hook_resid_post",
        },
    "embedding": {
        "token embed": "hook_embed",
        "positional embed": "hook_pos_embed",
    },
    "layer norm": {
        "ln1_scale": "ln1.hook_scale",
        "ln1_normalized": "ln1.hook_normalized",
        "ln2_scale": "ln2.hook_scale",
        "ln2_normalized": "ln2.hook_normalized",
    }

}


def get_activation_aliases(act_type, hook_dict: dict) -> list[str]:
    activations = [key for key, value in act_aliases[act_type].items() if any([value in k for k in hook_dict.keys()])]

    return activations


def get_layer_indices(act_type: str, act_name: str, hook_dict: dict):
    if "embed" in act_name:
        return None
    hook_keys = [key for key in hook_dict if act_aliases[act_type][act_name] in key and "blocks." in key]
    layer_indices = sorted([int(key.split(".")[1]) for key in hook_keys])

    return layer_indices


def get_hook_name(hook_dict: dict, act_alias: str, layer_idx: int = None) -> str:
    if "blocks." in act_alias or "embed" in act_alias:
        hook_name = act_alias
    else:
        if layer_idx is None:
            raise ValueError("Layer index must be provided for block activations")
        hook_name = f"blocks.{layer_idx}.{act_alias}"
    if hook_name not in hook_dict:
        raise ValueError(f"Hook {hook_name} not found in model")

    return hook_name
