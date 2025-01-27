from typing import Callable, Tuple

import torch
from jaxtyping import Float
from transformer_lens.hook_points import HookPoint


def hook_example(
        value: Float[torch.Tensor, "batch pos head_index d_head"],
        hook: HookPoint
) -> Float[torch.Tensor, "batch pos head_index d_head"]:
    value[:, :, 0, :] = value[:, :, 0, :] * 0.
    return value
