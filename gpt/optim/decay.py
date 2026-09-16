from typing import List, Tuple
import torch
import torch.nn as nn


def configure_optimizers(
    model: nn.Module,
    weight_decay: float,
    learning_rate: float,
    betas: Tuple[float, float],
    device_type: str = "cpu",
) -> torch.optim.AdamW:
    """Separate parameters into decayed (2D weights) and non-decayed (1D biases, norms).

    All weight tensors in matmuls + embeddings will be weight decayed.
    All biases and layernorms will NOT be weight decayed per standard GPT-2 recipe.
    """
    decay_params = []
    nodecay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # 2D and above parameters will be weight decayed
        if param.dim() >= 2:
            decay_params.append(param)
        else:
            nodecay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    # Use fused AdamW if available on CUDA
    fused_available = "fused" in torch.optim.AdamW.__init__.__code__.co_varnames
    use_fused = fused_available and device_type == "cuda"
    extra_args = dict(fused=True) if use_fused else dict()

    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=betas,
        **extra_args,
    )
    return optimizer
