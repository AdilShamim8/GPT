import math
import torch
import torch.nn as nn
from typing import Dict


def analyze_weight_initialization(model: nn.Module) -> Dict[str, Dict[str, float]]:
    """Analyze mean and standard deviation of all parameters across layers."""
    stats = {}
    for name, param in model.named_parameters():
        data = param.data.float()
        stats[name] = {
            "mean": data.mean().item(),
            "std": data.std().item(),
            "min": data.min().item(),
            "max": data.max().item(),
            "numel": data.numel(),
        }
    return stats


def reinitialize_weights(model: nn.Module, n_layer: int) -> None:
    """Apply scaled GPT-2 initialization with residual projection dampening."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            std = 0.02
            if name.endswith("c_proj"):
                std = 0.02 / math.sqrt(2 * n_layer)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
