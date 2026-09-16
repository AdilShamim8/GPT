import math
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuickGELU(nn.Module):
    """Fast approximation of GELU used in modern vision-language models."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit activation used in LLaMA, PaLM, and modern LLMs."""

    def __init__(self, in_features: int, hidden_features: int, bias: bool = False):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(self.w1(x)) * self.w2(x)


def get_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Retrieve activation module or function by name."""
    name = name.lower()
    if name == "gelu":
        return F.gelu
    elif name == "relu":
        return F.relu
    elif name == "silu" or name == "swish":
        return F.silu
    elif name == "quick_gelu":
        return QuickGELU()
    else:
        raise ValueError(f"Unknown activation function '{name}'")
