import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """LayerNorm with optional learnable bias (PyTorch's default LayerNorm always has bias)."""

    def __init__(self, ndim: int, bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.layer_norm(
            x,
            self.weight.shape,
            self.weight,
            self.bias,
            self.eps,
        )


class RMSNorm(nn.Module):
    """Root Mean Square Normalization (RMSNorm) used in LLaMA, Mistral, and Gemma."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def get_norm_layer(norm_type: str, dim: int, bias: bool = True, eps: float = 1e-5) -> nn.Module:
    """Factory helper for normalization layers."""
    if norm_type == "layernorm":
        return LayerNorm(dim, bias=bias, eps=eps)
    elif norm_type == "rmsnorm":
        return RMSNorm(dim, eps=eps)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")
