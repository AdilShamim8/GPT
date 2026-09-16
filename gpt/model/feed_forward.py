import torch
import torch.nn as nn
import torch.nn.functional as F
from gpt.config.model_config import ModelConfig
from gpt.model.activations import get_activation


class FeedForward(nn.Module):
    """Multi-Layer Perceptron (MLP) block for Transformer models."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.activation_type = config.activation

        if config.activation == "swiglu":
            # SwiGLU requires 2/3 * 4 = 8/3 hidden dim to match parameter count
            hidden_dim = int(2 * (4 * config.n_embd) / 3)
            # Round to multiple of 256 for tensor core efficiency
            hidden_dim = 256 * ((hidden_dim + 256 - 1) // 256)

            self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
            self.w2 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
            self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        else:
            self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            self.act = get_activation(config.activation)
            self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_type == "swiglu":
            x = F.silu(self.w1(x)) * self.w2(x)
            x = self.c_proj(x)
        else:
            x = self.c_fc(x)
            x = self.act(x)
            x = self.c_proj(x)
        return self.dropout(x)
