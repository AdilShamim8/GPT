from typing import Optional, Tuple
import torch
import torch.nn as nn
from gpt.config.model_config import ModelConfig
from gpt.model.attention import CausalSelfAttention
from gpt.model.feed_forward import FeedForward
from gpt.model.normalization import get_norm_layer


class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer decoder block."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = get_norm_layer(config.norm_type, config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = get_norm_layer(config.norm_type, config.n_embd, bias=config.bias)
        self.mlp = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-norm Self-Attention with residual connection
        attn_out, present = self.attn(
            self.ln_1(x),
            use_cache=use_cache,
            layer_past=layer_past,
        )
        x = x + attn_out

        # Pre-norm FeedForward with residual connection
        x = x + self.mlp(self.ln_2(x))
        return x, present
