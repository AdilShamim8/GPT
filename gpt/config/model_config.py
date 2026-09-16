from dataclasses import dataclass
from typing import Optional
from gpt.config.base import BaseConfig


@dataclass
class ModelConfig(BaseConfig):
    """Configuration for GPT Transformer Architecture.

    Attributes:
        vocab_size: Size of vocabulary.
        block_size: Maximum sequence context length.
        n_layer: Number of transformer blocks.
        n_head: Number of attention query heads.
        n_kv_head: Number of key/value heads for GQA/MQA. If None, equals n_head.
        n_embd: Dimensionality of embeddings and hidden states.
        dropout: Dropout probability.
        bias: Whether to use bias in linear and layer norm layers.
        norm_type: Normalization type ('layernorm' or 'rmsnorm').
        activation: Activation function ('gelu', 'relu', 'swiglu', 'quick_gelu').
        use_rope: Whether to use Rotary Position Embeddings (RoPE) instead of learned positional embeddings.
        tie_weights: Whether to tie token embedding weights with output LM head.
    """

    vocab_size: int = 50257
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: Optional[int] = None
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    norm_type: str = "layernorm"
    activation: str = "gelu"
    use_rope: bool = False
    tie_weights: bool = True

    def validate(self) -> None:
        """Validate model configuration parameters."""
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if self.n_layer <= 0:
            raise ValueError(f"n_layer must be positive, got {self.n_layer}")
        if self.n_head <= 0:
            raise ValueError(f"n_head must be positive, got {self.n_head}")
        if self.n_embd <= 0:
            raise ValueError(f"n_embd must be positive, got {self.n_embd}")
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        if self.n_kv_head is not None:
            if self.n_kv_head <= 0:
                raise ValueError(f"n_kv_head must be positive, got {self.n_kv_head}")
            if self.n_head % self.n_kv_head != 0:
                raise ValueError(
                    f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})"
                )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}")
        if self.norm_type not in {"layernorm", "rmsnorm"}:
            raise ValueError(f"Unsupported norm_type '{self.norm_type}'. Must be 'layernorm' or 'rmsnorm'.")
        if self.activation not in {"gelu", "relu", "swiglu", "quick_gelu"}:
            raise ValueError(f"Unsupported activation '{self.activation}'.")

    @property
    def head_size(self) -> int:
        """Dimensionality of each attention head."""
        return self.n_embd // self.n_head
