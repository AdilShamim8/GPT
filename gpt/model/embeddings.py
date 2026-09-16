import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Token embedding layer mapping discrete token indices to dense vectors."""

    def __init__(self, vocab_size: int, n_embd: int):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, n_embd)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.wte(idx)


class PositionalEmbedding(nn.Module):
    """Learned absolute positional embedding layer (standard in GPT-2)."""

    def __init__(self, block_size: int, n_embd: int):
        super().__init__()
        self.wpe = nn.Embedding(block_size, n_embd)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        return self.wpe(pos)
