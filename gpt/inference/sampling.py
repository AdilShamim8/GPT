from typing import Optional
import torch
import torch.nn.functional as F


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Scale logits by temperature (t -> 0: greedy / argmax, t > 1: more uniform)."""
    if temperature <= 0.0:
        raise ValueError(f"Temperature must be strictly positive, got {temperature}")
    return logits / temperature


def top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Filter logits leaving only top_k candidates, setting remaining positions to -inf."""
    if top_k <= 0 or top_k >= logits.size(-1):
        return logits
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    pivot = v[:, [-1]]
    return torch.where(logits < pivot, torch.tensor(float("-inf"), device=logits.device), logits)
