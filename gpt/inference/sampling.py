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


def top_p_filtering(logits: torch.Tensor, top_p: float = 0.95) -> torch.Tensor:
    """Nucleus (Top-p) filtering: keeps the cumulative top_p probability mass."""
    if top_p >= 1.0 or top_p <= 0.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift indices to keep first token above threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # Scatter back to original indices
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    return logits.masked_fill(indices_to_remove, float("-inf"))


def min_p_filtering(logits: torch.Tensor, min_p: float = 0.05) -> torch.Tensor:
    """Min-p filtering: drops tokens whose probability is below min_p * max_prob."""
    if min_p <= 0.0:
        return logits

    probs = F.softmax(logits, dim=-1)
    max_probs, _ = torch.max(probs, dim=-1, keepdim=True)
    threshold = max_probs * min_p
    return logits.masked_fill(probs < threshold, float("-inf"))


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    min_p: float = 0.0,
    greedy: bool = False,
) -> torch.Tensor:
    """Combine temperature, top-k, top-p, min-p, and sample next token."""
    if greedy or temperature < 1e-4:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = apply_temperature(logits, temperature)
    if top_k is not None and top_k > 0:
        logits = top_k_filtering(logits, top_k)
    if top_p < 1.0:
        logits = top_p_filtering(logits, top_p)
    if min_p > 0.0:
        logits = min_p_filtering(logits, min_p)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
