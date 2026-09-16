from typing import Dict, List
import torch


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: List[int],
    penalty: float = 1.2,
) -> torch.Tensor:
    """Apply repetition penalty to logits for tokens that have already appeared."""
    if penalty == 1.0 or not generated_tokens:
        return logits

    unique_tokens = list(set(generated_tokens))
    for tok in unique_tokens:
        if tok < logits.size(-1):
            val = logits[..., tok]
            logits[..., tok] = torch.where(val > 0, val / penalty, val * penalty)
    return logits


def apply_frequency_presence_penalty(
    logits: torch.Tensor,
    token_counts: Dict[int, int],
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
) -> torch.Tensor:
    """Apply frequency and presence penalties (matching OpenAI API specification)."""
    if frequency_penalty == 0.0 and presence_penalty == 0.0:
        return logits

    for tok, count in token_counts.items():
        if tok < logits.size(-1):
            logits[..., tok] -= (frequency_penalty * count) + (presence_penalty * (1.0 if count > 0 else 0.0))
    return logits
