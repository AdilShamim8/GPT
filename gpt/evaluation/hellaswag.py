from typing import Dict, List
import torch
import torch.nn.functional as F
from gpt.model.gpt import GPT
from gpt.tokenizer.base import BaseTokenizer


@torch.no_grad()
def evaluate_multiple_choice_example(
    model: GPT,
    tokenizer: BaseTokenizer,
    prompt: str,
    completions: List[str],
    device: str = "cpu",
) -> int:
    """Evaluate multiple-choice options by log-likelihood of each completion following prompt.

    Returns the index of the predicted option with highest likelihood.
    """
    model.eval()
    model.to(device)

    prompt_ids = tokenizer.encode(prompt)
    best_log_likelihood = float("-inf")
    best_idx = 0

    for idx, completion in enumerate(completions):
        comp_ids = tokenizer.encode(completion)
        full_ids = prompt_ids + comp_ids
        if len(full_ids) > model.config.block_size:
            full_ids = full_ids[-model.config.block_size :]

        x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)

        logits, _, _ = model(x, return_all_logits=True)
        # Log probabilities: (1, T, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)

        # Sum log probabilities only for completion tokens
        comp_len = len(comp_ids)
        target_tokens = y[0, -comp_len:].unsqueeze(1)  # (comp_len, 1)
        selected_log_probs = log_probs[0, -comp_len:, :].gather(1, target_tokens).squeeze(1)

        total_ll = selected_log_probs.sum().item()
        if total_ll > best_log_likelihood:
            best_log_likelihood = total_ll
            best_idx = idx

    return best_idx
