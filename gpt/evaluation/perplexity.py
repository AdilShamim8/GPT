import math
from typing import Optional
import torch
import torch.nn.functional as F
from gpt.model.gpt import GPT
from gpt.tokenizer.base import BaseTokenizer


@torch.no_grad()
def calculate_perplexity(
    model: GPT,
    tokenizer: BaseTokenizer,
    text: str,
    device: str = "cpu",
    max_length: Optional[int] = None,
    stride: int = 512,
) -> float:
    """Calculate exact perplexity over a continuous text document using a sliding window."""
    model.eval()
    model.to(device)

    encodings = tokenizer.encode(text)
    seq_len = len(encodings)
    max_len = max_length or model.config.block_size

    if seq_len < 2:
        return 1.0

    nlls = []
    for i in range(0, seq_len, stride):
        begin_loc = max(i + stride - max_len, 0)
        end_loc = min(i + stride, seq_len)
        trg_len = end_loc - i  # Length of target sequence for this step

        input_ids = torch.tensor([encodings[begin_loc:end_loc]], dtype=torch.long, device=device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -1  # Mask out context overlap

        logits, loss, _ = model(input_ids[:, :-1], targets=target_ids[:, 1:])
        if loss is not None and not torch.isnan(loss):
            nlls.append(loss.item() * trg_len)

    total_tokens = sum(min(seq_len - i, stride) for i in range(0, seq_len, stride))
    if not nlls or total_tokens == 0:
        return 1.0

    avg_nll = sum(nlls) / total_tokens
    return math.exp(min(avg_nll, 100.0))
