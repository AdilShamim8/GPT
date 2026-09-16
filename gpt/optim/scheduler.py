import math
from typing import Callable
import torch
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_iters: int,
    max_iters: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Create learning rate scheduler with linear warmup followed by cosine decay."""

    def lr_lambda(current_iter: int) -> float:
        # 1) Linear warmup
        if current_iter < warmup_iters:
            return float(current_iter) / float(max(1, warmup_iters))
        # 2) If beyond max_iters, return min learning rate
        if current_iter > max_iters:
            return min_lr_ratio
        # 3) In between, cosine decay down to min_lr_ratio
        decay_ratio = float(current_iter - warmup_iters) / float(
            max(1, max_iters - warmup_iters)
        )
        assert 0.0 <= decay_ratio <= 1.0
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr_ratio + coeff * (1.0 - min_lr_ratio)

    return LambdaLR(optimizer, lr_lambda)
