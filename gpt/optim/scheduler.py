import math
from typing import Callable, Optional
import torch
from torch.optim.lr_scheduler import LambdaLR, StepLR, OneCycleLR
from gpt.config.train_config import TrainingConfig


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_iters: int,
    max_iters: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Create learning rate scheduler with linear warmup followed by cosine decay."""

    def lr_lambda(current_iter: int) -> float:
        if current_iter < warmup_iters:
            return float(current_iter) / float(max(1, warmup_iters))
        if current_iter > max_iters:
            return min_lr_ratio
        decay_ratio = float(current_iter - warmup_iters) / float(
            max(1, max_iters - warmup_iters)
        )
        assert 0.0 <= decay_ratio <= 1.0
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr_ratio + coeff * (1.0 - min_lr_ratio)

    return LambdaLR(optimizer, lr_lambda)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    scheduler_type: str = "cosine",
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Factory to construct learning rate scheduler based on config."""
    min_ratio = config.min_lr / config.learning_rate if config.learning_rate > 0 else 0.1

    if scheduler_type.lower() == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            warmup_iters=config.warmup_iters,
            max_iters=config.max_iters,
            min_lr_ratio=min_ratio,
        )
    elif scheduler_type.lower() == "step":
        return StepLR(optimizer, step_size=max(1, config.max_iters // 4), gamma=0.5)
    elif scheduler_type.lower() == "constant":
        return None
    else:
        raise ValueError(f"Unknown scheduler_type '{scheduler_type}'")
