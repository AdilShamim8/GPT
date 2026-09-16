from dataclasses import dataclass
from typing import Optional
from gpt.config.base import BaseConfig


@dataclass
class TrainingConfig(BaseConfig):
    """Configuration for model training and optimization.

    Attributes:
        batch_size: Micro-batch size per optimization step.
        grad_accum_steps: Number of gradient accumulation steps.
        learning_rate: Maximum/peak learning rate.
        min_lr: Minimum learning rate floor after decay.
        weight_decay: L2 weight decay coefficient for 2D weights.
        beta1: AdamW beta1 parameter.
        beta2: AdamW beta2 parameter.
        grad_clip: Maximum gradient norm for clipping (0.0 to disable).
        warmup_iters: Number of linear warmup iterations.
        max_iters: Total number of training iterations.
        eval_interval: Frequency of evaluation runs in iterations.
        eval_iters: Number of batches evaluated during each eval step.
        save_interval: Frequency of checkpoint saving in iterations.
        checkpoint_dir: Directory where model checkpoints will be stored.
        device: Execution target ('cuda', 'cpu', 'mps', or 'auto').
        mixed_precision: Mixed precision mode ('none', 'fp16', 'bf16').
        seed: Random seed for reproducibility.
        log_interval: Frequency of console/metric logging in iterations.
    """

    batch_size: int = 64
    grad_accum_steps: int = 1
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_iters: int = 200
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200
    save_interval: int = 1000
    checkpoint_dir: str = "checkpoints"
    device: str = "auto"
    mixed_precision: str = "none"
    seed: int = 1337
    log_interval: int = 10

    def validate(self) -> None:
        """Validate training configuration parameters."""
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.grad_accum_steps <= 0:
            raise ValueError(f"grad_accum_steps must be positive, got {self.grad_accum_steps}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.min_lr < 0:
            raise ValueError(f"min_lr cannot be negative, got {self.min_lr}")
        if self.max_iters <= 0:
            raise ValueError(f"max_iters must be positive, got {self.max_iters}")
        if self.mixed_precision not in {"none", "fp16", "bf16"}:
            raise ValueError(f"Unsupported mixed_precision '{self.mixed_precision}'.")

    @property
    def total_batch_size(self) -> int:
        """Effective batch size with gradient accumulation."""
        return self.batch_size * self.grad_accum_steps
