from contextlib import nullcontext
from typing import Any, ContextManager
import torch


class MixedPrecisionManager:
    """Manages Automatic Mixed Precision (AMP) and gradient scaling across CPU, CUDA, and MPS."""

    def __init__(self, mode: str = "none", device_type: str = "cpu"):
        self.mode = mode.lower()
        self.device_type = device_type

        # Determine target dtype
        if self.mode == "fp16":
            self.dtype = torch.float16
        elif self.mode == "bf16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        # Initialize GradScaler only for CUDA fp16
        use_scaler = self.mode == "fp16" and self.device_type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    def autocast_context(self) -> ContextManager[Any]:
        """Return PyTorch autocast context manager for forward pass."""
        if self.mode in {"fp16", "bf16"} and self.device_type in {"cuda", "cpu"}:
            return torch.autocast(device_type=self.device_type, dtype=self.dtype)
        return nullcontext()

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Perform optimizer step through gradient scaler."""
        self.scaler.step(optimizer)
        self.scaler.update()

    def backward(self, loss: torch.Tensor) -> None:
        """Scale and backward loss."""
        self.scaler.scale(loss).backward()

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        """Unscale gradients prior to gradient clipping."""
        self.scaler.unscale_(optimizer)
