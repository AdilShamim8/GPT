import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Low-Rank Adaptation (LoRA) linear wrapper around a frozen base linear layer."""

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        # Freeze base layer weights
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # Trainable low-rank decomposition matrices: B @ A
        self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))

        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        # Initialize A with Gaussian and B with zero so initially Delta W == 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base frozen forward pass
        result = self.base_layer(x)

        # Low-rank adapted pathway: (x @ A.T) @ B.T * scaling
        lora_out = (self.lora_dropout(x) @ self.lora_A.T) @ self.lora_B.T
        result = result + lora_out * self.scaling
        return result

    def merge_weights(self) -> None:
        """Merge delta weights B @ A directly into base layer and disable adaptation."""
        with torch.no_grad():
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.add_(delta_w)
