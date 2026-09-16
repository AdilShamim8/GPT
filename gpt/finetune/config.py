from dataclasses import dataclass, field
from typing import List
from gpt.config.base import BaseConfig


@dataclass
class LoRAConfig(BaseConfig):
    """Configuration for Low-Rank Adaptation (LoRA) parameter-efficient fine-tuning.

    Attributes:
        r: Low-rank dimension (rank).
        lora_alpha: Scaling factor alpha (effective scaling is alpha / r).
        lora_dropout: Dropout probability applied to low-rank inputs.
        target_modules: List of substring module names to replace with LoRALinear.
    """

    r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj", "c_attn"]
    )

    def validate(self) -> None:
        if self.r <= 0:
            raise ValueError(f"LoRA rank r must be positive, got {self.r}")
        if self.lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be positive, got {self.lora_alpha}")
        if not (0.0 <= self.lora_dropout < 1.0):
            raise ValueError(f"lora_dropout must be in [0.0, 1.0), got {self.lora_dropout}")
