from dataclasses import dataclass, field
from typing import List, Optional
from gpt.config.base import BaseConfig


@dataclass
class InferenceConfig(BaseConfig):
    """Configuration for text generation and decoding.

    Attributes:
        max_new_tokens: Maximum number of tokens to generate.
        temperature: Softmax sampling temperature (lower is more deterministic).
        top_k: Top-k filtering threshold (0 to disable).
        top_p: Nucleus sampling probability mass threshold (1.0 to disable).
        min_p: Minimum probability threshold relative to the most likely token.
        repetition_penalty: Multiplier penalty applied to already generated tokens (1.0 = no penalty).
        frequency_penalty: Linear penalty based on frequency of appearance.
        presence_penalty: Flat penalty for presence in context.
        do_sample: If False, performs greedy decoding.
        stream: Whether to stream tokens incrementally.
        stop_tokens: Optional list of token strings that halt generation.
    """

    max_new_tokens: int = 500
    temperature: float = 0.8
    top_k: Optional[int] = 50
    top_p: float = 0.95
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    do_sample: bool = True
    stream: bool = False
    stop_tokens: List[str] = field(default_factory=list)

    def validate(self) -> None:
        """Validate inference configuration parameters."""
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {self.max_new_tokens}")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be strictly positive, got {self.temperature}")
        if self.top_k is not None and self.top_k < 0:
            raise ValueError(f"top_k cannot be negative, got {self.top_k}")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0.0, 1.0], got {self.top_p}")
        if not (0.0 <= self.min_p <= 1.0):
            raise ValueError(f"min_p must be in [0.0, 1.0], got {self.min_p}")
        if self.repetition_penalty <= 0:
            raise ValueError(f"repetition_penalty must be positive, got {self.repetition_penalty}")
