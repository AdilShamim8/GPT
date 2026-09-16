from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generator, List, Optional
import torch


@dataclass
class GenerationResult:
    """Stores the generated token IDs, decoded text, and generation latency."""

    text: str
    token_ids: List[int]
    num_generated: int
    elapsed_seconds: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        return self.num_generated / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0


class BaseGenerator(ABC):
    """Abstract interface for text generation engines."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: float = 0.95,
    ) -> GenerationResult:
        pass

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: float = 0.95,
    ) -> Generator[str, None, None]:
        pass
