from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch
import torch.nn as nn
from gpt.config.model_config import ModelConfig


class BaseModel(nn.Module, ABC):
    """Abstract base class for generative language models."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass computing logits and optional cross-entropy loss."""
        pass

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return the number of parameters in the model.

        Args:
            non_embedding: If True, subtract position and token embedding parameters.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            if hasattr(self, "transformer") and hasattr(self.transformer, "wpe"):
                if self.transformer.wpe is not None:
                    n_params -= self.transformer.wpe.weight.numel()
        return n_params
