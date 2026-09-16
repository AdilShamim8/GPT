from abc import ABC, abstractmethod
from typing import Tuple
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """Abstract base class for language model sequence datasets."""

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (x, y) input sequence and target sequence shifted by 1."""
        pass
