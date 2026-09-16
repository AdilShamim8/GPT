from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Union


class BaseTokenizer(ABC):
    """Abstract base class for all tokenizers."""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return the total number of tokens in the vocabulary."""
        pass

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode string to list of token IDs."""
        pass

    @abstractmethod
    def decode(self, ids: List[int]) -> str:
        """Decode list of token IDs back to string."""
        pass

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Serialize tokenizer state and vocabulary to file/directory."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Union[str, Path]) -> "BaseTokenizer":
        """Deserialize tokenizer from file/directory."""
        pass

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(vocab_size={self.vocab_size})"
