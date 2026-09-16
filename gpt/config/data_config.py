from dataclasses import dataclass
from typing import Optional
from gpt.config.base import BaseConfig


@dataclass
class DataConfig(BaseConfig):
    """Configuration for dataset loading, tokenization, and processing.

    Attributes:
        dataset_path: Path to raw or binary dataset file.
        data_dir: Directory containing datasets.
        tokenizer_type: Tokenizer type ('char', 'bpe', 'tiktoken').
        block_size: Sequence context window length.
        train_split: Fraction of data reserved for training (e.g. 0.9).
        use_memmap: Whether to use memory-mapped numpy binary files for large corpora.
        num_workers: Number of background data loader workers.
        pin_memory: Whether to pin host memory for accelerated CUDA transfers.
    """

    dataset_path: str = "input.txt"
    data_dir: str = "data"
    tokenizer_type: str = "char"
    block_size: int = 256
    train_split: float = 0.9
    use_memmap: bool = False
    num_workers: int = 0
    pin_memory: bool = False

    def validate(self) -> None:
        """Validate data configuration parameters."""
        if not (0.0 < self.train_split < 1.0):
            raise ValueError(f"train_split must be in (0.0, 1.0), got {self.train_split}")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if self.tokenizer_type not in {"char", "bpe", "tiktoken"}:
            raise ValueError(f"Unsupported tokenizer_type '{self.tokenizer_type}'.")
