from typing import List, Tuple, Union
import torch
from gpt.data.base import BaseDataset
from gpt.tokenizer.base import BaseTokenizer


class InMemoryDataset(BaseDataset):
    """Dataset storing token IDs in memory for fast iteration on small-to-medium corpora."""

    def __init__(self, data: Union[torch.Tensor, List[int]], block_size: int):
        if not isinstance(data, torch.Tensor):
            self.data = torch.tensor(data, dtype=torch.long)
        else:
            self.data = data.to(dtype=torch.long)

        self.block_size = block_size
        if len(self.data) <= block_size:
            raise ValueError(
                f"Data length ({len(self.data)}) must be strictly greater than block_size ({block_size})"
            )

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y

    @classmethod
    def from_text(
        cls,
        text: str,
        tokenizer: BaseTokenizer,
        block_size: int,
        split_ratio: float = 0.9,
    ) -> Tuple["InMemoryDataset", "InMemoryDataset"]:
        """Encode raw text with tokenizer and return (train_dataset, val_dataset)."""
        tokens = tokenizer.encode(text)
        n = int(len(tokens) * split_ratio)
        train_data = tokens[:n]
        val_data = tokens[n:]
        return cls(train_data, block_size), cls(val_data, block_size)
