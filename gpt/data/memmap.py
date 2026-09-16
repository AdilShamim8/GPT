from pathlib import Path
from typing import Tuple, Union
import numpy as np
import torch
from gpt.data.base import BaseDataset


class MemoryMappedDataset(BaseDataset):
    """Memory-mapped binary dataset for zero-copy streaming of large-scale token files."""

    def __init__(
        self,
        bin_path: Union[str, Path],
        block_size: int,
        dtype: np.dtype = np.uint16,
    ):
        self.bin_path = Path(bin_path)
        if not self.bin_path.exists():
            raise FileNotFoundError(f"Binary dataset file does not exist: {self.bin_path}")

        self.block_size = block_size
        self.dtype = dtype
        self.data = np.memmap(str(self.bin_path), dtype=dtype, mode="r")

        if len(self.data) <= block_size:
            raise ValueError(
                f"Dataset length ({len(self.data)}) must be strictly greater than block_size ({block_size})"
            )

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Fast zero-copy slicing from memory map
        chunk = self.data[idx : idx + self.block_size + 1].astype(np.int64)
        x = torch.from_numpy(chunk[: self.block_size])
        y = torch.from_numpy(chunk[1 : self.block_size + 1])
        return x, y

    def close(self) -> None:
        """Close memory mapped file handle (important for Windows OS file releases)."""
        if hasattr(self, "data") and hasattr(self.data, "_mmap") and self.data._mmap is not None:
            self.data._mmap.close()
            del self.data
