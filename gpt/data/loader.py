from typing import Optional, Tuple
import torch
from torch.utils.data import DataLoader
from gpt.data.base import BaseDataset
from gpt.data.collator import DataCollatorWithPadding


def create_dataloader(
    dataset: BaseDataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    pad_token_id: Optional[int] = None,
) -> DataLoader:
    """Build optimized PyTorch DataLoader for training or evaluation."""
    collator = DataCollatorWithPadding(pad_token_id=pad_token_id) if pad_token_id is not None else None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
        drop_last=True,
    )
    return loader


def get_batch_from_dataset(
    dataset: BaseDataset,
    batch_size: int,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a random micro-batch directly from dataset on target device."""
    indices = torch.randint(0, len(dataset), (batch_size,))
    xs, ys = [], []
    for idx in indices:
        x, y = dataset[idx.item()]
        xs.append(x)
        ys.append(y)
    x_batch = torch.stack(xs).to(device)
    y_batch = torch.stack(ys).to(device)
    return x_batch, y_batch
