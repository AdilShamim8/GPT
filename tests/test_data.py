import tempfile
from pathlib import Path
import numpy as np
import torch
from gpt.data import (
    DataCollatorWithPadding,
    InMemoryDataset,
    MemoryMappedDataset,
    deduplicate_documents,
    normalize_whitespace,
)


def test_in_memory_dataset():
    tokens = list(range(100))
    ds = InMemoryDataset(tokens, block_size=10)
    assert len(ds) == 90

    x, y = ds[0]
    assert x.tolist() == list(range(10))
    assert y.tolist() == list(range(1, 11))


def test_memmap_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        bin_path = Path(tmpdir) / "test.bin"
        arr = np.arange(200, dtype=np.uint16)
        arr.tofile(bin_path)

        ds = MemoryMappedDataset(bin_path, block_size=20)
        assert len(ds) == 180

        x, y = ds[5]
        assert x.shape == (20,)
        assert y.shape == (20,)
        assert x[0].item() == 5
        assert y[0].item() == 6
        ds.close()


def test_data_collator():
    collator = DataCollatorWithPadding(pad_token_id=0)
    batch = [
        (torch.tensor([1, 2, 3]), torch.tensor([2, 3, 4])),
        (torch.tensor([5, 6]), torch.tensor([6, 7])),
    ]
    x_pad, y_pad = collator(batch)
    assert x_pad.shape == (2, 3)
    assert y_pad.shape == (2, 3)
    assert x_pad[1, 2].item() == 0  # padded
    assert y_pad[1, 2].item() == -1  # target padded with -1 ignore index


def test_cleaner():
    raw = "  Hello   world!  \r\n\r\n\n  How are you?   "
    clean = normalize_whitespace(raw)
    assert clean == "Hello world!\n\nHow are you?"

    docs = ["doc A", "doc B", "doc A", "doc C", "doc B"]
    deduped = list(deduplicate_documents(docs))
    assert deduped == ["doc A", "doc B", "doc C"]


if __name__ == "__main__":
    test_in_memory_dataset()
    test_memmap_dataset()
    test_data_collator()
    test_cleaner()
    print("ALL DATA TESTS PASSED!")
