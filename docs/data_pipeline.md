# Scalable Data Pipeline in nano-gpt-prod

The data module provides two distinct paradigms for loading tokenized text sequences:

## 1. In-Memory Loading (`InMemoryDataset`)
- **Best for:** Small to medium corpora (< 100MB) such as Tiny Shakespeare.
- **Mechanism:** Holds the full integer token array in RAM as a PyTorch tensor. Slices subsequences of length `block_size` on the fly.
- **Fast split:** Automatically partitions sequences into train and validation sets with a user-configured ratio.

## 2. Memory-Mapped Binary Datasets (`MemoryMappedDataset`)
- **Best for:** Massive, multi-gigabyte or billion-token datasets (e.g. OpenWebText, FineWeb-Edu, Wikipedia).
- **Mechanism:** Uses `numpy.memmap` over raw binary files (`train.bin`, `val.bin`) with `uint16` encoding.
- **Key Advantage:** Consumes negligible RAM overhead regardless of dataset size; the operating system virtual memory manager pages in chunks on-demand.

## Binary Dataset Preparation

To compile raw text files into memory-mapped binaries, run:

```bash
# Prepare Shakespeare
python scripts/prepare_shakespeare.py --data_dir data/shakespeare

# Prepare custom text corpus
python scripts/prepare_dataset.py \
  --input my_corpus.txt \
  --output_dir data/my_dataset \
  --type bpe \
  --vocab_size 5000 \
  --split 0.95
```

## Batching and Sequence Collation

When sequences have variable lengths, `DataCollatorWithPadding` dynamically pads input tensors to the maximum length of the batch with `pad_token_id`, and pads targets with `-1` (PyTorch CrossEntropyLoss ignore index).
