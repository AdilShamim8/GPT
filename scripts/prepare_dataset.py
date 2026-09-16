#!/usr/bin/env python
"""
General-purpose binary dataset compiler.
Converts arbitrary text files into memory-mapped train.bin and val.bin files using Char or BPE tokenizers.
"""

import argparse
from pathlib import Path
import numpy as np
from gpt.tokenizer.factory import get_tokenizer


def compile_dataset(
    input_path: Path,
    output_dir: Path,
    tokenizer_type: str = "bpe",
    vocab_size: int = 1000,
    split_ratio: float = 0.9,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading {input_path}...")
    text = input_path.read_text(encoding="utf-8")
    print(f"Loaded {len(text):,} characters.")

    print(f"Building/loading tokenizer ({tokenizer_type})...")
    tokenizer = get_tokenizer(tokenizer_type, text=text, vocab_size=vocab_size)
    tok_path = output_dir / "tokenizer.json"
    tokenizer.save(tok_path)
    print(f"Saved tokenizer with vocab size {tokenizer.vocab_size} to {tok_path}")

    print("Encoding corpus into tokens...")
    tokens = tokenizer.encode(text)
    print(f"Total token count: {len(tokens):,}")

    tokens_arr = np.array(tokens, dtype=np.uint16 if tokenizer.vocab_size < 65535 else np.uint32)
    n = int(len(tokens_arr) * split_ratio)
    train_ids = tokens_arr[:n]
    val_ids = tokens_arr[n:]

    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"

    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    print(f"Saved {len(train_ids):,} train tokens to {train_path}")
    print(f"Saved {len(val_ids):,} val tokens to {val_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile raw text corpus to binary memmap format")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input text file")
    parser.add_argument("--output_dir", "-o", type=str, default="data/custom", help="Output directory")
    parser.add_argument("--type", "-t", choices=["char", "bpe"], default="bpe", help="Tokenizer algorithm")
    parser.add_argument("--vocab_size", "-v", type=int, default=1000, help="Target vocabulary size")
    parser.add_argument("--split", "-s", type=float, default=0.9, help="Train/val split ratio")
    args = parser.parse_args()

    compile_dataset(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        tokenizer_type=args.type,
        vocab_size=args.vocab_size,
        split_ratio=args.split,
    )
