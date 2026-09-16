#!/usr/bin/env python
"""
Automated downloader and preprocessor for Tiny Shakespeare dataset.
Compiles train.bin and val.bin for ultra-fast memory-mapped loading.
"""

import argparse
from pathlib import Path
import numpy as np
import requests
from gpt.tokenizer.char_tokenizer import CharTokenizer

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def prepare(data_dir: Path, split: float = 0.9) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    input_file = data_dir / "input.txt"

    # Download if not present locally
    if not input_file.exists():
        local_input = Path("input.txt")
        if local_input.exists():
            print("Found local input.txt, copying...")
            input_file.write_text(local_input.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            print(f"Downloading Tiny Shakespeare from {SHAKESPEARE_URL}...")
            resp = requests.get(SHAKESPEARE_URL, timeout=30)
            input_file.write_text(resp.text, encoding="utf-8")

    text = input_file.read_text(encoding="utf-8")
    print(f"Dataset length: {len(text):,} characters")

    # Fit character tokenizer
    tokenizer = CharTokenizer.from_text(text)
    tokenizer_path = data_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)
    print(f"Saved tokenizer with vocab size {tokenizer.vocab_size} to {tokenizer_path}")

    # Encode full dataset
    encoded = np.array(tokenizer.encode(text), dtype=np.uint16)
    n = int(len(encoded) * split)
    train_ids = encoded[:n]
    val_ids = encoded[n:]

    train_path = data_dir / "train.bin"
    val_path = data_dir / "val.bin"

    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    print(f"Successfully compiled binary datasets:")
    print(f"  Train: {len(train_ids):,} tokens -> {train_path} ({train_path.stat().st_size / 1024:.1f} KB)")
    print(f"  Val:   {len(val_ids):,} tokens -> {val_path} ({val_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Shakespeare binary dataset")
    parser.add_argument("--data_dir", type=str, default="data/shakespeare", help="Directory for output")
    args = parser.parse_args()
    prepare(Path(args.data_dir))
