#!/usr/bin/env python
"""
CLI script to train Byte-Pair Encoding (BPE) or Character tokenizers on text files.
"""

import argparse
from pathlib import Path
from gpt.tokenizer import BPETokenizer, CharTokenizer


def main():
    parser = argparse.ArgumentParser(description="Train custom tokenizer on text corpus")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input text file")
    parser.add_argument(
        "--output", "-o", type=str, default="data/tokenizer.json", help="Path to save tokenizer JSON"
    )
    parser.add_argument(
        "--type", "-t", choices=["bpe", "char"], default="bpe", help="Tokenizer algorithm"
    )
    parser.add_argument(
        "--vocab_size", "-v", type=int, default=1000, help="Target vocabulary size (for BPE)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    print(f"Reading text from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    print(f"Loaded {len(text):,} characters.")

    if args.type == "char":
        print("Training character tokenizer...")
        tokenizer = CharTokenizer.from_text(text)
    else:
        print(f"Training Byte-Pair Encoding (BPE) tokenizer (target vocab size {args.vocab_size})...")
        tokenizer = BPETokenizer.train_from_text(text, vocab_size=args.vocab_size)

    output_path = Path(args.output)
    tokenizer.save(output_path)
    print(f"Successfully saved tokenizer with vocab size {tokenizer.vocab_size} to {output_path}")

    # Validation check
    sample_text = "First Citizen:\nBefore we proceed any further, hear me speak."
    ids = tokenizer.encode(sample_text)
    decoded = tokenizer.decode(ids)
    print(f"Validation roundtrip test: {'SUCCESS' if decoded == sample_text else 'MISMATCH'}")
    print(f"Sample length: {len(sample_text)} chars -> {len(ids)} tokens (compression: {len(sample_text)/len(ids):.2f}x)")


if __name__ == "__main__":
    main()
