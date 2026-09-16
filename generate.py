#!/usr/bin/env python
"""
Interactive Text Generation CLI for nano-gpt-prod.
Streams tokens directly to stdout with configurable sampling parameters.
"""

import argparse
import sys
from pathlib import Path
import torch
from gpt.config.presets import get_preset
from gpt.inference.generator import TextGenerator
from gpt.model.gpt import GPT
from gpt.model.serialization import load_checkpoint
from gpt.tokenizer.factory import get_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Generate text using trained GPT model")
    parser.add_argument("--checkpoint", "-c", type=str, default=None, help="Path to checkpoint .pt")
    parser.add_argument("--preset", "-p", type=str, default="nano_shakespeare", help="Preset name")
    parser.add_argument("--tokenizer", "-t", type=str, default="data/shakespeare/tokenizer.json")
    parser.add_argument("--prompt", type=str, default="First Citizen:\n", help="Prompt text")
    parser.add_argument("--max_tokens", "-n", type=int, default=200, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k filtering threshold")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p nucleus threshold")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu')")
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Load model
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading checkpoint from {args.checkpoint}...")
        model, config, step, _ = load_checkpoint(args.checkpoint, device=device)
        print(f"Loaded checkpoint from step {step}")
    else:
        config = get_preset(args.preset)
        model = GPT(config)
        print(f"Initialized untrained model with preset '{args.preset}'")

    # Load tokenizer
    tok_path = Path(args.tokenizer)
    if tok_path.exists():
        tokenizer = get_tokenizer(tok_path)
    else:
        # Fallback to character tokenizer on prompt or input.txt
        input_txt = Path("input.txt")
        corpus = input_txt.read_text(encoding="utf-8") if input_txt.exists() else "abcdefghijklmnopqrstuvwxyz "
        tokenizer = get_tokenizer("char", text=corpus)

    generator = TextGenerator(model=model, tokenizer=tokenizer, device=device)

    print("=" * 60)
    print(f"Generating from prompt: {repr(args.prompt)}")
    print("=" * 60)
    sys.stdout.write(args.prompt)
    sys.stdout.flush()

    for token in generator.generate_stream(
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    ):
        sys.stdout.write(token)
        sys.stdout.flush()

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
