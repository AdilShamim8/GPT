#!/usr/bin/env python
"""
Standalone Model Evaluation & Benchmarking CLI for nano-gpt-prod.
Computes perplexity, latency profiles (TTFT, ITL), and exports structured evaluation reports.
"""

import argparse
import json
from pathlib import Path
import torch
from gpt.config.presets import get_preset
from gpt.evaluation.benchmark import benchmark_generation_speed
from gpt.evaluation.perplexity import calculate_perplexity
from gpt.model.gpt import GPT
from gpt.model.serialization import load_checkpoint
from gpt.tokenizer.factory import get_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate GPT Model Performance")
    parser.add_argument("--checkpoint", "-c", type=str, default=None, help="Path to checkpoint .pt")
    parser.add_argument("--preset", "-p", type=str, default="micro", help="Preset name if no checkpoint")
    parser.add_argument("--test_file", "-t", type=str, default="more.txt", help="Text file to compute perplexity")
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to tokenizer JSON")
    parser.add_argument("--output", "-o", type=str, default="eval_report.json", help="Report JSON output path")
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('cuda', 'cpu')")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("nano-gpt-prod :: Model Evaluation Suite")
    print("=" * 60)

    # 1. Load model
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading checkpoint: {args.checkpoint}...")
        model, config, step, _ = load_checkpoint(args.checkpoint, device=device)
    else:
        config = get_preset(args.preset)
        model = GPT(config)
        step = 0
        print(f"Initialized base model with preset '{args.preset}'")

    model.to(device)

    # 2. Load tokenizer
    if args.tokenizer and Path(args.tokenizer).exists():
        tokenizer = get_tokenizer(args.tokenizer)
    else:
        input_txt = Path("input.txt")
        corpus = input_txt.read_text(encoding="utf-8") if input_txt.exists() else "abcdefghijklmnopqrstuvwxyz "
        tokenizer = get_tokenizer("char", text=corpus)

    # 3. Compute Perplexity if text file exists
    test_path = Path(args.test_file)
    ppl = None
    if test_path.exists():
        print(f"Calculating perplexity on {test_path}...")
        test_text = test_path.read_text(encoding="utf-8")
        ppl = calculate_perplexity(model, tokenizer, test_text, device=device)
        print(f"Perplexity: {ppl:.2f}")

    # 4. Run Speed & Latency Benchmark
    print("Running generation speed & latency benchmark...")
    benchmark = benchmark_generation_speed(
        model=model,
        tokenizer=tokenizer,
        num_tokens=50,
        device=device,
    )

    report = {
        "step": step,
        "parameters": model.get_num_params(),
        "perplexity": ppl,
        "benchmark": benchmark.to_dict(),
    }

    print("\n--- Benchmark Summary ---")
    for k, v in benchmark.to_dict().items():
        print(f"  {k}: {v}")

    out_path = Path(args.output)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved evaluation report to {out_path}")


if __name__ == "__main__":
    main()
