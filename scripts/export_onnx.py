#!/usr/bin/env python
"""
Export trained GPT model to ONNX format for high-throughput serving and edge deployment.
"""

import argparse
from pathlib import Path
import torch
from gpt.config.presets import get_preset
from gpt.model.gpt import GPT
from gpt.model.serialization import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Export GPT model to ONNX format")
    parser.add_argument("--checkpoint", "-c", type=str, default=None, help="Path to checkpoint .pt")
    parser.add_argument("--preset", "-p", type=str, default="micro", help="Preset name if no checkpoint")
    parser.add_argument("--output", "-o", type=str, default="weights/model.onnx", help="Output path")
    parser.add_argument("--seq_len", type=int, default=64, help="Dummy input sequence length")
    args = parser.parse_args()

    if args.checkpoint and Path(args.checkpoint).exists():
        model, config, _, _ = load_checkpoint(args.checkpoint)
    else:
        config = get_preset(args.preset)
        model = GPT(config)

    model.eval()
    dummy_input = torch.randint(0, config.vocab_size, (1, args.seq_len), dtype=torch.long)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting model to ONNX at {out_path}...")
    torch.onnx.export(
        model,
        (dummy_input,),
        str(out_path),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch_size", 1: "sequence_length"}},
        opset_version=14,
    )
    print("ONNX export complete!")


if __name__ == "__main__":
    main()
