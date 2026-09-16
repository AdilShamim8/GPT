#!/usr/bin/env python
"""
Export trained GPT model to TorchScript (TorchScript Tracing) for C++ / production runtime.
"""

import argparse
from pathlib import Path
import torch
from gpt.config.presets import get_preset
from gpt.model.gpt import GPT
from gpt.model.serialization import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Export GPT model to TorchScript format")
    parser.add_argument("--checkpoint", "-c", type=str, default=None, help="Path to checkpoint .pt")
    parser.add_argument("--preset", "-p", type=str, default="micro", help="Preset name if no checkpoint")
    parser.add_argument(
        "--output", "-o", type=str, default="weights/model.torchscript.pt", help="Output path"
    )
    args = parser.parse_args()

    if args.checkpoint and Path(args.checkpoint).exists():
        model, config, _, _ = load_checkpoint(args.checkpoint)
    else:
        config = get_preset(args.preset)
        model = GPT(config)

    model.eval()
    dummy_input = torch.randint(0, config.vocab_size, (1, 32), dtype=torch.long)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Tracing model to TorchScript at {out_path}...")
    with torch.no_grad():
        traced = torch.jit.trace(model, dummy_input)
        traced.save(str(out_path))

    print("TorchScript export complete!")


if __name__ == "__main__":
    main()
