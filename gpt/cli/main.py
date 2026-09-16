#!/usr/bin/env python
"""
Unified CLI Entrypoint for nano-gpt-prod.
Dispatches commands: train, generate, serve, evaluate, finetune, export.
"""

import sys
import subprocess


def print_help():
    print("nano-gpt-prod :: Unified Command-Line Interface\n")
    print("Usage: gpt <command> [options]\n")
    print("Available Commands:")
    print("  train       Train a GPT model using a YAML config or preset")
    print("  generate    Sample text completions with streaming stdout")
    print("  serve       Launch OpenAI-compatible REST API server & Web Studio")
    print("  evaluate    Compute perplexity, TTFT, and generation benchmarks")
    print("  finetune    Fine-tune model using Low-Rank Adaptation (LoRA)")
    print("  export      Export checkpoint to ONNX or TorchScript")
    print("  help        Show this help message\n")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help", "help"}:
        print_help()
        sys.exit(0)

    cmd = sys.argv[1].lower()
    args = sys.argv[2:]

    scripts = {
        "train": "train.py",
        "generate": "generate.py",
        "serve": "serve.py",
        "evaluate": "evaluate.py",
        "finetune": "finetune.py",
        "export": "scripts/export_onnx.py",
    }

    if cmd not in scripts:
        print(f"Error: Unknown command '{cmd}'.")
        print_help()
        sys.exit(1)

    target_script = scripts[cmd]
    cmd_line = [sys.executable, target_script] + args
    res = subprocess.run(cmd_line)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
