#!/usr/bin/env python
"""
Parameter-Efficient Fine-Tuning (LoRA) CLI script for nano-gpt-prod.
Adapts pretrained or base checkpoints on instruction datasets or custom text corpora.
"""

import argparse
from pathlib import Path
import torch
from gpt.config.train_config import TrainingConfig
from gpt.finetune.config import LoRAConfig
from gpt.finetune.dataset import InstructionDataset
from gpt.finetune.injection import apply_lora_to_model
from gpt.finetune.serialization import merge_lora_weights, save_lora_weights
from gpt.model.serialization import load_checkpoint
from gpt.tokenizer.factory import get_tokenizer
from gpt.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune GPT with LoRA")
    parser.add_argument("--checkpoint", "-c", type=str, required=True, help="Base model checkpoint .pt")
    parser.add_argument("--data", "-d", type=str, required=True, help="Instruction JSONL dataset")
    parser.add_argument("--output_dir", "-o", type=str, default="checkpoints/lora_adapter")
    parser.add_argument("--rank", "-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--alpha", "-a", type=float, default=16.0, help="LoRA alpha scaling")
    parser.add_argument("--epochs", type=int, default=3, help="Fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=2e-4, help="Fine-tuning learning rate")
    parser.add_argument("--merge", action="store_true", help="Merge LoRA weights back into base checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("nano-gpt-prod :: LoRA Fine-Tuning Engine")
    print("=" * 60)

    # 1. Load Base Model
    print(f"Loading base checkpoint: {args.checkpoint}...")
    model, model_cfg, _, _ = load_checkpoint(args.checkpoint, device=device)

    # 2. Inject LoRA Adapters
    lora_cfg = LoRAConfig(r=args.rank, lora_alpha=args.alpha)
    model, trainable_p, total_p = apply_lora_to_model(model, lora_cfg)

    # 3. Load Tokenizer & Dataset
    tokenizer = get_tokenizer("char", text=open("input.txt", "r", encoding="utf-8").read())
    print(f"Loading instruction dataset from {args.data}...")
    dataset = InstructionDataset.from_jsonl(args.data, tokenizer, block_size=model_cfg.block_size)
    print(f"Loaded {len(dataset)} instruction pairs.")

    # 4. Train LoRA
    train_cfg = TrainingConfig(
        learning_rate=args.lr,
        max_iters=len(dataset) * args.epochs,
        batch_size=min(16, len(dataset)),
        checkpoint_dir=args.output_dir,
    )

    trainer = Trainer(model=model, config=train_cfg, train_dataset=dataset)
    trainer.train()

    # 5. Save LoRA Adapter
    out_path = Path(args.output_dir) / "adapter.pt"
    save_lora_weights(model, lora_cfg, out_path)

    # 6. Optional Merge
    if args.merge:
        print("Merging LoRA weights into base model...")
        merge_lora_weights(model)
        merged_path = Path(args.output_dir) / "merged_model.pt"
        torch.save({"model_state_dict": model.state_dict(), "config": model_cfg.to_dict()}, merged_path)
        print(f"Saved merged standalone checkpoint to {merged_path}")


if __name__ == "__main__":
    main()
