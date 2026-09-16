#!/usr/bin/env python
"""
Production Training Script for nano-gpt-prod.
Supports YAML configuration profiles, CLI flag overrides, memmap binary datasets, and AMP.
"""

import argparse
from pathlib import Path
import torch
from gpt.config.data_config import DataConfig
from gpt.config.model_config import ModelConfig
from gpt.config.presets import get_preset
from gpt.config.serialization import load_full_experiment_config
from gpt.config.train_config import TrainingConfig
from gpt.data.in_memory import InMemoryDataset
from gpt.data.memmap import MemoryMappedDataset
from gpt.model.gpt import GPT
from gpt.tokenizer.factory import get_tokenizer
from gpt.training.logger import TrainingLogger
from gpt.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT Language Model")
    parser.add_argument("--config", "-c", type=str, default=None, help="Path to YAML/JSON config file")
    parser.add_argument("--preset", "-p", type=str, default=None, help="Model preset name")
    parser.add_argument("--dataset", "-d", type=str, default=None, help="Path to text corpus or binary")
    parser.add_argument("--max_iters", type=int, default=None, help="Override maximum iterations")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="Override device ('cuda', 'cpu', 'auto')")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load Configurations
    if args.config:
        model_cfg, train_cfg, data_cfg = load_full_experiment_config(args.config)
    else:
        preset_name = args.preset or "nano_shakespeare"
        model_cfg = get_preset(preset_name)
        train_cfg = TrainingConfig()
        data_cfg = DataConfig()

    # 2. CLI Overrides
    if args.dataset:
        data_cfg.dataset_path = args.dataset
    if args.max_iters:
        train_cfg.max_iters = args.max_iters
    if args.batch_size:
        train_cfg.batch_size = args.batch_size
    if args.lr:
        train_cfg.learning_rate = args.lr
    if args.device:
        train_cfg.device = args.device

    # Set random seeds
    torch.manual_seed(train_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_cfg.seed)

    print("=" * 60)
    print("nano-gpt-prod :: Production Training Engine")
    print("=" * 60)

    # 3. Load or prepare Dataset
    ds_path = Path(data_cfg.dataset_path)
    if not ds_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {ds_path}")

    if ds_path.is_dir() or ds_path.suffix == ".bin":
        # Binary memmap dataset mode
        train_bin = ds_path if ds_path.suffix == ".bin" else ds_path / "train.bin"
        val_bin = ds_path.parent / "val.bin" if ds_path.suffix == ".bin" else ds_path / "val.bin"

        print(f"Loading memory-mapped binary dataset from {train_bin}...")
        train_ds = MemoryMappedDataset(train_bin, block_size=model_cfg.block_size)
        val_ds = MemoryMappedDataset(val_bin, block_size=model_cfg.block_size) if val_bin.exists() else None
    else:
        # Text file mode
        print(f"Loading raw text dataset from {ds_path}...")
        text = ds_path.read_text(encoding="utf-8")
        tokenizer = get_tokenizer(data_cfg.tokenizer_type, text=text)
        model_cfg.vocab_size = tokenizer.vocab_size
        print(f"Fitted {data_cfg.tokenizer_type} tokenizer: vocab size {tokenizer.vocab_size}")

        train_ds, val_ds = InMemoryDataset.from_text(
            text=text,
            tokenizer=tokenizer,
            block_size=model_cfg.block_size,
            split_ratio=data_cfg.train_split,
        )

    print(f"Train dataset sequences: {len(train_ds):,}")
    if val_ds:
        print(f"Val dataset sequences:   {len(val_ds):,}")

    # 4. Initialize Model
    model = GPT(model_cfg)
    logger = TrainingLogger(log_dir=train_cfg.checkpoint_dir)

    # 5. Initialize Trainer & Train
    trainer = Trainer(
        model=model,
        config=train_cfg,
        train_dataset=train_ds,
        val_dataset=val_ds,
        logger=logger,
    )

    trainer.train()


if __name__ == "__main__":
    main()
