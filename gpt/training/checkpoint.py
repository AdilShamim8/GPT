import shutil
import tempfile
from pathlib import Path
from typing import Optional, Tuple
import torch
from gpt.model.gpt import GPT
from gpt.model.serialization import load_checkpoint, save_checkpoint


class CheckpointManager:
    """Manages atomic saving, tracking best checkpoints, and resuming training."""

    def __init__(self, checkpoint_dir: str, save_best_only: bool = False):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.best_loss = float("inf")

    def save(
        self,
        model: GPT,
        optimizer: torch.optim.Optimizer,
        step: int,
        val_loss: float,
        is_best: bool = False,
    ) -> Path:
        """Atomically save checkpoint to prevent corrupted partial files on crash."""
        filename = f"ckpt_step_{step:07d}.pt"
        target_path = self.checkpoint_dir / filename

        # Write to temporary file in same filesystem first
        with tempfile.NamedTemporaryFile(
            dir=self.checkpoint_dir, delete=False, suffix=".tmp"
        ) as tmp:
            tmp_path = Path(tmp.name)

        save_checkpoint(
            model=model,
            path=tmp_path,
            optimizer=optimizer,
            step=step,
            loss=val_loss,
        )
        # Atomic rename
        tmp_path.replace(target_path)

        # Update latest pointer
        latest_path = self.checkpoint_dir / "ckpt_latest.pt"
        shutil.copyfile(target_path, latest_path)

        # Update best pointer
        if is_best or val_loss < self.best_loss:
            self.best_loss = val_loss
            best_path = self.checkpoint_dir / "ckpt_best.pt"
            shutil.copyfile(target_path, best_path)

        return target_path

    def load_latest(self, device: str = "cpu") -> Optional[Tuple[GPT, int, float]]:
        """Load latest checkpoint if available."""
        latest_path = self.checkpoint_dir / "ckpt_latest.pt"
        if not latest_path.exists():
            return None
        model, _, step, loss = load_checkpoint(latest_path, device=device)
        return model, step, loss
