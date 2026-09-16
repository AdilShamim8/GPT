import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional


class TrainingLogger:
    """Structured training logger supporting stdout formatting and JSONL logging."""

    def __init__(self, log_dir: Optional[str] = None, experiment_name: str = "gpt_run"):
        self.experiment_name = experiment_name
        self.log_file = None

        if log_dir is not None:
            path = Path(log_dir)
            path.mkdir(parents=True, exist_ok=True)
            self.log_file = open(path / f"{experiment_name}.jsonl", "a", encoding="utf-8")

        self.logger = logging.getLogger("gpt.training")

    def log_metrics(self, step: int, metrics: Dict[str, Any], lr: Optional[float] = None) -> None:
        """Log iteration metrics to console and JSONL file."""
        timestamp = time.time()
        record = {"step": step, "timestamp": timestamp, **metrics}
        if lr is not None:
            record["learning_rate"] = lr

        # Write to JSONL
        if self.log_file is not None:
            self.log_file.write(json.dumps(record) + "\n")
            self.log_file.flush()

        # Format console log
        parts = [f"Step {step:6d}"]
        if "train_loss" in metrics:
            parts.append(f"train loss: {metrics['train_loss']:.4f}")
        elif "loss" in metrics:
            parts.append(f"loss: {metrics['loss']:.4f}")
        if "val_loss" in metrics:
            parts.append(f"val loss: {metrics['val_loss']:.4f}")
        if "val_ppl" in metrics:
            parts.append(f"val ppl: {metrics['val_ppl']:.2f}")
        if lr is not None:
            parts.append(f"lr: {lr:.2e}")
        if "tok_per_sec" in metrics:
            parts.append(f"tok/s: {metrics['tok_per_sec']:,.0f}")

        self.logger.info(" | ".join(parts))

    def close(self) -> None:
        if self.log_file is not None and not self.log_file.closed:
            self.log_file.close()
