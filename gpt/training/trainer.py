import time
from typing import Dict, Optional, Tuple
import torch
from gpt.config.train_config import TrainingConfig
from gpt.data.base import BaseDataset
from gpt.data.loader import get_batch_from_dataset
from gpt.model.gpt import GPT
from gpt.optim.amp import MixedPrecisionManager
from gpt.optim.decay import configure_optimizers
from gpt.optim.scheduler import build_scheduler
from gpt.training.checkpoint import CheckpointManager
from gpt.training.logger import TrainingLogger
from gpt.training.metrics import MetricTracker


class Trainer:
    """Production training engine for Transformer language models."""

    def __init__(
        self,
        model: GPT,
        config: TrainingConfig,
        train_dataset: BaseDataset,
        val_dataset: Optional[BaseDataset] = None,
        logger: Optional[TrainingLogger] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Setup compute device
        if config.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = config.device

        self.model.to(self.device)

        # Build optimizer & scheduler
        self.optimizer = configure_optimizers(
            model=self.model,
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            betas=(config.beta1, config.beta2),
            device_type=self.device,
        )
        self.scheduler = build_scheduler(self.optimizer, config)

        # Precision manager
        self.amp = MixedPrecisionManager(
            mode=config.mixed_precision,
            device_type=self.device,
        )

        # Checkpoints & logging
        self.checkpoint_mgr = CheckpointManager(config.checkpoint_dir)
        self.logger = logger or TrainingLogger()
        self.metrics = MetricTracker()

        self.step = 0

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Estimate average loss across validation and training splits."""
        out = {}
        self.model.eval()
        datasets = {"train": self.train_dataset}
        if self.val_dataset is not None:
            datasets["val"] = self.val_dataset

        for split, ds in datasets.items():
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                x, y = get_batch_from_dataset(ds, self.config.batch_size, device=self.device)
                with self.amp.autocast_context():
                    _, loss, _ = self.model(x, targets=y)
                losses[k] = loss.item()
            out[f"{split}_loss"] = losses.mean().item()

        self.model.train()
        return out

    def train(self) -> None:
        """Run full training loop with gradient accumulation and evaluation."""
        self.model.train()
        start_time = time.time()
        tokens_per_step = self.config.total_batch_size * self.model.config.block_size

        print(f"Starting training run on {self.device.upper()}...")
        print(f"Model parameters: {self.model.get_num_params() / 1e6:.2f}M")
        print(f"Total training iterations: {self.config.max_iters}")
        print(f"Tokens per optimization step: {tokens_per_step:,}")

        while self.step < self.config.max_iters:
            t0 = time.time()

            # Periodic evaluation
            if self.step % self.config.eval_interval == 0 or self.step == self.config.max_iters - 1:
                eval_metrics = self.evaluate()
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.logger.log_metrics(self.step, eval_metrics, lr=current_lr)

            # Periodic checkpoint save
            if self.step > 0 and self.step % self.config.save_interval == 0:
                current_loss = eval_metrics.get("val_loss", 0.0)
                self.checkpoint_mgr.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    step=self.step,
                    val_loss=current_loss,
                )

            # Optimization step with gradient accumulation
            self.optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0

            for micro_step in range(self.config.grad_accum_steps):
                x, y = get_batch_from_dataset(
                    self.train_dataset, self.config.batch_size, device=self.device
                )
                with self.amp.autocast_context():
                    _, loss, _ = self.model(x, targets=y)
                    # Scale down loss for gradient accumulation
                    loss = loss / self.config.grad_accum_steps

                self.amp.backward(loss)
                accum_loss += loss.item()

            # Gradient clipping
            if self.config.grad_clip > 0.0:
                self.amp.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            # Optimizer step & scaler update
            self.amp.step(self.optimizer)
            if self.scheduler is not None:
                self.scheduler.step()

            # Measure timing and throughput
            dt = time.time() - t0
            tok_per_sec = tokens_per_step / dt if dt > 0 else 0.0
            self.metrics.update("loss", accum_loss)
            self.metrics.update("tok_per_sec", tok_per_sec)

            if self.step % self.config.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.logger.log_metrics(
                    self.step,
                    {"loss": accum_loss, "tok_per_sec": tok_per_sec},
                    lr=current_lr,
                )

            self.step += 1

        print(f"Training completed in {time.time() - start_time:.1f} seconds.")
