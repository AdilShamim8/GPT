import tempfile
from pathlib import Path
import torch
import torch.nn as nn
from gpt.config import ModelConfig, TrainingConfig
from gpt.model.gpt import GPT
from gpt.optim import configure_optimizers, get_cosine_schedule_with_warmup
from gpt.training import CheckpointManager, MetricTracker


def test_optimizer_weight_decay_split():
    cfg = ModelConfig(
        vocab_size=50,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
        bias=True,
    )
    model = GPT(cfg)
    optim = configure_optimizers(
        model=model,
        weight_decay=0.1,
        learning_rate=1e-3,
        betas=(0.9, 0.95),
        device_type="cpu",
    )

    # Param group 0: 2D weights with weight_decay = 0.1
    # Param group 1: 1D biases/norms with weight_decay = 0.0
    assert len(optim.param_groups) == 2
    assert optim.param_groups[0]["weight_decay"] == 0.1
    assert optim.param_groups[1]["weight_decay"] == 0.0


def test_cosine_scheduler_warmup():
    linear = nn.Linear(10, 10)
    optim = torch.optim.Adam(linear.parameters(), lr=1.0)
    sched = get_cosine_schedule_with_warmup(
        optim, warmup_iters=10, max_iters=100, min_lr_ratio=0.1
    )

    # Step 0: lr = 0.0
    assert sched.get_last_lr()[0] == 0.0

    # Step 5: halfway through warmup -> lr = 0.5
    for _ in range(5):
        sched.step()
    assert abs(sched.get_last_lr()[0] - 0.5) < 1e-4

    # Step 10: warmup complete -> lr = 1.0
    for _ in range(5):
        sched.step()
    assert abs(sched.get_last_lr()[0] - 1.0) < 1e-4


def test_checkpoint_manager_atomic_save():
    cfg = ModelConfig(
        vocab_size=50,
        block_size=16,
        n_layer=1,
        n_head=1,
        n_embd=16,
    )
    model = GPT(cfg)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = CheckpointManager(tmpdir)
        save_path = mgr.save(model, optim, step=42, val_loss=1.85)

        assert save_path.exists()
        assert (Path(tmpdir) / "ckpt_latest.pt").exists()
        assert (Path(tmpdir) / "ckpt_best.pt").exists()

        loaded_model, step, loss = mgr.load_latest(device="cpu")
        assert step == 42
        assert abs(loss - 1.85) < 1e-4


def test_metric_tracker():
    tracker = MetricTracker(window_size=3)
    tracker.update("loss", 2.0)
    tracker.update("loss", 4.0)
    assert tracker.get_window_avg("loss") == 3.0
    assert tracker.get_global_avg("loss") == 3.0


if __name__ == "__main__":
    test_optimizer_weight_decay_split()
    test_cosine_scheduler_warmup()
    test_checkpoint_manager_atomic_save()
    test_metric_tracker()
    print("ALL TRAINING TESTS PASSED!")
