from pathlib import Path
from typing import Optional, Tuple, Union
import torch
from gpt.config.model_config import ModelConfig
from gpt.model.gpt import GPT


def save_checkpoint(
    model: GPT,
    path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    step: int = 0,
    loss: float = 0.0,
    use_safetensors: bool = False,
) -> None:
    """Save model weights, configuration, and optimizer state to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if use_safetensors:
        try:
            from safetensors.torch import save_file
            save_file(model.state_dict(), str(path.with_suffix(".safetensors")))
            path.with_suffix(".json").write_text(model.config.to_json(), encoding="utf-8")
            return
        except ImportError:
            pass

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": model.config.to_dict(),
        "step": step,
        "loss": loss,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(
    path: Union[str, Path],
    device: str = "cpu",
) -> Tuple[GPT, ModelConfig, int, float]:
    """Load model, configuration, and step from checkpoint file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {path}")

    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        config_path = path.with_suffix(".json")
        config = ModelConfig.from_json(config_path.read_text(encoding="utf-8"))
        model = GPT(config)
        state_dict = load_file(str(path), device=device)
        model.load_state_dict(state_dict)
        model.to(device)
        return model, config, 0, 0.0

    checkpoint = torch.load(path, map_location=device)
    config = ModelConfig.from_dict(checkpoint["config"])
    model = GPT(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    step = checkpoint.get("step", 0)
    loss = checkpoint.get("loss", 0.0)
    return model, config, step, loss
