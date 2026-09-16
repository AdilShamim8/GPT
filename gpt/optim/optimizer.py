from typing import Tuple
import torch
import torch.nn as nn
from gpt.config.train_config import TrainingConfig
from gpt.optim.decay import configure_optimizers


def build_optimizer(
    model: nn.Module,
    config: TrainingConfig,
    optimizer_type: str = "adamw",
) -> torch.optim.Optimizer:
    """Build optimizer according to training configuration and parameter groups."""
    device_type = "cuda" if "cuda" in config.device else "cpu"

    if optimizer_type.lower() == "adamw":
        return configure_optimizers(
            model=model,
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            betas=(config.beta1, config.beta2),
            device_type=device_type,
        )
    elif optimizer_type.lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.beta1,
            weight_decay=config.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer type '{optimizer_type}'")
