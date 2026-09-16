from pathlib import Path
from typing import Dict, Union
import torch
from gpt.finetune.config import LoRAConfig
from gpt.finetune.lora import LoRALinear
from gpt.model.gpt import GPT


def get_lora_state_dict(model: GPT) -> Dict[str, torch.Tensor]:
    """Extract only the trainable LoRA parameters (A and B matrices)."""
    return {
        name: param.cpu()
        for name, param in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    }


def save_lora_weights(model: GPT, config: LoRAConfig, path: Union[str, Path]) -> None:
    """Save LoRA adapter weights and configuration to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "lora_state_dict": get_lora_state_dict(model),
        "config": config.to_dict(),
    }
    torch.save(state, path)
    print(f"Saved LoRA adapter weights to {path}")


def load_lora_weights(model: GPT, path: Union[str, Path]) -> LoRAConfig:
    """Load and inject saved LoRA adapter weights into model."""
    path = Path(path)
    state = torch.load(path, map_location="cpu")
    config = LoRAConfig.from_dict(state["config"])

    lora_sd = state["lora_state_dict"]
    model_sd = model.state_dict()

    for k, v in lora_sd.items():
        if k in model_sd:
            model_sd[k].copy_(v)

    return config


def merge_lora_weights(model: GPT) -> None:
    """Merge all active LoRA layers back into the base linear weights."""
    merged = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge_weights()
            merged += 1
    print(f"Successfully merged {merged} LoRA layers into base model.")
