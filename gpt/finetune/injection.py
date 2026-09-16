from typing import List, Tuple
import torch.nn as nn
from gpt.finetune.config import LoRAConfig
from gpt.finetune.lora import LoRALinear
from gpt.model.gpt import GPT


def apply_lora_to_model(model: GPT, lora_config: LoRAConfig) -> Tuple[GPT, int, int]:
    """Inject LoRALinear modules into target linear layers and freeze all other parameters.

    Returns:
        model: Adapted model
        trainable_params: Count of trainable LoRA parameters
        total_params: Count of all parameters
    """
    # 1. Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    replaced_count = 0

    # 2. Search and replace target linear modules
    for name, module in list(model.named_modules()):
        for target in lora_config.target_modules:
            if target in name and isinstance(module, nn.Linear) and not isinstance(module, LoRALinear):
                # Locate parent and child attribute name
                parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
                parent = model.get_submodule(parent_name) if parent_name else model

                lora_layer = LoRALinear(
                    base_layer=module,
                    r=lora_config.r,
                    lora_alpha=lora_config.lora_alpha,
                    lora_dropout=lora_config.lora_dropout,
                )
                setattr(parent, child_name, lora_layer)
                replaced_count += 1
                break

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Applied LoRA (rank={lora_config.r}, alpha={lora_config.lora_alpha}) to {replaced_count} layers.")
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return model, trainable_params, total_params
