from typing import Dict, List, Tuple, Union
import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_model_memory_mb(model: nn.Module, dtype: torch.dtype = torch.float32) -> float:
    """Estimate model memory footprint in Megabytes (MB)."""
    bytes_per_elem = 4 if dtype in {torch.float32, torch.int32} else 2
    total_params, _ = count_parameters(model)
    return (total_params * bytes_per_elem) / (1024 * 1024)


def get_layer_parameter_breakdown(model: nn.Module) -> List[Dict[str, Union[str, int, float]]]:
    """Generate detailed parameter distribution breakdown across model components."""
    breakdown = []
    for name, module in model.named_children():
        num_params = sum(p.numel() for p in module.parameters())
        if num_params > 0:
            breakdown.append({
                "module": name,
                "type": module.__class__.__name__,
                "params": num_params,
                "params_m": round(num_params / 1e6, 3),
            })
    return breakdown
