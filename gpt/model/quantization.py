import torch
import torch.nn as nn
from gpt.model.gpt import GPT


def quantize_dynamic_int8(model: GPT) -> nn.Module:
    """Apply PyTorch dynamic int8 quantization to Linear layers for accelerated CPU inference."""
    try:
        quantized_model = torch.ao.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8,
        )
        return quantized_model
    except Exception as e:
        # Fallback to standard torch.quantization if ao namespace differs
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8,
        )
        return quantized_model


def convert_to_half_precision(model: GPT, dtype: torch.dtype = torch.float16) -> GPT:
    """Cast model parameters to half precision (float16 or bfloat16)."""
    return model.to(dtype=dtype)
