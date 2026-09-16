import re
from typing import Dict
import torch
from gpt.config.model_config import ModelConfig
from gpt.config.presets import get_preset
from gpt.model.gpt import GPT


def load_huggingface_gpt2_weights(model_type: str = "gpt2") -> GPT:
    """Load pretrained GPT-2 weights from Hugging Face Transformers into our GPT model."""
    try:
        from transformers import GPT2LMHeadModel
    except ImportError:
        raise ImportError(
            "The 'transformers' package is required to load HF weights. Run `pip install transformers`."
        )

    print(f"Loading weights from Hugging Face model: {model_type}...")
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()

    # Map model name to our ModelConfig preset
    config = get_preset(model_type)
    model = GPT(config)
    sd = model.state_dict()

    # Copy weights, transposing Conv1D weights from HF to standard Linear weights
    transposed = [
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    ]

    for k in sd_hf.keys():
        if any(k.endswith(w) for w in ["attn.masked_bias", "attn.bias"]):
            # Skip attention buffer masks
            continue

        target_k = k
        if target_k in sd:
            if any(k.endswith(w) for w in transposed):
                # PyTorch Conv1D in HF stores weights transposed compared to Linear
                assert sd_hf[k].shape[::-1] == sd[target_k].shape
                with torch.no_grad():
                    sd[target_k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[target_k].shape
                with torch.no_grad():
                    sd[target_k].copy_(sd_hf[k])

    print(f"Successfully loaded and transposed weights for {model_type}!")
    return model
