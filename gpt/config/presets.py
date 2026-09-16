from typing import Dict
from gpt.config.model_config import ModelConfig


def get_preset(name: str) -> ModelConfig:
    """Retrieve standard ModelConfig by preset name."""
    presets: Dict[str, ModelConfig] = {
        # Micro model for fast testing and CPU debugging
        "micro": ModelConfig(
            vocab_size=65,
            block_size=64,
            n_layer=2,
            n_head=2,
            n_embd=64,
            dropout=0.0,
            bias=False,
        ),
        # Karpathy nanoGPT character-level Shakespeare configuration
        "nano_shakespeare": ModelConfig(
            vocab_size=65,
            block_size=256,
            n_layer=6,
            n_head=6,
            n_embd=384,
            dropout=0.2,
            bias=False,
        ),
        # OpenAI GPT-2 124M Base architecture
        "gpt2": ModelConfig(
            vocab_size=50257,
            block_size=1024,
            n_layer=12,
            n_head=12,
            n_embd=768,
            dropout=0.0,
            bias=True,
        ),
        "gpt2_medium": ModelConfig(
            vocab_size=50257,
            block_size=1024,
            n_layer=24,
            n_head=16,
            n_embd=1024,
            dropout=0.0,
            bias=True,
        ),
        "gpt2_large": ModelConfig(
            vocab_size=50257,
            block_size=1024,
            n_layer=36,
            n_head=20,
            n_embd=1280,
            dropout=0.0,
            bias=True,
        ),
        "gpt2_xl": ModelConfig(
            vocab_size=50257,
            block_size=1024,
            n_layer=48,
            n_head=25,
            n_embd=1600,
            dropout=0.0,
            bias=True,
        ),
        # LLaMA-style modern architecture: RoPE, RMSNorm, SwiGLU, no bias
        "llama_nano": ModelConfig(
            vocab_size=32000,
            block_size=1024,
            n_layer=6,
            n_head=8,
            n_kv_head=4,
            n_embd=512,
            dropout=0.0,
            bias=False,
            norm_type="rmsnorm",
            activation="swiglu",
            use_rope=True,
            tie_weights=False,
        ),
    }

    if name not in presets:
        available = ", ".join(presets.keys())
        raise KeyError(f"Unknown preset '{name}'. Available presets: {available}")
    return presets[name]
