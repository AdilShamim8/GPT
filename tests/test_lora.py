import torch
import torch.nn as nn
from gpt.config.model_config import ModelConfig
from gpt.finetune import LoRAConfig, LoRALinear, apply_lora_to_model, merge_lora_weights
from gpt.model.gpt import GPT


def test_lora_initialization_equivalence():
    """Verify that initially LoRALinear outputs EXACTLY match base linear layer because B=0."""
    linear = nn.Linear(32, 64)
    x = torch.randn(4, 32)
    base_out = linear(x)

    lora_layer = LoRALinear(linear, r=4, lora_alpha=8.0, lora_dropout=0.0)
    lora_out = lora_layer(x)

    assert torch.allclose(base_out, lora_out, atol=1e-6)


def test_lora_parameter_freezing():
    """Verify base model weights are frozen and only LoRA weights have gradients."""
    cfg = ModelConfig(
        vocab_size=50,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
    )
    model = GPT(cfg)
    lora_cfg = LoRAConfig(r=4, target_modules=["q_proj", "v_proj"])
    model, trainable_p, total_p = apply_lora_to_model(model, lora_cfg)

    assert trainable_p < total_p
    assert trainable_p > 0

    # Ensure wte embedding has no gradient
    assert not model.transformer.wte.weight.requires_grad


def test_lora_weight_merging():
    """Verify weight merging produces equivalent output and dissolves LoRA layers."""
    linear = nn.Linear(16, 16)
    x = torch.randn(2, 16)

    lora = LoRALinear(linear, r=2, lora_alpha=4.0, lora_dropout=0.0)
    # Give some non-zero weight to lora_B
    with torch.no_grad():
        lora.lora_B.fill_(0.5)

    out_before_merge = lora(x)
    lora.merge_weights()
    out_after_merge = lora.base_layer(x)

    assert torch.allclose(out_before_merge, out_after_merge, atol=1e-5)


if __name__ == "__main__":
    test_lora_initialization_equivalence()
    test_lora_parameter_freezing()
    test_lora_weight_merging()
    print("ALL LORA TESTS PASSED!")
