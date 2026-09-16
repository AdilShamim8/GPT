import torch
from gpt.config.model_config import ModelConfig
from gpt.model.gpt import GPT
from gpt.model.utils import count_parameters


def test_model_forward_shapes():
    cfg = ModelConfig(
        vocab_size=100,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=64,
        bias=False,
    )
    model = GPT(cfg)
    model.eval()

    # Input batch (B=2, T=16)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss, _ = model(idx)

    # In eval mode without targets, returns last position logits (B, 1, vocab_size)
    assert logits.shape == (2, 1, 100)
    assert loss is None


def test_model_loss_computation():
    cfg = ModelConfig(
        vocab_size=50,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
    )
    model = GPT(cfg)
    x = torch.randint(0, cfg.vocab_size, (4, 8))
    y = torch.randint(0, cfg.vocab_size, (4, 8))

    logits, loss, _ = model(x, targets=y)
    assert loss is not None
    assert loss.item() > 0.0
    assert not torch.isnan(loss)


def test_model_kv_cache():
    cfg = ModelConfig(
        vocab_size=60,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=32,
    )
    model = GPT(cfg)
    model.eval()

    prompt = torch.randint(0, cfg.vocab_size, (1, 4))

    # Forward prompt to populate cache
    logits_prompt, _, kv_cache = model(prompt, use_cache=True)

    # Next token
    next_tok = torch.randint(0, cfg.vocab_size, (1, 1))
    logits_cached, _, _ = model(next_tok, use_cache=True, past_key_values=kv_cache)

    # Compare against full forward pass of (prompt + next_tok)
    full_seq = torch.cat([prompt, next_tok], dim=1)
    logits_full, _, _ = model(full_seq, use_cache=False)

    # Logits at the last token should match closely
    assert torch.allclose(logits_cached, logits_full, atol=1e-4)


def test_parameter_counting():
    cfg = ModelConfig(
        vocab_size=100,
        block_size=32,
        n_layer=2,
        n_head=2,
        n_embd=64,
    )
    model = GPT(cfg)
    total, trainable = count_parameters(model)
    assert total > 0
    assert total == trainable


if __name__ == "__main__":
    test_model_forward_shapes()
    test_model_loss_computation()
    test_model_kv_cache()
    test_parameter_counting()
    print("ALL MODEL TESTS PASSED!")
