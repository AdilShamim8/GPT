import torch
from gpt.config.model_config import ModelConfig
from gpt.model.gpt import GPT


def test_causal_masking_no_future_leakage():
    """Verify strictly that future tokens do not influence past token predictions.

    Perturbing token at index j should have ZERO impact on logits at index i for all i < j.
    """
    cfg = ModelConfig(
        vocab_size=50,
        block_size=16,
        n_layer=2,
        n_head=2,
        n_embd=32,
    )
    model = GPT(cfg)
    model.eval()

    # Create base sequence with valid token IDs < vocab_size (50)
    seq1 = torch.tensor([[10, 20, 30, 40, 15]], dtype=torch.long)
    # Modify last token
    seq2 = torch.tensor([[10, 20, 30, 40, 45]], dtype=torch.long)

    # Full forward pass computing all intermediate logits
    tok_emb1 = model.transformer.wte(seq1) + model.transformer.wpe(torch.arange(5))
    tok_emb2 = model.transformer.wte(seq2) + model.transformer.wpe(torch.arange(5))

    x1 = model.transformer.drop(tok_emb1)
    x2 = model.transformer.drop(tok_emb2)

    for block in model.transformer.h:
        x1, _ = block(x1)
        x2, _ = block(x2)

    out1 = model.lm_head(model.transformer.ln_f(x1))
    out2 = model.lm_head(model.transformer.ln_f(x2))

    # Logits at positions 0, 1, 2, 3 must be strictly identical between seq1 and seq2
    diff_prefix = (out1[:, :4, :] - out2[:, :4, :]).abs().max().item()
    assert diff_prefix < 1e-5, f"Information leakage detected! Diff: {diff_prefix}"

    # Logits at position 4 must differ because token 4 was changed
    diff_target = (out1[:, 4, :] - out2[:, 4, :]).abs().max().item()
    assert diff_target > 1e-4, "Expected difference at mutated position!"


if __name__ == "__main__":
    test_causal_masking_no_future_leakage()
    print("CAUSALITY VERIFICATION TEST PASSED!")
