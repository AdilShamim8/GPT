import torch
from gpt.config.model_config import ModelConfig
from gpt.evaluation import (
    calculate_perplexity,
    evaluate_multiple_choice_example,
)
from gpt.model.gpt import GPT
from gpt.tokenizer.char_tokenizer import CharTokenizer


def test_perplexity_computation():
    text = "the quick brown fox jumps over the lazy dog"
    tok = CharTokenizer.from_text(text, add_special_tokens=False)
    cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        block_size=16,
        n_layer=1,
        n_head=1,
        n_embd=16,
    )
    model = GPT(cfg)
    ppl = calculate_perplexity(model, tok, text, device="cpu", stride=8)

    assert ppl > 0.0
    assert not torch.isnan(torch.tensor(ppl))


def test_multiple_choice_scoring():
    text = "the cat sat on the mat"
    tok = CharTokenizer.from_text(text, add_special_tokens=False)
    cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        block_size=32,
        n_layer=1,
        n_head=1,
        n_embd=16,
    )
    model = GPT(cfg)
    pred_idx = evaluate_multiple_choice_example(
        model=model,
        tokenizer=tok,
        prompt="the cat sat ",
        completions=["on the mat", "on the moon"],
        device="cpu",
    )
    assert pred_idx in {0, 1}


if __name__ == "__main__":
    test_perplexity_computation()
    test_multiple_choice_scoring()
    print("ALL EVALUATION TESTS PASSED!")
