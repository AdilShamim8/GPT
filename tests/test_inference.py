import torch
from gpt.config.model_config import ModelConfig
from gpt.inference import (
    TextGenerator,
    apply_temperature,
    sample_token,
    top_k_filtering,
    top_p_filtering,
)
from gpt.model.gpt import GPT
from gpt.tokenizer.char_tokenizer import CharTokenizer


def test_temperature_scaling():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    scaled_cold = apply_temperature(logits, 0.5)
    scaled_hot = apply_temperature(logits, 2.0)

    # Low temperature sharpens differences
    assert (scaled_cold[0, 2] - scaled_cold[0, 0]) > (logits[0, 2] - logits[0, 0])
    # High temperature dampens differences
    assert (scaled_hot[0, 2] - scaled_hot[0, 0]) < (logits[0, 2] - logits[0, 0])


def test_top_k_filtering():
    logits = torch.tensor([[1.0, 5.0, 2.0, 8.0, 3.0]])
    filtered = top_k_filtering(logits, top_k=2)

    # Only 8.0 and 5.0 should remain, rest must be -inf
    assert filtered[0, 3] == 8.0
    assert filtered[0, 1] == 5.0
    assert filtered[0, 0] == float("-inf")
    assert filtered[0, 2] == float("-inf")
    assert filtered[0, 4] == float("-inf")


def test_top_p_filtering():
    logits = torch.tensor([[10.0, 1.0, 0.0, -5.0]])
    filtered = top_p_filtering(logits, top_p=0.8)
    # The dominant token (10.0) covers > 80% mass, others should be -inf
    assert filtered[0, 0] == 10.0
    assert filtered[0, 3] == float("-inf")


def test_text_generator_stream():
    text = "abcdefghijklmnopqrstuvwxyz "
    tokenizer = CharTokenizer.from_text(text, add_special_tokens=False)
    cfg = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=16,
        n_layer=1,
        n_head=1,
        n_embd=16,
    )
    model = GPT(cfg)
    gen = TextGenerator(model, tokenizer, device="cpu")

    stream_tokens = list(gen.generate_stream("abc", max_new_tokens=5))
    assert len(stream_tokens) == 5

    res = gen.generate("abc", max_new_tokens=5)
    assert len(res.text) == 8  # 3 chars prompt + 5 generated


if __name__ == "__main__":
    test_temperature_scaling()
    test_top_k_filtering()
    test_top_p_filtering()
    test_text_generator_stream()
    print("ALL INFERENCE TESTS PASSED!")
