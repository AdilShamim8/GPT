import tempfile
from pathlib import Path
from gpt.tokenizer import BPETokenizer, CharTokenizer, get_tokenizer


def test_char_tokenizer_basic():
    corpus = "hello world! 123"
    tok = CharTokenizer.from_text(corpus)
    assert tok.vocab_size >= len(set(corpus))

    encoded = tok.encode("hello")
    decoded = tok.decode(encoded)
    assert decoded == "hello"


def test_char_tokenizer_specials():
    corpus = "abc"
    tok = CharTokenizer.from_text(corpus, add_special_tokens=True)
    text = "a<|endoftext|>b"
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)
    assert decoded == text


def test_char_tokenizer_save_load():
    corpus = "To be or not to be"
    tok = CharTokenizer.from_text(corpus)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "char_tok.json"
        tok.save(save_path)

        loaded_tok = CharTokenizer.load(save_path)
        assert loaded_tok.vocab_size == tok.vocab_size
        assert loaded_tok.encode("To be") == tok.encode("To be")


def test_bpe_tokenizer_training():
    corpus = "banana band bandit banner banter bandana" * 5
    tok = BPETokenizer.train_from_text(corpus, vocab_size=270)
    assert tok.vocab_size >= 270

    test_str = "banana band"
    encoded = tok.encode(test_str)
    decoded = tok.decode(encoded)
    assert decoded == test_str


def test_bpe_tokenizer_save_load():
    corpus = "transformer attention multi-head self-attention" * 4
    tok = BPETokenizer.train_from_text(corpus, vocab_size=265)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "bpe_tok.json"
        tok.save(save_path)

        loaded_tok = BPETokenizer.load(save_path)
        assert loaded_tok.vocab_size == tok.vocab_size
        assert loaded_tok.encode("attention") == tok.encode("attention")


def test_tokenizer_factory():
    tok = get_tokenizer("char", text="abcdef")
    assert isinstance(tok, CharTokenizer)

    bpe_tok = get_tokenizer("bpe", text="abcdef 123", vocab_size=260)
    assert isinstance(bpe_tok, BPETokenizer)


if __name__ == "__main__":
    test_char_tokenizer_basic()
    test_char_tokenizer_specials()
    test_char_tokenizer_save_load()
    test_bpe_tokenizer_training()
    test_bpe_tokenizer_save_load()
    test_tokenizer_factory()
    print("ALL TOKENIZER TESTS PASSED!")
