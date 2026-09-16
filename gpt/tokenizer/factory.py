import json
from pathlib import Path
from typing import Union
from gpt.tokenizer.base import BaseTokenizer
from gpt.tokenizer.bpe_tokenizer import BPETokenizer
from gpt.tokenizer.char_tokenizer import CharTokenizer
from gpt.tokenizer.tiktoken_tokenizer import TiktokenTokenizer


def get_tokenizer(
    tokenizer_type_or_path: Union[str, Path] = "char",
    **kwargs,
) -> BaseTokenizer:
    """Instantiate or load tokenizer by type name or checkpoint path.

    Args:
        tokenizer_type_or_path: Either a type ('char', 'bpe', 'tiktoken', 'gpt2')
                                or a path to a saved tokenizer JSON.
    """
    path = Path(str(tokenizer_type_or_path))

    # If it's an existing file, inspect and load it
    if path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        tok_type = meta.get("type")
        if tok_type == "char":
            return CharTokenizer.load(path)
        elif tok_type == "bpe":
            return BPETokenizer.load(path)
        elif tok_type == "tiktoken":
            return TiktokenTokenizer.load(path)
        else:
            raise ValueError(f"Unknown saved tokenizer type '{tok_type}' in {path}")

    # Otherwise instantiate from standard name
    name = str(tokenizer_type_or_path).lower()
    if name in {"char", "character"}:
        if "text" in kwargs:
            return CharTokenizer.from_text(kwargs["text"])
        return CharTokenizer(**kwargs)
    elif name in {"bpe", "byte_pair"}:
        if "text" in kwargs:
            return BPETokenizer.train_from_text(
                kwargs["text"],
                vocab_size=kwargs.get("vocab_size", 1000),
            )
        return BPETokenizer(**kwargs)
    elif name in {"tiktoken", "gpt2", "gpt4"}:
        enc_name = "gpt2" if name in {"tiktoken", "gpt2"} else name
        return TiktokenTokenizer(encoding_name=enc_name)
    else:
        raise ValueError(
            f"Unsupported tokenizer '{tokenizer_type_or_path}'. "
            "Supported types: 'char', 'bpe', 'tiktoken', 'gpt2', or path to file."
        )
