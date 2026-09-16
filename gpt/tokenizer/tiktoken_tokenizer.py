from pathlib import Path
from typing import List, Optional, Union
from gpt.tokenizer.base import BaseTokenizer


class TiktokenTokenizer(BaseTokenizer):
    """Wrapper around OpenAI's official tiktoken library for GPT-2 / GPT-4 encoding."""

    def __init__(self, encoding_name: str = "gpt2"):
        self.encoding_name = encoding_name
        try:
            import tiktoken
            self._enc = tiktoken.get_encoding(encoding_name)
        except ImportError:
            raise ImportError(
                "tiktoken is required for TiktokenTokenizer. Run `pip install tiktoken`."
            )

    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab

    @property
    def eos_token_id(self) -> int:
        return self._enc.eot_token

    def encode(self, text: str, allowed_special: str = "all") -> List[int]:
        """Encode text using tiktoken BPE."""
        if allowed_special == "all":
            return self._enc.encode(text, allowed_special={"<|endoftext|>"})
        return self._enc.encode(text)

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text."""
        return self._enc.decode(ids)

    def save(self, path: Union[str, Path]) -> None:
        """Save configuration reference."""
        import json
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "tiktoken", "encoding_name": self.encoding_name}, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TiktokenTokenizer":
        """Load TiktokenTokenizer from config file."""
        import json
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(encoding_name=data.get("encoding_name", "gpt2"))
