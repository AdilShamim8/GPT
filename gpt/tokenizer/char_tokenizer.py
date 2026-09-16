import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from gpt.tokenizer.base import BaseTokenizer


class CharTokenizer(BaseTokenizer):
    """Character-level tokenizer for small-scale and educational LLMs."""

    def __init__(
        self,
        chars: Optional[List[str]] = None,
        stoi: Optional[Dict[str, int]] = None,
        itos: Optional[Dict[int, str]] = None,
    ):
        if stoi is not None and itos is not None:
            self._stoi = stoi
            self._itos = itos
        elif chars is not None:
            sorted_chars = sorted(list(set(chars)))
            self._stoi = {ch: i for i, ch in enumerate(sorted_chars)}
            self._itos = {i: ch for i, ch in enumerate(sorted_chars)}
        else:
            self._stoi = {}
            self._itos = {}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        """Build character vocabulary directly from training text."""
        chars = sorted(list(set(text)))
        return cls(chars=chars)

    @property
    def vocab_size(self) -> int:
        return len(self._stoi)

    def encode(self, text: str) -> List[int]:
        """Encode text to character IDs. Skips unknown characters or raises KeyError."""
        encoded = []
        for ch in text:
            if ch in self._stoi:
                encoded.append(self._stoi[ch])
            else:
                # Default to unk if available, else raise
                if "<|unk|>" in self._stoi:
                    encoded.append(self._stoi["<|unk|>"])
                else:
                    raise KeyError(f"Character {repr(ch)} not in tokenizer vocabulary.")
        return encoded

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back into string."""
        return "".join([self._itos.get(i, "") for i in ids])

    def save(self, path: Union[str, Path]) -> None:
        """Save vocabulary mapping to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "type": "char",
            "stoi": self._stoi,
            "itos": {str(k): v for k, v in self._itos.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CharTokenizer":
        """Load vocabulary mapping from a JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        stoi = data["stoi"]
        itos = {int(k): v for k, v in data["itos"].items()}
        return cls(stoi=stoi, itos=itos)
