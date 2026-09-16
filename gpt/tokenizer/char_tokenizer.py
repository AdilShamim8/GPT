import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
from gpt.tokenizer.base import BaseTokenizer

SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>", "<|unk|>"]


class CharTokenizer(BaseTokenizer):
    """Character-level tokenizer with support for special control tokens."""

    def __init__(
        self,
        chars: Optional[List[str]] = None,
        stoi: Optional[Dict[str, int]] = None,
        itos: Optional[Dict[int, str]] = None,
        special_tokens: Optional[List[str]] = None,
    ):
        self.special_tokens = special_tokens if special_tokens is not None else SPECIAL_TOKENS.copy()

        if stoi is not None and itos is not None:
            self._stoi = stoi
            self._itos = itos
        elif chars is not None:
            sorted_chars = sorted(list(set(chars)))
            all_tokens = sorted_chars + [t for t in self.special_tokens if t not in sorted_chars]
            self._stoi = {ch: i for i, ch in enumerate(all_tokens)}
            self._itos = {i: ch for i, ch in enumerate(all_tokens)}
        else:
            self._stoi = {}
            self._itos = {}

    @classmethod
    def from_text(cls, text: str, add_special_tokens: bool = True) -> "CharTokenizer":
        """Build character vocabulary directly from training text."""
        chars = sorted(list(set(text)))
        specials = SPECIAL_TOKENS if add_special_tokens else []
        return cls(chars=chars, special_tokens=specials)

    @property
    def vocab_size(self) -> int:
        return len(self._stoi)

    @property
    def pad_token_id(self) -> Optional[int]:
        return self._stoi.get("<|pad|>")

    @property
    def eos_token_id(self) -> Optional[int]:
        return self._stoi.get("<|endoftext|>")

    @property
    def unk_token_id(self) -> Optional[int]:
        return self._stoi.get("<|unk|>")

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs with support for special tokens and fallback."""
        encoded = []
        i = 0
        n = len(text)
        while i < n:
            matched_special = False
            for spec in self.special_tokens:
                if text.startswith(spec, i):
                    encoded.append(self._stoi[spec])
                    i += len(spec)
                    matched_special = True
                    break
            if not matched_special:
                ch = text[i]
                if ch in self._stoi:
                    encoded.append(self._stoi[ch])
                elif self.unk_token_id is not None:
                    encoded.append(self.unk_token_id)
                else:
                    raise KeyError(f"Character {repr(ch)} not in vocabulary.")
                i += 1
        return encoded

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back into string."""
        return "".join([self._itos.get(i, "") for i in ids])

    def save(self, path: Union[str, Path]) -> None:
        """Save vocabulary mapping and special tokens to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "type": "char",
            "special_tokens": self.special_tokens,
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
        specials = data.get("special_tokens", SPECIAL_TOKENS)
        return cls(stoi=stoi, itos=itos, special_tokens=specials)
