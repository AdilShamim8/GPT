import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from gpt.tokenizer.base import BaseTokenizer
from gpt.tokenizer.bpe import merge, train_bpe


class BPETokenizer(BaseTokenizer):
    """Production Byte-Pair Encoding (BPE) Tokenizer with byte fallback."""

    def __init__(
        self,
        merges: Optional[Dict[Tuple[int, int], int]] = None,
        vocab: Optional[Dict[int, bytes]] = None,
        special_tokens: Optional[List[str]] = None,
    ):
        self.merges: Dict[Tuple[int, int], int] = merges or {}
        self.vocab: Dict[int, bytes] = vocab or {i: bytes([i]) for i in range(256)}
        self.special_tokens: List[str] = special_tokens or ["<|endoftext|>", "<|pad|>"]

        # Assign IDs for special tokens
        start_id = max(self.vocab.keys(), default=255) + 1
        self.special_to_id: Dict[str, int] = {}
        for idx, token in enumerate(self.special_tokens):
            t_id = start_id + idx
            self.special_to_id[token] = t_id
            self.vocab[t_id] = token.encode("utf-8")

        self.id_to_special: Dict[int, str] = {v: k for k, v in self.special_to_id.items()}

    @classmethod
    def train_from_text(
        cls,
        text: str,
        vocab_size: int = 1000,
        special_tokens: Optional[List[str]] = None,
    ) -> "BPETokenizer":
        """Train a BPE tokenizer from raw text corpus."""
        merges, vocab = train_bpe(text, target_vocab_size=vocab_size)
        return cls(merges=merges, vocab=vocab, special_tokens=special_tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def eos_token_id(self) -> Optional[int]:
        return self.special_to_id.get("<|endoftext|>")

    @property
    def pad_token_id(self) -> Optional[int]:
        return self.special_to_id.get("<|pad|>")

    def encode(self, text: str) -> List[int]:
        """Encode text to BPE token IDs with special token support."""
        if not text:
            return []

        # Split on special tokens if present
        chunks = [text]
        for spec in self.special_tokens:
            new_chunks = []
            for ch in chunks:
                if isinstance(ch, str):
                    parts = ch.split(spec)
                    for idx, part in enumerate(parts):
                        if part:
                            new_chunks.append(part)
                        if idx < len(parts) - 1:
                            new_chunks.append(self.special_to_id[spec])
                else:
                    new_chunks.append(ch)
            chunks = new_chunks

        encoded_ids: List[int] = []
        for chunk in chunks:
            if isinstance(chunk, int):
                encoded_ids.append(chunk)
            else:
                chunk_bytes = list(chunk.encode("utf-8"))
                # Iteratively apply merges in order of occurrence
                while len(chunk_bytes) >= 2:
                    stats = {}
                    for pair in zip(chunk_bytes, chunk_bytes[1:]):
                        if pair in self.merges:
                            stats[pair] = self.merges[pair]
                    if not stats:
                        break
                    # Find merge with lowest resulting ID (earliest learned merge)
                    best_pair = min(stats, key=stats.get)
                    chunk_bytes = merge(chunk_bytes, best_pair, self.merges[best_pair])
                encoded_ids.extend(chunk_bytes)

        return encoded_ids

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to UTF-8 text, replacing malformed bytes."""
        byte_chunks = []
        for i in ids:
            if i in self.id_to_special:
                byte_chunks.append(self.id_to_special[i].encode("utf-8"))
            elif i in self.vocab:
                byte_chunks.append(self.vocab[i])
            else:
                byte_chunks.append(b"")
        return b"".join(byte_chunks).decode("utf-8", errors="replace")

    def save(self, path: Union[str, Path]) -> None:
        """Save BPE merges and vocabulary to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        serializable_merges = [
            {"p0": p[0], "p1": p[1], "idx": idx} for p, idx in self.merges.items()
        ]
        data = {
            "type": "bpe",
            "merges": serializable_merges,
            "special_tokens": self.special_tokens,
            "vocab_size": self.vocab_size,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BPETokenizer":
        """Load BPE tokenizer from saved JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        merges = {}
        vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        for item in data["merges"]:
            pair = (item["p0"], item["p1"])
            idx = item["idx"]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        return cls(
            merges=merges,
            vocab=vocab,
            special_tokens=data.get("special_tokens"),
        )
