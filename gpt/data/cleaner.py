import hashlib
import re
from typing import Iterable, Iterator, List, Set


def normalize_whitespace(text: str) -> str:
    """Normalize redundant horizontal whitespace and carriage returns while preserving newlines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_control_characters(text: str) -> str:
    """Strip non-printable control characters except standard whitespace characters."""
    return "".join(
        ch for ch in text if ch in {"\n", "\t", " "} or (ord(ch) >= 32 and ord(ch) != 127)
    )


def deduplicate_documents(documents: Iterable[str]) -> Iterator[str]:
    """Yield unique documents based on MD5 content hashes to eliminate duplicate training examples."""
    seen_hashes: Set[str] = set()
    for doc in documents:
        doc_hash = hashlib.md5(doc.strip().encode("utf-8")).hexdigest()
        if doc_hash not in seen_hashes:
            seen_hashes.add(doc_hash)
            yield doc
