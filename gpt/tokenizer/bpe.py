import collections
from typing import Dict, List, Tuple


def get_pair_stats(ids: List[int]) -> Dict[Tuple[int, int], int]:
    """Count frequency of adjacent token pairs in a list of IDs."""
    counts = collections.defaultdict(int)
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts


def merge(ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
    """Replace all consecutive occurrences of `pair` with new token ID `idx`."""
    newids = []
    i = 0
    n = len(ids)
    p0, p1 = pair
    while i < n:
        if i < n - 1 and ids[i] == p0 and ids[i + 1] == p1:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


def train_bpe(
    text: str,
    target_vocab_size: int = 500,
    verbose: bool = False,
) -> Tuple[Dict[Tuple[int, int], int], Dict[int, bytes]]:
    """Train Byte-Pair Encoding on raw text starting from base bytes (0..255).

    Returns:
        merges: Mapping from (id1, id2) -> new_id
        vocab: Mapping from token_id -> byte representation
    """
    raw_bytes = list(text.encode("utf-8"))
    num_merges = target_vocab_size - 256
    if num_merges <= 0:
        raise ValueError(
            f"target_vocab_size ({target_vocab_size}) must be > 256 (base byte count)."
        )

    ids = list(raw_bytes)
    merges: Dict[Tuple[int, int], int] = {}
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    for i in range(num_merges):
        stats = get_pair_stats(ids)
        if not stats:
            break
        best_pair = max(stats, key=stats.get)
        idx = 256 + i
        ids = merge(ids, best_pair, idx)
        merges[best_pair] = idx
        vocab[idx] = vocab[best_pair[0]] + vocab[best_pair[1]]
        if verbose and (i + 1) % 50 == 0:
            print(f"BPE merge {i+1}/{num_merges}: {best_pair} -> {idx}")

    return merges, vocab
