"""MinHash-based near-duplicate detection for text datasets.

Uses datasketch's MinHashLSH to greedily drop rows whose text is highly similar to (but not
necessarily byte-identical with) an earlier row — paraphrases, minor edits, whitespace/punctuation
variants that survive the exact-match dedup in dataset_loaders.normalized_hash. Kept separate from
main_train_set_consolidation.py because building and querying the LSH index is the heavy, iterative
part of that script.
"""

import re

import pandas as pd
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

NUM_PERM = 128               # MinHash permutations — higher = more accurate, slower
SHINGLE_SIZE = 3             # word n-gram size used to build each MinHash
SIMILARITY_THRESHOLD = 0.8   # Jaccard similarity at/above which two rows count as near-duplicates


def _shingles(text: str, k: int = SHINGLE_SIZE) -> set[str]:
    """Word k-shingles of the normalized text, used as the MinHash input set."""
    words = re.sub(r"\s+", " ", str(text).strip().lower()).split(" ")
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def _minhash(text: str, num_perm: int = NUM_PERM) -> MinHash:
    mh = MinHash(num_perm=num_perm)
    for shingle in _shingles(text):
        mh.update(shingle.encode("utf-8"))
    return mh


def remove_near_duplicates(
    df: pd.DataFrame,
    text_col: str = "text",
    threshold: float = SIMILARITY_THRESHOLD,
    num_perm: int = NUM_PERM,
) -> pd.DataFrame:
    """Greedily drop rows whose text is a near-duplicate (Jaccard similarity >= threshold) of an
    earlier row, keeping the first occurrence — same convention as the exact-match dedup in
    dataset_loaders.normalized_hash."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    keep_mask = [False] * len(df)

    for i, text in enumerate(tqdm(df[text_col], desc="Near-duplicate scan")):
        mh = _minhash(text, num_perm=num_perm)
        if lsh.query(mh):
            continue
        lsh.insert(str(i), mh)
        keep_mask[i] = True

    kept = df[keep_mask].reset_index(drop=True)
    print(f"Dropped {len(df) - len(kept)} near-duplicates — {len(kept)} rows remain")
    return kept
