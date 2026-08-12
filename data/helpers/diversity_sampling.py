"""Per-source, per-label diversity sampling for the consolidated training set.

Embeds each row with a sentence-transformers encoder, PCA-reduces the embeddings, then keeps a
diverse subset of each `source` x `label` group via greedy max-min dispersion (the same objective
submodlib's DisparityMinFunction optimizes: repeatedly pick the point farthest — in embedding
space — from everything already picked). Stratifying by label as well as source preserves each
source's label balance, since dispersion sampling by itself has no notion of label. Kept separate
from main_train_set_consolidation.py because the embedding model, PCA fit, and dispersion search
are the heavy, slow part of that script.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from tqdm import tqdm

# On CPU (no GPU here) this 3-layer model runs ~2.5-3x faster than the 6-layer all-MiniLM-L6-v2
# for a modest quality tradeoff, and small batches beat the default 32 — larger batches were
# consistently *slower* in benchmarking (more padding waste, no CPU parallelism payoff).
EMBEDDING_MODEL = "sentence-transformers/paraphrase-MiniLM-L3-v2"
EMBEDDING_BATCH_SIZE = 8
PCA_DIMS = 100              # target embedding dimensionality after PCA vs 384 for paraphrase-MiniLM-L3-v2 (keeps 81.2%)

# Fraction of each source kept by dispersion sampling — set per source since sources vary wildly in
# size/redundancy (jayavibhav dominates the pool and is heavily downsampled; tensor_trust_hijacking
# is small and kept whole). Sources not listed here fall back to DEFAULT_SAMPLE_FRACTION.
SOURCE_SAMPLE_FRACTIONS = {
    "tensor_trust_hijacking": 1.0,
    "neuralchemy/Prompt-injection-dataset": 0.75,
    "jayavibhav/prompt-injection-safety": 0.15,
}
DEFAULT_SAMPLE_FRACTION = 1.0

MIN_SOURCE_SIZE_TO_SAMPLE = 20  # sources smaller than this are kept whole — nothing to diversify away
VIZ_SAMPLE_SIZE = 600          # points per panel in the comparison plot
RANDOM_SEED = 0


def embed(texts: list[str], model_name: str = EMBEDDING_MODEL, batch_size: int = EMBEDDING_BATCH_SIZE) -> np.ndarray:
    """Encode texts with a sentence-transformers model and L2-normalize the resulting vectors."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def reduce_dimensionality(embeddings: np.ndarray, n_components: int = PCA_DIMS) -> np.ndarray:
    """PCA down to n_components (capped to whatever's actually available). Prints the fraction of
    total variance retained, since n_components is a lossy tradeoff (see EMBEDDING_MODEL/PCA_DIMS
    comment) — worth knowing how much signal each run actually keeps."""
    n_components = min(n_components, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    reduced = pca.fit_transform(embeddings)
    explained = pca.explained_variance_ratio_.sum()
    print(f"PCA: {n_components} dims retain {explained:.1%} of total variance")
    return reduced


def max_min_dispersion_sample(
    vectors: np.ndarray, k: int, seed: int = RANDOM_SEED, desc: str = "Dispersion sampling"
) -> list[int]:
    """Greedily select k rows maximizing the minimum pairwise distance to what's already selected
    (max-min dispersion — the diversity baseline submodlib's DisparityMinFunction targets).

    Starts from a random point, then repeatedly adds whichever remaining point is farthest from
    its nearest already-selected neighbor. Squared euclidean distance is tracked via the identity
    ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b so each round is a single BLAS matrix-vector product
    instead of an O(n*d) elementwise pass.
    """
    n = vectors.shape[0]
    if k >= n:
        return list(range(n))

    rng = np.random.default_rng(seed)
    norms_sq = np.einsum("ij,ij->i", vectors, vectors)

    first = int(rng.integers(n))
    selected = [first]
    min_dist = norms_sq + norms_sq[first] - 2 * (vectors @ vectors[first])
    min_dist[first] = -1

    for _ in tqdm(range(k - 1), desc=desc):
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        new_dist = norms_sq + norms_sq[nxt] - 2 * (vectors @ vectors[nxt])
        np.minimum(min_dist, new_dist, out=min_dist)
        min_dist[nxt] = -1

    return selected


def sample_per_source(
    df: pd.DataFrame,
    vectors: np.ndarray,
    fractions: dict[str, float] = SOURCE_SAMPLE_FRACTIONS,
    default_fraction: float = DEFAULT_SAMPLE_FRACTION,
    min_source_size: int = MIN_SOURCE_SIZE_TO_SAMPLE,
) -> np.ndarray:
    """Run max-min dispersion sampling independently within each (`source`, `label`) group, keeping
    `fractions[source]` of each group (or `default_fraction` if the source isn't listed). Stratifying
    by label too — not just source — keeps each source's label balance intact, since dispersion
    sampling has no notion of label and could otherwise skew it. Returns a boolean keep-mask aligned
    with df/vectors. Groups smaller than min_source_size are kept whole."""
    keep_mask = np.zeros(len(df), dtype=bool)

    for (source, label), group in df.groupby(["source", "label"]):
        idx = group.index.to_numpy()
        if len(idx) < min_source_size:
            keep_mask[idx] = True
            print(f"  {source} (label={label}): kept all {len(idx)} rows (below min_source_size)")
            continue

        fraction = fractions.get(source, default_fraction)
        k = max(1, round(len(idx) * fraction))
        local_selected = max_min_dispersion_sample(
            vectors[idx], k, desc=f"Dispersion sampling ({source}, label={label})"
        )
        keep_mask[idx[local_selected]] = True
        print(f"  {source} (label={label}): kept {len(local_selected)}/{len(idx)} rows (fraction={fraction})")

    return keep_mask


def _scatter_by_source(ax: plt.Axes, sources: np.ndarray, coords_2d: np.ndarray, title: str) -> None:
    """Scatter coords_2d on ax, colored by source, with one legend entry per source."""
    for source in sorted(set(sources)):
        mask = sources == source
        ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1], s=14, alpha=0.7, label=source)
    ax.set_title(title)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend(fontsize=7, markerscale=1.5)

def vendi(V: np.ndarray, q: float = 1.0) -> float:
    """Vendi score of a set of vectors (Friedman & Dieng, 2022) — an effective-diversity-count
    metric: exp of the (Renyi-q) entropy of the eigenvalues of the normalized similarity matrix.
    Computed on the d x d Gram matrix V.T @ V rather than the usual n x n similarity matrix V @ V.T
    — for a linear/cosine kernel they share the same nonzero eigenvalues, but this is O(d^3) instead
    of O(n^3), which matters since d (PCA_DIMS) stays fixed while n can be tens of thousands of rows.
    """
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    n = V.shape[0]
    G = (V.T @ V) / n                 # (d,d), trace = 1
    lam = np.linalg.eigvalsh(G)
    lam = lam[lam > 1e-12]            # drop numerical zeros/negatives
    if q == 1.0:      H = -np.sum(lam * np.log(lam))       # Shannon
    elif np.isinf(q): H = -np.log(lam.max())               # min-entropy
    else:             H = np.log(np.sum(lam**q)) / (1 - q) # Rényi-q
    return np.exp(H)


def save_comparison_plot(
    df: pd.DataFrame,
    coords_2d: np.ndarray,
    keep_mask: np.ndarray,
    path: str,
    sample_size: int = VIZ_SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
) -> None:
    """Save a 2-panel figure: a random sample vs. the diversity-sampled rows, both projected on the
    first 2 PCA dimensions. The "random" panel is drawn with the same per-source sample sizes the
    diversity sampling actually kept (rather than one flat random draw over the whole df), so the
    only difference between panels is *how* rows are picked within each source — not the source
    mix — isolating the effect of dispersion sampling. The diversity-sampled panel should visibly
    spread out more, since it drops near-neighbors in embedding space. Each panel's title reports its
    Vendi score, computed on the full (PCA_DIMS-dimensional) embedding matrix — not just the 2
    plotted dimensions — and the figure title reports the full (pre-sampling) dataset's Vendi score
    for reference."""
    rng = np.random.default_rng(seed)
    sources = df["source"].to_numpy()

    kept_per_source = pd.Series(sources[keep_mask]).value_counts()
    original_idx = np.concatenate([
        rng.choice(np.flatnonzero(sources == source), size=n_kept, replace=False)
        for source, n_kept in kept_per_source.items()
    ])
    if len(original_idx) > sample_size:
        original_idx = rng.choice(original_idx, size=sample_size, replace=False)

    sampled_pool = np.flatnonzero(keep_mask)
    sampled_idx = rng.choice(sampled_pool, size=min(sample_size, len(sampled_pool)), replace=False)

    full_vendi = vendi(coords_2d)
    original_vendi = vendi(coords_2d[original_idx])
    sampled_vendi = vendi(coords_2d[sampled_idx])
    print(
        f"Vendi scores — full dataset (n={len(df)}): {full_vendi:.1f}, "
        f"random matched-ratio sample (n={len(original_idx)}): {original_vendi:.1f}, "
        f"diversity-sampled (n={len(sampled_idx)}): {sampled_vendi:.1f}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    _scatter_by_source(
        axes[0], sources[original_idx], coords_2d[original_idx],
        f"Random sample (matched per-source ratio)\nVendi score: {original_vendi:.1f}",
    )
    _scatter_by_source(
        axes[1], sources[sampled_idx], coords_2d[sampled_idx],
        f"Max-min dispersion sample\nVendi score: {sampled_vendi:.1f}",
    )
    fig.suptitle(
        f"PCA (first 2 components): random vs. diversity-sampled subset\n"
        f"Full dataset Vendi score: {full_vendi:.1f} (n={len(df)})"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved diversity sampling comparison plot to {path}")
