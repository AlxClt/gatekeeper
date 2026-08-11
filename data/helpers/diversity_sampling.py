"""Per-source diversity sampling for the consolidated training set.

Embeds each row with a sentence-transformers encoder, PCA-reduces the embeddings, then keeps a
diverse subset of each `source` group via greedy max-min dispersion (the same objective submodlib's
DisparityMinFunction optimizes: repeatedly pick the point farthest — in embedding space — from
everything already picked). Kept separate from main_train_set_consolidation.py because the
embedding model, PCA fit, and dispersion search are the heavy, slow part of that script.
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
PCA_DIMS = 75                  # target embedding dimensionality after PCA (50-100 range)

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
    """PCA down to n_components (capped to whatever's actually available)."""
    n_components = min(n_components, embeddings.shape[0], embeddings.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    return pca.fit_transform(embeddings)


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
    """Run max-min dispersion sampling independently within each `source` group, keeping
    `fractions[source]` of each group (or `default_fraction` if the source isn't listed). Returns a
    boolean keep-mask aligned with df/vectors. Sources smaller than min_source_size are kept whole."""
    keep_mask = np.zeros(len(df), dtype=bool)

    for source, group in df.groupby("source"):
        idx = group.index.to_numpy()
        if len(idx) < min_source_size:
            keep_mask[idx] = True
            print(f"  {source}: kept all {len(idx)} rows (below min_source_size)")
            continue

        fraction = fractions.get(source, default_fraction)
        k = max(1, round(len(idx) * fraction))
        local_selected = max_min_dispersion_sample(vectors[idx], k, desc=f"Dispersion sampling ({source})")
        keep_mask[idx[local_selected]] = True
        print(f"  {source}: kept {len(local_selected)}/{len(idx)} rows (fraction={fraction})")

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


def save_comparison_plot(
    df: pd.DataFrame,
    coords_2d: np.ndarray,
    keep_mask: np.ndarray,
    path: str,
    sample_size: int = VIZ_SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
) -> None:
    """Save a 2-panel figure: a random sample of the original rows vs. a random sample of the
    diversity-sampled rows, both projected on the first 2 PCA dimensions. The diversity-sampled
    panel should visibly spread out more, since it drops near-neighbors in embedding space."""
    rng = np.random.default_rng(seed)

    original_idx = rng.choice(len(df), size=min(sample_size, len(df)), replace=False)
    sampled_pool = np.flatnonzero(keep_mask)
    sampled_idx = rng.choice(sampled_pool, size=min(sample_size, len(sampled_pool)), replace=False)

    sources = df["source"].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    _scatter_by_source(axes[0], sources[original_idx], coords_2d[original_idx], "Random sample (original)")
    _scatter_by_source(axes[1], sources[sampled_idx], coords_2d[sampled_idx], "Max-min dispersion sample")
    fig.suptitle("PCA (first 2 components): random vs. diversity-sampled subset")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved diversity sampling comparison plot to {path}")
