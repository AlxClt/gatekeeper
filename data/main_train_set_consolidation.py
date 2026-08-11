"""Consolidates train_distilled.parquet into the final fine-tuning training set.

Loads the distilled training pool, removes near duplicates (MinHash LSH, see
helpers/near_duplicates.py), applies per-source stratified diversity sampling to keep the pool
balanced, then appends the handcrafted LLM07 examples (which skip both dedup and sampling — they're
small, hand-audited, and meant to be kept in full). Run from within this directory:
`python main_train_set_consolidation.py`.
"""

import pandas as pd

from helpers.dataset_loaders import SCHEMA_COLUMNS
from helpers.diversity_sampling import embed, reduce_dimensionality, sample_per_source, save_comparison_plot
from helpers.near_duplicates import remove_near_duplicates

TRAIN_DISTILLED_PATH = "train_distilled.parquet"
HANDCRAFTED_LLM07_PATH = "handcrafted_llm07.csv"
TRAIN_CONSOLIDATED_PATH = "train_consolidated.parquet"
DIVERSITY_PLOT_PATH = "diversity_sampling_comparison.png"


def stratified_diversity_sampling(df: pd.DataFrame) -> pd.DataFrame:
    """Embed each row, PCA-reduce the embeddings, then keep a diverse subset of each source via
    max-min dispersion sampling (see helpers/diversity_sampling.py). Also saves a before/after scatter plot
    of the first 2 PCA dimensions to DIVERSITY_PLOT_PATH."""
    embeddings = embed(df["text"].tolist())
    coords = reduce_dimensionality(embeddings)

    print("Sampling per source:")
    keep_mask = sample_per_source(df, coords)

    save_comparison_plot(df, coords, keep_mask, DIVERSITY_PLOT_PATH)
    return df[keep_mask].reset_index(drop=True)


def load_handcrafted_llm07() -> pd.DataFrame:
    """Load the handcrafted LLM07 examples, already shaped to the target schema."""
    df = pd.read_csv(HANDCRAFTED_LLM07_PATH)
    return df[SCHEMA_COLUMNS]


def main() -> None:
    df = pd.read_parquet(TRAIN_DISTILLED_PATH)
    print(f"Loaded {len(df)} rows from {TRAIN_DISTILLED_PATH}")

    df = remove_near_duplicates(df)  # prints its own drop count

    df = stratified_diversity_sampling(df)
    print(f"{len(df)} rows remain after stratified diversity sampling")

    handcrafted = load_handcrafted_llm07()
    print(f"Appending {len(handcrafted)} handcrafted LLM07 rows from {HANDCRAFTED_LLM07_PATH}")

    consolidated = pd.concat([df, handcrafted], ignore_index=True)[SCHEMA_COLUMNS]
    consolidated.to_parquet(TRAIN_CONSOLIDATED_PATH, index=False)
    print(f"Saved {len(consolidated)} rows to {TRAIN_CONSOLIDATED_PATH}")


if __name__ == "__main__":
    main()
