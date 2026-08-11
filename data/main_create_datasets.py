"""Builds eval_dataset_clean.parquet and train_raw.parquet for the LLM01/LLM07 classifier.

Assembles data from the sources listed in readme.md, normalizes them into a common schema, dedups
across all sources combined, then partitions the result into a clean held-out evaluation set and
a raw pool left for future fine-tuning. Run from within this directory: `python main_create_datasets.py`.
"""

import os

import pandas as pd
from huggingface_hub import login

from helpers.dataset_loaders import (
    SCHEMA_COLUMNS,
    load_deepset,
    load_gandalf,
    load_jayavibhav,
    load_neuralchemy,
    load_notinject,
    load_tensor_trust_extraction,
    load_tensor_trust_hijacking,
    load_wildguard,
    load_xstest,
    normalized_hash,
    read_env,
)

_TENSOR_TRUST_BASE = "https://raw.githubusercontent.com/HumanCompatibleAI/tensor-trust-data/main/benchmarks"
TENSOR_TRUST_HIJACK_URL = f"{_TENSOR_TRUST_BASE}/hijacking-robustness/v1/hijacking_robustness_dataset.jsonl"
TENSOR_TRUST_EXTRACT_URL = f"{_TENSOR_TRUST_BASE}/extraction-robustness/v1/extraction_robustness_dataset.jsonl"

DEEPSET_RELABELED_PATH = "deepset_relabeled.csv"
CLEAN_EVAL_PATH = "eval_dataset_clean.parquet"
TRAIN_RAW_PATH = "train_raw.parquet"

# Smallest, cleanest, most-audited sources (or slices) — held out entirely from training and used
# as-is for zero-shot evaluation. Everything else is pooled into train_raw. See readme.md.
CLEAN_EVAL_SOURCES = {
    "tensor_trust_extraction",
    "deepset/prompt-injections",
    "leolee99/NotInject",
    "natolambert/xstest-v2-copy",
    "Lakera/gandalf_ignore_instructions",
    "allenai/wildguardmix",
    "neuralchemy/Prompt-injection-dataset:clean",
}


def build_frames() -> dict[str, pd.DataFrame]:
    """Load every source and return each already reshaped into the target schema."""
    hf_token = os.environ.get("HF_TOKEN") or read_env().get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    # Insertion order matters: the cross-source dedup in combine_and_dedup() keeps the first
    # occurrence of each normalized text, so this must match the source order the eval/train_raw
    # parquet files were originally built with.
    frames = {
        "tensor_trust_hijacking": load_tensor_trust_hijacking(TENSOR_TRUST_HIJACK_URL),
        "tensor_trust_extraction": load_tensor_trust_extraction(TENSOR_TRUST_EXTRACT_URL),
        "jayavibhav/prompt-injection-safety": load_jayavibhav(),
        **load_neuralchemy(),
        "Lakera/gandalf_ignore_instructions": load_gandalf(),
        "leolee99/NotInject": load_notinject(),
        "allenai/wildguardmix": load_wildguard(),
        "deepset/prompt-injections": load_deepset(DEEPSET_RELABELED_PATH),
        "natolambert/xstest-v2-copy": load_xstest(),
    }
    return frames


def combine_and_dedup(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate every source, drop empty text, then dedup exact matches on normalized text
    across all sources combined (keeping the first occurrence)."""
    combined = pd.concat(frames.values(), ignore_index=True)
    combined = combined.dropna(subset=["text"]).reset_index(drop=True)
    print(f"Combined (pre-dedup): {len(combined)} rows")

    combined["_hash"] = combined["text"].map(normalized_hash)
    before = len(combined)
    df = combined.drop_duplicates(subset=["_hash"], keep="first").drop(columns=["_hash"]).reset_index(drop=True)
    print(f"Dropped {before - len(df)} cross-source duplicates — {len(df)} rows remain")
    return df


def partition(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into the clean eval set (CLEAN_EVAL_SOURCES) and the train_raw pool (everything else)."""
    clean_eval_df = df[df["source"].isin(CLEAN_EVAL_SOURCES)].reset_index(drop=True)
    train_raw_df = df[~df["source"].isin(CLEAN_EVAL_SOURCES)].reset_index(drop=True)
    print(f"Clean eval set: {len(clean_eval_df)} rows")
    print(f"train_raw pool: {len(train_raw_df)} rows")
    return clean_eval_df, train_raw_df


def save_and_validate(clean_eval_df: pd.DataFrame, train_raw_df: pd.DataFrame) -> None:
    """Write both partitions to parquet, then reload each to confirm the round trip."""
    clean_eval_df[SCHEMA_COLUMNS].to_parquet(CLEAN_EVAL_PATH, index=False)
    train_raw_df[SCHEMA_COLUMNS].to_parquet(TRAIN_RAW_PATH, index=False)
    print(f"Saved {len(clean_eval_df)} rows to {CLEAN_EVAL_PATH}")
    print(f"Saved {len(train_raw_df)} rows to {TRAIN_RAW_PATH}")

    for path, expected_len in [(CLEAN_EVAL_PATH, len(clean_eval_df)), (TRAIN_RAW_PATH, len(train_raw_df))]:
        reloaded = pd.read_parquet(path)
        assert list(reloaded.columns) == SCHEMA_COLUMNS
        assert len(reloaded) == expected_len
        print(f"Round-trip OK: {path} {reloaded.shape}")


def main() -> None:
    frames = build_frames()
    df = combine_and_dedup(frames)
    clean_eval_df, train_raw_df = partition(df)
    save_and_validate(clean_eval_df, train_raw_df)


if __name__ == "__main__":
    main()
