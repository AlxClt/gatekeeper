"""Per-source loaders for the gatekeeper evaluation/training data build.

Each loader returns a DataFrame already reshaped into the target schema
(text, label, source, threat_class, in_scope) so main_create_datasets.py only
has to combine, dedup, partition, and save. See readme.md for the dataset
list and the rationale behind each source's scope/mapping.
"""

import hashlib
import re
from pathlib import Path

import pandas as pd
from datasets import concatenate_datasets, load_dataset

SCHEMA_COLUMNS = ["text", "label", "source", "threat_class", "in_scope"]


def read_env(env_path: str = "../.env") -> dict:
    """Parse simple KEY=VALUE lines from a .env file."""
    path = Path(env_path)
    if not path.exists():
        return {}
    env_vars = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env_vars[key.strip()] = value.strip()
    return env_vars


def normalized_hash(text: str) -> str:
    """Lowercase + collapse whitespace + sha1, used for cross-source dedup."""
    normalized = re.sub(r"\s+", " ", str(text).strip().lower())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def load_full_dataset(path: str, config: str | None = None) -> pd.DataFrame:
    """Load every split of a HF dataset (optionally a named config) and concatenate into one frame."""
    loaded = load_dataset(path, config) if config else load_dataset(path)
    return concatenate_datasets([loaded[split] for split in loaded.keys()]).to_pandas()


def try_fetch(load_fn, source: str, hint: str) -> pd.DataFrame | None:
    """Run `load_fn()`; on failure (gated access, network issue, moved file), print `hint` and
    return None instead of raising, so one unavailable source doesn't break the whole build."""
    try:
        return load_fn()
    except Exception as exc:
        print(f"Skipping {source} — could not load.\n{hint}\nOriginal error: {exc}")
        return None


def _aligned(value, n: int):
    """Reset a Series' index to 0..n-1 (positional alignment); pass constants through untouched."""
    return value.reset_index(drop=True) if isinstance(value, pd.Series) else value


def finalize(df: pd.DataFrame, *, text_col: str, label, source: str, threat_class, in_scope) -> pd.DataFrame:
    """Select/rename into {text, label, source, threat_class, in_scope}.

    `text_col` is always a column name pulled from `df`. `label`, `threat_class`, and `in_scope`
    are each either a constant (applied to every row) or an already-computed `pd.Series` aligned
    positionally to `df` (e.g. `df["label"]` or `df["label"].map(...)`).
    """
    n = len(df)
    out = pd.DataFrame({"text": _aligned(df[text_col], n)})
    out["label"] = _aligned(label, n)
    out["source"] = source
    out["threat_class"] = _aligned(threat_class, n)
    out["in_scope"] = _aligned(in_scope, n)
    return out[SCHEMA_COLUMNS]


# ── Tensor Trust — hijacking (LLM01 anchor) + extraction (LLM07 anchor) ─────
#
# Purpose-built, human-generated, manually cleaned adversarial sets, published as small JSONL
# files at https://github.com/HumanCompatibleAI/tensor-trust-data (not on the HF Hub). Neither
# file ships a label column — every row is an attack, so both are treated as all-positive.

_TENSOR_TRUST_HINT = (
    "Check that https://github.com/HumanCompatibleAI/tensor-trust-data is reachable "
    "and the file path hasn't moved."
)


def load_tensor_trust_hijacking(url: str) -> pd.DataFrame:
    raw = try_fetch(lambda: pd.read_json(url, lines=True), source="tensor_trust_hijacking", hint=_TENSOR_TRUST_HINT)
    if raw is None:
        return pd.DataFrame(columns=SCHEMA_COLUMNS)
    return finalize(raw, text_col="attack", label=1, source="tensor_trust_hijacking", threat_class="LLM01", in_scope=True)


def load_tensor_trust_extraction(url: str) -> pd.DataFrame:
    raw = try_fetch(lambda: pd.read_json(url, lines=True), source="tensor_trust_extraction", hint=_TENSOR_TRUST_HINT)
    if raw is None:
        return pd.DataFrame(columns=SCHEMA_COLUMNS)
    return finalize(raw, text_col="attack", label=1, source="tensor_trust_extraction", threat_class="LLM07", in_scope=True)


# ── jayavibhav/prompt-injection-safety ──────────────────────────────────────
#
# The only large set that pre-separates harmful content from injection at the label level:
# 0=benign, 1=prompt injection, 2=direct harmful request. Label 2 is retained for audit but
# excluded from scoring (in_scope=False).

_JAYAVIBHAV_LABEL = {0: 0, 1: 1, 2: 0}  # harmful_content (2) scores as 0 — no injection present
_JAYAVIBHAV_THREAT_CLASS = {0: "benign", 1: "LLM01", 2: "harmful_content"}
_JAYAVIBHAV_IN_SCOPE = {0: True, 1: True, 2: False}


def load_jayavibhav() -> pd.DataFrame:
    raw = load_full_dataset("jayavibhav/prompt-injection-safety")
    return finalize(
        raw,
        text_col="text",
        label=raw["label"].map(_JAYAVIBHAV_LABEL),
        source="jayavibhav/prompt-injection-safety",
        threat_class=raw["label"].map(_JAYAVIBHAV_THREAT_CLASS),
        in_scope=raw["label"].map(_JAYAVIBHAV_IN_SCOPE),
    )


# ── neuralchemy/Prompt-injection-dataset ────────────────────────────────────
#
# Multi-category taxonomy dataset. Uses the `core` config only (originals, no paraphrase
# augmentation) and filters `category` to an explicit whitelist, mapped to threat_class and split
# into a clean eval slice and a train_raw slice — see readme.md for the full category breakdown
# and why each category lands where it does.

_NEURALCHEMY_CATEGORY_THREAT_CLASS = {
    "benign": "benign",
    "control": "benign",
    "crescendo": "benign",
    "edge_case": "benign",
    "agent_manipulation": "LLM01",
    "code_execution": "LLM01",
    "context_confusion": "LLM01",
    "indirect_injection": "LLM01",
    "instruction_override": "LLM01",
    "output_manipulation": "LLM01",
    "prompt_injection": "LLM01",
    "rag_poisoning": "LLM01",
    "token_smuggling": "LLM01",
    "direct_injection": "LLM01",
    "prompt_extraction": "LLM07",
    "system_extraction": "LLM07",
}

_NEURALCHEMY_CLEAN_CATEGORIES = [
    "prompt_extraction", "system_extraction", "agent_manipulation", "code_execution",
    "context_confusion", "indirect_injection", "instruction_override",
    "output_manipulation", "prompt_injection", "rag_poisoning", "token_smuggling",
]

_NEURALCHEMY_TRAIN_CATEGORIES = ["direct_injection", "benign", "control", "crescendo", "edge_case"]


def load_neuralchemy() -> dict[str, pd.DataFrame]:
    """Returns two sources: the clean LLM01/LLM07 slice (eval-bound) and the rest (train_raw-bound)."""
    raw = load_full_dataset("neuralchemy/Prompt-injection-dataset", config="core")
    scoped = raw[(~raw["augmented"]) & raw["category"].isin(_NEURALCHEMY_CATEGORY_THREAT_CLASS)].reset_index(drop=True)

    clean = scoped[scoped["category"].isin(_NEURALCHEMY_CLEAN_CATEGORIES)]
    rest = scoped[scoped["category"].isin(_NEURALCHEMY_TRAIN_CATEGORIES)]

    return {
        "neuralchemy/Prompt-injection-dataset:clean": finalize(
            clean,
            text_col="text",
            label=clean["label"],
            source="neuralchemy/Prompt-injection-dataset:clean",
            threat_class=clean["category"].map(_NEURALCHEMY_CATEGORY_THREAT_CLASS),
            in_scope=True,
        ),
        "neuralchemy/Prompt-injection-dataset": finalize(
            rest,
            text_col="text",
            label=rest["label"],
            source="neuralchemy/Prompt-injection-dataset",
            threat_class=rest["category"].map(_NEURALCHEMY_CATEGORY_THREAT_CLASS),
            in_scope=True,
        ),
    }


# ── Lakera/gandalf_ignore_instructions ──────────────────────────────────────
#
# Ships no label column — every row attacks Gandalf to leak its hidden password/system prompt,
# so all rows are label=1. All-positive (no negatives).


def load_gandalf() -> pd.DataFrame:
    raw = load_full_dataset("Lakera/gandalf_ignore_instructions")
    return finalize(raw, text_col="text", label=1, source="Lakera/gandalf_ignore_instructions", threat_class="LLM07", in_scope=True)


# ── leolee99/NotInject — over-defense / FPR probe ───────────────────────────
#
# Benign prompts loaded with injection trigger words ("ignore", "system", etc.) — the single
# most valuable negative set for measuring over-firing.


def load_notinject() -> pd.DataFrame:
    raw = load_full_dataset("leolee99/NotInject")
    return finalize(raw, text_col="prompt", label=0, source="leolee99/NotInject", threat_class="benign", in_scope=True)


# ── allenai/wildguardmix (wildguardtest config) — plain benign negatives ───
#
# Gated — requires accepting terms on HF and a token. Skips gracefully with instructions if
# unavailable, rather than failing the whole build.


def load_wildguard() -> pd.DataFrame:
    raw = try_fetch(
        lambda: load_full_dataset("allenai/wildguardmix", config="wildguardtest"),
        source="allenai/wildguardmix",
        hint=(
            "Accept the terms at https://huggingface.co/datasets/allenai/wildguardmix, "
            "set HF_TOKEN in the environment, then re-run."
        ),
    )
    if raw is None:
        return pd.DataFrame(columns=SCHEMA_COLUMNS)

    benign = raw[raw["prompt_harm_label"] == "unharmful"]
    return finalize(benign, text_col="prompt", label=0, source="allenai/wildguardmix", threat_class="benign", in_scope=True)


# ── deepset/prompt-injections — regression anchor ───────────────────────────
#
# Loaded from a local, manually relabeled copy (deepset_relabeled.csv) instead of straight from
# HuggingFace — a pass over the native labels found some that didn't hold up on inspection, so
# they were corrected by hand. Already in the target schema.


def load_deepset(csv_path: str) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, index_col=0)
    return raw[SCHEMA_COLUMNS].reset_index(drop=True)


# ── natolambert/xstest-v2-copy — legitimate meta-questions (hard negatives) ─
#
# Benign questions that sound risky (e.g. "how do I kill a process?"). Only the bare `prompts`
# split is used; the `type` column marks unsafe contrast rows with a `contrast_` prefix.


def load_xstest() -> pd.DataFrame:
    prompts = load_dataset("natolambert/xstest-v2-copy")["prompts"].to_pandas()
    safe = prompts[~prompts["type"].str.startswith("contrast_")]
    return finalize(safe, text_col="prompt", label=0, source="natolambert/xstest-v2-copy", threat_class="benign", in_scope=True)
