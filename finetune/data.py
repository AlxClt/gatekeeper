"""Data loading and tokenization for the gatekeeper fine-tune.

Loads data/train_consolidated.parquet and turns each (text, label) row into a single-turn
Gemma-2 chat example: one user turn holding the classification prompt (prompt.yaml, same
{{input}} substitution convention as app/verification/verifier.py), one model turn holding the
single-character label. The loss is masked to the label token + <end_of_turn> only — everything
else (the prompt) is context, not a training target.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

PROMPT_PATH = Path(__file__).parent / "prompt.yaml"

# Mirrors app/verification/preprocessing.py's _MAX_LENGTH: the served model never sees more than
# 4000 chars of input, so training text is capped the same way to keep the two distributions
# consistent.
TEXT_MAX_CHARS = 4000

_PROMPT_TEMPLATE = yaml.safe_load(PROMPT_PATH.read_text())


def build_user_content(text: str) -> str:
    """Fills the fine-tuning prompt template with a (truncated) input text."""
    return _PROMPT_TEMPLATE.replace("{{input}}", text[:TEXT_MAX_CHARS])


def load_split(parquet_path: str, val_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads train_consolidated.parquet and returns a stratified (by label) train/val split.

    This val split only monitors training (early stopping, loss/metric curves) — it is drawn from
    the same pool the model trains on. The real held-out benchmark is
    evaluation/eval_dataset_clean.parquet, scored separately by evaluation/evaluation.ipynb once
    this model is deployed behind the gatekeeper API.
    """
    df = pd.read_parquet(parquet_path)
    train_df, val_df = train_test_split(
        df, test_size=val_fraction, stratify=df["label"], random_state=seed,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


class GatekeeperDataset(Dataset):
    """Tokenizes each row into a Gemma-2 chat turn (user: prompt, model: label) with the loss
    masked to everything except the label digit + <end_of_turn>."""

    def __init__(self, df: pd.DataFrame, tokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        examples = [self._tokenize(row.text, int(row.label)) for row in df.itertuples()]
        n_dropped = sum(1 for ex in examples if ex is None or len(ex["input_ids"]) > max_seq_len)
        self.examples = [ex for ex in examples if ex is not None and len(ex["input_ids"]) <= max_seq_len]
        if n_dropped:
            print(f"Dropped {n_dropped}/{len(df)} rows exceeding max_seq_len={max_seq_len} tokens")

    def _tokenize(self, text: str, label: int) -> dict:
        messages = [{"role": "user", "content": build_user_content(text)}]
        prompt_str = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        full_str = prompt_str + f"{label}<end_of_turn>"

        # Special-token literals (e.g. <bos>, <end_of_turn>) rendered by apply_chat_template are
        # recognized back into their ids on encode regardless of add_special_tokens — only the
        # automatic prepending of extra tokens is what that flag controls, so this stays exact.
        prompt_ids = self.tokenizer(prompt_str, add_special_tokens=False).input_ids
        full_ids = self.tokenizer(full_str, add_special_tokens=False).input_ids

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]
        return {"input_ids": full_ids, "labels": labels}

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


@dataclass
class DataCollator:
    """Dynamically pads a batch to its longest sequence; pads labels with -100 so padding never
    contributes to the loss."""

    tokenizer: object

    def __call__(self, batch: list[dict]) -> dict:
        max_len = max(len(ex["input_ids"]) for ex in batch)
        pad_id = self.tokenizer.pad_token_id
        input_ids, attention_mask, labels = [], [], []
        for ex in batch:
            pad_len = max_len - len(ex["input_ids"])
            input_ids.append(ex["input_ids"] + [pad_id] * pad_len)
            attention_mask.append([1] * len(ex["input_ids"]) + [0] * pad_len)
            labels.append(ex["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }
