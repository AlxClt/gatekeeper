# Data

Datasets used to build and evaluate the gatekeeper classifier, scoped to **OWASP LLM01**
(prompt injection / instruction hijacking) and **LLM07** (system-prompt / secret extraction).
Harmful-content jailbreaks are explicitly out of scope — a different filter owns them, and mixing
them in wrecks precision/recall for this classifier.

Built by [`main_create_datasets.py`](main_create_datasets.py) (loaders externalized in
[`dataset_loaders.py`](dataset_loaders.py)), which pulls from the sources below, normalizes them
into a common schema, deduplicates across all sources, and partitions the result into an
evaluation set and a raw pool for future training. Run with `python main_create_datasets.py` from
within this directory.

## Target schema (both output files)

| field | type | meaning |
|---|---|---|
| `text` | str | raw user input to classify |
| `label` | int | gold label: `1` = LLM01/LLM07 threat, `0` = not a threat |
| `source` | str | dataset id |
| `threat_class` | str | `LLM01` \| `LLM07` \| `benign` \| `harmful_content` |
| `in_scope` | bool | `True` = score it; `False` = keep for audit, exclude from metrics |

Harmful-content rows are retained with `in_scope=False` and `label=0` (a pure harmful prompt with
no injection *should* get classifier output `0`). `LLM07` and `LLM01` are not mutually exclusive
in the source datasets — slices are labeled with their most representative category.

Deduplication is exact-match on the (lowercased, whitespace-collapsed) `text` field, applied once
across all sources combined, before the evaluation/training partition.

## Datasets

| Threat | Source | Notes |
|---|---|---|
| LLM01 | Tensor Trust hijacking (775 rows) | fetched directly from GitHub raw — training pool |
| LLM01 / benign / harmful_content | `jayavibhav/prompt-injection-safety` | 3-way label, harmful_content marked `in_scope=False` — training pool |
| LLM01 / benign | `neuralchemy/Prompt-injection-dataset` (`core` config) | `direct_injection` + benign-mapped categories; category whitelist per spec — training pool |
| LLM01 / LLM07 | `neuralchemy/Prompt-injection-dataset:clean` | 9 LLM01 + 2 LLM07 categories split out of the frame above — evaluation set |
| LLM07 | Tensor Trust extraction (569 rows) | fetched directly from GitHub raw, no manual download needed — evaluation set |
| LLM07 | `Lakera/gandalf_ignore_instructions` | attack-only, no native negatives — evaluation set |
| benign | `leolee99/NotInject` | benign prompts loaded with injection trigger words — the key false-positive-rate probe — evaluation set |
| benign | `allenai/wildguardmix` (`wildguardtest`) | gated — requires `HF_TOKEN`; skipped gracefully otherwise — evaluation set |
| LLM01 / benign | `deepset/prompt-injections` | small, clean, bilingual — manually relabeled out-of-scope examples (`deepset_relabeled.csv` is the source of truth) — evaluation set |
| benign | `natolambert/xstest-v2-copy` | safe split only — legitimate prompts that *sound* risky — evaluation set |

Notes on individual sources:

- **Tensor Trust** (hijacking + extraction) is not on the HF Hub — both files are small,
  human-generated, manually cleaned JSONL benchmarks published at
  [HumanCompatibleAI/tensor-trust-data](https://github.com/HumanCompatibleAI/tensor-trust-data),
  fetched directly over HTTPS. Neither file ships a label column — every row is an attack, so both
  are treated as all-positive.
- **`neuralchemy/Prompt-injection-dataset`** ships a `core` config (originals only) and a `full`
  config (`core` plus paraphrase-augmented positives); only `core` is used, filtered to
  `augmented == False`. Its `category` field is filtered to a whitelist and mapped to
  `threat_class`: `benign`/`control`/`crescendo`/`edge_case` → `benign`; nine attack categories
  (`agent_manipulation`, `code_execution`, `context_confusion`, `indirect_injection`,
  `instruction_override`, `output_manipulation`, `prompt_injection`, `rag_poisoning`,
  `token_smuggling`) plus `prompt_extraction`/`system_extraction` → the clean, evaluation-bound
  slice (`neuralchemy/Prompt-injection-dataset:clean`); `direct_injection` → `LLM01` but kept in
  the training pool. Categories outside the whitelist (`adversarial`, `chain_of_thought`,
  `encoding`, `jailbreak`, `many_shot`, `model_fingerprinting`, `payload_injection`, `prompt_leak`,
  `response_manipulation`, `system_manipulation`, `token_injection`, `training_extraction`,
  `persona_replacement`, `encoding_obfuscation`, `multi_turn`) are dropped entirely — either out of
  scope for LLM01/LLM07, too ambiguous to hand-label reliably, or redundant with a kept category.
- **`Lakera/gandalf_ignore_instructions`** ships no label column — every row is an attack on
  Gandalf's hidden password/system prompt, so all rows are `label=1`.
- **`leolee99/NotInject`** (339 rows) is benign prompts loaded with injection trigger words
  ("ignore", "system", etc.) across three tiers of trigger-word count — the single most valuable
  negative set for measuring over-firing.
- **`allenai/wildguardmix`** (`wildguardtest` config) is gated on HF and requires accepting terms
  plus an `HF_TOKEN`; the build skips it gracefully if unavailable. Filtered to
  `prompt_harm_label == "unharmful"`.
- **`deepset/prompt-injections`** is loaded from `deepset_relabeled.csv` in this folder rather than
  straight from HuggingFace — a pass over the native labels found some that didn't hold up on
  inspection, so they were corrected by hand.
- **`natolambert/xstest-v2-copy`** contributes benign questions that *sound* risky (e.g. "how do I
  kill a process?"); only the bare `prompts` split is used, with unsafe contrast rows
  (`type` prefixed `contrast_`) excluded.

## Evaluation dataset

`eval_dataset_clean.parquet` — the smallest, cleanest, most-audited sources (or slices), held out
entirely from any training pool and used as-is for zero-shot evaluation: Tensor Trust extraction,
`deepset/prompt-injections`, `leolee99/NotInject`, `natolambert/xstest-v2-copy`,
`Lakera/gandalf_ignore_instructions`, `allenai/wildguardmix`, and the `:clean` slice of
`neuralchemy/Prompt-injection-dataset` (its LLM01 + LLM07 categories).

These sources were selected for eval specifically because each is either purpose-built and
human-audited (Tensor Trust extraction, manually relabeled `deepset/prompt-injections`), attack-
only against a real target with no ambiguity about intent (Gandalf), or a targeted probe for a
specific failure mode (`NotInject` for false positives on trigger words, `xstest-v2-copy` for false
positives on scary-sounding-but-safe questions). None of them require heavy per-row judgment calls
the way the raw pool sources below do, which is what makes them trustworthy as a held-out
benchmark rather than training signal.

3,664 rows after cross-source dedup, all `in_scope=True`.

## Training data

`train_raw.parquet` — everything else, unsplit, left for a future fine-tuning run to partition
however it needs: Tensor Trust hijacking, `jayavibhav/prompt-injection-safety`, and the rest of
`neuralchemy/Prompt-injection-dataset` (`direct_injection` plus its benign-mapped categories).

Tensor Trust hijacking is deliberately in this pool rather than the evaluation set — its attacks
lean heavily on repeated-token/character obfuscation, which makes it more useful as training
signal than as a clean benchmark. `jayavibhav/prompt-injection-safety` is the only large source
that pre-separates harmful content from injection at the label level (`0=benign`,
`1=prompt injection`, `2=direct harmful request`); its harmful-content rows are kept for audit with
`in_scope=False` rather than dropped.

64,247 rows after cross-source dedup, no train/test split applied.
