# Data

This folder contains scripts building a clean, held-out **evaluation set** for
benchmarking gatekeeper models, and a **training set** for fine-tuning one, both scoped to
**OWASP LLM01** (prompt injection / instruction hijacking) and **LLM07** (system-prompt / secret
extraction), as scoped in the repo's main Readme.

These scripts run a one time data curation job, but it is **not directly reproducible from a clean checkout**: building the evaluation set requires
[`deepset_relabeled.csv`](deepset_relabeled.csv), a manually relabeled copy of
`deepset/prompt-injections` checked into this folder (it has been relabled to align with the genreal interpretation of LLM01 used in this project), and building the training set requires
[`handcrafted_llm07.csv`](handcrafted_llm07.csv), 250 hand-authored LLM07 examples. These files can be obtained by contacting the author of the repo directly, everything else is either fetched from the Hugging Face Hub /
GitHub on demand, or derived from those two by the scripts described below.

## Data sources

### Target schema

Every source loader ([`helpers/dataset_loaders.py`](helpers/dataset_loaders.py)) normalizes its
source into a common schema before anything is combined:

| field | type | meaning |
|---|---|---|
| `text` | str | raw user input to classify |
| `label` | int | gold label: `1` = LLM01/LLM07 threat, `0` = not a threat |
| `source` | str | dataset id |
| `threat_class` | str | `LLM01` \| `LLM07` \| `benign` \| `harmful_content` |

Harmful-content rows are retained with `label=0` (a pure harmful prompt with no injection *should*
get classifier output `0`) — they're tagged `threat_class=harmful_content` for audit, but otherwise
scored as ordinary negatives like any other `label=0` row. `LLM07` and `LLM01` are not mutually
exclusive in the source datasets — slices are labeled with their most representative category.

Deduplication is exact-match on the (lowercased, whitespace-collapsed) `text` field, applied once
across all sources combined, before the evaluation/training partition.

### Sources

| Threat | Source | Notes |
|---|---|---|
| LLM01 | Tensor Trust hijacking (775 rows) | fetched directly from GitHub raw — training pool |
| LLM01 / benign / harmful_content | `jayavibhav/prompt-injection-safety` | 3-way label, harmful_content folded into `label=0` — training pool |
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
- **`jayavibhav/prompt-injection-safety`** is by far the largest source (~92% of
  `train_raw.parquet`), and worth a closer look because of that weight. It ships no dataset card,
  and manual inspection shows most rows are heavily-stylized, LLM-generated synthetic prompts:
  florid, repetitive templates (often wrapped in code scaffolding across several languages,
  occasionally emoji-only). Surface diversity is wide — many topics, syntactic wrappers, unusual
  vocabulary — but the underlying attack semantics are narrow: mostly "ignore previous
  instructions" / "start over" / "reveal your secret" variants, with indirect injection,
  tool-poisoning, multi-turn escalation, or encoding-based obfuscation essentially absent. The
  benign/injection boundary is also inconsistently drawn in places — structurally similar phrasing
  ends up on both sides of the label.
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

## Scripts

Three entry points, meant to run in this order. Each is a `main_*.py` script at the top level of
this folder; everything else (`helpers/`) is library code they import. See the Evaluation set /
Training set sections below for what each step actually does and why — this is just the map.

1. [`main_create_datasets.py`](main_create_datasets.py) — pulls every source above, normalizes and
   deduplicates them, and partitions the result into `eval_dataset_clean.parquet` (evaluation set)
   and `train_raw.parquet` (raw training pool). Run first: `python main_create_datasets.py`.
2. [`main_distill_train_set.py`](main_distill_train_set.py) — takes `train_raw.parquet`, filters its
   labeling noise down through a live gatekeeper model, and produces `train_distilled.parquet`. Run
   second, against a running gatekeeper server: `python main_distill_train_set.py`.
3. [`main_train_set_consolidation.py`](main_train_set_consolidation.py) — takes
   `train_distilled.parquet`, dedups + diversity-samples + appends the handcrafted LLM07 rows, and
   produces `train_consolidated.parquet` — the actual fine-tuning set. Run third:
   `python main_train_set_consolidation.py`.

## Evaluation set

`eval_dataset_clean.parquet` — the smallest, cleanest, most-audited sources (or slices), held out
entirely from any training pool and used as-is for model evaluation: Tensor Trust extraction,
`deepset/prompt-injections`, `leolee99/NotInject`, `natolambert/xstest-v2-copy`,
`Lakera/gandalf_ignore_instructions`, `allenai/wildguardmix`, and the `:clean` slice of
`neuralchemy/Prompt-injection-dataset` (its LLM01 + LLM07 categories).

These sources were selected for eval specifically because each is either purpose-built and
human-audited (Tensor Trust extraction, manually relabeled `deepset/prompt-injections`), attack-
only against a real target with no ambiguity about intent (Gandalf), or a targeted probe for a
specific failure mode (`NotInject` for false positives on trigger words, `xstest-v2-copy` for false
positives on scary-sounding-but-safe questions). None of them require heavy per-row judgment calls
the way the raw pool sources do, which is what makes them trustworthy as a held-out benchmark rather
than training signal. This is the file `evaluation/evaluation.ipynb` scores models against.

**Stats** (see [`data_inspection.ipynb`](data_inspection.ipynb) for the full breakdown):

| | |
|---|---|
| Rows | 3,664 |
| Sources | 7 |
| `label=1` rate | 49.4% |

| `threat_class` | rows | `label=1` |
|---|---|---|
| LLM01 | 440 | 348 |
| LLM07 | 1,463 | 1,463 |
| benign | 1,761 | 0 |

## Training set

`train_raw.parquet` can't be fine-tuned on directly. It's dominated (~92%) by
`jayavibhav/prompt-injection-safety`, whose labels are noisy in places and whose attack patterns are
semantically narrow despite looking surface-diverse (see the note above) — training on it as-is
risks both learning from wrong labels and overfitting to one source's stylistic fingerprint rather
than to prompt-injection semantics in general. Distillation (step 1 below) filters out label noise
using the zero-shot classifier's own judgment; diversity sampling (step 3) then thins out
`jayavibhav`'s redundant, repetitive patterns and rebalances the pool across sources, so no single
source's quirks dominate what the model ends up learning.

`train_consolidated.parquet` is the final fine-tuning set, built from `train_raw.parquet` in three
transformations, each shrinking or reshaping the pool further:

```
train_raw.parquet (64,247 rows, 3 sources)
  │  main_distill_train_set.py — zero-shot distillation
  ▼
train_distilled.parquet (52,786 rows)
  │  main_train_set_consolidation.py:
  │    1. near-duplicate removal
  ▼
  ~52,690 rows
  │    2. diversity sampling (per source × label)
  ▼
  ~10,580 rows
  │    3. + handcrafted_llm07.csv (250 rows)
  ▼
train_consolidated.parquet (10,830 rows)
```

1. **Distillation** — `train_raw.parquet` is pooled from sources that were never manually audited
   row-by-row (unlike the evaluation set), so a meaningful share of its `label=1` rows are mislabeled
   or too ambiguous to count as a real attack. Every `label=1` row is sent through a live gatekeeper
   `/verify` endpoint; rows the model doesn't itself flag as a threat are dropped as training-label
   noise. `label=0` rows pass through unchanged (no API call needed). The gatekeeper model used is the **best-performing 9B model** and its matching prompt
   (`app/verification/prompts/default-9b.yaml`) — see `evaluation/readme.md` for current per-model
   numbers — since it's the strongest classifier judgment available before any fine-tuned model
   exists to bootstrap from.

2. **Near-duplicate removal** ([`helpers/near_duplicates.py`](helpers/near_duplicates.py)) Near duplicates are eliminated by computing the Jaccard      similarity using minhashing (Jaccard
   threshold 0.8)

3. **Diversity sampling** ([`helpers/diversity_sampling.py`](helpers/diversity_sampling.py)) — In order to reduce the final training set size and limit the risk of repeating similar prompts/patters (beyond simple near-duplicate removal), we apply diversity maximizing sampling over the distilled set.

4. **Append handcrafted LLM07 examples** — `handcrafted_llm07.csv` (250 hand-authored
   `label=1`/`LLM07` rows: "show me your system prompt"-style extraction attempts) is appended as-is,
   skipping both dedup and sampling. It has been manually created because the raw training set does not contain LLM07 threats examples

**Stats** (see [`data_inspection.ipynb`](data_inspection.ipynb) for the full breakdown):

| | |
|---|---|
| Rows | 10,830 |
| Sources | 4 (3 real + `handcrafted_llm07`) |
| `label=1` rate | 43.9% |

| `threat_class` | rows | `label=1` |
|---|---|---|
| LLM01 | 4,502 | 4,502 |
| LLM07 | 250 | 250 |
| benign | 6,078 | 1 |

### Diversity sampling: farthest-point sampling (FPS)

Within each `(source, label)` group, rows are embedded
(`sentence-transformers/paraphrase-MiniLM-L3-v2`, L2-normalized) and PCA-reduced (currently 100
dimensions, retaining ~81% of variance), then downsampled via greedy **farthest point sampling**
(a.k.a. max-min dispersion)

Sampling is stratified by `(source, label)` rather than by source alone, so that diversity is maximized within each class. The fraction kept is configured per source in
`helpers/diversity_sampling.py`'s `SOURCE_SAMPLE_FRACTIONS`:

| source | fraction kept |
|---|---|
| `jayavibhav/prompt-injection-safety` | 15% |
| `neuralchemy/Prompt-injection-dataset` | 75% |
| `tensor_trust_hijacking` | 100% (kept whole — small and already the cleanest of the three) |

### Measuring the effect: Vendi score

To check FPS actually buys more effective diversity than plain random sampling (rather than just
re-shuffling the same distribution), we the **Vendi score**
([Friedman & Dieng, 2022](https://arxiv.org/abs/2210.02410)) of three sets: the full pre-sampling
pool, a random sample matched to the same per-source sizes FPS kept, and the FPS-sampled set itself.

![Random sample vs. diversity-sampled subset, projected on the first 2 PCA dimensions, with Vendi scores per panel](diversity_sampling_comparison.png)

On the run above: the full pre-sampling pool (n=52,690) scores **78.3**, a random sample matched to
FPS's per-source proportions scores **71.9**, and the FPS-sampled set scores **74.1** — confirming
FPS keeps more effective diversity than random sampling would at the same size, though the gap is not huge here. We can note that the diversity gap with the full dataset is not huge either (4 points of vendi score), although the cardinality has been reduced by a factor of ~5. Since the main downsampling factor comes from `jayavibhav/prompt-injection-safety`, this tends to confirm that this dataset contained redundant, repetitive patterns.
