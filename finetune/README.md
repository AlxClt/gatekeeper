# Fine-tuning

Fine-tunes a 9B model to perform the gatekeeper verification task directly — output `1` if the
input is an OWASP LLM01 (prompt injection) or LLM07 (system-prompt/secret leakage) threat, `0`
otherwise — instead of relying on the ~9,000-character calibrated prompt the zero-shot 9B models
need (`app/verification/prompts/default-9b.yaml`). Trained on `data/train_consolidated.parquet`
(10,830 rows, see `data/readme.md` for how it was built).

## Files

| File | Purpose |
| --- | --- |
| `train.py` | Entry point: loads the model, builds the dataset, runs training, serves TensorBoard. `python finetune/train.py` |
| `data.py` | Loading/splitting `train_consolidated.parquet` and tokenizing rows into loss-masked chat examples |
| `report.py` | Exports the monitoring snapshot (loss plot + metrics/timing summary) to `--output_dir`, called after every eval |
| `prompt.yaml` | The (short) fine-tuning prompt — same `{{input}}` substitution convention as `app/verification/prompts/*.yaml`. Doubles as the prompt to serve the merged model with afterwards |
| `requirements.txt` | `pip install -r finetune/requirements.txt` |
| `RUNPOD.md` | Step-by-step guide for running the finetuning using RunPod.io - the cloud platform that have been used by the author for this project (pod setup, getting the data on, monitoring, pulling outputs off) |

Deploying this on RunPod specifically? Skip ahead to **[RUNPOD.md](RUNPOD.md)** for the concrete
steps — this file is the *why*, that one is the *how*.

No test suite here either, consistent with the rest of the repo — verification is the eval run
described at the bottom of this file.

## Strategy

### Base model: `google/gemma-2-9b-it`

`gemma2:9b` is the strongest 9B zero-shot model evaluated so far (best balance — see
`evaluation/readme.md`)

### Method: QLoRA

QLoRA is the industry standard for finetuning as it almost achieves full finetuning performance with considerably lower memory usage

For a 9B model, QLoRA fits a
single 24-32GB GPU using parameters value LoRA rank 16 /
alpha 32 / dropout 0.05. Those are conservative choices, since they are standard QLoRA defaults and a task this narrow (binary
classification, not open-ended generation) — do not need a larger rank.

**However, the actual memory bottleneck isn't the quantized weights — it's the classification head.**
Gemma-2's vocabulary is ~256k tokens, so a training step's `(batch, seq_len, vocab_size)` logits
tensor is enormous compared to what most QLoRA memory-sizing guides assume (they're usually
written against a 32k–128k vocab), and `CrossEntropyLoss` upcasts that tensor to fp32 internally
right at the point of peak memory use.

So in order to fit the fine tuning process on a 32GB GPU (note that it is also possible to fine tune on a bigger GPU), the following parameters values will be set:

- `--batch_size 2` / `--grad_accum 8` (this effective batch size is 16, but memory usage is reduced),
- `--eval_accumulation_steps 5`
- non-reentrant gradient checkpointing
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set automatically to cut down on the allocator
fragmentation 

If using a ≥40GB GPU, it is possible to raise
`--batch_size` back up (lowering `--grad_accum` to compensate) for faster training.

### Prompt

We will use `prompt.yaml`, which is a short prompt that, in comparison to the zero shot prompts used, keeps only what can't be learned from labels alone — the two-sentence definitions of LLM01/LLM07 and the
scope boundary (content vs. instruction-attack)

Gemma-2's chat template has no `system` role, so `prompt.yaml` (with `{{input}}` substituted) is
sent as the single `user` turn, per Google's documented convention for Gemma system instructions.

### Target format and loss masking

Each example is one chat turn pair: `user` = the filled prompt, `model` = the single label
character (`"0"` or `"1"`) followed by `<end_of_turn>`. The loss is masked (`-100`) over every
token except that label + `<end_of_turn>`

### Data handling

- Text is truncated to 4,000 characters — the same cap `app/verification/preprocessing.py` applies
  before any prompt reaches the LLM in production (`_MAX_LENGTH`). Keeping train and inference
  truncation identical avoids training the model on a distribution it will never see served.
  Only ~20/10,830 rows are affected.
- `train_consolidated.parquet` is close to balanced already (43.9% `label=1` — see
  `data/readme.md`), so no class weighting or oversampling is applied.
- A 5% stratified (by `label`) split is held out as `val` purely to monitor training (early
  stopping, loss/metric curves) — coming from the *same* pool as train. The actual held-out test set is
  `evaluation/eval_dataset_clean.parquet` (3,664 rows, entirely disjoint sources), scored the same
  way the zero-shot models were via `evaluation/evaluation.ipynb`, once this model is deployed
  behind the gatekeeper API. The final, real eval numbers are in the
  `evaluation/readme.md` comparison table.

### Hyperparameters

| Setting | Value | Why |
| --- | --- | --- |
| Epochs | 3 (cap; early stopping usually ends it sooner) | Small dataset (~10.8k rows) fine-tuned with LoRA overfits quickly; 3 passes is already generous |
| Effective batch size | 16 (`batch_size=2 × grad_accum=8`) | `max_seq_len=1024` combined with Gemma-2's ~256k vocab makes the per-step loss tensor the real memory constraint (see "Method: QLoRA" above) — `batch_size=2` is the safe default for a 24-32GB GPU |
| Learning rate | 2e-4, cosine schedule, 3% warmup | Standard QLoRA LR |
| Optimizer | `paged_adamw_8bit` | Keeps optimizer state memory low, standard for QLoRA |
| Max grad norm | 0.3 | Standard QLoRA clipping value |
| `max_seq_len` | 1024 tokens | Covers >99% of rows (`text` ≤ 4,000 chars + the ~700-char prompt); longer rows are dropped, not truncated mid-sequence, to avoid cutting the label off.|

### Stopping criterion

`eval_loss` (cross-entropy on the masked label token) every `eval_steps` (default 200, ~9-10 evals
over 3 epochs at these defaults), with `load_best_model_at_end=True` and
`EarlyStoppingCallback(patience=3)`: training stops once `eval_loss` hasn't improved for 3
consecutive evals, and the best checkpoint (by `eval_loss`) is what gets saved — not the last one.

 `precision`,
`recall`, `f1`, `fpr`, and `malformed_rate` (rate at which the model's argmax token was neither
`"0"` nor `"1"`) are computed every eval
too and logged to TensorBoard alongside it.

### Monitoring: TensorBoard on an exposed port

`train.py` launches `tensorboard --logdir <output_dir>/tensorboard --host 0.0.0.0 --port 6006` as
a plain subprocess at startup.

## Outputs

Everything lands under `--output_dir` (default `finetune/output/`):

- `checkpoints/` — Trainer checkpoints (top 3 by `eval_loss`, per `save_total_limit`)
- `adapter/` — the best LoRA adapter + tokenizer, standalone (small, loadable with `peft`)
- `merged/` — the LoRA adapter merged into the base weights in bf16 (skip with `--skip_merge`).
- `training_config.json` — the exact CLI args used, for reproducibility
- `tensorboard/` — TensorBoard event files
- **Monitoring snapshot** (`report.py`, see below) — `training_curves.png`, `summary.md`,
  `summary.json`, `log_history.json`

### Monitoring snapshot

`report.py` writes a static snapshot of training and eval metrics,
regenerated after **every** eval via `SnapshotCallback` in `train.py`.

- `training_curves.png` — train loss (every `logging_steps`) and val loss (every `eval_steps`) vs. step
- `summary.md` elapsed time, steps/epochs completed, the latest eval's
  `precision`/`recall`/`f1`/`fpr`/`malformed_rate`, and the run's config, with the plot embedded
- `summary.json` — the same content, machine-readable
- `log_history.json` — the raw Trainer log history (every logged train/eval entry), for
  re-plotting or re-analyzing offline without needing the GPU environment

`elapsed_seconds`/`elapsed_human` in the summary times the training loop itself (from just before
`trainer.train()` starts), not model loading or the final merge step.

## Requirements

```bash
pip install -r finetune/requirements.txt
```

Needs a CUDA GPU with bf16 support (Ampere or newer — A5000/A6000/4090/A100/L40S-class), 24-32GB
VRAM at the default `batch_size`/`max_seq_len` (see "Method: QLoRA" above for why Gemma-2's large
vocab makes this tighter than a typical 9B QLoRA fine-tune), and access to the gated
`google/gemma-2-9b-it` weights (`huggingface-cli login` or set `HF_TOKEN`, same variable the rest
of the repo uses).
