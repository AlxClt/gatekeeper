# Fine-tuning

Fine-tunes a 9B model to perform the gatekeeper verification task directly — output `1` if the
input is an OWASP LLM01 (prompt injection) or LLM07 (system-prompt/secret leakage) threat, `0`
otherwise — instead of relying on the ~9,000-character calibrated prompt the zero-shot 9B models
need (`app/verification/prompts/default-9b.yaml`). Trained on `data/train_consolidated.parquet`
(10,830 rows, see `data/readme.md` for how it was built).

## Files

| File | Purpose |
|---|---|
| `train.py` | Entry point: loads the model, builds the dataset, runs training, serves TensorBoard. `python finetune/train.py` |
| `data.py` | Loading/splitting `train_consolidated.parquet` and tokenizing rows into loss-masked chat examples |
| `report.py` | Exports the monitoring snapshot (loss plot + metrics/timing summary) to `--output_dir`, called after every eval |
| `prompt.yaml` | The (short) fine-tuning prompt — same `{{input}}` substitution convention as `app/verification/prompts/*.yaml`. Doubles as the prompt to serve the merged model with afterwards |
| `requirements.txt` | `pip install -r finetune/requirements.txt` |
| `RUNPOD.md` | Step-by-step mechanics of running this on a RunPod.io pod (pod setup, getting the data on, monitoring, pulling outputs off) |

Deploying this on RunPod specifically? Skip ahead to **[RUNPOD.md](RUNPOD.md)** for the concrete
steps — this file is the *why*, that one is the *how*.

No test suite here either, consistent with the rest of the repo — verification is the eval run
described at the bottom of this file.

## Strategy

### Base model: `google/gemma-2-9b-it`

`gemma2:9b` is the strongest 9B zero-shot model evaluated so far (F1 0.940, FPR 0.038 — see
`evaluation/readme.md`), and Ollama's `gemma2:9b` tag *is* this instruction-tuned checkpoint. Fine-
tuning the already-best base keeps the comparison to the zero-shot table meaningful (same weights,
different training) and gives the fine-tune a head start on following the task.

### Method: QLoRA

Full fine-tuning a 9B model needs multiple high-memory GPUs; QLoRA (4-bit NF4 base weights via
`bitsandbytes`, trainable LoRA adapters via `peft` on all attention + MLP projections) fits a
single 24-32GB GPU, which is the class of pod this is meant to run on (RunPod, one GPU, no
multi-node/distributed training — out of scope here to keep the script simple). LoRA rank 16 /
alpha 32 / dropout 0.05 are conservative, standard QLoRA defaults for a task this narrow (binary
classification, not open-ended generation) — no need for a larger rank.

**The actual memory bottleneck isn't the quantized weights — it's the classification head.**
Gemma-2's vocabulary is ~256k tokens, so a training step's `(batch, seq_len, vocab_size)` logits
tensor is enormous compared to what most QLoRA memory-sizing guides assume (they're usually
written against a 32k–128k vocab), and `CrossEntropyLoss` upcasts that tensor to fp32 internally
right at the point of peak memory use — on a batch of 4 at `max_seq_len=1024` that single tensor
is several GB by itself, on top of the ~5-7GB of quantized weights + tied embeddings + activations.
That's what actually pushed a 32GB card over the edge at the original defaults. The defaults now
account for this: `--batch_size 2` / `--grad_accum 8` (same effective batch size as before, 16),
`--eval_accumulation_steps 5`, non-reentrant gradient checkpointing, and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set automatically to cut down on the allocator
fragmentation the OOM error reports as "reserved but unallocated". If you have a ≥40GB GPU, raise
`--batch_size` back up (lowering `--grad_accum` to compensate) for faster training.

### Prompt: short, not the zero-shot prompt

`default-9b.yaml` is long because it has to teach the *rules* to a model at inference time, in-
context, every single call — its taxonomy sections and ~30 calibration examples exist to
compensate for the model never having seen labeled examples. Fine-tuning inverts that: the model
sees 10,000+ labeled examples during training and bakes the decision boundary into its weights,
so the calibration examples aren't needed at inference anymore. `prompt.yaml` keeps only what
still can't be learned from labels alone — the two-sentence definitions of LLM01/LLM07 and the
scope boundary (content vs. instruction-attack) — cutting prompt tokens from ~9,000 to ~700
characters per request. This is a deliberate choice, not an oversight: training with the full
zero-shot prompt would work too, but at ~10x the tokens per example for no expected accuracy gain,
since the whole point of the calibration examples is to substitute for exactly what fine-tuning
already provides.

Gemma-2's chat template has no `system` role, so `prompt.yaml` (with `{{input}}` substituted) is
sent as the single `user` turn, per Google's documented convention for Gemma system instructions.

### Target format and loss masking

Each example is one chat turn pair: `user` = the filled prompt, `model` = the single label
character (`"0"` or `"1"`) followed by `<end_of_turn>`. The loss is masked (`-100`) over every
token except that label + `<end_of_turn>` — the prompt is context, not something to learn to
reproduce. This exactly matches how the zero-shot models are queried in production
(`OnlineAdapter`/`LocalAdapter` request one token, temperature 0, and parse `"0"`/`"1"`), so the
fine-tune is trained to do precisely the thing `Verifier._classify` expects.

### Data handling

- Text is truncated to 4,000 characters — the same cap `app/verification/preprocessing.py` applies
  before any prompt reaches the LLM in production (`_MAX_LENGTH`). Keeping train and inference
  truncation identical avoids training the model on a distribution it will never see served.
  Only ~20/10,830 rows are affected.
- `train_consolidated.parquet` is close to balanced already (43.9% `label=1` — see
  `data/readme.md`), so no class weighting or oversampling is applied.
- A 5% stratified (by `label`) split is held out as `val` purely to monitor training (early
  stopping, loss/metric curves) — it comes from the *same* pool as train, so it is **not** a
  substitute for a real benchmark. The actual held-out test set is
  `evaluation/eval_dataset_clean.parquet` (3,664 rows, entirely disjoint sources), scored the same
  way the zero-shot models were via `evaluation/evaluation.ipynb`, once this model is deployed
  behind the gatekeeper API. Don't read the in-training val metrics as the final numbers for the
  `evaluation/readme.md` comparison table.

### Hyperparameters

| Setting | Value | Why |
|---|---|---|
| Epochs | 3 (cap; early stopping usually ends it sooner) | Small dataset (~10.8k rows) fine-tuned with LoRA overfits quickly; 3 passes is already generous |
| Effective batch size | 16 (`batch_size=2 × grad_accum=8`) | `max_seq_len=1024` combined with Gemma-2's ~256k vocab makes the per-step loss tensor the real memory constraint (see "Method: QLoRA" above), not the model weights — `batch_size=2` is the safe default for a 24-32GB GPU; raise it on a bigger card |
| Learning rate | 2e-4, cosine schedule, 3% warmup | Standard QLoRA LR — LoRA adapters tolerate higher LR than full fine-tuning since the base weights are frozen |
| Optimizer | `paged_adamw_8bit` | Keeps optimizer state memory low, standard for QLoRA |
| Max grad norm | 0.3 | Standard QLoRA clipping value; the 4-bit base makes training more sensitive to gradient spikes |
| `max_seq_len` | 1024 tokens | Covers >99% of rows (`text` ≤ 4,000 chars + the ~700-char prompt); longer rows are dropped, not truncated mid-sequence, to avoid cutting the label off. Lower this first if you still hit OOM after reducing `--batch_size` |

### Stopping criterion

`eval_loss` (cross-entropy on the masked label token) every `eval_steps` (default 200, ~9-10 evals
over 3 epochs at these defaults), with `load_best_model_at_end=True` and
`EarlyStoppingCallback(patience=3)`: training stops once `eval_loss` hasn't improved for 3
consecutive evals, and the best checkpoint (by `eval_loss`) is what gets saved — not the last one.
`eval_loss` was picked over `eval_f1` as the stopping metric because it's the smoother, less noisy
signal at this eval-set size (~540 rows) and it's exactly the quantity being optimized (the
label token's cross-entropy), so it can't be gamed by a metric/loss mismatch. `precision`,
`recall`, `f1`, `fpr`, and `malformed_rate` (rate at which the model's argmax token was neither
`"0"` nor `"1"` — should collapse to ~0 within the first few hundred steps) are computed every eval
too and logged to TensorBoard alongside it, using the exact same definitions as
`evaluation/readme.md`, for a human-readable sanity check on top of the loss curve.

### Monitoring: TensorBoard on an exposed port, no Docker

`train.py` launches `tensorboard --logdir <output_dir>/tensorboard --host 0.0.0.0 --port 6006` as
a plain subprocess at startup (RunPod pods already run inside Docker themselves, so this
deliberately avoids Docker-in-Docker). To watch it from your local machine on RunPod:

1. When creating/editing the pod, expose port `6006` as an **HTTP** port (not just TCP) —
   RunPod's pod page will then show a proxy link like `https://<pod-id>-6006.proxy.runpod.net`.
2. Run `python finetune/train.py` (optionally `--tensorboard_port` to use a different port).
3. Open the proxy URL — loss, learning rate, and the eval metrics above update live as training
   progresses.

## Outputs

Everything lands under `--output_dir` (default `finetune/output/`):

- `checkpoints/` — Trainer checkpoints (top 3 by `eval_loss`, per `save_total_limit`)
- `adapter/` — the best LoRA adapter + tokenizer, standalone (small, loadable with `peft`)
- `merged/` — the LoRA adapter merged into the base weights in bf16 (skip with `--skip_merge`).
  Merging happens as a separate reload of the unquantized base model, since LoRA can't be merged
  directly into 4-bit weights. This is the artifact to convert (e.g. to GGUF for Ollama) for
  serving behind `LocalAdapter`/`OnlineAdapter`.
- `training_config.json` — the exact CLI args used, for reproducibility
- `tensorboard/` — TensorBoard event files
- **Monitoring snapshot** (`report.py`, see below) — `training_curves.png`, `summary.md`,
  `summary.json`, `log_history.json`

### Monitoring snapshot

TensorBoard (see below) is only reachable while the pod is up — nothing about it survives past
`docker stop`/pod termination unless you export it. `report.py` writes a static snapshot instead,
regenerated after **every** eval (not just at the end, via `SnapshotCallback` in `train.py`), so
it's pull-able off the pod at any point during the run, not only once it finishes:

- `training_curves.png` — train loss (every `logging_steps`) and val loss (every `eval_steps`) vs. step
- `summary.md` — human-readable: elapsed time, steps/epochs completed, the latest eval's
  `precision`/`recall`/`f1`/`fpr`/`malformed_rate`, and the run's config, with the plot embedded
- `summary.json` — the same content, machine-readable
- `log_history.json` — the raw Trainer log history (every logged train/eval entry), for
  re-plotting or re-analyzing offline without needing the GPU environment

`elapsed_seconds`/`elapsed_human` in the summary times the training loop itself (from just before
`trainer.train()` starts), not model loading or the final merge step. If you use
`--resume_from_checkpoint`, elapsed time resets to the current run segment — it does not carry
over from a previous, interrupted session.

## After training

Deploying the merged model and scoring it with `evaluation/evaluation.ipynb` against
`eval_dataset_clean.parquet` (same procedure as the zero-shot models) is what fills in the
`Fine-tuned` row of `evaluation/readme.md` — that comparison is out of scope for this script.
`prompt.yaml` here is the prompt to serve it with (e.g. as a new
`app/verification/prompts/finetuned-9b.yaml`).

## Requirements

```bash
pip install -r finetune/requirements.txt
```

Needs a CUDA GPU with bf16 support (Ampere or newer — A5000/A6000/4090/A100/L40S-class), 24-32GB
VRAM at the default `batch_size`/`max_seq_len` (see "Method: QLoRA" above for why Gemma-2's large
vocab makes this tighter than a typical 9B QLoRA fine-tune), and access to the gated
`google/gemma-2-9b-it` weights (`huggingface-cli login` or set `HF_TOKEN`, same variable the rest
of the repo uses).

**Still OOM at the defaults?** Lower `--max_seq_len` next (e.g. `768`) — it shrinks the same
per-step loss tensor `--batch_size` does. `--batch_size 1`/`--grad_accum 16` is the floor before
`max_seq_len` is the only lever left.
