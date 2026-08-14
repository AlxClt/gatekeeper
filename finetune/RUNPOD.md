# Running the fine-tune on RunPod.io

Step-by-step guide to run `finetune/train.py` on a RunPod GPU pod. For *why* the script is built
the way it is (model choice, QLoRA, prompt design, hyperparameters, stopping criterion), see
`finetune/README.md` — this file is only the mechanics of getting it running on RunPod.

## 0. Before you start

- A RunPod account with billing set up.
- A Hugging Face account with access to the gated `google/gemma-2-9b-it` weights (step 1 below).
- `data/train_consolidated.parquet` available on your local machine. It is **not** committed to
  git (`*.parquet` is gitignored) — see step 5 for getting it onto the pod.

## 1. Accept the Gemma-2 license and create an HF token

1. Log into Hugging Face, open [google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it),
   and accept the license (one-time, gates you into `google/gemma-*` model downloads).
2. Create a **read** access token at
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). You'll export this as
   `HF_TOKEN` on the pod in step 7 — same variable name the rest of this repo already uses.

## 2. Launch the pod

In the RunPod console, **Deploy a Pod**:

- **GPU**: 24-32GB at the defaults (RTX A5000/A6000, RTX 4090, L4/L40, RTX PRO 4500/6000, etc.),
  more headroom on ≥40GB (A100/L40S) — QLoRA fits a single GPU here, but Gemma-2's large vocabulary
  makes the per-step loss tensor tighter than a typical 9B QLoRA fine-tune; see `finetune/README.md`'s
  "Method: QLoRA" note before assuming 24GB has slack to spare. No multi-GPU needed.
- **Template**: an official RunPod PyTorch template (e.g. "RunPod PyTorch 2.x") — comes with CUDA
  and a matching torch build already installed, which `bitsandbytes` needs.
- **Disk**: 60–80GB container/volume disk. `google/gemma-2-9b-it` is ~18GB in bf16, downloaded
  once and reused for both the quantized training load and the final merge; the merged output
  adds another ~18GB; checkpoints are LoRA-only and small (tens–hundreds of MB each, `save_total_limit=3`
  caps how many stick around).
- **Exposed ports** — this is the step that makes monitoring work. Add port **6006** as an
  **HTTP** port (not a TCP port mapping). RunPod will give it a proxy URL of the form
  `https://<pod-id>-6006.proxy.runpod.net` once the pod is running — that's what you'll open to
  watch TensorBoard from your local machine. (If you change `--tensorboard_port`, expose that port
  instead.)
- If your pod template includes SSH, also make sure the SSH port is enabled — useful for step 5.

Deploy the pod and wait for it to reach "Running".

## 3. Connect to the pod

Use either the RunPod web terminal (Pod → **Connect** → **Start Web Terminal**) or SSH (Pod →
**Connect** → copy the `ssh root@... -p ...` command). SSH is more convenient for the file
transfer in step 5.

## 4. Get the code onto the pod

```bash
cd workspace
git clone <your-gatekeeper-repo-url>
cd gatekeeper
```

If the repo is private, either clone over HTTPS with a GitHub personal access token in the URL, or
add an SSH deploy key to the pod first — same as cloning a private repo anywhere else.

## 5. Get the training data onto the pod

`data/train_consolidated.parquet` isn't in git, so it needs a separate transfer. Two options:

**Option A — `runpodctl send`/`receive`** (no SSH key setup needed, works through RunPod's relay):
install `runpodctl` on both your local machine and the pod (prebuilt binaries at
[github.com/runpod/runpodctl/releases](https://github.com/runpod/runpodctl/releases)), then:

```bash
# local machine
runpodctl send data/train_consolidated.parquet
# prints a one-time code like "8338-galileo-tango-4926"

# on the pod
runpodctl receive 8338-galileo-tango-4926
mv train_consolidated.parquet gatekeeper/data/
```

**Option B — `scp`** (if you already have SSH access from step 3):

```bash
scp -P <ssh-port> data/train_consolidated.parquet root@<pod-ip>:/workspace/gatekeeper/data/
```

Either way, confirm it landed at `gatekeeper/data/train_consolidated.parquet` — that's the default
path `finetune/train.py` reads (`--train_data` to override).

## 6. Install dependencies

```bash
pip install -r finetune/requirements.txt
```

## 7. Set your HF token

```bash
export HF_TOKEN=hf_...
```

## 8. Start training

Run it inside `tmux` (or `nohup ... &`) so it survives a dropped web-terminal/SSH connection —
this is a 1–2.5 hour run (see the earlier time estimate for an RTX PRO 4500-class GPU), and a
disconnected shell will kill a foreground process otherwise:

```bash
tmux new -s finetune
python finetune/train.py
# detach with Ctrl+B then D; reattach later with: tmux attach -t finetune
```

Defaults (`--batch_size 2`) are already sized for a 24-32GB GPU. If you still hit an out-of-memory
error, lower `--max_seq_len` next (e.g. `768`) — see `finetune/README.md`'s "Method: QLoRA" note
for why Gemma-2's vocab size, not the model weights, is what actually drives memory use here.

## 9. Monitor from your local machine

Open the proxy URL RunPod showed you in step 2 (`https://<pod-id>-6006.proxy.runpod.net`).
TensorBoard is started automatically by `train.py` itself — no separate command needed. You'll see
training loss, learning rate, and (every `--eval_steps`) `eval_loss` plus `precision`/`recall`/
`f1`/`fpr`/`malformed_rate` on the validation split, updating live.

Training stops on its own once early stopping triggers (`eval_loss` hasn't improved for
`--early_stopping_patience` evals) or the epoch cap is reached — see `finetune/README.md`'s
"Stopping criterion" section.

## 10. Retrieve the outputs

`finetune/output/` on the pod ends up with a few things of very different size — pull them
separately rather than the whole directory at once:

**The monitoring snapshot** (`training_curves.png`, `summary.md`, `summary.json`,
`log_history.json`) is a few hundred KB total and is regenerated after every eval, not just at the
end — so it's worth pulling this even *before* training finishes, e.g. to sanity-check progress
without leaving the TensorBoard tab open, or to grab a last-known-good snapshot if a spot pod gets
preempted mid-run:

```bash
# runpodctl, from the pod
runpodctl send finetune/output/summary.md finetune/output/training_curves.png finetune/output/summary.json
# then on your local machine
runpodctl receive <code>

# or scp, from your local machine, any time during or after the run
scp -P <ssh-port> "root@<pod-ip>:/workspace/gatekeeper/finetune/output/{summary.md,summary.json,training_curves.png,log_history.json}" ./
```

**The model itself** — `adapter/` (LoRA-only, small) and `merged/` (full bf16 model, ~18GB, unless
you ran with `--skip_merge`) — only exist once training has finished:

```bash
# runpodctl, from the pod
runpodctl send finetune/output/adapter
# then on your local machine
runpodctl receive <code>

# or scp, from your local machine
scp -r -P <ssh-port> root@<pod-ip>:/workspace/gatekeeper/finetune/output/merged ./
```

Pulling only `adapter/` is enough if you plan to merge or serve elsewhere; pull `merged/` if you
want the ready-to-convert bf16 checkpoint (e.g. to GGUF for Ollama — see `finetune/README.md`'s
"After training" section for wiring it back into the gatekeeper API).

## 11. Shut down the pod

RunPod bills by the hour while the pod is running, regardless of whether anything is using the
GPU. **Stop or terminate the pod as soon as you've pulled the outputs you need** — "Stop" keeps the
disk (and lets you resume later at storage-only cost); "Terminate" deletes it entirely.

## Troubleshooting

| Symptom | Likely cause / fix |
| --- | --- |
| `401`/`403` downloading `google/gemma-2-9b-it` | License not accepted on that HF account, or `HF_TOKEN` not exported / wrong token — redo step 1 and 7 |
| TensorBoard proxy URL doesn't load | Port 6006 was added as **TCP** instead of **HTTP** in the pod's port settings — HTTP ports are what get a `*.proxy.runpod.net` URL |
| Training dies when the terminal disconnects | Wasn't run inside `tmux`/`nohup` — see step 8 |
| CUDA out of memory | Defaults are already sized for a 24-32GB card (`--batch_size 2`) — if you still hit it, lower `--max_seq_len` next (e.g. `768`), then `--batch_size 1` (raising `--grad_accum` to compensate) as a last resort. See `finetune/README.md`'s "Method: QLoRA" note for why Gemma-2's vocab size is the actual driver here |
| `FileNotFoundError` for `train_consolidated.parquet` | Data transfer (step 5) didn't land at the path `train.py` expects — check `--train_data` or move the file to `gatekeeper/data/` |
