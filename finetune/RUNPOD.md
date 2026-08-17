# Running the fine-tune on RunPod.io

Step-by-step guide to run `finetune/train.py` on a RunPod GPU pod. This is the cloud platfor that has been used by the author of this repo, but finetuning can be performed on any other cloud platform, or on a local GPU.

## 0. Before you start

- A RunPod account with billing set up.
- A Hugging Face account with access to the gated `google/gemma-2-9b-it` weights.
- `data/train_consolidated.parquet` available on your local machine. It is **not** committed to git.

## 1. Accept the Gemma-2 license and create an HF token

1. Log into Hugging Face, open [google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it),
   and accept the license (one-time, gates you into `google/gemma-*` model downloads).
2. Create a **read** access token at
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). You'll export this as
   `HF_TOKEN` on the pod in step 7 — same variable name the rest of this repo already uses.

## 2. Launch the pod

In the RunPod console, **Deploy a Pod**:

- **GPU**: 24-32GB at the defaults (RTX A5000/A6000, RTX 4090, L4/L40, RTX PRO 4500/6000, etc.), more headroom on ≥40GB (A100/L40S) — but anyway QLoRA fits a single GPU here.
- **Template**: an official RunPod PyTorch template (e.g. "RunPod PyTorch 2.x") — comes with CUDA and a matching torch build already installed, which `bitsandbytes` needs.
- **Disk**: **100GB+** if you'll also do the GGUF conversion (step 11) on this pod — 60–80GB is enough for training alone otherwise.
- **Exposed ports**: Add port **6006** as an
  **HTTP** port. RunPod will give it a proxy URL of the form
  `https://<pod-id>-6006.proxy.runpod.net` once the pod is running to watch TensorBoard from a local machine.
- Make sure the SSH port is enabled 

## 3. Connect to the pod

Follow Runpod's guidelines to connect to the port using SSH.

## 4. Get the code onto the pod

```bash
cd workspace
git clone <your-gatekeeper-repo-url>
cd gatekeeper
```

## 5. Get the training data onto the pod

`data/train_consolidated.parquet` needs to be transferred onto the pod for finetuning:

```bash
scp -P <ssh-port> data/train_consolidated.parquet root@<pod-ip>:/workspace/gatekeeper/data/
```

## 6. Install dependencies

```bash
pip install -r finetune/requirements.txt
```

## 7. Set your HF token

```bash
export HF_TOKEN=hf_...
```

## 8. Start training

Start a new `tmux` session so it survives a dropped web-terminal/SSH connection.

```bash
tmux new -s finetune
python finetune/train.py
# detach with Ctrl+B then D; reattach later with: tmux attach -t finetune
```

## 9. Monitoring

Opening the proxy URL RunPod showed in step 2 (`https://<pod-id>-6006.proxy.runpod.net`) will show tensorboard.

## 10. Retrieving the outputs

Outputs will be saved to `finetune/output/` on the pod. Not everything is worth transferring:

**The monitoring snapshot** (`training_curves.png`, `summary.md`, `summary.json`,
`log_history.json`, can also be pulled anytime during training for an in-process snapshot):

```bash

scp -P <ssh-port> "root@<pod-ip>:/workspace/gatekeeper/finetune/output/{summary.md,summary.json,training_curves.png,log_history.json}" ./
```

**The model itself** — `adapter/` (LoRA-only, small) and `merged/` (full bf16 model, ~18GB):

```bash
scp -r -P <ssh-port> root@<pod-ip>:/workspace/gatekeeper/finetune/output/merged ./
```

## 11. Serve the merged model with Ollama

The author of this repo is serving the model using Ollama. This sections describes the necessary steps.

Ollama runs GGUF, not raw HF safetensors, so `finetune/output/merged/` needs converting first. The best solution is to do this on the same pod right after training, before shutting it
down.

**Convert to GGUF** using llama.cpp's conversion script:

```bash
git clone https://github.com/ggml-org/llama.cpp   
cd llama.cpp
pip install -r requirements/requirements-convert_hf_to_gguf.txt

python convert_hf_to_gguf.py /workspace/gatekeeper/finetune/output/merged \
    --outfile /workspace/gatekeeper-finetuned-f16.gguf \
    --outtype f16
```

`f16` keeps full training precision

The next steps should be performed on the machine where the gatekeeper model should be served. If it is not the same one, download and transfer the GGUF model before.

**Write a Modelfile** — deliberately minimal, no `SYSTEM` line:

```dockerfile
FROM /workspace/gatekeeper-finetuned-f16.gguf

PARAMETER temperature 0
PARAMETER num_predict 1
PARAMETER stop "<end_of_turn>"
```

No `TEMPLATE` override needed — Gemma-2 has no system role, so the fine-tune was trained with
`prompt.yaml`'s *entire* filled content as one `user` turn (see `finetune/README.md`'s "Prompt"
section). `num_predict 1` matches the single-token target the
model was trained on; `stop` is a cheap safety net.

**Importing in ollama:**

```bash
ollama create gatekeeper-finetuned -f Modelfile
```

**Verify before wiring up your wrapper:**

```bash
ollama show gatekeeper-finetuned --modelfile   

curl http://localhost:11434/api/generate -d '{
  "model": "gatekeeper-finetuned",
  "prompt": "<paste a prompt.yaml-filled example here, e.g. a filled <text>...</text> prompt>",
  "stream": false,
  "options": {"temperature": 0, "num_predict": 1}
}'
# response.response should be exactly "0" or "1"
```

Then Ollama's'model-name config should point at `gatekeeper-finetuned`. `prompt.yaml`'s filled template should be used as the prompt.

