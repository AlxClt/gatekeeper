"""Fine-tunes a 9B model (default: google/gemma-2-9b-it) to perform the gatekeeper verification
task: output a single character, 1 (threat) or 0 (safe), for OWASP LLM01/LLM07 classification —
see the repo's README.md and evaluation/readme.md for the task definition and the zero-shot
baselines this is meant to beat. Strategy and hyperparameter rationale: see finetune/README.md.

Run (from the repo root, on a single CUDA GPU — e.g. a RunPod pod):
    pip install -r finetune/requirements.txt
    python finetune/train.py

Training status is served live over HTTP via TensorBoard on --tensorboard_port (default 6006) —
expose that port on the pod to watch loss/eval curves from your local machine (see README).
"""

import os

# Must be set before torch's CUDA allocator initializes. Gemma-2's ~256k vocab makes the
# (batch, seq, vocab) logits tensor unusually large for a 9B model, and the fp32 upcast
# CrossEntropyLoss does internally roughly doubles that at the exact peak-memory moment of each
# step — this reduces the fragmentation (reserved-but-unallocated memory) that otherwise turns a
# borderline-fitting peak into an actual OOM on 24-32GB cards. See finetune/README.md.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import atexit
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

from data import DataCollator, GatekeeperDataset, load_split
from report import export_snapshot

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base_model", default="google/gemma-2-9b-it")
    p.add_argument("--train_data", default=str(REPO_ROOT / "data" / "train_consolidated.parquet"))
    p.add_argument("--output_dir", default=str(Path(__file__).parent / "output"))
    p.add_argument("--val_fraction", type=float, default=0.05)
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--epochs", type=float, default=3.0)
    # batch_size=2/grad_accum=8 (effective batch 16, same as batch_size=4/grad_accum=4) is the
    # safe default for 24-32GB cards — see finetune/README.md's memory notes for why gemma-2's
    # ~256k vocab makes the (batch, seq, vocab) loss tensor the actual memory bottleneck, not the
    # model weights. Raise --batch_size (and lower --grad_accum to compensate) on a ≥40GB GPU.
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--eval_accumulation_steps", type=int, default=5, help="Lower = less GPU memory held during eval (offloads to CPU more often), at some eval-speed cost.")
    p.add_argument("--early_stopping_patience", type=int, default=3)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--attn_implementation", default="eager", help="'eager' is the safe default for Gemma-2's attention logit softcapping; try 'sdpa' for speed once verified.")
    p.add_argument("--tensorboard_port", type=int, default=6006)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume_from_checkpoint", default=None)
    p.add_argument("--skip_merge", action="store_true", help="Skip producing the merged fp16 model at the end (keep LoRA adapter only).")
    return p.parse_args()


def launch_tensorboard(logdir: Path, port: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        ["tensorboard", "--logdir", str(logdir), "--host", "0.0.0.0", "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    atexit.register(proc.terminate)
    print(f"TensorBoard serving at http://0.0.0.0:{port} — expose this port on the pod to monitor training remotely.")
    return proc


def load_model_and_tokenizer(args: argparse.Namespace):
    hf_token = os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map={"": 0},
        attn_implementation=args.attn_implementation,
        token=hf_token,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


def make_metrics_fns(tokenizer):
    """Returns (preprocess_logits_for_metrics, compute_metrics) for a single-token binary label.

    preprocess_logits_for_metrics reduces each step's (batch, seq, vocab) logits to (batch, seq)
    argmax token ids before Trainer accumulates them across the eval set — gemma-2's ~256k vocab
    makes accumulating full logits prohibitive in memory.
    """
    zero_ids = tokenizer("0", add_special_tokens=False).input_ids
    one_ids = tokenizer("1", add_special_tokens=False).input_ids
    assert len(zero_ids) == 1 and len(one_ids) == 1, (
        f"Expected '0'/'1' to each tokenize to a single token, got {zero_ids} / {one_ids} — "
        "the label encoding in data.py assumes this."
    )
    zero_id, one_id = zero_ids[0], one_ids[0]

    def preprocess_logits_for_metrics(logits, labels):
        return logits.argmax(dim=-1)

    def compute_metrics(eval_pred) -> dict:
        predicted_ids, label_ids = eval_pred.predictions, eval_pred.label_ids
        y_true, y_pred, malformed = [], [], 0
        for row_pred, row_labels in zip(predicted_ids, label_ids):
            label_positions = np.where(row_labels != -100)[0]
            if len(label_positions) == 0:
                continue
            pos = int(label_positions[0])  # the label digit is always the first unmasked token
            if pos == 0 or pos - 1 >= len(row_pred):
                continue
            true = 1 if row_labels[pos] == one_id else 0
            pred_token = row_pred[pos - 1]  # causal LM: logits at pos-1 predict the token at pos
            if pred_token == one_id:
                pred = 1
            elif pred_token == zero_id:
                pred = 0
            else:
                malformed += 1
                pred = 1 - true  # neither "0" nor "1" was predicted: always counts as an error
            y_true.append(true)
            y_pred.append(pred)

        return {
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "fpr": sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1) / max(1, sum(1 for t in y_true if t == 0)),
            "malformed_rate": malformed / max(1, len(y_true)),
        }

    return preprocess_logits_for_metrics, compute_metrics


class SnapshotCallback(TrainerCallback):
    """Regenerates the loss-curve plot + metrics/timing summary (report.py) after every eval, so a
    monitoring snapshot is always on disk — not just visible in the live TensorBoard tab, which
    disappears the moment the pod stops. Cheap enough (a matplotlib plot + a couple JSON dumps) to
    run on every eval rather than only at the end."""

    def __init__(self, output_dir: Path, config: dict, start_time: float):
        self.output_dir = output_dir
        self.config = config
        self.start_time = start_time

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        export_snapshot(state.log_history, time.time() - self.start_time, metrics or {}, self.config, self.output_dir)


def merge_and_save(base_model: str, adapter_dir: Path, merged_dir: Path) -> None:
    """Reloads the base model unquantized and merges the trained LoRA adapter into it — merging
    directly into 4-bit weights isn't supported, so this is a separate, one-off fp16 reload."""
    print("Reloading base model in bf16 to merge LoRA adapters...")
    hf_token = os.getenv("HF_TOKEN")
    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map={"": 0}, token=hf_token)
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    merged.save_pretrained(merged_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(base_model, token=hf_token).save_pretrained(merged_dir)
    _copy_sentencepiece_model(base_model, merged_dir, hf_token)
    print(f"Merged model saved to {merged_dir}")


def _copy_sentencepiece_model(base_model: str, merged_dir: Path, hf_token: str | None) -> None:
    """The fast tokenizer's save_pretrained doesn't write out the raw SentencePiece
    tokenizer.model — but llama.cpp's convert_hf_to_gguf.py needs that exact file for Gemma's
    vocab (it isn't recoverable from tokenizer.json alone), so fetch it from the Hub directly and
    drop it alongside the merged model. See finetune/RUNPOD.md's "Serve the merged model with
    Ollama" step."""
    try:
        path = hf_hub_download(base_model, filename="tokenizer.model", token=hf_token)
        shutil.copy(path, merged_dir / "tokenizer.model")
    except Exception as exc:
        print(f"Warning: couldn't fetch tokenizer.model ({exc}) — GGUF conversion may need it copied in manually.")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_config.json").write_text(json.dumps(vars(args), indent=2))

    tb_logdir = output_dir / "tensorboard"
    # transformers>=5 dropped TrainingArguments(logging_dir=...) — TensorBoardCallback now reads
    # this env var instead (defaults to output_dir/runs/... if unset).
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(tb_logdir)
    launch_tensorboard(tb_logdir, args.tensorboard_port)

    model, tokenizer = load_model_and_tokenizer(args)

    train_df, val_df = load_split(args.train_data, args.val_fraction, args.seed)
    print(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows")
    train_ds = GatekeeperDataset(train_df, tokenizer, args.max_seq_len)
    val_ds = GatekeeperDataset(val_df, tokenizer, args.max_seq_len)

    preprocess_logits_for_metrics, compute_metrics = make_metrics_fns(tokenizer)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_ratio,
        bf16=True,
        gradient_checkpointing=True,
        # non-reentrant is the modern recommended mode (lower memory, avoids some known reentrant-
        # checkpointing + frozen-base-model interactions); compatible with prepare_model_for_kbit_training's
        # enable_input_require_grads() call in load_model_and_tokenizer.
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        eval_accumulation_steps=args.eval_accumulation_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["tensorboard"],
        seed=args.seed,
        remove_unused_columns=False,
    )

    start_time = time.time()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollator(tokenizer),
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
            SnapshotCallback(output_dir, vars(args), start_time),
        ],
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_metrics = trainer.evaluate()
    print("Final validation metrics:", final_metrics)
    export_snapshot(trainer.state.log_history, time.time() - start_time, final_metrics, vars(args), output_dir)
    print(f"Monitoring snapshot written to {output_dir} (training_curves.png, summary.md, summary.json, log_history.json)")

    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)
    print(f"LoRA adapter saved to {adapter_dir}")

    if not args.skip_merge:
        del model, trainer
        torch.cuda.empty_cache()
        merge_and_save(args.base_model, adapter_dir, output_dir / "merged")


if __name__ == "__main__":
    main()
