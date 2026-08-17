# Fine-tuning snapshot

Elapsed: **2h38m06s** (1600 steps, 2.493 epochs, 9 evals so far)

![loss curves](training_curves.png)

## Latest evaluated metrics

| metric | value |
|---|---|
| `eval_loss` | 0.0004 |
| `eval_precision` | 1.0000 |
| `eval_recall` | 1.0000 |
| `eval_f1` | 1.0000 |
| `eval_fpr` | 0.0000 |
| `eval_malformed_rate` | 0.0000 |

## Config

```json
{
  "base_model": "google/gemma-2-9b-it",
  "train_data": "/workspace/gatekeeper/data/train_consolidated.parquet",
  "output_dir": "/workspace/gatekeeper/finetune/output",
  "val_fraction": 0.05,
  "max_seq_len": 1024,
  "epochs": 3.0,
  "batch_size": 2,
  "grad_accum": 8,
  "lr": 0.0002,
  "warmup_ratio": 0.03,
  "eval_steps": 200,
  "eval_accumulation_steps": 5,
  "early_stopping_patience": 3,
  "lora_r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "attn_implementation": "eager",
  "tensorboard_port": 6006,
  "seed": 42,
  "resume_from_checkpoint": null,
  "skip_merge": false
}
```
