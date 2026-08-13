"""Exports a monitoring snapshot to disk: a train/val loss plot, a metrics + timing summary (JSON
and Markdown), and the raw log history. TensorBoard is only reachable while the pod is up — this
is what's left once it (or the pod) is gone. Deliberately has no torch/transformers dependency so
it can also be re-run offline against a saved log_history.json.

Called by SnapshotCallback in train.py after every eval, so a snapshot is on disk (and pull-able)
at any point during training, not only once the run finishes.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_METRIC_KEYS = ["eval_loss", "eval_precision", "eval_recall", "eval_f1", "eval_fpr", "eval_malformed_rate"]


def export_snapshot(log_history: list[dict], elapsed_seconds: float, latest_metrics: dict, config: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "log_history.json").write_text(json.dumps(log_history, indent=2))
    _plot_curves(log_history, output_dir / "training_curves.png")

    summary = _build_summary(log_history, elapsed_seconds, latest_metrics, config)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "summary.md").write_text(_render_markdown(summary))


def _plot_curves(log_history: list[dict], out_path: Path) -> None:
    train_steps = [e["step"] for e in log_history if "loss" in e]
    train_loss = [e["loss"] for e in log_history if "loss" in e]
    eval_steps = [e["step"] for e in log_history if "eval_loss" in e]
    eval_loss = [e["eval_loss"] for e in log_history if "eval_loss" in e]
    if not train_steps and not eval_steps:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    if train_steps:
        ax.plot(train_steps, train_loss, label="train loss", alpha=0.7, linewidth=1)
    if eval_steps:
        ax.plot(eval_steps, eval_loss, label="val loss", marker="o")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Gatekeeper fine-tune — train/val loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _build_summary(log_history: list[dict], elapsed_seconds: float, latest_metrics: dict, config: dict) -> dict:
    step_entries = [e for e in log_history if "step" in e]
    last_step = step_entries[-1]["step"] if step_entries else 0
    last_epoch = step_entries[-1].get("epoch", 0) if step_entries else 0
    eval_entries = [e for e in log_history if "eval_loss" in e]
    return {
        "elapsed_seconds": round(elapsed_seconds, 1),
        "elapsed_human": _format_duration(elapsed_seconds),
        "steps_completed": last_step,
        "epochs_completed": round(last_epoch, 3),
        "n_evals": len(eval_entries),
        "latest_metrics": {k: latest_metrics[k] for k in _METRIC_KEYS if k in latest_metrics},
        "config": config,
    }


def _render_markdown(summary: dict) -> str:
    metrics_rows = "\n".join(f"| `{k}` | {v:.4f} |" for k, v in summary["latest_metrics"].items())
    return f"""# Fine-tuning snapshot

Elapsed: **{summary['elapsed_human']}** ({summary['steps_completed']} steps, {summary['epochs_completed']} epochs, {summary['n_evals']} evals so far)

![loss curves](training_curves.png)

## Latest evaluated metrics

| metric | value |
|---|---|
{metrics_rows}

## Config

```json
{json.dumps(summary['config'], indent=2)}
```
"""


def _format_duration(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s"
