"""
Gatekeeper prompt-injection & system-prompt-leakage evaluation.

    python evaluation/evaluation.py

Runs the clean, held-out eval set (`../data/eval_dataset_clean.parquet`, built by
`../data/main_create_datasets.py`) through the one-pass `POST /verify` endpoint of a live
gatekeeper server, covering two OWASP threat classes: LLM01 (prompt injection / instruction
hijacking) and LLM07 (system-prompt / secret extraction).

Precision, recall, F1 and FPR are reported per dataset and as one overall binary-task summary.
Recall is additionally broken out per threat class: in this dataset `threat_class` is only set to
LLM01/LLM07 on positive rows (every negative is tagged `benign`), so recall is the only one of the
four metrics that is well-defined per threat class.

Outputs go to `results/<YYYYMMDD>_<model_name>/`:
    metrics.csv                  per-dataset rows + an `overall` row
    recall_by_threat_class.csv   LLM01 vs. LLM07 recall
    errors.csv                   every false positive / false negative
    run_info.json                model, sample count, inference timing

`model_name` is read from `../.env` (`ONLINE_LLM_MODEL` or `LOCAL_LLM_MODEL` depending on
`LLM_BACKEND`) unless overridden with `--model-name`. The server URL and API token come from
`GATEKEEPER_URL` / `GATEKEEPER_API_TOKEN` (environment first, then `../.env`).
"""

import argparse
import asyncio
import json
import os
import re
import time
from datetime import date
from pathlib import Path

import httpx
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
EVAL_DATASET_PATH = ROOT / "data" / "eval_dataset_clean.parquet"
ENV_PATH = ROOT / ".env"
RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ── Configuration ─────────────────────────────────────────────────────────────

def read_env_file(env_path: Path = ENV_PATH) -> dict[str, str]:
    env_vars = {}
    if not env_path.exists():
        return env_vars
    for line in env_path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env_vars[key.strip()] = value.strip()
    return env_vars


def model_name_from_env(env_vars: dict[str, str]) -> str:
    backend = env_vars.get("LLM_BACKEND", "online")
    model_name = env_vars.get("LOCAL_LLM_MODEL") if backend == "local" else env_vars.get("ONLINE_LLM_MODEL")
    return model_name or "unknown_model"


# ── Inference ─────────────────────────────────────────────────────────────────

async def _classify_one(client: httpx.AsyncClient, url: str, text: str, sem: asyncio.Semaphore) -> int:
    async with sem:
        resp = await client.post(url, json={"prompt": text})
        resp.raise_for_status()
        return resp.json()["result"]


async def run_inference(texts: list[str], base_url: str, token: str, concurrency: int) -> list[int]:
    sem = asyncio.Semaphore(concurrency)
    results: list[int] = [None] * len(texts)
    url = f"{base_url.rstrip('/')}/verify"
    headers = {"Authorization": f"Bearer {token}"}

    with tqdm(total=len(texts), desc="Classifying") as pbar:
        async with httpx.AsyncClient(timeout=500, headers=headers) as client:
            async def _task(i: int, text: str):
                results[i] = await _classify_one(client, url, text, sem)
                pbar.update(1)

            await asyncio.gather(*[_task(i, t) for i, t in enumerate(texts)])

    return results


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(sub: pd.DataFrame) -> dict:
    y_true = sub["label"].to_numpy()
    y_pred = sub["prediction"].to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else float("nan")

    return {
        "n": len(sub), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
    }


def metrics_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-source metrics plus an `overall` row treating every sample as one binary task."""
    rows = [{"source": key, **compute_metrics(sub)} for key, sub in frame.groupby("source")]
    rows.append({"source": "overall", **compute_metrics(frame)})
    return pd.DataFrame(rows).set_index("source")


def recall_by_threat_class(frame: pd.DataFrame) -> pd.DataFrame:
    """Recall only, grouped by threat_class — the only metric well-defined when a group has no
    native negatives (see module docstring)."""
    positives = frame[frame["label"] == 1]
    rows = [
        {"threat_class": key, "n": len(sub), "recall": recall_score(sub["label"], sub["prediction"], zero_division=0)}
        for key, sub in positives.groupby("threat_class")
    ]
    return pd.DataFrame(rows).set_index("threat_class")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    env_vars = read_env_file()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=os.getenv("GATEKEEPER_URL", env_vars.get("GATEKEEPER_URL", "http://localhost:8000")))
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--model-name", default=model_name_from_env(env_vars),
                        help="Name used for the results folder (default: read from ../.env)")
    args = parser.parse_args()

    token = os.getenv("GATEKEEPER_API_TOKEN", env_vars.get("GATEKEEPER_API_TOKEN", ""))

    df = pd.read_parquet(EVAL_DATASET_PATH)
    df = df.dropna(subset=["text"]).reset_index(drop=True)

    print(f"Loaded {len(df)} samples from {EVAL_DATASET_PATH}\n")
    print("Per source (label counts):")
    print(df.groupby("source")["label"].value_counts().rename("count").to_string())
    print("\nPer threat class (label counts):")
    print(df.groupby("threat_class")["label"].value_counts().rename("count").to_string())
    print()

    inference_start = time.perf_counter()
    df["prediction"] = asyncio.run(run_inference(df["text"].tolist(), args.url, token, args.concurrency))
    inference_elapsed = time.perf_counter() - inference_start
    avg_time_per_request = inference_elapsed / len(df)
    print(f"\nInference time: {inference_elapsed:.2f}s total for {len(df)} requests "
          f"({avg_time_per_request:.3f}s/request avg)\n")

    metrics = metrics_table(df)
    threat_recall = recall_by_threat_class(df)
    print(metrics.to_string(float_format="%.3f"))
    print()
    print(threat_recall.to_string(float_format="%.3f"))

    errors_df = df[df["prediction"] != df["label"]].copy()
    errors_df["error_type"] = errors_df["prediction"].map({1: "false_positive", 0: "false_negative"})

    safe_model_name = re.sub(r'[<>:"/\\|?*]', "-", args.model_name)
    out_dir = RESULTS_DIR / f"{date.today():%Y%m%d}_{safe_model_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(out_dir / "metrics.csv")
    threat_recall.to_csv(out_dir / "recall_by_threat_class.csv")
    errors_df[["source", "threat_class", "error_type", "label", "prediction", "text"]].to_csv(
        out_dir / "errors.csv", index=False
    )
    (out_dir / "run_info.json").write_text(json.dumps({
        "model_name": args.model_name,
        "date": date.today().isoformat(),
        "url": args.url,
        "n": len(df),
        "inference_time_s": round(inference_elapsed, 2),
        "avg_time_per_request_s": round(avg_time_per_request, 3),
    }, indent=2))

    print(f"\nSaved metrics and {len(errors_df)} classification errors to {out_dir}")


if __name__ == "__main__":
    main()
