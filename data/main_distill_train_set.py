"""Filters train_raw.parquet down to rows the gatekeeper model actually agrees are threats.

Drops out-of-scope rows (`in_scope=False`) entirely, then sends only the remaining `label=1` rows
through a live gatekeeper `/verify` endpoint (same one-pass call as evaluation/evaluation.ipynb) —
`label=0` rows don't need a model opinion, so they're kept as-is without an API call. Any `label=1`
row the model predicts as `0` is training-label noise the model doesn't even recognize as an
attack under its own zero-shot judgment, and gets dropped too.

Requires a running gatekeeper server — point BASE_URL at it before running:
`python main_distill_train_set.py`.
"""

import asyncio
import time

import httpx
import pandas as pd
from tqdm import tqdm

from helpers.dataset_loaders import SCHEMA_COLUMNS

BASE_URL = "http://localhost:8000"  # gatekeeper server
CONCURRENCY = 10                    # concurrent requests

TRAIN_RAW_PATH = "train_raw.parquet"
TRAIN_DISTILLED_PATH = "train_distilled.parquet"


async def _classify_one(client: httpx.AsyncClient, text: str, sem: asyncio.Semaphore) -> int:
    async with sem:
        resp = await client.post(f"{BASE_URL}/verify", json={"prompt": text})
        resp.raise_for_status()
        return resp.json()["result"]


async def run_inference(texts: list[str]) -> list[int]:
    """Classify every text concurrently (bounded by CONCURRENCY), with a progress bar."""
    sem = asyncio.Semaphore(CONCURRENCY)
    results: list[int] = [None] * len(texts)
    pbar = tqdm(total=len(texts), desc="Classifying")

    async with httpx.AsyncClient(timeout=500) as client:
        async def _task(i: int, text: str):
            results[i] = await _classify_one(client, text, sem)
            pbar.update(1)

        await asyncio.gather(*[_task(i, t) for i, t in enumerate(texts)])

    pbar.close()
    return results


def distill(positives: pd.DataFrame, predictions: list[int], negatives: pd.DataFrame) -> pd.DataFrame:
    """Drop label=1 rows the model predicted as 0 (not a threat); label=0 rows pass through untouched."""
    positives = positives.copy()
    positives["prediction"] = predictions

    missed = positives["prediction"] == 0
    kept_positives = positives[~missed].drop(columns=["prediction"])
    print(f"Dropped {missed.sum()} label=1 rows the model didn't flag as a threat — {len(kept_positives)} kept")

    combined = pd.concat([negatives, kept_positives], ignore_index=True)
    return combined[SCHEMA_COLUMNS].reset_index(drop=True)


def main() -> None:
    df = pd.read_parquet(TRAIN_RAW_PATH)
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    print(f"Loaded {len(df)} rows from {TRAIN_RAW_PATH}")

    df = df[df["in_scope"]].reset_index(drop=True)
    print(f"{len(df)} rows remain after dropping in_scope=False")

    positives = df[df["label"] == 1].reset_index(drop=True)
    negatives = df[df["label"] == 0].reset_index(drop=True)
    print(f"Sending {len(positives)} label=1 rows to the verifier ({len(negatives)} label=0 rows kept without inference)")

    inference_start = time.perf_counter()
    predictions = asyncio.run(run_inference(positives["text"].tolist()))
    inference_elapsed = time.perf_counter() - inference_start
    avg_time_per_request = inference_elapsed / len(positives) if len(positives) else 0.0
    print(f"Inference time: {inference_elapsed:.2f}s total for {len(positives)} requests ({avg_time_per_request:.3f}s/request avg)")

    distilled = distill(positives, predictions, negatives)
    distilled.to_parquet(TRAIN_DISTILLED_PATH, index=False)
    print(f"Saved {len(distilled)} rows to {TRAIN_DISTILLED_PATH}")


if __name__ == "__main__":
    main()
