# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Gatekeeper is a Dockerized FastAPI service that classifies text prompts as threats (`1`) or benign (`0`) using an LLM as the classifier. It targets OWASP LLM01 (prompt injection) and LLM07 (system prompt leakage), scoped to explicit attempts to hijack an AI's instructions or extract its system prompt/secrets — not general jailbreak/harmful-content detection (see README.md's "Note" section for the exact scope boundary).

## Running the service

```bash
cp .env.example .env
```

**Local LLM** (Ollama container, no DB logging):
```bash
docker compose -f docker-compose.yml -f docker-compose.local-llm.yml --profile local-llm up
```

**Online LLM** (OpenAI-compatible endpoint, requires `ONLINE_LLM_API_KEY` in `.env`):
```bash
docker compose up
```

**Prod overlay** (adds Postgres logging, combine with either backend above):
```bash
docker compose -f docker-compose.yml -f docker-compose.local-llm.yml -f docker-compose.prod.yml --profile local-llm up
docker compose -f docker-compose.yml -f docker-compose.prod.yml up
```

Compose files are additive overlays, not standalone alternatives — `docker-compose.local-llm.yml` and `docker-compose.prod.yml` only set env vars / add services on top of the base `docker-compose.yml`.

There is no test suite (no pytest/unittest anywhere in the repo). Verification is done via the scripts in `demo/`:

```bash
python demo/preprocessing_demo.py   # exercises the 7-step preprocessing pipeline, no running services needed
python demo/demo.py                 # single prompt end-to-end against a live /verify endpoint
python demo/attack_demo.py          # 25 prompts across 4 attack categories against both endpoints
```

All three demo scripts use only the Python standard library — no dependency install needed to run them. Point them at a non-default host with `GATEKEEPER_URL=http://your-server:8000`.

## Architecture

Request flow: `app/api/routes.py` → `app/verification/verifier.py` (`Verifier`) → `app/verification/preprocessing.py` + `app/llm/llm_adaptater.py` → `app/db/logger.py`.

- **`app/main.py`** — FastAPI app + lifespan. On startup it builds the `Verifier` once via `create_llm()` and stashes it on `app.state.verifier`; routes pull it from there rather than constructing it per-request. If `LLM_BACKEND=local`, startup blocks until Ollama reports the model pulled *and* has run one warmup generation (`_wait_for_local_model`) — this is why local-backend startup is slow. Online backends only warm up if `ONLINE_LLM_WARMUP=true` (for self-hosted OpenAI-compatible servers with Ollama-style cold starts; real hosted APIs skip this).

- **`app/llm/llm_adaptater.py`** — `LLMInterface` ABC with two implementations: `LocalAdapter` (talks to Ollama's `/api/generate`) and `OnlineAdapter` (talks to any OpenAI-compatible `/chat/completions`, keyed by `ONLINE_LLM_API_KEY`/`ONLINE_LLM_BASE_URL`). `app/llm/factory.py` picks one based on `LLM_BACKEND`. The two adapters have different knobs for disabling reasoning output on hybrid-thinking models (`think` boolean for Ollama vs `reasoning_effort` field for OpenAI-compatible) — `gpt-oss` models don't accept a boolean at all and log a warning instead (`_warn_if_gpt_oss`).

- **`app/verification/verifier.py`** — `Verifier.verify()` (single-pass: preprocess then classify) and `Verifier.verify_raw()` (two-pass: classify raw text AND preprocessed text, threat if either fires). `_classify()` loads the YAML prompt template fresh on every call (not cached — "kept for future extensibility" per the comment), substitutes `{{input}}`, and retries up to `LLM_RETRY_CALLS` times on `httpx.HTTPError` or a malformed (non-`0`/`1`) LLM response. On retry exhaustion for a malformed response, it fails closed (defaults to `1`, threat); on retry exhaustion for an HTTP error, it re-raises. The LLM's system prompt is selected via `PROMPT_NAME` env var → `app/verification/prompts/<name>.yaml`; there is no `default.yaml`, only `default-3b.yaml` and `default-9b.yaml` (sized to the target model's parameter count — see the calibration examples/threat taxonomy inside those files before editing).

- **`app/verification/preprocessing.py`** — pure functions, no I/O. Pipeline order matters: truncate → strip invisible/control chars → decode obfuscation layers (URL-encoding, HTML entities, base64, hex, applied in sequence) → NFKC unicode normalization (defeats fullwidth lookalike chars) → strip fake model markup (`<|system|>`-style control tokens and `<system>`-style tag delimiters) → regex pre-screen for PII/credential patterns (logged, not stripped). Returns a `PreprocessingResult(text, pattern_hits)`. `verify_raw`'s raw pass intentionally skips this pipeline so prompts that legitimately contain `<system>`-style tags can still be checked as-is.

- **`app/db/logger.py`** — `DBLogger` is a no-op wrapper when `LOG_TO_DB=false` (the default); only connects to Postgres and inserts into the `logs` table (schema in `db/init.sql`) when explicitly enabled via the prod compose overlay.

- Two endpoints in `app/api/routes.py`: `POST /verify` (single-pass, returns the preprocessed prompt for safe forwarding downstream) and `POST /verify-raw` (two-pass, no preprocessed text returned — use when the untouched original must be forwarded). Both map `httpx.HTTPError` from the verifier to a `502`.

## Working in this repo

- `app/` runs with `app/` as the import root inside its container (Dockerfile does `COPY . .` into `/app` then `uvicorn main:app`) — internal imports are absolute from that root (`from api.routes import router`, `from llm.factory import create_llm`), not `app.api.routes`.
- When touching a prompt YAML in `app/verification/prompts/`, re-run `demo/attack_demo.py` against a live server to sanity-check the four attack categories (harmless / direct attack / verifier evasion / preprocessing) before considering the change done — the taxonomy and calibration examples embedded in the prompt file are the actual spec for what should and shouldn't classify as a threat.
- `data/` is a separate concern with its own `requirements.txt` (pandas/datasets/huggingface_hub/etc.) — only needed for rebuilding the eval/train datasets, not for running the API itself. `data/eval_dataset_clean.parquet` is excluded from version control; see `data/main_create_datasets.py` (source loaders in `data/helpers/dataset_loaders.py`) for how it's built.
- `evaluation/` is also a separate concern with its own `requirements.txt` (pandas/scikit-learn/httpx/etc.) — only needed for running `evaluation/evaluation.ipynb` against a live server.
- Model choice tradeoffs (precision/recall/F1/FPR per model) live in the README and `evaluation/readme.md` — check there before assuming a given local/online model is "good enough" for a change.
