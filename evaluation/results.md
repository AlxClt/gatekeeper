# Gatekeeper Evaluation Results

Results from [`evaluation.ipynb`](evaluation.ipynb) — prompt-injection (LLM01) and system-prompt-leakage (LLM07), run one-pass (`POST /verify`) against:

- `deepset/prompt-injections` (LLM01)
- `xTRam1/safe-guard-prompt-injection` (LLM01)
- `Lakera/gandalf_ignore_instructions` (LLM07 - not 100% of the attacks are LLM07)

Models compared:
3B parameters category: **llama3.2:3b**, **qwen2.5:3b**, **gemma3:4b**, **phi4-mini**
9B parameters category: **gemma2:9b**, **llama3.1:8b**, **qwen3:8b**, **glm4:9b**

Same system prompt used for all ~9b models: app/llm/prompts/default-9b.yaml  
Same system prompt used for all ~3b models: app/llm/prompts/default-3b.yaml  

All self hosted models (**Model 1**, **Model 2**, **Model 3**, **Model 4**) have been run on a RTX PRO 4500 GPU with 62GB RAM

Date of evaluation run: _TBD_

---

## Per-dataset metrics

### `deepset/prompt-injections` (LLM01)

**n=622**

3B models

| Model | precision | recall | f1 | fpr |
|---|---|---|---|---|
| llama3.2 | 0.798 | 0.825 | 0.811 | 0.138 |
| qwen2.5:3b | | | | |
| Model 3 | | | | |
| Model 4 | | | | |
| Model 5 | | | | |
| Model 6 | | | | |


9B models 

| Model | precision | recall | f1 | fpr |
|---|---|---|---|---|
| Model 1 | | | | |
| Model 2 | | | | |
| gemma2:9b | | | | |
| Model 4 | | | | |
| Model 5 | | | | |
| Model 6 | | | | |

### `xTRam1/safe-guard-prompt-injection` (LLM01)

**n=10296**

3B models

| Model | precision | recall | f1 | fpr |
|---|---|---|---|---|
| llama3.2 | 0.780 | 0.962 | 0.862 | 0.119 |
| qwen2.5:3b | | | | |
| Model 3 | | | | |
| Model 4 | | | | |
| Model 5 | | | | |
| Model 6 | | | | |

9B models

| Model | precision | recall | f1 | fpr |
|---|---|---|---|---|
| Model 1 | | | | |
| Model 2 | | | | |
| gemma2:9b | | | | |
| Model 4 | | | | |
| Model 5 | | | | |
| Model 6 | | | | |

### `Lakera/gandalf_ignore_instructions` (LLM07)

**n=1400**

3B models

| Model | precision | recall | f1 | fpr |
|---|---|---|---|---|
| llama3.2 | 0.955 | 1.000 | 0.977 | 0.118 |
| qwen2.5:3b | | | | |
| Model 3 | | | | |
| Model 4 | | | | |
| Model 5 | | | | |
| Model 6 | | | | |

9B models

| Model | precision | recall | f1 | fpr |
|---|---|---|---|---|
| Model 1 | | | | |
| Model 2 | | | | |
| gemma2:9b | | | | |
| Model 4 | | | | |
| Model 5 | | | | |
| Model 6 | | | | |

---

## Per-threat-class metrics

Note: LLM07's negatives are entirely the 400 prompts borrowed from `xTRam1` (see notebook §1) — `Lakera/gandalf` itself contributes no safe prompts, so its FPR reflects the borrowed sample, not native negatives.

### LLM01 (Prompt Injection)

**n=10958**

3B models

| Model | precision | recall | f1 | fpr |
|---|---|---|---|---|
| llama3.2 | 0.781 | 0.951 | 0.858 | 0.120 |
| qwen2.5:3b | | | | |
| Model 3 | | | | |
| Model 4 | | | | |
| Model 5 | | | | |
| Model 6 | | | | |

9B models

| Model | precision | recall | f1 | fpr |
|---|---|---|---|---|
| Model 1 | | | | |
| Model 2 | | | | |
| gemma2:9b | | | | |
| Model 4 | | | | |
| Model 5 | | | | |
| Model 6 | | | | |

### LLM07 (System Prompt Leakage)

Note: we only use the `Lakera/gandalf_ignore_instructions` dataset for this class so the results are the same than the per dataset results

**n=1400**

3B models

| Model | precision | recall | f1 | fpr |
|---|---|---|---|---|
| llama3.2 | 0.955 | 1.000 | 0.977 | 0.118 |
| qwen2.5:3b | | | | |
| Model 3 | | | | |
| Model 4 | | | | |
| Model 5 | | | | |
| Model 6 | | | | |

9B models

| Model | precision | recall | f1 | fpr |
|---|---|---|---|---|
| Model 1 | | | | |
| Model 2 | | | | |
| gemma2:9b | | | | |
| Model 4 | | | | |
| Model 5 | | | | |
| Model 6 | | | | |

---

## Time per request

Note: Comparison is relevant for models hosted on the same GPU. 

3B models

| Model | average time per request (seconds) |
|---|---|
| llama3.2 | 0.085 |
| qwen2.5:3b | | | | |
| Model 3 | | | | |
| Model 4 | |
| Model 5 | |
| Model 6 | |

9B models

| Model | average time per request (seconds) |
|---|---|
| Model 1 | |
| Model 2 | |
| gemma2:9b | |
| Model 4 | |
| Model 5 | |
| Model 6 | |

---

## Manual inspection notes

Observations from the false positive / false negative review (§4 of the notebook), per model.

### Model 1
- False positives:
- False negatives:

### Model 2
- False positives:
- False negatives:

### Model 3
- False positives:
- False negatives:

### Model 4
- False positives:
- False negatives:

### Model 5
- False positives:
- False negatives:

### Model 6
- False positives:
- False negatives:

---

## Summary / takeaways

_TBD_
