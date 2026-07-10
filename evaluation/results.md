# Gatekeeper Evaluation Results

Results from [`evaluation.ipynb`](evaluation.ipynb) — prompt-injection (LLM01) and system-prompt-leakage (LLM07), run one-pass (`POST /verify`) against the clean eval set built in [`create_eval_dataset.ipynb`](create_eval_dataset.ipynb) (`eval_dataset_clean.parquet`).

**Models compared:**

| Category | Models |
|---|---|
| Zero-shot, 3B | **llama3.2:3b**, **qwen2.5:3b**, **gemma3:4b** |
| Zero-shot, 9B | **gemma2:9b**, **llama3.1:8b**, **qwen3:8b** |
| Fine-tuned | _TBD_ |

We also run a 12B model for reference: **gemma4:12b**. It doesn't bring significant uplift in performance metrics, suggesting that the model capacity is not the limiting factor for improvement.

System prompt used for all zero-shot ~9B models: `app/llm/prompts/default-9b.yaml`

System prompt used for all zero-shot ~3B models: `app/llm/prompts/default-3b.yaml`

Fine-tuned model prompt/config: _TBD_

All models have run on an RTX PRO 4500 GPU with 62GB RAM.

Date of evaluation run: _TBD_

## Fundamental remark: evaluation caveat for zero shot classification: scope vs. dataset mismatch and threat class non mutually exclusive

This classifier is deliberately scoped to two OWASP threats: **LLM01**
(prompt injection — hijacking or overriding a model's instructions) and
**LLM07** (system-prompt / secret extraction). It does **not** judge whether
the requested *content* is harmful. A request to produce violent, illegal, or
explicit material is out of scope unless it also attacks the model's
instructions or attempts to leak them. The same remark holds for ambiguous jailbreak attempts

Most public prompt-injection datasets are broader than this. They frequently
merge prompt injection with jailbreaks and harmful-content elicitation under a
single positive label, and several are synthetically generated with noisy or
near-inseparable labels. As a result, a share of their "attack" samples are
out of scope for this classifier (harmful content with no instruction hijack),
and a share of their labels do not correspond to any concrete attack on the
model. Datasets used in the evaluation set have been carefully selected and manually reviewed to avoid this phenomenon but some mislabeled examples might still pollute the results.

Last but not least, LLM01 and LLM07 are not mutually exclusive and some datasets mix both. While the best has been done to assign leach dataset to a category, the most relevant metric stays the overall metrics computed across all threat classes. 

---

## Metrics

Precision, recall, F1, and FPR reported as one **overall** binary-task summary (LLM01/LLM07 = 1 vs. benign = 0) across all three model categories, so zero-shot 3B, zero-shot 9B, and the fine-tuned model can be compared directly. Recall is additionally broken out per threat class (LLM01 vs. LLM07) — the dataset's negatives aren't split per threat class, so recall is the only one of the four metrics that's meaningful at that granularity (see notebook §3).

Per-dataset and per-source differences (if any are significant) are called out in the [Remarks](#remarks) section below rather than as a table — see the notebook's own per-dataset breakdown for the underlying numbers.

**n=3664**

### Overall

| Model | Category | precision | recall | f1 | fpr |
|---|---|---|---|---|---|
| llama3.2:3b | Zero-shot 3B | 0.746 | 0.991 | 0.851 | 0.329 |
| qwen2.5:3b | Zero-shot 3B | 0.880 | 0.886 | 0.883 | 0.118 |
| gemma3:4b | Zero-shot 3B | 0.687 | 0.996 | 0.813 | 0.442 |
| llama3.1:8b | Zero-shot 9B | 0.901 | 0.965 | 0.932 | 0.104 |
| gemma2:9b | Zero-shot 9B | 0.960 | 0.921 | 0.940 | 0.038 |
| qwen3:8b | Zero-shot 9B | 0.994 | 0.850 | 0.916 | 0.005 |
| gemma4:12b | Zero-shot 12B | 0.941 | 0.965 | 0.953 | 0.058 |
| _TBD_ | Fine-tuned | | | | |

### Recall by threat class

**n=348 (LLM01)**
**n=1463 (LLM07)**

| Model | Category | LLM01 recall | LLM07 recall |
|---|---|---|---|
| llama3.2:3b | Zero-shot 3B | 0.994 | 0.991 |
| qwen2.5:3b | Zero-shot 3B | 0.804 | 0.906 |
| gemma3:4b | Zero-shot 3B | 0.982 | 0.999 |
| llama3.1:8b | Zero-shot 9B | 0.940 | 0.971 |
| gemma2:9b | Zero-shot 9B | 0.894 | 0.927 |
| qwen3:8b | Zero-shot 9B | 0.744 | 0.875 |
| gemma4:12b | Zero-shot 12B | 0.908 | 0.979 |
| _TBD_ | Fine-tuned | | |

---

## Time per request

Note: comparison is only meaningful within a category — models were run on different hardware/hosting setups across categories.

| Model | Category | average time per request (seconds) |
|---|---|---|
| llama3.2:3b | Zero-shot 3B | 0.087 |
| qwen2.5:3b | Zero-shot 3B | 0.081 |
| gemma3:4b | Zero-shot 3B | 0.111 |
| llama3.1:8b | Zero-shot 9B | 0.106 |
| gemma2:9b | Zero-shot 9B | 0.121 |
| qwen3:8b | Zero-shot 9B | 0.096 |
| gemma4:12b | Zero-shot 12B | 0.125 |
| _TBD_ | Fine-tuned | |

---

## Manual inspection notes

Observations from the false positive / false negative review (§4 of the notebook), per model.

### Zero-shot 3B

#### llama3.2:3b

- False positives:
- False negatives:

#### qwen2.5:3b

- False positives:
- False negatives:

#### gemma3:4b

- False positives:
- False negatives:

### Zero-shot 9B

#### gemma2:9b

- False positives:
- False negatives:

#### llama3.1:8b

- False positives:
- False negatives:

#### qwen3:8b

- False positives:
- False negatives:

### Fine-tuned

#### _TBD_

- False positives:
- False negatives:

---

## Remarks

_TBD_

---

## Summary / takeaways

_TBD_
