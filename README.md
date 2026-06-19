# PhenoAgent

A three-stage agentic LLM pipeline for high-precision phenotyping from Japanese discharge summaries.

Companion code for *PhenoAgent: agentic LLM framework for phenotyping electronic health records via structured query decomposition and self-correction*, npj Digital Medicine (in submission).

## Pipeline

PhenoAgent operates in three stages:

1. **Structured Query Decomposition** (Semantic Parser) -- An LLM converts a free-form natural-language clinical query into a structured JSON schema of N boolean criteria. The query can be any clinical question; the LLM determines the appropriate decomposition.
2. **Self-Correcting Semantic Evaluation** (Semantic Evaluator) -- A second LLM pass applies the schema to each discharge summary, producing per-criterion true/false judgements with natural-language reasoning. Malformed JSON outputs trigger an automatic retry mechanism (up to 3 attempts with progressively stricter formatting instructions).
3. **Unanimous Ensemble Voting** -- A case is classified as positive only when **all three** constituent LLMs independently agree (AND vote). The PhenoAgent ensemble consists of DeepSeek-32B, GPT-OSS-20B, and Llama-3.1-Swallow-70B, selected by exhaustive search over all 63 three-model subsets of six candidate LLMs.

Inference runs locally on a single Apple Silicon workstation via MLX -- no external API calls.

## Repository structure

```
src/pipeline/      Three-stage agentic pipeline (end2end.py + utils.py)
prompts/           Per-model Japanese prompt templates
configs/           Per-model configurations (HuggingFace model paths)
scripts/           generate_dummy_data.py, run_pipeline.sh, run_end2end.sh
data_sample/       Synthetic dummy data (see "Quick start")
```

## Installation

```bash
git clone https://github.com/javasparrows/PhenoAgent.git
cd PhenoAgent
uv sync
```

Target environment: Apple Silicon with the MLX backend.

## Quick start with synthetic dummy data

Real discharge summaries are **not** distributed (see [Data availability](#data-availability)). The demo generates synthetic dummy data and runs the PhenoAgent pipeline on it.

```bash
# One-command demo: generate dummy data + run end-to-end pipeline
bash scripts/run_pipeline.sh tohoku

# Or step by step:
uv run python scripts/generate_dummy_data.py         # Create synthetic data
uv run python -m src.pipeline.end2end \
    --dataset tohoku \
    --config configs/config_gpt_oss_20b.py \
    --query "心停止の患者さんの中で、ECMOが使われた患者は何人？" \
    --label-column "心停止+ECMO" \
    --data-dir data_sample/tohoku
```

The pipeline requires an MLX-compatible LLM checkpoint. Model weights are not redistributed; download the checkpoints into the `mlx-lm` cache (see [Models](#models)).

## Running with a free-form query

The `--query` argument accepts any natural-language clinical question. The pipeline decomposes it into N boolean criteria (Stage 1), evaluates each record against the schema (Stage 2), and classifies a case as positive when all criteria are true.

```bash
# Example: run a custom query on synthetic data
uv run python -m src.pipeline.end2end \
    --dataset tohoku \
    --config configs/config_gpt_oss_20b.py \
    --query "Are there patients with cardiac arrest who received ECMO?" \
    --data-dir data_sample/tohoku
```

The `--label-column` argument is optional and only needed for evaluation against ground-truth labels.

### Batch run (12 paper queries)

```bash
./scripts/run_end2end.sh tohoku            # Per-dataset LLM inference
./scripts/run_end2end.sh okayama
```

## Models

| Internal identifier | HuggingFace ID | PhenoAgent ensemble |
| --- | --- | :---: |
| `GPT-OSS-20B`           | `openai/gpt-oss-20b`                                | yes |
| `CALM3-22B`             | `cyberagent/calm3-22b-chat`                         |   |
| `DeepSeek-32B`          | `cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese`  | yes |
| `DeepSeek-70B`          | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B`         |   |
| `Llama-3.1-Swallow-70B` | `tokyotech-llm/Llama-3.1-Swallow-70B-Instruct-v0.3` | yes |
| `MedLLM-72B`            | `pfnet/Preferred-MedLLM-Qwen-72B`                   |   |

## Clinical query set

Twelve queries on the Tohoku cohort; seven of them on Okayama.

| ID | English | Japanese |
| --- | --- | --- |
| 1  | Cardiac arrest + ECMO                     | 心停止 + ECMO                |
| 2  | Cardiac arrest + Hypoxic encephalopathy   | 心停止 + 低酸素性脳症       |
| 3  | Abdominal stab wound + Laparotomy         | 腹部刺創 + 開腹術           |
| 4  | Hypothermia + ECMO                        | 低体温 + ECMO                |
| 5  | Myocardial infarction + ECMO              | 心筋梗塞 + ECMO              |
| 6  | Heart failure + NPPV                      | 心不全 + NPPV                |
| 7  | Sepsis + AKI                              | 敗血症 + 急性腎障害         |
| 8  | Sepsis + DIC                              | 敗血症 + 播種性血管内凝固   |
| 9  | Acute drug poisoning + Activated charcoal | 急性薬物中毒 + 活性炭投与   |
| 10 | Pelvic fracture + External fixation       | 骨盤骨折 + 創外固定         |
| 11 | Pelvic fracture + IVR                     | 骨盤骨折 + IVR               |
| 12 | Pelvic fracture + Hemorrhagic shock       | 骨盤骨折 + 出血性ショック   |

Abbreviations: AKI, acute kidney injury; DIC, disseminated intravascular coagulation; ECMO, extracorporeal membrane oxygenation; IVR, interventional radiology; NPPV, non-invasive positive pressure ventilation.

## Data availability

Discharge summaries from Tohoku University Hospital and Okayama University Hospital contain protected patient information and cannot be released. `data_sample/` ships with **synthetic dummy data** for code-execution testing only. The paper's numbers cannot be reproduced from synthetic data alone. Requests for the source clinical text require ethics-committee approval at the requester's institution; direct enquiries to the corresponding author.

## Citation

```bibtex
@article{phenoagent2026,
  title   = {PhenoAgent: agentic LLM framework for phenotyping electronic health records via structured query decomposition and self-correction},
  author  = {Kashiwada, Yuki and others},
  journal = {npj Digital Medicine},
  year    = {2026},
  note    = {In submission}
}
```

The BibTeX entry will be updated once the DOI is assigned.

## License

This software is released under the MIT License -- see `LICENSE`. HuggingFace model weights retain their original licenses -- see `NOTICE`.
