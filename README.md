# PhenoAgent

A three-stage agentic LLM pipeline for high-precision phenotyping from Japanese discharge summaries.

Companion code for *PhenoAgent: agentic LLM framework for phenotyping electronic health records via structured query decomposition and self-correction*, npj Digital Medicine (in submission).

## Pipeline

PhenoAgent operates in three stages:

1. **Structured Query Decomposition** (Semantic Parser) -- An LLM converts a free-form natural-language clinical query into a structured JSON schema of N boolean criteria. The query can be any clinical question; the LLM determines the appropriate decomposition.
2. **Self-Correcting Semantic Evaluation** (Semantic Evaluator) -- A second LLM pass applies the schema to each discharge summary, producing per-criterion true/false judgements with natural-language reasoning. Malformed JSON outputs trigger an automatic retry mechanism (up to 3 attempts with progressively stricter formatting instructions).
3. **Unanimous Ensemble Voting** -- A case is classified as positive only when **all three** constituent LLMs independently agree (AND vote). The PhenoAgent ensemble consists of GPT-OSS-20B, DeepSeek-32B, and Llama-3.3-Swallow-70B, selected by exhaustive search over all 63 non-empty subsets of the six candidate LLMs on the TUH-DEM data.

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

Target environment: Apple Silicon with the MLX backend. `uv sync` uses Python 3.13 (`.python-version`); the pinned mlx 0.29.0 provides wheels for Python 3.11-3.13 only.

### Environment used in the paper

- Hardware: Apple M4 Max MacBook Pro, 128 GB unified memory
- Software: Python 3.13, mlx 0.29.0, mlx-metal 0.29.0, mlx-lm 0.27.1 (pinned in `pyproject.toml` / `uv.lock`)
- Quantization: 8-bit affine, group size 64 (see [Models](#models))
- Decoding: greedy (temperature 0) via `mlx_lm.generate`, up to 512 new tokens per call

Other versions of mlx / mlx-lm (even mlx 0.29.2) can change the generated text and therefore the predictions.

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

The pipeline requires an MLX-compatible LLM checkpoint. Model weights are not redistributed; the pre-quantized checkpoints are downloaded on first use, and the other models must be converted locally (see [Models](#models)).

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

### Optional inference mode: chat template and JSON early stop

By default the pipeline reproduces the submitted generation path: the prompt is passed as a raw string (no chat template) and each call generates until EOS or 512 tokens. Two opt-in flags change this:

- `--chat-template` renders prompts with the model's own chat template (system turns are merged into the user turn when the template has no system role; models without a template, e.g. Preferred-MedLLM-Qwen-72B, fall back to the raw prompt).
- `--early-stop` ends generation as soon as a complete, parseable JSON value has been produced after any reasoning section (`</think>` or the GPT-OSS analysis channel).

With either flag, per-call runtime (seconds, prompt/generated tokens, finish reason `eos` / `early_stop` / `length`, attempt number) is written to `call_stats.csv` in the log directory. Self-correction retries are unchanged.

### Batch run (12 paper queries)

```bash
./scripts/run_end2end.sh tohoku            # Per-dataset LLM inference
./scripts/run_end2end.sh okayama
```

## Models

| Internal identifier | HuggingFace ID | MLX 8-bit checkpoint | Config | PhenoAgent ensemble |
| --- | --- | --- | --- | :---: |
| `GPT-OSS-20B`           | `openai/gpt-oss-20b`                                | `lmstudio-community/gpt-oss-20b-MLX-8bit` (pre-quantized)              | `config_gpt_oss_20b.py`                 | yes |
| `CALM3-22B`             | `cyberagent/calm3-22b-chat`                         | converted locally                                                      | `config_calm3_8bit.py`                  |   |
| `DeepSeek-32B`          | `cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese`  | converted locally                                                      | `config_ca_deepseek32B.py`              | yes |
| `DeepSeek-70B`          | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B`         | converted locally                                                      | `config_deepseek70B.py`                 |   |
| `Llama-3.3-Swallow-70B` | `tokyotech-llm/Llama-3.3-Swallow-70B-Instruct-v0.4` | `mlx-community/Llama-3.3-Swallow-70B-Instruct-v0.4-8bit` (pre-quantized) | `config_llama33_swallow_70b_instruct.py` | yes |
| `MedLLM-72B`            | `pfnet/Preferred-MedLLM-Qwen-72B`                   | converted locally                                                      | `config_preferred_qwen.py`              |   |

All models were run with 8-bit affine quantization (group size 64). The "converted locally" models were quantized with

```bash
uv run mlx_lm.convert --hf-path <HuggingFace ID> --mlx-path <dir> --quantize --q-bits 8
```

and `MODEL_PATH` in the corresponding config must point to `<dir>` to reproduce the paper's setting (the configs default to the HuggingFace ID, which loads unquantized weights).

## Clinical query set

Twelve queries on TUH-DEM (Tohoku University Hospital development cohort, 842 admissions; dataset key `tohoku`); seven evaluable queries (Q1, Q2, Q7, Q9-Q12) on OUH-DEM (Okayama University Hospital external evaluation cohort, 196 admissions; dataset key `okayama`). The Japanese names correspond to the pipeline's label columns (written without spaces, e.g. `心停止+ECMO`).

| Query | English | Japanese |
| --- | --- | --- |
| Q1  | Cardiac arrest + ECMO                     | 心停止 + ECMO                |
| Q2  | Cardiac arrest + Hypoxic encephalopathy   | 心停止 + 低酸素性脳症       |
| Q3  | Abdominal stab wound + Laparotomy         | 腹部刺創 + 開腹術           |
| Q4  | Hypothermia + ECMO                        | 低体温 + ECMO                |
| Q5  | Myocardial infarction + ECMO              | 心筋梗塞 + ECMO              |
| Q6  | Heart failure + NPPV                      | 心不全 + NPPV                |
| Q7  | Sepsis + AKI                              | 敗血症 + 急性腎障害         |
| Q8  | Sepsis + DIC                              | 敗血症 + 播種性血管内凝固   |
| Q9  | Acute drug poisoning + Activated charcoal | 急性薬物中毒 + 活性炭投与   |
| Q10 | Pelvic fracture + External fixation       | 骨盤骨折 + 創外固定         |
| Q11 | Pelvic fracture + IVR                     | 骨盤骨折 + IVR               |
| Q12 | Pelvic fracture + Hemorrhagic shock       | 骨盤骨折 + 出血性ショック   |

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
