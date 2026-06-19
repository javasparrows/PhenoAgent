# data_sample/

> **WARNING**: The CSV files in this directory are **SYNTHETIC DUMMY DATA**
> generated for code-execution verification only. They do NOT correspond to
> real patient data and the numerical results reported in the paper CANNOT be
> reproduced from them. Real clinical data cannot be shared publicly due to
> patient privacy requirements (see paper Data Availability statement).

## Regenerating dummy data

```bash
uv run python scripts/generate_dummy_data.py
```

This produces:

### Master CSV files (analysis pipeline schema)
- `tohoku_vs_doctors/master.csv` -- 30 synthetic cases x 12 queries x 10 methods = 3,600 rows
- `okayama_vs_doctor/master.csv` -- 20 synthetic cases x 7 queries x 10 methods = 1,400 rows

### End-to-end pipeline input files
- `tohoku/label_data/40_result_df.csv` -- 30 synthetic cases with mock clinical text + 12 label columns
- `okayama/label_data/40_result_df.csv` -- 20 synthetic cases with mock clinical text + 7 label columns

## Schema

### `master.csv` (long format)

Columns:
- `case_id` -- synthetic record identifier
- `query_id` -- clinical query number (1--12)
- `query_name` -- Japanese query label (e.g., "心停止+ECMO")
- `site` -- dataset site (`tohoku` or `okayama`)
- `reference_type` -- `doctor` (simulated two-physician AND consensus)
- `method` -- method identifier (paper-canonical names)
- `y_true` -- reference standard label (0 or 1)
- `y_pred` -- method prediction (0 or 1)

### `40_result_df.csv` (wide format)

Columns:
- Patient ID (`文書番号` for Tohoku, `カルテNo` for Okayama)
- Clinical text fields (`主訴`, `入院までの経過`, `入院中経過`, `退院時所見` for Tohoku)
- Binary label columns for each clinical query (e.g., `心停止+ECMO`)

All clinical text is obviously MOCK (`【MOCK】` prefix) and does not derive
from any real patient record.

## Purpose

These dummy files allow verification that the PhenoAgent end-to-end pipeline
runs without data-loading errors. They are NOT suitable for reproducing any
numerical result in the paper.
