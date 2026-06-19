"""Generate DUMMY synthetic data for code-execution testing only.

NOT real clinical data. Numbers in the paper cannot be reproduced from this.

This script generates:
1. Small-scale synthetic binary prediction matrices (master.csv) that mimic
   the schema of the real master.csv files for analysis-pipeline testing.
2. Synthetic 40_result_df.csv files that mimic the end-to-end pipeline's
   input format (patient ID + clinical text fields + label columns).

Usage:
    uv run python scripts/generate_dummy_data.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

# ============================================================
# Configuration
# ============================================================

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data_sample"

METHODS: list[str] = [
    "keyword_search",
    "semantic_search",
    "bm25",
    "GPT-OSS-20B",
    "CALM3-22B",
    "DeepSeek-32B",
    "DeepSeek-70B",
    "Llama-3.1-Swallow-70B",
    "MedLLM-72B",
    "ensembled",
]

QUERY_MAP: dict[int, str] = {
    1: "心停止+ECMO",
    2: "心停止+低酸素性脳症",
    3: "腹部刺創+開腹術",
    4: "低体温+ECMO",
    5: "心筋梗塞+ECMO",
    6: "心不全+NPPV",
    7: "敗血症+急性腎障害",
    8: "敗血症+播種性血管内凝固",
    9: "急性薬物中毒+活性炭投与",
    10: "骨盤骨折+創外固定",
    11: "骨盤骨折+IVR",
    12: "骨盤骨折+出血性ショック",
}

TOHOKU_QUERIES: list[int] = list(range(1, 13))
OKAYAMA_QUERIES: list[int] = [1, 2, 7, 9, 10, 11, 12]

TOHOKU_N_CASES = 30
OKAYAMA_N_CASES = 20

# Correlation between y_true and y_pred per method (higher = better method)
METHOD_CORRELATION: dict[str, float] = {
    "keyword_search": 0.50,
    "semantic_search": 0.55,
    "bm25": 0.55,
    "GPT-OSS-20B": 0.60,
    "CALM3-22B": 0.60,
    "DeepSeek-32B": 0.65,
    "DeepSeek-70B": 0.70,
    "Llama-3.1-Swallow-70B": 0.70,
    "MedLLM-72B": 0.72,
    "ensembled": 0.78,
}

MASTER_COLUMNS: list[str] = [
    "case_id",
    "query_id",
    "query_name",
    "site",
    "reference_type",
    "method",
    "y_true",
    "y_pred",
]

# ---------------------------------------------------------------------------
# Mock clinical text templates (obviously synthetic, deterministic)
# All text is fabricated and does NOT derive from any real patient record.
# ---------------------------------------------------------------------------

MOCK_CHIEF_COMPLAINTS: list[str] = [
    "【MOCK】胸痛および呼吸困難を主訴に搬送。",
    "【MOCK】腹部痛と嘔吐を主訴に来院。",
    "【MOCK】意識障害にて救急搬送。",
    "【MOCK】発熱、全身倦怠感を主訴に受診。",
    "【MOCK】外傷による骨盤部痛にて搬送。",
]

MOCK_PREADMISSION: list[str] = [
    "【MOCK】発症当日に救急要請、搬送中バイタル安定。合成データ。",
    "【MOCK】数日前より症状出現、徐々に増悪し来院。合成データ。",
    "【MOCK】目撃者により通報、救急隊到着時GCS 8。合成データ。",
    "【MOCK】前医にて初期治療後、精査目的に紹介搬送。合成データ。",
    "【MOCK】事故現場より直接搬送。受傷機転は転落。合成データ。",
]

MOCK_HOSPITAL_COURSE: list[str] = [
    "【MOCK】入院後、輸液管理および経過観察を実施。合成データ。",
    "【MOCK】集中治療室にて全身管理を継続。合成データ。",
    "【MOCK】手術施行し、術後経過良好。合成データ。",
    "【MOCK】保存的加療にて症状改善傾向。合成データ。",
    "【MOCK】人工呼吸器管理を要したが、離脱成功。合成データ。",
]

MOCK_DISCHARGE_FINDINGS: list[str] = [
    "【MOCK】全身状態改善し、独歩退院。合成データ。",
    "【MOCK】ADL自立にてリハビリ転院。合成データ。",
    "【MOCK】軽度後遺症残存も日常生活可能な状態で退院。合成データ。",
    "【MOCK】外来フォローアップを予定し退院。合成データ。",
    "【MOCK】原疾患は寛解し、退院後通院継続。合成データ。",
]


# ============================================================
# Master CSV generation logic (analysis pipeline)
# ============================================================


def generate_correlated_predictions(
    y_true: np.ndarray,
    correlation: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate binary predictions correlated with y_true.

    Args:
        y_true: Ground truth binary labels.
        correlation: Probability of matching y_true (agreement rate).
        rng: NumPy random generator.

    Returns:
        Binary prediction array of the same shape.
    """
    n = len(y_true)
    agree = rng.random(n) < correlation
    y_pred = np.where(agree, y_true, 1 - y_true)
    return y_pred.astype(int)


def generate_dataset(
    site: str,
    n_cases: int,
    queries: list[int],
    rng: np.random.Generator,
) -> tuple[list[list], dict[int, np.ndarray]]:
    """Generate a full dummy dataset for one site.

    Args:
        site: Site name (tohoku or okayama).
        n_cases: Number of cases to generate.
        queries: List of query IDs.
        rng: NumPy random generator.

    Returns:
        Tuple of (master_rows, ground_truth_map) where ground_truth_map
        maps query_id -> y_true array for building 40_result_df.
    """
    rows: list[list] = []
    ground_truth: dict[int, np.ndarray] = {}

    for query_id in queries:
        query_name = QUERY_MAP[query_id]
        positive_rate = rng.uniform(0.05, 0.20)
        y_true = (rng.random(n_cases) < positive_rate).astype(int)

        if y_true.sum() == 0:
            y_true[rng.integers(0, n_cases)] = 1
        if y_true.sum() == n_cases:
            y_true[rng.integers(0, n_cases)] = 0

        ground_truth[query_id] = y_true

        for method in METHODS:
            correlation = METHOD_CORRELATION[method]
            y_pred = generate_correlated_predictions(y_true, correlation, rng)

            for case_idx in range(n_cases):
                case_id: int | str
                if site == "okayama":
                    case_id = f"okayama_{case_idx + 1:03d}"
                else:
                    case_id = case_idx + 1

                rows.append(
                    [
                        case_id,
                        query_id,
                        query_name,
                        site,
                        "doctor",
                        method,
                        int(y_true[case_idx]),
                        int(y_pred[case_idx]),
                    ]
                )

    return rows, ground_truth


def write_csv(filepath: Path, header: list[str], rows: list[list]) -> None:
    """Write rows to CSV file with header.

    Args:
        filepath: Output CSV path.
        header: Column names.
        rows: Data rows (without header).
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


# ============================================================
# Synthetic 40_result_df.csv generation (end-to-end pipeline)
# ============================================================


def generate_end2end_input(
    site: str,
    n_cases: int,
    queries: list[int],
    ground_truth: dict[int, np.ndarray],
    rng: np.random.Generator,
) -> None:
    """Generate a synthetic 40_result_df.csv for end2end.py.

    Creates a CSV with the patient_id column, text fields, and label
    columns expected by the end-to-end pipeline. All text is obviously
    MOCK clinical text.

    Args:
        site: Site name (tohoku or okayama).
        n_cases: Number of cases.
        queries: List of query IDs present in this dataset.
        ground_truth: Map of query_id -> y_true array from master generation.
        rng: NumPy random generator.
    """
    if site == "tohoku":
        id_col = "文書番号"
        text_fields = ["主訴", "入院までの経過", "入院中経過", "退院時所見"]
    else:
        id_col = "カルテNo"
        text_fields = ["主訴", "入院までの経過", "入院中経過"]

    label_columns = [QUERY_MAP[q] for q in queries]
    header = [id_col] + text_fields + label_columns

    rows: list[list] = []
    for case_idx in range(n_cases):
        if site == "okayama":
            case_id = f"okayama_{case_idx + 1:03d}"
        else:
            case_id = case_idx + 1

        # Deterministic mock text selection based on case index
        chief = MOCK_CHIEF_COMPLAINTS[case_idx % len(MOCK_CHIEF_COMPLAINTS)]
        preadm = MOCK_PREADMISSION[case_idx % len(MOCK_PREADMISSION)]
        course = MOCK_HOSPITAL_COURSE[case_idx % len(MOCK_HOSPITAL_COURSE)]

        text_values = [chief, preadm, course]
        if site == "tohoku":
            discharge = MOCK_DISCHARGE_FINDINGS[case_idx % len(MOCK_DISCHARGE_FINDINGS)]
            text_values.append(discharge)

        # Label values from ground truth
        labels = [int(ground_truth[q][case_idx]) for q in queries]

        rows.append([case_id] + text_values + labels)

    out_path = OUTPUT_DIR / site / "label_data" / "40_result_df.csv"
    write_csv(out_path, header, rows)
    print(f"  End2end input: {n_cases} rows -> {out_path}")


# ============================================================
# Main
# ============================================================


def main() -> None:
    """Generate dummy data for both sites."""
    rng = np.random.default_rng(SEED)

    # ------------------------------------------------------------------
    # Tohoku
    # ------------------------------------------------------------------
    tohoku_rows, tohoku_gt = generate_dataset(
        "tohoku", TOHOKU_N_CASES, TOHOKU_QUERIES, rng
    )
    tohoku_master = OUTPUT_DIR / "tohoku_vs_doctors" / "master.csv"
    write_csv(tohoku_master, MASTER_COLUMNS, tohoku_rows)
    print(f"  Tohoku master: {len(tohoku_rows)} rows -> {tohoku_master}")

    generate_end2end_input("tohoku", TOHOKU_N_CASES, TOHOKU_QUERIES, tohoku_gt, rng)

    # ------------------------------------------------------------------
    # Okayama
    # ------------------------------------------------------------------
    okayama_rows, okayama_gt = generate_dataset(
        "okayama", OKAYAMA_N_CASES, OKAYAMA_QUERIES, rng
    )
    okayama_master = OUTPUT_DIR / "okayama_vs_doctor" / "master.csv"
    write_csv(okayama_master, MASTER_COLUMNS, okayama_rows)
    print(f"  Okayama master: {len(okayama_rows)} rows -> {okayama_master}")

    generate_end2end_input("okayama", OKAYAMA_N_CASES, OKAYAMA_QUERIES, okayama_gt, rng)

    # Summary
    print("\nDone. Wrote SYNTHETIC dummy data for testing only.")
    print("  master.csv files: analysis pipeline schema")
    print("  40_result_df.csv files: end-to-end pipeline input schema")


if __name__ == "__main__":
    main()
