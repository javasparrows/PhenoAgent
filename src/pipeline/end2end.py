import argparse
import importlib
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .utils import (
    extract_json_from_response,
    load_model_and_tokenizer,
    remove_before_think,
    run_inference,
)

# ---------------------------------------------------------------------------
# ANSI color definitions
# ---------------------------------------------------------------------------
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_RED = "\033[91m"
COLOR_RESET = "\033[0m"
COLOR_ORANGE = "\033[38;5;208m"

# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
DATASET_CONFIG = {
    "tohoku": {
        "patient_id_col": "文書番号",
        "text_fields": ["主訴", "入院までの経過", "入院中経過", "退院時所見"],
        "default_data_dir": "data/tohoku",
    },
    "okayama": {
        "patient_id_col": "カルテNo",
        "text_fields": ["主訴", "入院までの経過", "入院中経過"],
        "default_data_dir": "data/okayama",
    },
}

# ---------------------------------------------------------------------------
# Label pairs (unified standard: numbered prefix)
# ---------------------------------------------------------------------------
EXAMPLE_QUERIES = [
    {
        "no": 1,
        "query": "心停止の患者さんの中で、ECMOが使われた患者は何人？",
        "label_column": "心停止+ECMO",
    },
    {
        "no": 2,
        "query": "心停止の患者さんの中で、低酸素性脳症が使われた患者は何人？",
        "label_column": "心停止+低酸素性脳症",
    },
    {
        "no": 3,
        "query": "腹部刺創の患者さんの中で、開腹術が使われた患者は何人？",
        "label_column": "腹部刺創+開腹術",
    },
    {
        "no": 4,
        "query": "低体温の患者さんの中で、ECMOが使われた患者は何人？",
        "label_column": "低体温+ECMO",
    },
    {
        "no": 5,
        "query": "心筋梗塞の患者さんの中で、ECMOが使われた患者は何人？",
        "label_column": "心筋梗塞+ECMO",
    },
    {
        "no": 6,
        "query": "心不全の患者さんの中で、NPPVが使われた患者は何人？",
        "label_column": "心不全+NPPV",
    },
    {
        "no": 7,
        "query": "敗血症の患者さんの中で、急性腎障害が使われた患者は何人？",
        "label_column": "敗血症+急性腎障害",
    },
    {
        "no": 8,
        "query": "敗血症の患者さんの中で、播種性血管内凝固が使われた患者は何人？",
        "label_column": "敗血症+播種性血管内凝固",
    },
    {
        "no": 9,
        "query": "急性薬物中毒の患者さんの中で、活性炭投与が使われた患者は何人？",
        "label_column": "急性薬物中毒+活性炭投与",
    },
    {
        "no": 10,
        "query": "骨盤骨折の患者さんの中で、創外固定が使われた患者は何人？",
        "label_column": "骨盤骨折+創外固定",
    },
    {
        "no": 11,
        "query": "骨盤骨折の患者さんの中で、IVRが使われた患者は何人？",
        "label_column": "骨盤骨折+IVR",
    },
    {
        "no": 12,
        "query": "骨盤骨折の患者さんの中で、出血性ショックが使われた患者は何人？",
        "label_column": "骨盤骨折+出血性ショック",
    },
]

# ---------------------------------------------------------------------------
# Pydantic models for structured output
# ---------------------------------------------------------------------------


class MedicalFieldResult(BaseModel):
    value: bool = Field(description="該当するかどうかの真偽値")
    reason: str = Field(description="その判断の詳細な理由")


class MedicalAnalysisResult(BaseModel):
    """Pydantic model for medical data analysis results."""

    model_config = ConfigDict(extra="allow")


# ---------------------------------------------------------------------------
# Global variables (populated from config / CLI args)
# ---------------------------------------------------------------------------
QUERY: str = ""
LABEL_COLUMN: str | None = None
DATASET: str = ""
DATA_DIR: Path = Path()
LOG_DIR: str = ""
LOG_FILENAME: str = ""
MODEL_PATH: str = ""
build_first_prompt = None
build_second_prompt = None


# ---------------------------------------------------------------------------
# Configuration & logging helpers
# ---------------------------------------------------------------------------


def load_config(config_path: str) -> None:
    """Load a configuration file dynamically."""
    global MODEL_PATH, LOG_DIR, build_first_prompt, build_second_prompt

    # Strip .py extension if present
    if config_path.endswith(".py"):
        config_path = config_path[:-3]

    # Convert path separators to module path dots
    config_module_name = config_path.replace("/", ".")

    try:
        config_module = importlib.import_module(config_module_name)
    except ImportError as e:
        raise ImportError(
            f"Failed to load configuration file {config_path}: {e}"
        ) from e

    MODEL_PATH = config_module.MODEL_PATH
    log_dir_suffix = config_module.LOG_DIR_SUFFIX

    # Build LOG_DIR under data/{dataset}/logs/
    log_base = DATA_DIR / "logs" / log_dir_suffix
    log_base.mkdir(parents=True, exist_ok=True)

    datetime_str = datetime.now().strftime("%Y%m%d%H%M%S")
    label_slug = (
        LABEL_COLUMN if LABEL_COLUMN else QUERY.replace("/", "_").replace(" ", "_")
    )
    LOG_DIR = str(log_base / f"log_{datetime_str}_{label_slug}")

    # Dynamically import prompt builder functions
    first_prompt_module = importlib.import_module(config_module.PROMPT_MODULE_FIRST)
    second_prompt_module = importlib.import_module(config_module.PROMPT_MODULE_SECOND)

    build_first_prompt = first_prompt_module.build_first_prompt
    build_second_prompt = second_prompt_module.build_second_prompt


def setup_logging() -> None:
    """Set up file-based logging."""
    global LOG_FILENAME, logger

    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
    LOG_FILENAME = str(Path(LOG_DIR) / "log.txt")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(LOG_FILENAME, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)


# ---------------------------------------------------------------------------
# Data verification (from Okayama version)
# ---------------------------------------------------------------------------


def verify_data_consistency(df_input: pd.DataFrame, df_label: pd.DataFrame) -> dict:
    """Verify correspondence between input data and label data.

    Args:
        df_input: Input data (e.g. label_data/40_result_df.csv).
        df_label: Label data (e.g. doctors_answers/integrated_all.csv).

    Returns:
        dict with keys:
            - 'is_consistent': bool
            - 'input_shape': tuple
            - 'label_shape': tuple
            - 'common_key_column': str | None
            - 'matched_count': int
            - 'mismatched_rows': list
            - 'warnings': list[str]
    """
    result: dict = {
        "is_consistent": True,
        "input_shape": df_input.shape,
        "label_shape": df_label.shape,
        "common_key_column": None,
        "matched_count": 0,
        "mismatched_rows": [],
        "warnings": [],
    }

    if df_input.shape[0] != df_label.shape[0]:
        result["warnings"].append(
            f"Warning: input data has {df_input.shape[0]} rows "
            f"but label data has {df_label.shape[0]} rows"
        )

    # Detect common key columns
    common_columns: list[str] = []
    for col_name in ("カルテNo", "Document_ID", "患者No", "文書番号"):
        if col_name in df_input.columns and col_name in df_label.columns:
            common_columns.append(col_name)

    # Use the dataset-specific patient_id_col first, then fallback
    patient_id_col = DATASET_CONFIG.get(DATASET, {}).get("patient_id_col")

    # Determine key column priority
    key_col: str | None = None
    if patient_id_col and patient_id_col in common_columns:
        key_col = patient_id_col
    elif "患者No" in common_columns:
        key_col = "患者No"
    elif "カルテNo" in common_columns:
        key_col = "カルテNo"
    elif common_columns:
        key_col = common_columns[0]

    if key_col is not None:
        result["common_key_column"] = key_col
        matched_count = 0
        mismatched_indices: list[dict] = []

        for idx in range(min(len(df_input), len(df_label))):
            input_val = df_input.iloc[idx].get(key_col, None)
            label_val = df_label.iloc[idx].get(key_col, None)

            if input_val == label_val:
                matched_count += 1
            else:
                mismatched_indices.append(
                    {
                        "index": idx,
                        f"input_{key_col}": input_val,
                        f"label_{key_col}": label_val,
                    }
                )

        result["matched_count"] = matched_count
        result["mismatched_rows"] = mismatched_indices

        if mismatched_indices:
            result["is_consistent"] = False
            result["warnings"].append(
                f"Error: {len(mismatched_indices)} rows have mismatched "
                f"{key_col} values"
            )
    else:
        result["is_consistent"] = False
        result["warnings"].append(
            "Error: no common key column found between input and label data"
        )

    return result


# ---------------------------------------------------------------------------
# Text processing helpers
# ---------------------------------------------------------------------------


def extract_text_before_endoftext(text: str) -> str:
    """Extract text before the <|endoftext|> token."""
    if "<|endoftext|>" not in text:
        return text
    pattern = r"(.*?)<\|endoftext\|>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else text


def save_response_to_file(response: str, output_file: str, model_path: str) -> None:
    """Write a model response to a file inside LOG_DIR."""
    out_path = Path(LOG_DIR) / output_file
    out_path.write_text(f"{model_path}\n\n{response}", encoding="utf-8")


def make_document_text(df_one_document: pd.Series, text_fields: list[str]) -> str:
    """Build a formatted document text from a DataFrame row.

    Uses the dataset-specific text_fields list. Skips fields whose
    values are NaN.
    """
    sections: list[str] = []
    for field in text_fields:
        if field in df_one_document.index and pd.notna(df_one_document[field]):
            sections.append(f"# {field}\n{df_one_document[field]}")
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Structured output generation with retry
# ---------------------------------------------------------------------------


def generate_structured_output_with_retry(
    model,
    tokenizer,
    chat_prompt: list | str,
    max_retries: int = 3,
    temp: float = 0.0,
) -> tuple:
    """Generate structured output via Pydantic validation, retrying on failure.

    Modifies the prompt dynamically on retries to encourage correct JSON
    formatting even when temperature is 0.0.

    Args:
        model: MLX model.
        tokenizer: MLX tokenizer.
        chat_prompt: Chat-style prompt (list of dicts or plain string).
        max_retries: Maximum number of attempts (default 3).
        temp: Sampling temperature (default 0.0).

    Returns:
        tuple: (parsed_result, retry_count, success, dynamic_prompt_resolved)
    """
    # Self-correction retry prompts — matches Supplementary Box 3 verbatim.
    retry_prompt_additions = [
        "",  # First attempt uses the original prompt as-is
        "\n\n再確認：必ずJSON形式の出力で、正確に2つの診断項目のみを含めてください。",
        (
            "\n\n再度確認：以下の形式を厳密に守ってください：\n"
            '{"項目1": {"value": true/false, "reason": "詳細な理由"}, '
            '"項目2": {"value": true/false, "reason": "詳細な理由"}}'
        ),
    ]

    if isinstance(chat_prompt, str):
        original_chat_prompt = chat_prompt
        is_string_prompt = True
    else:
        original_chat_prompt = chat_prompt.copy()
        is_string_prompt = False

    dynamic_prompt_resolved = False
    for retry_count in range(max_retries):
        try:
            # Modify prompt dynamically on retries
            if retry_count > 0 and retry_count < len(retry_prompt_additions):
                if is_string_prompt:
                    current_chat_prompt = (
                        original_chat_prompt + retry_prompt_additions[retry_count]
                    )
                else:
                    current_chat_prompt = original_chat_prompt.copy()
                    current_chat_prompt[-1]["content"] += retry_prompt_additions[
                        retry_count
                    ]
                print(
                    f"{COLOR_ORANGE}Modifying prompt for attempt "
                    f"{retry_count + 1}/{max_retries}...{COLOR_RESET}"
                )
            else:
                current_chat_prompt = chat_prompt

            # Run inference
            response = run_inference(
                model, tokenizer, current_chat_prompt, temp=temp, top_p=1.0
            )
            response = extract_text_before_endoftext(response)
            response = remove_before_think(response)

            # Parse JSON
            parsed_json = extract_json_from_response(response)

            if "error" in parsed_json:
                if retry_count < max_retries - 1:
                    print(
                        f"{COLOR_ORANGE}JSON parse attempt "
                        f"{retry_count + 1}/{max_retries} failed, "
                        f"retrying...{COLOR_RESET}"
                    )
                    continue
                else:
                    print(
                        f"{COLOR_RED}JSON parse failed after "
                        f"{max_retries} attempts{COLOR_RESET}"
                    )
                    return parsed_json, retry_count + 1, False, dynamic_prompt_resolved

            # Pydantic validation
            try:
                validated_fields: dict = {}
                for key, value in parsed_json.items():
                    if (
                        isinstance(value, dict)
                        and "value" in value
                        and "reason" in value
                    ):
                        field_result = MedicalFieldResult(**value)
                        validated_fields[key] = field_result.model_dump()
                    elif isinstance(value, bool):
                        field_result = MedicalFieldResult(
                            value=value, reason="推論結果"
                        )
                        validated_fields[key] = field_result.model_dump()
                    else:
                        validated_fields[key] = value

                validated_result = MedicalAnalysisResult(**validated_fields)

                # Verify at least 1 boolean field (N-field generalization)
                bool_count = sum(
                    1
                    for v in validated_fields.values()
                    if isinstance(v, dict) and "value" in v
                )

                if bool_count < 1:
                    if retry_count < max_retries - 1:
                        print(
                            f"{COLOR_ORANGE}Expected at least 1 boolean "
                            f"field, got {bool_count} on attempt "
                            f"{retry_count + 1}/{max_retries}, "
                            f"retrying...{COLOR_RESET}"
                        )
                        continue
                    else:
                        print(
                            f"{COLOR_RED}Expected at least 1 boolean "
                            f"field, got {bool_count} after "
                            f"{max_retries} attempts{COLOR_RESET}"
                        )
                        return (
                            parsed_json,
                            retry_count + 1,
                            False,
                            dynamic_prompt_resolved,
                        )

                if retry_count > 0:
                    dynamic_prompt_resolved = True
                    print(
                        f"{COLOR_GREEN}JSON parse succeeded on attempt "
                        f"{retry_count + 1}/{max_retries} with dynamic "
                        f"prompt modification{COLOR_RESET}"
                    )

                return (
                    validated_result.model_dump(),
                    retry_count + 1,
                    True,
                    dynamic_prompt_resolved,
                )

            except Exception as e:
                if retry_count < max_retries - 1:
                    print(
                        f"{COLOR_ORANGE}Pydantic validation attempt "
                        f"{retry_count + 1}/{max_retries} failed: "
                        f"{e!s}, retrying...{COLOR_RESET}"
                    )
                    continue
                else:
                    print(
                        f"{COLOR_RED}Pydantic validation failed after "
                        f"{max_retries} attempts: {e!s}{COLOR_RESET}"
                    )
                    return parsed_json, retry_count + 1, False, dynamic_prompt_resolved

        except Exception as e:
            if retry_count < max_retries - 1:
                print(
                    f"{COLOR_ORANGE}Inference attempt "
                    f"{retry_count + 1}/{max_retries} failed: "
                    f"{e!s}, retrying...{COLOR_RESET}"
                )
                continue
            else:
                print(
                    f"{COLOR_RED}Inference failed after "
                    f"{max_retries} attempts: {e!s}{COLOR_RESET}"
                )
                return (
                    {"error": str(e), "raw_response": ""},
                    retry_count + 1,
                    False,
                    dynamic_prompt_resolved,
                )

    return (
        {"error": "Max retries exceeded", "raw_response": ""},
        max_retries,
        False,
        dynamic_prompt_resolved,
    )


# ---------------------------------------------------------------------------
# Label column helpers
# ---------------------------------------------------------------------------


def find_label_column(df: pd.DataFrame, label_column: str | None) -> str | None:
    """Find the ground-truth label column in the DataFrame.

    Matches both numbered-prefix format (e.g. "1.心停止+ECMO") and plain
    format (e.g. "心停止+ECMO"). Returns None if label_column is not given
    or no match is found.
    """
    if not label_column:
        return None
    for col in df.columns:
        if col.endswith(label_column):
            return col
    if label_column in df.columns:
        return label_column
    return None


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def check_all_true(bool_list: list[bool]) -> bool:
    """Return True if all elements are True, otherwise False."""
    return all(bool_list)


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------


def main(filter_label_true: bool = False) -> None:
    """Main processing loop.

    Loads model, runs first-stage inference, reads CSV data, and runs
    second-stage inference for each document row. Results are saved
    incrementally.

    Args:
        filter_label_true: When True, only process rows whose ground-truth
            label is True.
    """
    total_start_time = time.time()

    dataset_cfg = DATASET_CONFIG[DATASET]
    patient_id_col = dataset_cfg["patient_id_col"]
    text_fields = dataset_cfg["text_fields"]

    logger.info("===== START MAIN =====")
    logger.info(f"Using model: {MODEL_PATH}")
    logger.info(f"Dataset: {DATASET}")
    print(f"{COLOR_BLUE}--- Using model: {MODEL_PATH} ---{COLOR_RESET}")
    print(f"{COLOR_BLUE}--- Dataset: {DATASET} ---{COLOR_RESET}")
    print()

    # -----------------------------------------------------------------
    # 1) Load model and tokenizer
    # -----------------------------------------------------------------
    step1_start = time.time()
    text_1_1 = "[STEP1] Loading model and tokenizer ..."
    logger.info(text_1_1)
    print(f"{COLOR_BLUE}--- {text_1_1} ---{COLOR_RESET}")
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    step1_end = time.time()
    text_1_2 = f"[STEP1] Finished in {step1_end - step1_start:.3f} seconds\n"
    logger.info(text_1_2)
    print(f"{COLOR_BLUE}--- {text_1_2} ---{COLOR_RESET}")

    # -----------------------------------------------------------------
    # 2) Build first prompt and run inference
    # -----------------------------------------------------------------
    step2_start = time.time()
    text_2_1 = "[STEP2] Building first prompt and running inference ..."
    logger.info(text_2_1)
    print(f"{COLOR_BLUE}--- {text_2_1} ---{COLOR_RESET}")
    chat_first = build_first_prompt(QUERY)
    response1 = run_inference(model, tokenizer, chat_first, temp=0.0, top_p=1.0)
    response1 = extract_text_before_endoftext(response1)
    response1 = remove_before_think(response1)
    text_2_2 = f"[STEP2] First inference result:\n{response1}"
    logger.info(text_2_2)
    print(f"{COLOR_BLUE}--- {text_2_2} ---{COLOR_RESET}")
    step2_end = time.time()
    text_2_3 = f"[STEP2] Finished in {step2_end - step2_start:.3f} seconds\n"
    logger.info(text_2_3)
    print(f"{COLOR_BLUE}--- {text_2_3} ---{COLOR_RESET}")

    # -----------------------------------------------------------------
    # 3) Save first inference response to file
    # -----------------------------------------------------------------
    step3_start = time.time()
    text_3_1 = "[STEP3] Saving first inference result to file ..."
    logger.info(text_3_1)
    print(f"{COLOR_BLUE}--- {text_3_1} ---{COLOR_RESET}")
    save_response_to_file(response1, "response_v01.txt", MODEL_PATH)
    step3_end = time.time()
    text_3_2 = f"[STEP3] Finished in {step3_end - step3_start:.3f} seconds\n"
    logger.info(text_3_2)
    print(f"{COLOR_BLUE}--- {text_3_2} ---{COLOR_RESET}")

    # -----------------------------------------------------------------
    # 4) Read CSV and loop over rows for second-stage inference
    # -----------------------------------------------------------------
    step4_start = time.time()
    logger.info("[STEP4] Reading CSV and looping over rows ...")

    # Try multiple CSV path candidates (Tohoku fallback support)
    csv_candidates = [
        DATA_DIR / "label_data" / "40_result_df.csv",
        DATA_DIR / "40_result_df.csv",
    ]
    df = None
    csv_used: Path | None = None
    for csv_path in csv_candidates:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            csv_used = csv_path
            break

    if df is None:
        raise FileNotFoundError(
            "CSV data file not found. Checked: "
            + ", ".join(str(p) for p in csv_candidates)
        )

    logger.info(f"Loaded CSV from: {csv_used}")
    print(f"{COLOR_BLUE}--- Loaded CSV from: {csv_used} ---{COLOR_RESET}")

    # Data consistency verification (Okayama-style)
    input_data_paths = [
        DATA_DIR / "label_data" / "40_result_df.csv",
    ]
    for input_path in input_data_paths:
        if input_path.exists():
            df_input = pd.read_csv(input_path)
            consistency_result = verify_data_consistency(df_input, df)
            text_consistency = "[DATA VERIFICATION] Data consistency check"
            logger.info(text_consistency)
            print(f"{COLOR_BLUE}--- {text_consistency} ---{COLOR_RESET}")
            print(f"  Input data shape: {consistency_result['input_shape']}")
            print(f"  Label data shape: {consistency_result['label_shape']}")
            print(f"  Common key column: {consistency_result['common_key_column']}")
            print(
                f"  Matched count: {consistency_result['matched_count']}/"
                f"{consistency_result['label_shape'][0]}"
            )

            for warning in consistency_result["warnings"]:
                print(f"  {warning}")
                logger.info(f"  {warning}")

            if not consistency_result["is_consistent"]:
                print(
                    f"{COLOR_RED}Warning: data correspondence may be "
                    f"incorrect.{COLOR_RESET}"
                )
                logger.warning("Warning: data correspondence may be incorrect.")
            else:
                print(
                    f"{COLOR_GREEN}Confirmed: data correspondence is "
                    f"correct.{COLOR_RESET}"
                )
                logger.info("Confirmed: data correspondence is correct.")
            break

    # Find label column (flexible matching for numbered prefix format)
    actual_label_column = find_label_column(df, LABEL_COLUMN)

    # Filtering: only process rows with True ground-truth labels
    if filter_label_true:
        if actual_label_column:
            original_len = len(df)
            true_count_total = df[actual_label_column].astype(bool).sum()
            text_true_count = (
                f"{COLOR_ORANGE}Total True label count: "
                f"{true_count_total}/{original_len}{COLOR_RESET}"
            )
            print(text_true_count)
            logger.info(f"Total True label count: {true_count_total}/{original_len}")

            df = df[df[actual_label_column].astype(bool)].reset_index(drop=True)
            filtered_len = len(df)
            text_filter = (
                f"{COLOR_YELLOW}Filtering result: {original_len} -> "
                f"{filtered_len} rows (True labels only){COLOR_RESET}"
            )
            print(text_filter)
            logger.info(text_filter)
            if filtered_len == 0:
                print(
                    f"{COLOR_RED}No rows with True labels found. Exiting.{COLOR_RESET}"
                )
                return
        else:
            print(
                f"{COLOR_RED}Warning: label column "
                f"'{LABEL_COLUMN}' not found. "
                f"Running on all data.{COLOR_RESET}"
            )

    print()
    text_4_1 = f"{COLOR_YELLOW}Row count:{COLOR_RESET} {len(df)}"
    print(text_4_1)
    logger.info(text_4_1)

    # Display patient/document count using dataset-specific ID column
    if patient_id_col in df.columns:
        id_nunique = df[patient_id_col].nunique()
        text_4_2 = f"{COLOR_YELLOW}{patient_id_col} count:{COLOR_RESET} {id_nunique}"
    elif "カルテNo" in df.columns:
        text_4_2 = (
            f"{COLOR_YELLOW}カルテNo count:{COLOR_RESET} {df['カルテNo'].nunique()}"
        )
    else:
        text_4_2 = f"{COLOR_YELLOW}Unique rows:{COLOR_RESET} {len(df)}"
    print(text_4_2)
    logger.info(text_4_2)
    print()

    result: list[tuple] = []
    make_bool_false_list: list[list] = []
    elapsed_time_list: list[float] = []

    # Determine the ID column to use for per-row identification
    row_id_col = (
        patient_id_col
        if patient_id_col in df.columns
        else ("カルテNo" if "カルテNo" in df.columns else None)
    )

    for idx in range(len(df)):
        iter_start_time = time.time()

        df_one_document = df.iloc[idx]

        # Get row identifier
        row_id = df_one_document[row_id_col] if row_id_col else idx

        # Get ground-truth label
        label = (
            df_one_document.get(actual_label_column, False)
            if actual_label_column
            else False
        )

        str_one_document = make_document_text(df_one_document, text_fields)

        chat_second = build_second_prompt(
            structured_question=response1, document_text=str_one_document
        )

        # Structured output with retry (max 3 attempts)
        (
            parsed_json,
            retry_count,
            json_parse_success,
            dynamic_prompt_resolved,
        ) = generate_structured_output_with_retry(
            model, tokenizer, chat_second, max_retries=3, temp=0.0
        )

        # Build response string for logging
        if json_parse_success:
            response2 = json.dumps(parsed_json, ensure_ascii=False, indent=2)
        else:
            response2 = (
                f"JSON parsing failed after {retry_count} attempts: {parsed_json}"
            )

        try:
            bool_list: list[bool] = []
            for key in parsed_json.keys():
                val = parsed_json[key]
                if isinstance(val, dict) and ("value" in val):
                    bool_list.append(val["value"])
                elif isinstance(val, bool):
                    bool_list.append(val)
                else:
                    bool_list.append(False)

            result_all_true = check_all_true(bool_list)
        except Exception:
            make_bool_false_list.append([idx, row_id])
            print(f"Error parsing JSON: {parsed_json}")
            continue

        print(
            f"{COLOR_BLUE}--- [Iteration {idx}/{len(df)}] "
            f"{row_id_col or 'idx'}: {row_id} processed ---{COLOR_RESET}"
        )
        parse_success_msg = (
            f"{COLOR_GREEN}SUCCESS{COLOR_RESET}"
            if json_parse_success
            else f"{COLOR_RED}FAILED{COLOR_RESET}"
        )
        retry_msg = (
            f" (attempts: {COLOR_ORANGE}{retry_count}{COLOR_RESET})"
            if retry_count > 1
            else ""
        )
        print(
            f"{COLOR_YELLOW}json_parse_success:{COLOR_RESET} "
            f"{parse_success_msg}{retry_msg}"
        )
        result_str = "正解" if result_all_true == label else "不正解"
        output_str = (
            f"{COLOR_YELLOW}check_all_true:{COLOR_RESET} {bool_list} -> "
            f"{result_all_true} | {COLOR_YELLOW}Ground truth:{COLOR_RESET} "
            f"{bool(label)} {QUERY} {MODEL_PATH}"
        )
        output_str += (
            f" | {COLOR_YELLOW}Result:{COLOR_RESET} "
            f"{COLOR_RED if result_all_true != label else COLOR_GREEN}"
            f"{result_str}{COLOR_RESET}"
        )
        print(output_str)

        logger.info(f"[Iteration {idx + 1}/{len(df)}] response2: {response2}")
        logger.info(f"[Iteration {idx + 1}/{len(df)}] parsed_json: {parsed_json}")
        logger.info(
            f"[Iteration {idx + 1}/{len(df)}] JSON parse success => "
            f"{json_parse_success} (attempts: {retry_count})"
        )
        logger.info(f"[Iteration {idx + 1}/{len(df)}] check_all_true => {output_str}")

        iter_end_time = time.time()
        elapsed_time = iter_end_time - iter_start_time
        elapsed_time_list.append(elapsed_time)
        avg_time = sum(elapsed_time_list) / len(elapsed_time_list)
        logger.info(
            f"[Iteration {idx + 1}/{len(df)}] took {elapsed_time:.2f} sec, "
            f"average {avg_time:.2f} sec\n"
        )
        print(
            f"{COLOR_BLUE}Took {elapsed_time:.2f} sec, "
            f"average {avg_time:.2f} sec{COLOR_RESET}"
        )

        # Store result (N-field: serialize bool_list as JSON string)
        result.append(
            (
                row_id,
                response2,
                parsed_json,
                json.dumps(bool_list),
                result_all_true,
                label,
                (result_all_true == label),
                json_parse_success,
                retry_count,
                dynamic_prompt_resolved,
            )
        )

        # Save result_df after every iteration
        result_df = pd.DataFrame(
            result,
            columns=[
                patient_id_col,
                "response2",
                "parsed_json",
                "bool_list_json",
                "result_all_true",
                "label",
                "result_all_true_eq_label",
                "json_parse_success",
                "retry_count",
                "dynamic_prompt_resolved",
            ],
        )
        result_df.to_csv(str(Path(LOG_DIR) / "result_df.csv"), index=False)

    step4_end = time.time()
    logger.info(f"[STEP4] All loop done in {step4_end - step4_start:.3f} seconds\n")

    total_end_time = time.time()
    logger.info(
        f"=== TOTAL processing time: "
        f"{total_end_time - total_start_time:.3f} seconds ==="
    )
    logger.info("===== END MAIN =====\n")

    make_bool_false_df = pd.DataFrame(
        make_bool_false_list, columns=["idx", patient_id_col]
    )
    make_bool_false_df.to_csv(
        str(Path(LOG_DIR) / "make_bool_false_df.csv"), index=False
    )

    print(
        f"{COLOR_GREEN}=== Total processing time: "
        f"{(total_end_time - total_start_time) / 60:.2f} min ==={COLOR_RESET}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified end-to-end processing script for Tohoku and Okayama datasets"
    )
    parser.add_argument(
        "--dataset",
        choices=["tohoku", "okayama"],
        required=True,
        help="Dataset to use: tohoku or okayama",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file (e.g. configs/config_calm3.py)",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Natural-language clinical query (e.g. '心停止の患者さんの中で、ECMOが使われた患者は何人？')",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default=None,
        help="Ground-truth label column name in the CSV for evaluation/filtering "
        "(e.g. '心停止+ECMO'). Optional; needed only with --filter-label-true",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory (default: data/{dataset})",
    )
    parser.add_argument(
        "--filter-label-true",
        action="store_true",
        help="Only process rows whose ground-truth label is True",
    )
    args = parser.parse_args()

    # Set global dataset
    DATASET = args.dataset

    # Set data directory
    if args.data_dir is not None:
        DATA_DIR = Path(args.data_dir)
    else:
        DATA_DIR = Path(DATASET_CONFIG[DATASET]["default_data_dir"])

    # Set query and (optional) ground-truth label column
    QUERY = args.query
    LABEL_COLUMN = args.label_column
    print(f"{COLOR_RED}QUERY: {QUERY}{COLOR_RESET}")
    print(f"{COLOR_RED}LABEL_COLUMN: {LABEL_COLUMN}{COLOR_RESET}")
    print(f"{COLOR_BLUE}Dataset: {DATASET}{COLOR_RESET}")
    print(f"{COLOR_BLUE}Data directory: {DATA_DIR}{COLOR_RESET}")

    # Filtering option
    FILTER_LABEL_TRUE = args.filter_label_true
    if FILTER_LABEL_TRUE:
        print(f"{COLOR_YELLOW}Processing only rows with True labels{COLOR_RESET}")

    # Load configuration
    load_config(args.config)
    print(f"{COLOR_BLUE}Config loaded from: {args.config}{COLOR_RESET}")
    print(f"{COLOR_BLUE}MODEL_PATH: {MODEL_PATH}{COLOR_RESET}")
    print(f"{COLOR_BLUE}LOG_DIR: {LOG_DIR}{COLOR_RESET}")

    # Set up logging
    setup_logging()

    # Run main processing
    main(filter_label_true=FILTER_LABEL_TRUE)
