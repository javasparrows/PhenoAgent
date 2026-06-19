#!/bin/bash
# =============================================================================
# PhenoAgent Demo Runner
#
# Generates synthetic dummy data, then runs the end-to-end PhenoAgent pipeline
# on the synthetic data. No real patient data is required.
#
# Usage:
#   ./scripts/run_pipeline.sh              # default: tohoku, first query only
#   ./scripts/run_pipeline.sh tohoku       # run on synthetic Tohoku data
#   ./scripts/run_pipeline.sh okayama      # run on synthetic Okayama data
#
# Prerequisites:
#   - uv sync (install dependencies)
#   - An MLX-compatible LLM checkpoint (see configs/ for model paths)
#
# What this script does:
#   1. Generates synthetic dummy data (master.csv + 40_result_df.csv)
#   2. Runs the PhenoAgent end-to-end pipeline on the first query pair
#      using the GPT-OSS-20B config (smallest model) on synthetic data
# =============================================================================
set -euo pipefail

DATASET="${1:-tohoku}"

if [[ "$DATASET" != "tohoku" && "$DATASET" != "okayama" ]]; then
    echo "Error: dataset must be 'tohoku' or 'okayama', got '$DATASET'"
    echo "Usage: $0 [tohoku|okayama]"
    exit 1
fi

echo "======================================================================"
echo "  PhenoAgent Demo Runner"
echo "======================================================================"
echo "  Dataset: $DATASET"
echo ""

# --------------------------------------------------------------------------
# Step 1: Generate synthetic dummy data
# --------------------------------------------------------------------------
echo "[Step 1/2] Generating synthetic dummy data..."
uv run python scripts/generate_dummy_data.py
echo ""

# --------------------------------------------------------------------------
# Step 2: Run end-to-end pipeline on synthetic data (single query demo)
# --------------------------------------------------------------------------
echo "[Step 2/2] Running PhenoAgent end-to-end pipeline on synthetic data..."
echo ""
echo "  NOTE: This requires an MLX-compatible LLM checkpoint to be available."
echo "  The pipeline will load the model specified in the config file."
echo "  If the model is not cached locally, it will be downloaded first."
echo ""

# Use the first query pair as the demo query
QUERY="心停止の患者さんの中で、ECMOが使われた患者は何人？"
LABEL_COL="心停止+ECMO"
CONFIG="configs/config_gpt_oss_20b.py"

echo "  Config : $CONFIG"
echo "  Query  : $QUERY"
echo "  Label  : $LABEL_COL"
echo "  Data   : data_sample/$DATASET"
echo ""

uv run python -m src.pipeline.end2end \
    --dataset "$DATASET" \
    --config "$CONFIG" \
    --query "$QUERY" \
    --label-column "$LABEL_COL" \
    --data-dir "data_sample/$DATASET"

echo ""
echo "======================================================================"
echo "  Demo complete."
echo "======================================================================"
exit 0
