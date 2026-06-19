#!/bin/bash
# =============================================================================
# Unified end-to-end batch runner for PhenoAgent experiments
#
# Usage:
#   ./scripts/run_end2end.sh tohoku
#   ./scripts/run_end2end.sh okayama
#
# Environment:
#   PHENOAGENT_CONFIGS  - Comma-separated list of config indices (1-6) to run.
#                         Default: all configs.
#                         Example: PHENOAGENT_CONFIGS=1,3,5 ./scripts/run_end2end.sh tohoku
#   PHENOAGENT_PAIRS    - Comma-separated list of label pair indices (1-12) to run.
#                         Default: all pairs.
#                         Example: PHENOAGENT_PAIRS=1,2,3 ./scripts/run_end2end.sh tohoku
# =============================================================================
set -euo pipefail

# ---- Argument validation ----------------------------------------------------
DATASET="${1:-}"
if [[ -z "$DATASET" ]]; then
    echo "Error: dataset argument required."
    echo "Usage: $0 [tohoku|okayama]"
    exit 1
fi

if [[ "$DATASET" != "tohoku" && "$DATASET" != "okayama" ]]; then
    echo "Error: dataset must be 'tohoku' or 'okayama', got '$DATASET'"
    exit 1
fi

# ---- Clinical queries (12 free-form queries from the paper) -----------------
# Format: "free-form query string|ground-truth label column"
LABEL_PAIRS=(
    "心停止の患者さんの中で、ECMOが使われた患者は何人？|心停止+ECMO"
    "低体温の患者さんの中で、ECMOが使われた患者は何人？|低体温+ECMO"
    "骨盤骨折の患者さんの中で、創外固定が使われた患者は何人？|骨盤骨折+創外固定"
    "心筋梗塞の患者さんの中で、ECMOが使われた患者は何人？|心筋梗塞+ECMO"
    "心不全の患者さんの中で、NPPVが使われた患者は何人？|心不全+NPPV"
    "腹部刺創の患者さんの中で、開腹術が使われた患者は何人？|腹部刺創+開腹術"
    "急性薬物中毒の患者さんの中で、活性炭投与が使われた患者は何人？|急性薬物中毒+活性炭投与"
    "骨盤骨折の患者さんの中で、IVRが使われた患者は何人？|骨盤骨折+IVR"
    "敗血症の患者さんの中で、急性腎障害が使われた患者は何人？|敗血症+急性腎障害"
    "敗血症の患者さんの中で、播種性血管内凝固が使われた患者は何人？|敗血症+播種性血管内凝固"
    "心停止の患者さんの中で、低酸素性脳症が使われた患者は何人？|心停止+低酸素性脳症"
    "骨盤骨折の患者さんの中で、出血性ショックが使われた患者は何人？|骨盤骨折+出血性ショック"
)

# ---- Model configs (6 configs) ----------------------------------------------
CONFIGS=(
    "config_calm3_8bit.py"
    "config_ca_deepseek32B.py"
    "config_deepseek70B.py"
    "config_gpt_oss_20b.py"
    "config_llama31_swallow_70b_instruct.py"
    "config_preferred_qwen.py"
)

# ---- Filter by environment variables if set ----------------------------------
if [[ -n "${PHENOAGENT_CONFIGS:-}" ]]; then
    IFS=',' read -ra SELECTED_CONFIGS <<< "$PHENOAGENT_CONFIGS"
else
    SELECTED_CONFIGS=()
    for i in $(seq 1 ${#CONFIGS[@]}); do
        SELECTED_CONFIGS+=("$i")
    done
fi

if [[ -n "${PHENOAGENT_PAIRS:-}" ]]; then
    IFS=',' read -ra SELECTED_PAIRS <<< "$PHENOAGENT_PAIRS"
else
    SELECTED_PAIRS=()
    for i in $(seq 1 ${#LABEL_PAIRS[@]}); do
        SELECTED_PAIRS+=("$i")
    done
fi

# ---- Utility: format duration ------------------------------------------------
format_duration() {
    local total_seconds=$1
    local hours=$((total_seconds / 3600))
    local minutes=$(((total_seconds % 3600) / 60))
    local seconds=$((total_seconds % 60))
    printf "%dh %dm %ds" "$hours" "$minutes" "$seconds"
}

# ---- Batch execution ---------------------------------------------------------
TOTAL_CONFIGS=${#SELECTED_CONFIGS[@]}
TOTAL_PAIRS=${#SELECTED_PAIRS[@]}
TOTAL_RUNS=$((TOTAL_CONFIGS * TOTAL_PAIRS))

batch_start_time=$(date +%s)
batch_start_datetime=$(date '+%Y-%m-%d %H:%M:%S')

echo "======================================================================"
echo "  PhenoAgent End-to-End Batch Runner"
echo "======================================================================"
echo "  Dataset    : $DATASET"
echo "  Configs    : $TOTAL_CONFIGS"
echo "  Label pairs: $TOTAL_PAIRS"
echo "  Total runs : $TOTAL_RUNS"
echo "  Started at : $batch_start_datetime"
echo "======================================================================"
echo ""

# Track results for summary
declare -a RUN_RESULTS=()
declare -a RUN_DURATIONS=()
declare -a RUN_DESCRIPTIONS=()
run_count=0
fail_count=0

for config_idx in "${SELECTED_CONFIGS[@]}"; do
    # Validate config index
    if [[ "$config_idx" -lt 1 || "$config_idx" -gt ${#CONFIGS[@]} ]]; then
        echo "Warning: skipping invalid config index $config_idx"
        continue
    fi
    CONFIG="${CONFIGS[$((config_idx - 1))]}"
    CONFIG_NAME="${CONFIG%.py}"

    echo "----------------------------------------------------------------------"
    echo "  Config [$config_idx/${#CONFIGS[@]}]: $CONFIG_NAME"
    echo "----------------------------------------------------------------------"

    for pair_idx in "${SELECTED_PAIRS[@]}"; do
        # Validate pair index
        if [[ "$pair_idx" -lt 1 || "$pair_idx" -gt ${#LABEL_PAIRS[@]} ]]; then
            echo "Warning: skipping invalid pair index $pair_idx"
            continue
        fi
        ENTRY="${LABEL_PAIRS[$((pair_idx - 1))]}"
        QUERY="${ENTRY%%|*}"
        LABEL_COL="${ENTRY#*|}"

        run_count=$((run_count + 1))
        desc="$CONFIG_NAME | $LABEL_COL"
        RUN_DESCRIPTIONS+=("$desc")

        echo ""
        echo "  [$run_count/$TOTAL_RUNS] $LABEL_COL"

        run_start=$(date +%s)

        if uv run python -u -m src.pipeline.end2end \
            --dataset "$DATASET" \
            --config "configs/$CONFIG" \
            --query "$QUERY" --label-column "$LABEL_COL"; then
            run_end=$(date +%s)
            run_duration=$((run_end - run_start))
            RUN_RESULTS+=("OK")
            RUN_DURATIONS+=("$run_duration")
            echo "  -> OK ($(format_duration $run_duration))"
        else
            run_end=$(date +%s)
            run_duration=$((run_end - run_start))
            RUN_RESULTS+=("FAIL")
            RUN_DURATIONS+=("$run_duration")
            fail_count=$((fail_count + 1))
            echo "  -> FAILED (exit code $?, $(format_duration $run_duration))"
        fi
    done
done

# ---- Summary -----------------------------------------------------------------
batch_end_time=$(date +%s)
batch_end_datetime=$(date '+%Y-%m-%d %H:%M:%S')
batch_duration=$((batch_end_time - batch_start_time))

echo ""
echo "======================================================================"
echo "  Batch Summary"
echo "======================================================================"
echo "  Dataset    : $DATASET"
echo "  Started at : $batch_start_datetime"
echo "  Finished at: $batch_end_datetime"
echo "  Total time : $(format_duration $batch_duration)"
echo "  Total runs : $run_count"
echo "  Succeeded  : $((run_count - fail_count))"
echo "  Failed     : $fail_count"
echo "----------------------------------------------------------------------"

# Print per-run details
for i in $(seq 0 $((run_count - 1))); do
    status="${RUN_RESULTS[$i]}"
    duration="${RUN_DURATIONS[$i]}"
    description="${RUN_DESCRIPTIONS[$i]}"
    if [[ "$status" == "FAIL" ]]; then
        printf "  [FAIL] %-60s %s\n" "$description" "$(format_duration $duration)"
    else
        printf "  [ OK ] %-60s %s\n" "$description" "$(format_duration $duration)"
    fi
done

echo "======================================================================"

# Exit with non-zero if any run failed
if [[ $fail_count -gt 0 ]]; then
    echo "WARNING: $fail_count run(s) failed."
    exit 1
fi

echo "All runs completed successfully."
exit 0
