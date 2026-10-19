# Configuration for CALM3-22B model
# HuggingFace: cyberagent/calm3-22b-chat
# Quantization: MLX 8-bit affine, group size 64, converted locally before inference:
#   mlx_lm.convert --hf-path <HF ID> --mlx-path <dir> --quantize --q-bits 8
# Set MODEL_PATH to <dir> to reproduce the paper; loading the HF ID directly
# runs the unquantized weights.

MODEL_PATH = "cyberagent/calm3-22b-chat"
LOG_DIR_SUFFIX = "CALM3-22B"
PROMPT_MODULE_FIRST = "src.pipeline.utils"
PROMPT_MODULE_SECOND = "src.pipeline.utils"
