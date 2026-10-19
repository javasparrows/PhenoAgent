# Configuration for Preferred MedLLM Qwen 72B model
# HuggingFace: pfnet/Preferred-MedLLM-Qwen-72B
# Quantization: MLX 8-bit affine, group size 64, converted locally before inference:
#   mlx_lm.convert --hf-path <HF ID> --mlx-path <dir> --quantize --q-bits 8
# Set MODEL_PATH to <dir> to reproduce the paper; loading the HF ID directly
# runs the unquantized weights.

MODEL_PATH = "pfnet/Preferred-MedLLM-Qwen-72B"
LOG_DIR_SUFFIX = "MedLLM-72B"
PROMPT_MODULE_FIRST = "src.pipeline.utils"
PROMPT_MODULE_SECOND = "src.pipeline.utils"
