# Configuration for Llama 3.3 Swallow 70B Instruct model
# HuggingFace: tokyotech-llm/Llama-3.3-Swallow-70B-Instruct-v0.4
# Quantization: pre-quantized MLX 8-bit checkpoint (group size 64), as used in
# the paper: mlx-community/Llama-3.3-Swallow-70B-Instruct-v0.4-8bit

MODEL_PATH = "mlx-community/Llama-3.3-Swallow-70B-Instruct-v0.4-8bit"
LOG_DIR_SUFFIX = "Llama-3.3-Swallow-70B"
PROMPT_MODULE_FIRST = "src.pipeline.utils"
PROMPT_MODULE_SECOND = "src.pipeline.utils"
