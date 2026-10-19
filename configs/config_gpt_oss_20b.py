# Configuration for GPT-OSS-20B model
# HuggingFace: openai/gpt-oss-20b
# Quantization: pre-quantized MLX 8-bit checkpoint (group size 64), as used in
# the paper: lmstudio-community/gpt-oss-20b-MLX-8bit

MODEL_PATH = "lmstudio-community/gpt-oss-20b-MLX-8bit"
LOG_DIR_SUFFIX = "GPT-OSS-20B"
PROMPT_MODULE_FIRST = "src.pipeline.utils"
PROMPT_MODULE_SECOND = "src.pipeline.utils"
