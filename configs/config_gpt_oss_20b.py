# Configuration for GPT-OSS-20B model
# HuggingFace: openai/gpt-oss-20b
# Quantization: MLX 8-bit (applied locally before inference)

MODEL_PATH = "openai/gpt-oss-20b"
LOG_DIR_SUFFIX = "GPT-OSS-20B"
PROMPT_MODULE_FIRST = "prompts.gpt_oss_20b.prompt_first"
PROMPT_MODULE_SECOND = "prompts.gpt_oss_20b.prompt_second"
