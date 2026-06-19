# Configuration for CALM3-22B model
# HuggingFace: cyberagent/calm3-22b-chat
# Quantization: MLX 8-bit (applied locally before inference)

MODEL_PATH = "cyberagent/calm3-22b-chat"
LOG_DIR_SUFFIX = "CALM3-22B"
PROMPT_MODULE_FIRST = "prompts.calm3.prompt_first"
PROMPT_MODULE_SECOND = "prompts.calm3.prompt_second"
