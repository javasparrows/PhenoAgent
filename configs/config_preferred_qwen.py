# Configuration for Preferred MedLLM Qwen 72B model
# HuggingFace: pfnet/Preferred-MedLLM-Qwen-72B
# Quantization: MLX 8-bit (applied locally before inference)

MODEL_PATH = "pfnet/Preferred-MedLLM-Qwen-72B"
LOG_DIR_SUFFIX = "MedLLM-72B"
PROMPT_MODULE_FIRST = "prompts.preferred_qwen.prompt_first"
PROMPT_MODULE_SECOND = "prompts.preferred_qwen.prompt_second"
