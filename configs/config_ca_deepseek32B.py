# Configuration for CyberAgent DeepSeek R1 Distill Qwen 32B Japanese model
# HuggingFace: cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese
# Quantization: MLX 8-bit (applied locally before inference)

MODEL_PATH = "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese"
LOG_DIR_SUFFIX = "DeepSeek-32B"
PROMPT_MODULE_FIRST = "prompts.ca_deepseek32b.prompt_first"
PROMPT_MODULE_SECOND = "prompts.ca_deepseek32b.prompt_second"
