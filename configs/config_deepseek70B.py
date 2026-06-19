# Configuration for DeepSeek R1 Distill Llama 70B model
# HuggingFace: deepseek-ai/DeepSeek-R1-Distill-Llama-70B
# Quantization: MLX 8-bit (applied locally before inference)

MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
LOG_DIR_SUFFIX = "DeepSeek-70B"
PROMPT_MODULE_FIRST = "prompts.deepseek70b.prompt_first"
PROMPT_MODULE_SECOND = "prompts.deepseek70b.prompt_second"
