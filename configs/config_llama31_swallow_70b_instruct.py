# Configuration for Llama 3.1 Swallow 70B Instruct model
# HuggingFace: tokyotech-llm/Llama-3.1-Swallow-70B-Instruct-v0.3
# Quantization: MLX 8-bit (applied locally before inference)

MODEL_PATH = "tokyotech-llm/Llama-3.1-Swallow-70B-Instruct-v0.3"
LOG_DIR_SUFFIX = "Llama-3.1-Swallow-70B"
PROMPT_MODULE_FIRST = "prompts.llama31_swallow_70b_instruct.prompt_first"
PROMPT_MODULE_SECOND = "prompts.llama31_swallow_70b_instruct.prompt_second"
