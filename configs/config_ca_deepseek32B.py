# Configuration for CyberAgent DeepSeek R1 Distill Qwen 32B Japanese model
# HuggingFace: cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese
# Quantization: MLX 8-bit affine, group size 64, converted locally before inference:
#   mlx_lm.convert --hf-path <HF ID> --mlx-path <dir> --quantize --q-bits 8
# Set MODEL_PATH to <dir> to reproduce the paper; loading the HF ID directly
# runs the unquantized weights.

MODEL_PATH = "cyberagent/DeepSeek-R1-Distill-Qwen-32B-Japanese"
LOG_DIR_SUFFIX = "DeepSeek-32B"
PROMPT_MODULE_FIRST = "prompts.ca_deepseek32b.prompt_first"
PROMPT_MODULE_SECOND = "prompts.ca_deepseek32b.prompt_second"
