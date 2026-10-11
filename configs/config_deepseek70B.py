# Configuration for DeepSeek R1 Distill Llama 70B model
# HuggingFace: deepseek-ai/DeepSeek-R1-Distill-Llama-70B
# Quantization: MLX 8-bit affine, group size 64, converted locally before inference:
#   mlx_lm.convert --hf-path <HF ID> --mlx-path <dir> --quantize --q-bits 8
# Set MODEL_PATH to <dir> to reproduce the paper; loading the HF ID directly
# runs the unquantized weights.

MODEL_PATH = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
LOG_DIR_SUFFIX = "DeepSeek-70B"
PROMPT_MODULE_FIRST = "prompts.deepseek70b.prompt_first"
PROMPT_MODULE_SECOND = "prompts.deepseek70b.prompt_second"
