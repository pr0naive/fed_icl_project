#!/usr/bin/env bash
# Serve Llama-3.1-8B-Instruct (full bf16 weights, as in Wang et al.) on port 8000.
# First start downloads ~16 GB of weights into /workspace/hf_cache.
set -euo pipefail
source /workspace/venv/bin/activate
export HF_HOME=/workspace/hf_cache
export HF_TOKEN="$(cat /workspace/.hf_token)"

vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --port 8000 2>&1 | tee /workspace/vllm.log
