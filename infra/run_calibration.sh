#!/usr/bin/env bash
# Calibration against Wang et al.'s rebuttal Table 1 (Llama-3.1-8B, MMLU, 114 queries):
#   zero-shot 74/114, Fed-ICL-GT 76/114, Fed-ICL 84/114, Fed-ICL-Free 84/114.
# One seed, all arms. Run after the vLLM server reports it is ready.
set -euo pipefail
source /workspace/venv/bin/activate
export HF_HOME=/workspace/hf_cache
cd /workspace/fed_icl_project
mkdir -p results_p2
export HF_TOKEN="$(cat /workspace/.hf_token)"
ALPHA="${1:-1.0}"
python -m p2.run --arms zeroshot,gt,fedicl,fedicl_free,solo_free --seed 42 --alpha "$ALPHA" \
  2>&1 | tee "results_p2/calibration_alpha${ALPHA}_seed42.log"
