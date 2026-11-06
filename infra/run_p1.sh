#!/usr/bin/env bash
# P1: all arms x 5 seeds at one alpha, each seed with its own 114 queries
# (570 distinct questions in total). About 45 min per seed on an A40.
# Usage: bash infra/run_p1.sh 1.0                  (alpha 1.0, all five seeds)
#        bash infra/run_p1.sh 1.0 "42 7"           (only some seeds)
#        P2_PROMPT_POOL=union bash infra/run_p1.sh 1.0   (alternative line-7 reading)
# Only some arms: P2_ARMS=fedicl bash infra/run_p1.sh 1.0
# Optional phone alert: export NTFY_TOPIC=<your-secret-topic> before running.
# On finish it writes DONE (or FAILED) into the results folder; the Mac-side
# watcher (infra/watch_run.sh) looks for that file.
set -euo pipefail
source /workspace/venv/bin/activate
export HF_HOME=/workspace/hf_cache
export HF_TOKEN="$(cat /workspace/.hf_token)"
export P2_QUERY_MODE=per_seed
cd /workspace/fed_icl_project
ALPHA="${1:-1.0}"
SEEDS="${2:-42 7 13 99 3}"
ARMS="${P2_ARMS:-zeroshot,randlabel,gt,fedicl,fedicl_free,solo_free}"
export P2_RESULTS="${P2_OUT_ROOT:-results_p2}/p1_alpha${ALPHA}_${P2_PROMPT_POOL:-both}"
mkdir -p "$P2_RESULTS"
rm -f "$P2_RESULTS/DONE" "$P2_RESULTS/FAILED"

notify() {
  if [ -n "${NTFY_TOPIC:-}" ]; then
    curl -s -H "Title: Fed-ICL run" -d "$1" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null || true
  fi
}
trap 'touch "$P2_RESULTS/FAILED"; notify "P1 FAILED (alpha $ALPHA). Check the pod."' ERR

for SEED in $SEEDS; do
  python -m p2.run --arms "$ARMS" --seed "$SEED" --alpha "$ALPHA" \
    2>&1 | tee "$P2_RESULTS/seed${SEED}.log"
  notify "Seed $SEED finished (alpha $ALPHA)."
done
touch "$P2_RESULTS/DONE"
notify "P1 DONE: alpha $ALPHA, seeds $SEEDS. Copy results and stop the pod."
echo "P1 done: $P2_RESULTS"
