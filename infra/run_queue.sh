#!/usr/bin/env bash
# Runs several experiments back to back, then writes DONE (or FAILED) in one
# folder, so a single Mac watcher can copy everything and stop the pod.
#   bash infra/run_queue.sh night2
# Watch it from the Mac with RESULTS_DIR = night2.
set -euo pipefail
cd /workspace/fed_icl_project
NAME="${1:-queue}"
export P2_OUT_ROOT="results_p2/$NAME"
mkdir -p "$P2_OUT_ROOT"
rm -f "$P2_OUT_ROOT/DONE" "$P2_OUT_ROOT/FAILED"
notify() {
  if [ -n "${NTFY_TOPIC:-}" ]; then
    curl -s -H "Title: Fed-ICL queue" -d "$1" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null || true
  fi
}
trap 'touch "$P2_OUT_ROOT/FAILED"; notify "Queue $NAME FAILED."' ERR

case "$NAME" in
  night2)
    # 1. Extreme heterogeneity: each subject sits on a single client.
    bash infra/run_p1.sh 0.001
    # 2. Alternative reading of Algorithm 1 line 7 (only Fed-ICL uses it).
    P2_PROMPT_POOL=union P2_ARMS=fedicl bash infra/run_p1.sh 1.0
    ;;
  *)
    echo "Unknown queue: $NAME"; exit 1 ;;
esac

touch "$P2_OUT_ROOT/DONE"
notify "Queue $NAME DONE."
echo "Queue $NAME done: $P2_OUT_ROOT"
