#!/usr/bin/env bash
# Runs on your MAC. Checks the pod every 2 minutes. When the run finishes it:
#   1. copies the results folder to your Mac (results_p2/),
#   2. sounds an alarm, speaks, and shows a macOS notification,
#   3. optionally stops the pod (only after the copy succeeded).
#
# Usage (from ~/Desktop/fed_icl_project):
#   caffeinate -i bash infra/watch_run.sh IP PORT RESULTS_DIR
#   caffeinate -i bash infra/watch_run.sh IP PORT RESULTS_DIR POD_ID
# RESULTS_DIR is the folder name on the pod, e.g. p1_alpha1.0_both.
# Auto-stop needs POD_ID and your RunPod API key saved in ~/.runpod_api_key.
# caffeinate keeps the Mac awake while the watcher runs.

IP="$1"; PORT="$2"; DIR="$3"; POD_ID="${4:-}"
KEY="$HOME/.ssh/id_ed25519"
REMOTE="/workspace/fed_icl_project/results_p2/$DIR"
LOCAL="results_p2"
SSH_OPTS="-p $PORT -i $KEY -o ConnectTimeout=15 -o BatchMode=yes"

alarm() {
  osascript -e "display notification \"$1\" with title \"Fed-ICL run\" sound name \"Glass\"" >/dev/null 2>&1
  say "$1" >/dev/null 2>&1 &
  for i in 1 2 3 4 5 6; do afplay /System/Library/Sounds/Glass.aiff; done
}

stop_pod() {
  if [ -z "$POD_ID" ]; then
    echo "No POD_ID given: remember to STOP the pod in the RunPod console."
    return
  fi
  if [ ! -f "$HOME/.runpod_api_key" ]; then
    echo "No ~/.runpod_api_key file: STOP the pod in the RunPod console."
    return
  fi
  code=$(curl -s -o /dev/null -w "%{http_code}" -X POST \
    "https://rest.runpod.io/v1/pods/$POD_ID/stop" \
    -H "Authorization: Bearer $(cat "$HOME/.runpod_api_key")")
  if [ "$code" = "200" ]; then
    echo "Pod $POD_ID stopped."
  else
    echo "Auto-stop FAILED (HTTP $code): STOP the pod in the RunPod console."
    alarm "Could not stop the pod. Stop it manually."
  fi
}

echo "Watching $REMOTE on $IP:$PORT (checks every 2 minutes; Ctrl-C to quit)"
fails=0
while true; do
  state=$(ssh $SSH_OPTS root@"$IP" \
    "if [ -f $REMOTE/DONE ]; then echo done; elif [ -f $REMOTE/FAILED ]; then echo failed; else echo running; fi; ls $REMOTE/*.json 2>/dev/null | wc -l" 2>/dev/null)
  if [ -z "$state" ]; then
    fails=$((fails + 1))
    echo "$(date +%H:%M) cannot reach the pod ($fails in a row)"
    if [ "$fails" -ge 5 ]; then
      alarm "Cannot reach the pod. Check RunPod."
      exit 1
    fi
    sleep "${WATCH_INTERVAL:-120}"
    continue
  fi
  fails=0
  status=$(echo "$state" | head -1)
  njson=$(echo "$state" | tail -1 | tr -d ' ')
  echo "$(date +%H:%M) $status, $njson result files so far"

  if [ "$status" = "running" ]; then
    sleep "${WATCH_INTERVAL:-120}"
    continue
  fi

  mkdir -p "$LOCAL"
  if scp -P "$PORT" -i "$KEY" -o BatchMode=yes -r root@"$IP":"$REMOTE" "$LOCAL/"; then
    copied=$(ls "$LOCAL/$DIR"/*.json 2>/dev/null | wc -l | tr -d ' ')
    echo "Copied $copied result files to $LOCAL/$DIR"
    if [ "$status" = "done" ] && [ "$copied" -ge 1 ]; then
      alarm "Your Fed-ICL run has finished and the results are on your Mac."
      stop_pod
    else
      alarm "The run FAILED. Partial results copied. Check the pod log."
      echo "Pod left running so you can inspect the error."
    fi
  else
    alarm "The run finished but copying results FAILED. Do not stop the pod."
  fi
  exit 0
done
