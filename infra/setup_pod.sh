#!/usr/bin/env bash
# One-time setup on a fresh RunPod pod. Everything lives on /workspace (the
# persistent volume), so stopping and restarting the pod keeps the venv, the
# repo and the downloaded model weights.
set -euo pipefail

cd /workspace
if [ ! -d fed_icl_project ]; then
  git clone https://github.com/pr0naive/fed_icl_project.git
fi
cd fed_icl_project
git fetch origin
git checkout paper2-security
git pull origin paper2-security

if [ ! -d /workspace/venv ]; then
  python3 -m venv /workspace/venv
fi
source /workspace/venv/bin/activate
pip install -U pip
pip install -r requirements_p2.txt

command -v tmux >/dev/null || (apt-get update && apt-get install -y tmux)

if [ ! -f /workspace/.hf_token ]; then
  echo
  echo "Paste your Hugging Face READ token, then press Enter (it is not shown):"
  read -rs token
  echo "$token" > /workspace/.hf_token
  chmod 600 /workspace/.hf_token
fi

pip freeze > /workspace/env_freeze.txt
python -m pytest -q tests_p2
echo
echo "Setup done. Next: start the model server with  bash infra/start_vllm.sh  inside tmux."
