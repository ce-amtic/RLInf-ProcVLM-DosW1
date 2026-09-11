#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$repo/.venv/bin/activate"
export REPO_PATH="$repo" EMBODIED_PATH="$repo/examples/embodiment"
export PYTHONPATH="$repo"
export RAY_USAGE_STATS_ENABLED=0
case "${1:-}" in
  head)
    export RLINF_NODE_RANK=0
    : "${GPU_SERVER_IP:?Set GPU_SERVER_IP to the GPU LAN address}"
    ray start --head --port="${RAY_HEAD_PORT:-6379}" --node-ip-address="$GPU_SERVER_IP" --include-dashboard=false
    ;;
  robot)
    export RLINF_NODE_RANK=1
    : "${GPU_SERVER_IP:?Set GPU_SERVER_IP to the GPU LAN address}"
    : "${ROBOT_NODE_IP:?Set ROBOT_NODE_IP to the robot LAN address}"
    ray start --address="$GPU_SERVER_IP:${RAY_HEAD_PORT:-6379}" --node-ip-address="$ROBOT_NODE_IP" --num-gpus=0
    ;;
  *) echo 'Usage: node.sh head|robot' >&2; exit 2 ;;
esac
