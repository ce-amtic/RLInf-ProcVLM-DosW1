#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

MODE="${1:-single}"
if (($# > 0)); then
  shift
fi

case "${MODE}" in
  single)
    CONFIG_NAME="dosw1_pick_sac_flow_procvlm"
    ;;
  multiview)
    CONFIG_NAME="dosw1_pick_sac_flow_procvlm_multiview"
    ;;
  *)
    echo "Usage: $0 [single|multiview] [Hydra overrides ...]" >&2
    exit 2
    ;;
esac

export REPO_PATH="${SCRIPT_DIR}"
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
export PROCVLM_REPO_PATH="${PROCVLM_REPO_PATH:-${REPO_PATH}/third_party/ProcVLM}"
: "${PROCVLM_PYTHON:?Set PROCVLM_PYTHON to a ProcVLM-compatible Python executable}"
: "${PROCVLM_REWARD_MODEL_PATH:?Set PROCVLM_REWARD_MODEL_PATH to the merged ProcVLM checkpoint}"
: "${DOSW1_POLICY_MODEL_PATH:?Set DOSW1_POLICY_MODEL_PATH to the DOS-W1 flow-policy checkpoint}"
: "${DOSW1_DEMO_BUFFER_PATH:?Set DOSW1_DEMO_BUFFER_PATH to the DOS-W1 RLPD demonstration buffer}"

RLINF_PYTHON="${RLINF_PYTHON:-${REPO_PATH}/.venv/bin/python}"
if [[ ! -x "${RLINF_PYTHON}" ]]; then
  echo "RLInf Python executable not found: ${RLINF_PYTHON}" >&2
  exit 1
fi
if [[ ! -x "${PROCVLM_PYTHON}" ]]; then
  echo "ProcVLM Python executable not found: ${PROCVLM_PYTHON}" >&2
  exit 1
fi
if [[ ! -d "${PROCVLM_REPO_PATH}/evqa" ]]; then
  echo "ProcVLM public checkout not found: ${PROCVLM_REPO_PATH}" >&2
  exit 1
fi
if [[ ! -d "${PROCVLM_REWARD_MODEL_PATH}" ]]; then
  echo "ProcVLM reward checkpoint not found: ${PROCVLM_REWARD_MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -e "${DOSW1_POLICY_MODEL_PATH}" ]]; then
  echo "DOS-W1 policy checkpoint not found: ${DOSW1_POLICY_MODEL_PATH}" >&2
  exit 1
fi
if [[ ! -e "${DOSW1_DEMO_BUFFER_PATH}" ]]; then
  echo "DOS-W1 demonstration buffer not found: ${DOSW1_DEMO_BUFFER_PATH}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_PATH}:${PROCVLM_REPO_PATH}:${PYTHONPATH:-}"

LOG_DIR="${LOG_DIR:-${REPO_PATH}/logs/$(date +'%Y%m%d-%H%M%S')-${CONFIG_NAME}}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/run_embodiment.log"

CMD=(
  "${RLINF_PYTHON}"
  "${EMBODIED_PATH}/train_embodied_agent.py"
  --config-path "${EMBODIED_PATH}/config"
  --config-name "${CONFIG_NAME}"
  "runner.logger.log_path=${LOG_DIR}"
  "$@"
)

printf '%q ' "${CMD[@]}" > "${LOG_FILE}"
printf '\n' >> "${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
