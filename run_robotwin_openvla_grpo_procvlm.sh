#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

source .venv/bin/activate

export EMBODIED_PATH="${SCRIPT_DIR}/examples/embodiment"
export REPO_PATH="${SCRIPT_DIR}"
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"
export CONFIG_PATH="${CONFIG_PATH:-${EMBODIED_PATH}/config}"
export PROCVLM_REPO_PATH="${PROCVLM_REPO_PATH:-${SCRIPT_DIR}/third_party/ProcVLM}"
export ROBOTWIN_PATH="${ROBOTWIN_PATH:-${SCRIPT_DIR}/third_party/RoboTwin}"
export ROBOTWIN_ASSETS_PATH="${ROBOTWIN_ASSETS_PATH:-${ROBOTWIN_PATH}}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${SCRIPT_DIR}/.hf-cache}"
: "${PROCVLM_PYTHON:?Set PROCVLM_PYTHON to the Python executable of a ProcVLM-compatible environment}"
export PROCVLM_PYTHON
export PROCVLM_SERVER_SCRIPT="${PROCVLM_SERVER_SCRIPT:-${SCRIPT_DIR}/rlinf/models/embodiment/reward/procvlm_reward_server.py}"
export USE_TF="${USE_TF:-0}"
export TRANSFORMERS_NO_TF="${TRANSFORMERS_NO_TF:-1}"
export USE_FLAX="${USE_FLAX:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export TF_ENABLE_ONEDNN_OPTS="${TF_ENABLE_ONEDNN_OPTS:-0}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.995}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-1}"
export RAY_enable_core_worker_task_event_to_gcs="${RAY_enable_core_worker_task_event_to_gcs:-0}"
export RAY_task_events_max_num_task_in_gcs="${RAY_task_events_max_num_task_in_gcs:-1024}"
export RAY_task_events_max_dropped_task_attempts_tracked_per_job_in_gcs="${RAY_task_events_max_dropped_task_attempts_tracked_per_job_in_gcs:-1024}"
export RAY_task_events_max_num_status_events_buffer_on_worker="${RAY_task_events_max_num_status_events_buffer_on_worker:-1024}"
export RAY_task_events_max_num_profile_events_per_task="${RAY_task_events_max_num_profile_events_per_task:-0}"
export RAY_task_events_max_num_profile_events_buffer_on_worker="${RAY_task_events_max_num_profile_events_buffer_on_worker:-0}"
export PROCVLM_LOG_EVERY_N_REQUESTS="${PROCVLM_LOG_EVERY_N_REQUESTS:-50}"
export PROCVLM_SERVER_LOG_EVERY_N_REQUESTS="${PROCVLM_SERVER_LOG_EVERY_N_REQUESTS:-50}"
export PROCVLM_SHOW_PROGRESS="${PROCVLM_SHOW_PROGRESS:-0}"
export ROBOTWIN_LOG_FILTER_ENABLED="${ROBOTWIN_LOG_FILTER_ENABLED:-1}"
export ROBOTWIN_LOG_FILTER_STRIP_ANSI="${ROBOTWIN_LOG_FILTER_STRIP_ANSI:-1}"
export ROBOTWIN_LOG_FILTER_DROP_STEP_PROGRESS="${ROBOTWIN_LOG_FILTER_DROP_STEP_PROGRESS:-1}"
export ROBOTWIN_LOG_FILTER_DROP_BLANK_LINES="${ROBOTWIN_LOG_FILTER_DROP_BLANK_LINES:-1}"
export ROBOTWIN_LOG_FILTER_DROP_EMPTY_WORKER_LINES="${ROBOTWIN_LOG_FILTER_DROP_EMPTY_WORKER_LINES:-1}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export VLLM_DO_NOT_TRACK="${VLLM_DO_NOT_TRACK:-1}"
export DO_NOT_TRACK="${DO_NOT_TRACK:-1}"
export PYTHONPATH="${REPO_PATH}:${ROBOTWIN_PATH}:${PROCVLM_REPO_PATH}:${PYTHONPATH:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

check_openvla_oft_deps() {
    "${PYTHON_BIN}" - <<'PY'
import sys
from pathlib import Path

import tokenizers
import transformers

transformers_path = Path(transformers.__file__).resolve()
errors = []
if transformers.__version__ != "4.40.1":
    errors.append(f"transformers version {transformers.__version__} != 4.40.1")
if "transformers-openvla-oft" not in str(transformers_path):
    errors.append(
        f"transformers path {transformers_path} is not the OpenVLA-OFT fork"
    )
if tokenizers.__version__ != "0.19.1":
    errors.append(f"tokenizers version {tokenizers.__version__} != 0.19.1")

if errors:
    for error in errors:
        print(f"[ERROR] {error}", file=sys.stderr)
    print(
        "[ERROR] Install with: python -m pip install 'tokenizers==0.19.1' && "
        "python -m pip install --no-deps -e .third_party_src/transformers-openvla-oft",
        file=sys.stderr,
    )
    raise SystemExit(1)

print(
    "[INFO] OpenVLA-OFT deps OK: "
    f"transformers={transformers.__version__} ({transformers_path}), "
    f"tokenizers={tokenizers.__version__}",
    flush=True,
)
PY
}

check_openvla_oft_deps

CONFIG_NAME="${1:-${CONFIG_NAME:-robotwin_beat_block_hammer_grpo_openvlaoft_procvlm_reward}}"
ROBOT_PLATFORM="${2:-${ROBOT_PLATFORM:-ALOHA}}"
if (($# > 2)); then
    EXTRA_ARGS=("${@:3}")
else
    EXTRA_ARGS=()
fi

case "${CONFIG_NAME}" in
    *handover_block*)
        TASK_NAME="${ROBOTWIN_TASK_NAME:-handover_block}"
        TASK_MAX_STEPS="${ROBOTWIN_TASK_MAX_STEPS:-400}"
        TASK_UNNORM_KEY="${ROBOTWIN_UNNORM_KEY:-handover_block_1k}"
        TASK_SFT_MODEL="${ROBOTWIN_SFT_MODEL:-${SCRIPT_DIR}/models/RLinf-OpenVLAOFT-RoboTwin-SFT-handover_block}"
        TASK_RM_DESCRIPTION="${ROBOTWIN_TASK_DESCRIPTION:-use the left arm to grasp the red block on the table, handover it to the right arm and place it on the blue pad}"
        ;;
    *beat_block_hammer*)
        TASK_NAME="${ROBOTWIN_TASK_NAME:-beat_block_hammer}"
        TASK_MAX_STEPS="${ROBOTWIN_TASK_MAX_STEPS:-400}"
        TASK_UNNORM_KEY="${ROBOTWIN_UNNORM_KEY:-beat_block_hammer_1k}"
        TASK_SFT_MODEL="${ROBOTWIN_SFT_MODEL:-${SCRIPT_DIR}/models/RLinf-OpenVLAOFT-RoboTwin-SFT-beat_block_hammer}"
        TASK_RM_DESCRIPTION="${ROBOTWIN_TASK_DESCRIPTION:-there is a hammer and a block on the table, use the arm to grab the hammer and beat the block}"
        ;;
    *pick_dual_bottles*)
        TASK_NAME="${ROBOTWIN_TASK_NAME:-pick_dual_bottles}"
        TASK_MAX_STEPS="${ROBOTWIN_TASK_MAX_STEPS:-100}"
        TASK_UNNORM_KEY="${ROBOTWIN_UNNORM_KEY:-pick_dual_bottles_1k}"
        TASK_SFT_MODEL="${ROBOTWIN_SFT_MODEL:-${SCRIPT_DIR}/models/RLinf-OpenVLAOFT-RoboTwin-SFT-pick_dual_bottles}"
        TASK_RM_DESCRIPTION="${ROBOTWIN_TASK_DESCRIPTION:-pick up one bottle with one arm, and pick up another bottle with the other arm}"
        ;;
    *place_container_plate*)
        TASK_NAME="${ROBOTWIN_TASK_NAME:-place_container_plate}"
        TASK_MAX_STEPS="${ROBOTWIN_TASK_MAX_STEPS:-150}"
        TASK_UNNORM_KEY="${ROBOTWIN_UNNORM_KEY:-place_container_plate_1k}"
        TASK_SFT_MODEL="${ROBOTWIN_SFT_MODEL:-${SCRIPT_DIR}/models/RLinf-OpenVLAOFT-RoboTwin-SFT-place_container_plate}"
        TASK_RM_DESCRIPTION="${ROBOTWIN_TASK_DESCRIPTION:-place the container onto the plate}"
        ;;
    *move_can_pot*)
        TASK_NAME="${ROBOTWIN_TASK_NAME:-move_can_pot}"
        TASK_MAX_STEPS="${ROBOTWIN_TASK_MAX_STEPS:-200}"
        TASK_UNNORM_KEY="${ROBOTWIN_UNNORM_KEY:-move_can_pot_1k}"
        TASK_SFT_MODEL="${ROBOTWIN_SFT_MODEL:-${SCRIPT_DIR}/models/RLinf-OpenVLAOFT-RoboTwin-SFT-move_can_pot}"
        TASK_RM_DESCRIPTION="${ROBOTWIN_TASK_DESCRIPTION:-there is a can and a pot on the table, use one arm to pick up the can and move it to beside the pot}"
        ;;
    *)
        TASK_NAME="${ROBOTWIN_TASK_NAME:-}"
        TASK_MAX_STEPS="${ROBOTWIN_TASK_MAX_STEPS:-200}"
        TASK_UNNORM_KEY="${ROBOTWIN_UNNORM_KEY:-}"
        TASK_SFT_MODEL="${ROBOTWIN_SFT_MODEL:-${SCRIPT_DIR}/models/RLinf-OpenVLAOFT-RoboTwin-SFT-${TASK_NAME}}"
        TASK_RM_DESCRIPTION="${ROBOTWIN_TASK_DESCRIPTION:-}"
        ;;
esac

if [[ "${CONFIG_NAME}" == *procvlm* && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
fi

export ROBOT_PLATFORM
export LOG_DIR="${LOG_DIR:-/data/fengyouhe/rlinf_logs/${CONFIG_NAME}}"
GROUP_SIZE="${GROUP_SIZE:-8}"

DEFAULT_ROLLOUT_EPOCH="${ROLLOUT_EPOCH:-5}"
DEFAULT_TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-96}"
DEFAULT_PIPELINE_STAGE_NUM="${ROLLOUT_PIPELINE_STAGE_NUM:-1}"
DEFAULT_ACTOR_MICRO_BATCH_SIZE="${ACTOR_MICRO_BATCH_SIZE:-24}"
DEFAULT_ACTOR_OFFLOAD_OPTIMIZER="${ACTOR_OFFLOAD_OPTIMIZER:-True}"
DEFAULT_ROLLOUT_OFFLOAD_STRATEGY="${ROLLOUT_OFFLOAD_STRATEGY:-destroy}"
DEFAULT_ACTION_EXEC_HORIZON="${ROBOTWIN_ACTION_EXEC_HORIZON:-8}"
DEFAULT_EVAL_ACTION_EXEC_HORIZON="${ROBOTWIN_EVAL_ACTION_EXEC_HORIZON:-${DEFAULT_ACTION_EXEC_HORIZON}}"
DEFAULT_EXECUTE_ACTION_PREFIX_INDIVIDUALLY="${ROBOTWIN_EXECUTE_ACTION_PREFIX_INDIVIDUALLY:-True}"
DEFAULT_EVAL_EXECUTE_ACTION_PREFIX_INDIVIDUALLY="${ROBOTWIN_EVAL_EXECUTE_ACTION_PREFIX_INDIVIDUALLY:-${DEFAULT_EXECUTE_ACTION_PREFIX_INDIVIDUALLY}}"
PROCVLM_RM_IMAGE_KEYS="${PROCVLM_RM_IMAGE_KEYS:-[\"reward_main_images\"]}"
PROCVLM_RM_HISTORY_KEYS="${PROCVLM_RM_HISTORY_KEYS:-${PROCVLM_RM_IMAGE_KEYS}}"

resolve_reward_routing_weights() {
    local spec="${PROCVLM_REWARD_ROUTING_WEIGHTS:-}"
    if [[ -z "${spec}" || "${spec}" == "none" || "${spec}" == "None" ]]; then
        return 0
    fi
    if [[ "${spec}" != "auto" ]]; then
        printf '%s\n' "${spec}"
        return 0
    fi

    local placement="${COMPONENT_PLACEMENT_REWARD:-0-1}"
    local reserved_mb="${PROCVLM_REWARD_ROUTING_RESERVED_MB:-32768}"
    "${PYTHON_BIN}" - "${placement}" "${reserved_mb}" <<'PY'
import os
import subprocess
import sys

placement = sys.argv[1]
reserved_mb = int(float(sys.argv[2]))

def expand_resource_ranks(spec: str) -> list[int]:
    ranks: list[int] = []
    seen: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        resource = part.split(":", 1)[0].strip()
        if not resource:
            continue
        for item in resource.split("+"):
            item = item.strip()
            if not item:
                continue
            if "-" in item:
                begin, end = [int(x) for x in item.split("-", 1)]
                values = range(begin, end + 1)
            else:
                values = [int(item)]
            for value in values:
                if value not in seen:
                    ranks.append(value)
                    seen.add(value)
    return ranks

local_ranks = expand_resource_ranks(placement)
if not local_ranks:
    raise SystemExit(0)

visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
if visible:
    visible_ids = [item.strip() for item in visible.split(",") if item.strip()]
else:
    visible_ids = []

physical_ids = []
for rank in local_ranks:
    if visible_ids and rank < len(visible_ids):
        physical_ids.append(visible_ids[rank])
    else:
        physical_ids.append(str(rank))

free_by_index: dict[str, int] = {}
try:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    for line in output.splitlines():
        if not line.strip():
            continue
        idx, free = [item.strip() for item in line.split(",", 1)]
        free_by_index[idx] = int(free)
except Exception:
    free_by_index = {}

weights = []
for idx in physical_ids:
    free_mb = free_by_index.get(idx)
    if free_mb is None:
        weights.append(1)
    else:
        weights.append(max(1, free_mb - reserved_mb))

print(",".join(str(weight) for weight in weights))
PY
}

RESOLVED_PROCVLM_REWARD_ROUTING_WEIGHTS="$(resolve_reward_routing_weights || true)"

DEFAULT_OVERRIDES=(
    "runner.val_check_interval=${RUNNER_VAL_CHECK_INTERVAL:-10}"
    "algorithm.eval_rollout_epoch=1"
    "runner.max_steps=${RUNNER_MAX_STEPS:-320}"
    "runner.save_interval=${RUNNER_SAVE_INTERVAL:-10}"
    "++runner.keep_latest_checkpoints=${RUNNER_KEEP_LATEST_CHECKPOINTS:-3}"
    "++runner.keep_best_checkpoint=${RUNNER_KEEP_BEST_CHECKPOINT:-True}"
    "++runner.best_checkpoint_metric=${RUNNER_BEST_CHECKPOINT_METRIC:-eval/success_once}"
    "++runner.best_checkpoint_fallback_metric=${RUNNER_BEST_CHECKPOINT_FALLBACK_METRIC:-env/success_once}"
    'runner.logger.logger_backends=["tensorboard","wandb"]'
    "algorithm.rollout_epoch=${DEFAULT_ROLLOUT_EPOCH}"
    "algorithm.group_size=${GROUP_SIZE}"
    "actor.global_batch_size=${ACTOR_GLOBAL_BATCH_SIZE:-960}"
    "actor.enable_offload=${ACTOR_ENABLE_OFFLOAD:-True}"
    "++actor.offload_optimizer=${DEFAULT_ACTOR_OFFLOAD_OPTIMIZER}"
    "actor.micro_batch_size=${DEFAULT_ACTOR_MICRO_BATCH_SIZE}"
    "++actor.log_every_n_global_batches=${ACTOR_LOG_EVERY_N_GLOBAL_BATCHES:-1}"
    "++actor.fsdp_config.enable_gradient_accumulation=${ACTOR_ENABLE_GRADIENT_ACCUMULATION:-True}"
    "++actor.fsdp_config.limit_all_gathers=${ACTOR_LIMIT_ALL_GATHERS:-True}"
    "++actor.fsdp_config.save_full_model_weights=${ACTOR_SAVE_FULL_MODEL_WEIGHTS:-False}"
    "actor.model.model_path=${TASK_SFT_MODEL}"
    "rollout.model.model_path=${TASK_SFT_MODEL}"
    "actor.model.unnorm_key=${TASK_UNNORM_KEY}"
    "actor.model.num_action_chunks=${ROBOTWIN_NUM_ACTION_CHUNKS:-25}"
    "rollout.enable_offload=${ROLLOUT_ENABLE_OFFLOAD:-True}"
    "++rollout.generation_backend=${ROLLOUT_GENERATION_BACKEND:-huggingface}"
    "++rollout.offload_strategy=${DEFAULT_ROLLOUT_OFFLOAD_STRATEGY}"
    "++rollout.infer_micro_batch_size=${ROLLOUT_INFER_MICRO_BATCH_SIZE:-16}"
    "++rollout.batch_pipeline_stages=${ROLLOUT_BATCH_PIPELINE_STAGES:-False}"
    "++rollout.env_rank_dispatch_mode=${ROLLOUT_ENV_RANK_DISPATCH_MODE:-streaming_ready}"
    "++rollout.env_rank_ready_min_batch_ranks=${ROLLOUT_ENV_RANK_READY_MIN_BATCH_RANKS:-2}"
    "++rollout.env_rank_ready_max_wait_s=${ROLLOUT_ENV_RANK_READY_MAX_WAIT_S:-0.02}"
    "rollout.pipeline_stage_num=${DEFAULT_PIPELINE_STAGE_NUM}"
    "++cluster.component_placement.actor=${COMPONENT_PLACEMENT_ACTOR:-0-3}"
    "++cluster.component_placement.rollout=${COMPONENT_PLACEMENT_ROLLOUT:-0-1}"
    "++cluster.component_placement.env=${COMPONENT_PLACEMENT_ENV:-2-3:0-11}"
    "++cluster.component_placement.reward=${COMPONENT_PLACEMENT_REWARD:-0-1}"
    "env.train.total_num_envs=${DEFAULT_TRAIN_NUM_ENVS}"
    "env.train.group_size=${ENV_TRAIN_GROUP_SIZE:-${GROUP_SIZE}}"
    "env.train.assets_path=${ROBOTWIN_ASSETS_PATH}"
    "env.eval.assets_path=${ROBOTWIN_ASSETS_PATH}"
    "env.train.max_steps_per_rollout_epoch=${TASK_MAX_STEPS}"
    "env.train.max_episode_steps=${TASK_MAX_STEPS}"
    "env.eval.max_steps_per_rollout_epoch=${TASK_MAX_STEPS}"
    "env.eval.max_episode_steps=${TASK_MAX_STEPS}"
    "++env.train.execute_action_prefix_individually=${DEFAULT_EXECUTE_ACTION_PREFIX_INDIVIDUALLY}"
    "++env.eval.execute_action_prefix_individually=${DEFAULT_EVAL_EXECUTE_ACTION_PREFIX_INDIVIDUALLY}"
    "env.train.task_config.step_lim=${TASK_MAX_STEPS}"
    "env.eval.task_config.step_lim=${TASK_MAX_STEPS}"
    "env.train.task_config.planner_backend=${ROBOTWIN_PLANNER_BACKEND:-mplib}"
    "env.eval.task_config.planner_backend=${ROBOTWIN_PLANNER_BACKEND:-mplib}"
    "++env.train.task_config.render_every_control_step=${ROBOTWIN_RENDER_EVERY_CONTROL_STEP:-False}"
    "++env.eval.task_config.render_every_control_step=${ROBOTWIN_EVAL_RENDER_EVERY_CONTROL_STEP:-False}"
    "env.train.task_config.data_type.pointcloud=False"
    "env.eval.task_config.data_type.pointcloud=False"
    "++env.train.offload_before_actor_handoff=${ENV_TRAIN_OFFLOAD_BEFORE_ACTOR_HANDOFF:-True}"
    "++env.train.spool_trajectories_before_actor_handoff=${ENV_TRAIN_SPOOL_TRAJECTORIES_BEFORE_ACTOR_HANDOFF:-True}"
    "++env.train.trajectory_spool_dir=${ENV_TRAIN_TRAJECTORY_SPOOL_DIR:-${LOG_DIR}/tmp/actor_handoff}"
    "++env.train.trajectory_spool_min_free_gb=${ENV_TRAIN_TRAJECTORY_SPOOL_MIN_FREE_GB:-40}"
    "env.train.video_cfg.save_video=False"
    "env.eval.total_num_envs=${EVAL_TOTAL_NUM_ENVS:-96}"
    "++env.eval.num_parallel_envs=${EVAL_NUM_PARALLEL_ENVS:-96}"
    "++env.eval.close_after_eval=${EVAL_CLOSE_AFTER_EVAL:-True}"
    "env.eval.video_cfg.save_video=False"
    "++reward.use_reward_model=${REWARD_USE_REWARD_MODEL:-True}"
    "++reward.group_name=${REWARD_GROUP_NAME:-RewardGroup}"
    "++reward.reward_mode=${PROCVLM_REWARD_MODE:-history_buffer}"
    "++reward.history_reward_assign=${PROCVLM_HISTORY_REWARD_ASSIGN:-True}"
    "++reward.enable_offload=${REWARD_ENABLE_OFFLOAD:-True}"
    "++reward.defer_init_until_after_first_weight_sync=${REWARD_DEFER_INIT_UNTIL_AFTER_FIRST_WEIGHT_SYNC:-True}"
    "++reward.async_queue.enabled=${PROCVLM_REWARD_ASYNC_QUEUE_ENABLED:-True}"
    "++reward.async_queue.max_pending=${PROCVLM_REWARD_ASYNC_MAX_PENDING:-16}"
    "++reward.async_queue.merge_max_requests=${PROCVLM_REWARD_ASYNC_MERGE_MAX_REQUESTS:-16}"
    "++reward.async_queue.merge_wait_timeout_s=${PROCVLM_REWARD_ASYNC_MERGE_WAIT_TIMEOUT_S:-0.01}"
    "++reward.reward_weight=${PROCVLM_REWARD_WEIGHT:-1.0}"
    "++reward.env_reward_weight=${PROCVLM_ENV_REWARD_WEIGHT:-0.1}"
    "++reward.model.model_path=${PROCVLM_REWARD_MODEL_PATH:-/home/fengyouhe/models/LoRA/procvlm/beat_block_hammer}"
    "++reward.model.model_type=${PROCVLM_REWARD_MODEL_TYPE:-procvlm_history_vlm}"
    "++reward.model.procvlm_repo_path=${PROCVLM_REPO_PATH}"
    "++reward.model.backend=${PROCVLM_REWARD_BACKEND:-value_head_subprocess}"
    "++reward.model.enable_value_head=${PROCVLM_REWARD_ENABLE_VALUE_HEAD:-False}"
    "++reward.model.procvlm_python=${PROCVLM_PYTHON}"
    "++reward.model.procvlm_server_script=${PROCVLM_SERVER_SCRIPT}"
    "++reward.model.precision=${PROCVLM_REWARD_PRECISION:-bf16}"
    "++reward.model.torch_dtype=${PROCVLM_REWARD_TORCH_DTYPE:-bf16}"
    "++reward.model.attn_implementation=${PROCVLM_REWARD_ATTN_IMPLEMENTATION:-sdpa}"
    "++reward.model.vllm_tp=${PROCVLM_REWARD_VLLM_TP:-1}"
    "++reward.model.vllm_engine_kwargs.dtype=${PROCVLM_REWARD_VLLM_DTYPE:-bfloat16}"
    "++reward.model.vllm_engine_kwargs.gpu_memory_utilization=${PROCVLM_REWARD_VLLM_GPU_MEMORY_UTILIZATION:-0.55}"
    "++reward.model.vllm_engine_kwargs.max_model_len=${PROCVLM_REWARD_VLLM_MAX_MODEL_LEN:-8192}"
    "++reward.model.vllm_engine_kwargs.process_workers=${PROCVLM_REWARD_VLLM_PROCESS_WORKERS:-1}"
    "++reward.model.vllm_engine_kwargs.pipeline_chunk_size=${PROCVLM_REWARD_VLLM_PIPELINE_CHUNK_SIZE:-8}"
    "++reward.model.reward_form=${PROCVLM_REWARD_FORM:-delta}"
    "++reward.model.pos_clip=${PROCVLM_REWARD_POS_CLIP:-0.2}"
    "++reward.model.neg_clip=${PROCVLM_REWARD_NEG_CLIP:-0.2}"
    "++reward.model.progress_smoothing.enabled=${PROCVLM_REWARD_PROGRESS_SMOOTHING_ENABLED:-False}"
    "++reward.model.progress_smoothing.alpha=${PROCVLM_REWARD_PROGRESS_SMOOTHING_ALPHA:-0.9}"
    "++reward.model.gt_success_bonus=${PROCVLM_REWARD_GT_SUCCESS_BONUS:-0.0}"
    "++reward.model.input_builder_name=${PROCVLM_REWARD_INPUT_BUILDER_NAME:-procvlm_progress_input_builder}"
    "++reward.model.input_builder_params.history_buffer_name=${PROCVLM_RM_HISTORY_BUFFER_NAME:-history_window}"
    "++reward.model.input_builder_params.image_keys=${PROCVLM_RM_IMAGE_KEYS}"
    "++reward.model.input_builder_params.default_task_description='${TASK_RM_DESCRIPTION}'"
    "++reward.model.input_builder_params.force_default_task_description=${PROCVLM_RM_FORCE_DEFAULT_TASK_DESCRIPTION:-True}"
    "++reward.model.reward_parser_name=${PROCVLM_REWARD_PARSER_NAME:-procvlm_progress_reward_parser}"
    "++reward.model.reward_parser_params.normalize=${PROCVLM_REWARD_PARSER_NORMALIZE:-True}"
    "++reward.model.reward_parser_params.invalid_progress=${PROCVLM_REWARD_PARSER_INVALID_PROGRESS:-0.0}"
    "++reward.model.history_buffers.history_window.history_size=${PROCVLM_RM_HISTORY_SIZE:-1}"
    "++reward.model.history_buffers.history_window.min_history_size=${PROCVLM_RM_MIN_HISTORY_SIZE:-1}"
    "++reward.model.history_buffers.history_window.input_interval=${PROCVLM_RM_HISTORY_INPUT_INTERVAL:-1}"
    "++reward.model.history_buffers.history_window.history_keys=${PROCVLM_RM_HISTORY_KEYS}"
    "++reward.model.history_buffers.history_window.input_on_done=${PROCVLM_RM_HISTORY_INPUT_ON_DONE:-True}"
    "++reward.model.interval_reward=${PROCVLM_REWARD_INTERVAL_REWARD:-0.0}"
    "++reward.model.infer_micro_batch_size=${PROCVLM_REWARD_INFER_MICRO_BATCH_SIZE:-0}"
    "++reward.model.max_new_tokens=${PROCVLM_REWARD_MAX_NEW_TOKENS:-128}"
    "++reward.model.do_sample=${PROCVLM_REWARD_DO_SAMPLE:-False}"
    "++reward.model.temperature=${PROCVLM_REWARD_TEMPERATURE:-0.0}"
    "++reward.model.eager_init=${PROCVLM_REWARD_EAGER_INIT:-True}"
    "++reward.model.warmup_on_init=${PROCVLM_REWARD_WARMUP_ON_INIT:-True}"
    "++reward.model.base_cost_mb=${PROCVLM_REWARD_BASE_COST_MB:-100}"
    "++reward.model.image_cost_mb=${PROCVLM_REWARD_IMAGE_COST_MB:-75}"
    "++reward.model.show_progress=${PROCVLM_REWARD_SHOW_PROGRESS:-False}"
    "++reward.model.debug_samples.enabled=${PROCVLM_RM_DEBUG_SAMPLES_ENABLED:-True}"
    "++reward.model.debug_samples.every_n_steps=${PROCVLM_RM_DEBUG_SAMPLES_EVERY_N_STEPS:-10}"
    "++reward.model.debug_samples.max_samples_per_step=${PROCVLM_RM_DEBUG_SAMPLES_MAX_PER_STEP:-1}"
    "++reward.model.debug_samples.output_dir=${PROCVLM_RM_DEBUG_SAMPLES_OUTPUT_DIR:-${LOG_DIR}/rm_samples}"
    "++reward.model.debug_samples.save_format=${PROCVLM_RM_DEBUG_SAMPLES_SAVE_FORMAT:-gif}"
    "++reward.model.debug_samples.fps=${PROCVLM_RM_DEBUG_SAMPLES_FPS:-2}"
    "++reward.model.debug_samples.include_prompt=${PROCVLM_RM_DEBUG_SAMPLES_INCLUDE_PROMPT:-True}"
    "++reward.model.debug_samples.trajectory_video.enabled=${PROCVLM_RM_TRAJECTORY_VIDEO_ENABLED:-True}"
    "++reward.model.debug_samples.trajectory_video.every_n_steps=${PROCVLM_RM_TRAJECTORY_VIDEO_EVERY_N_STEPS:-1}"
    "++reward.model.debug_samples.trajectory_video.output_dir=${PROCVLM_RM_TRAJECTORY_VIDEO_OUTPUT_DIR:-${LOG_DIR}/rm_rollout_videos}"
    "++reward.model.debug_samples.trajectory_video.target_reward_rank=${PROCVLM_RM_TRAJECTORY_VIDEO_TARGET_REWARD_RANK:-0}"
    "++reward.model.debug_samples.trajectory_video.target_env_id=${PROCVLM_RM_TRAJECTORY_VIDEO_TARGET_ENV_ID:-0}"
    "++reward.model.debug_samples.trajectory_video.fps=${PROCVLM_RM_TRAJECTORY_VIDEO_FPS:-4}"
    "++reward.model.debug_samples.trajectory_video.max_frames=${PROCVLM_RM_TRAJECTORY_VIDEO_MAX_FRAMES:-${TASK_MAX_STEPS}}"
    "++reward.model.debug_samples.trajectory_video.finalize_after_points=${PROCVLM_RM_TRAJECTORY_VIDEO_FINALIZE_AFTER_POINTS:-0}"
    "++reward.model.debug_samples.trajectory_video.min_width=${PROCVLM_RM_TRAJECTORY_VIDEO_MIN_WIDTH:-960}"
    "++reward.model.debug_samples.trajectory_video.min_height=${PROCVLM_RM_TRAJECTORY_VIDEO_MIN_HEIGHT:-540}"
    "++reward.model.debug_samples.trajectory_video.include_prompt=${PROCVLM_RM_TRAJECTORY_VIDEO_INCLUDE_PROMPT:-True}"
    "++reward.model.debug_samples.trajectory_video.include_raw_output=${PROCVLM_RM_TRAJECTORY_VIDEO_INCLUDE_RAW_OUTPUT:-True}"
)

if [[ -n "${DEFAULT_ACTION_EXEC_HORIZON}" ]]; then
    DEFAULT_OVERRIDES+=(
        "++env.train.action_exec_horizon=${DEFAULT_ACTION_EXEC_HORIZON}"
        "++env.train.task_config.rdt_step=${DEFAULT_ACTION_EXEC_HORIZON}"
    )
fi

if [[ -n "${DEFAULT_EVAL_ACTION_EXEC_HORIZON}" ]]; then
    DEFAULT_OVERRIDES+=(
        "++env.eval.action_exec_horizon=${DEFAULT_EVAL_ACTION_EXEC_HORIZON}"
        "++env.eval.task_config.rdt_step=${DEFAULT_EVAL_ACTION_EXEC_HORIZON}"
    )
fi

if [[ -n "${RESOLVED_PROCVLM_REWARD_ROUTING_WEIGHTS}" ]]; then
    DEFAULT_OVERRIDES+=(
        "++reward.routing_weights=[${RESOLVED_PROCVLM_REWARD_ROUTING_WEIGHTS}]"
    )
fi

if [[ -n "${TASK_NAME}" ]]; then
    DEFAULT_OVERRIDES+=(
        "env.train.task_config.task_name=${TASK_NAME}"
        "env.eval.task_config.task_name=${TASK_NAME}"
    )
fi

strip_hydra_quotes() {
    local value="$1"
    value="${value#\"}"
    value="${value%\"}"
    value="${value#\'}"
    value="${value%\'}"
    printf '%s' "${value}"
}

EXPERIMENT_NAME="${EXPERIMENT_NAME:-${CONFIG_NAME}}"
for override in "${DEFAULT_OVERRIDES[@]}" "${EXTRA_ARGS[@]}"; do
    case "${override}" in
        runner.logger.experiment_name=*)
            EXPERIMENT_NAME="$(strip_hydra_quotes "${override#runner.logger.experiment_name=}")"
            ;;
    esac
done

MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
rm -rf -- "${LOG_DIR}"
mkdir -p "${LOG_DIR}"

cleanup_latest_checkpoints() {
    [[ "${KEEP_LATEST_CHECKPOINT:-1}" == "1" ]] || return 0

    local checkpoint_root="${CHECKPOINT_ROOT:-${LOG_DIR}/${EXPERIMENT_NAME}/checkpoints}"
    [[ -d "${checkpoint_root}" ]] || return 0

    local dirs=()
    local dir base step keep_latest best_step=""
    keep_latest="${RUNNER_KEEP_LATEST_CHECKPOINTS:-3}"

    shopt -s nullglob
    dirs=("${checkpoint_root}"/global_step_*)
    shopt -u nullglob

    ((${#dirs[@]} > keep_latest)) || return 0

    if [[ -f "${checkpoint_root}/best_success.json" ]]; then
        best_step="$(
            "${PYTHON_BIN}" - "${checkpoint_root}/best_success.json" <<'PY' || true
import json
import sys

try:
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)
    step = data.get("step", "")
    if step != "":
        print(int(step))
except Exception:
    pass
PY
        )"
    fi

    IFS=$'\n' dirs=($(
        for dir in "${dirs[@]}"; do
            [[ -d "${dir}" ]] || continue
            base="$(basename "${dir}")"
            step="${base#global_step_}"
            [[ "${step}" =~ ^[0-9]+$ ]] || continue
            printf '%012d %s\n' "${step}" "${dir}"
        done | sort -rn | awk '{print $2}'
    ))
    unset IFS

    local kept=0
    for dir in "${dirs[@]}"; do
        [[ -d "${dir}" ]] || continue
        base="$(basename "${dir}")"
        step="${base#global_step_}"
        [[ "${step}" =~ ^[0-9]+$ ]] || continue
        if [[ -n "${best_step}" && "${step}" == "${best_step}" ]]; then
            continue
        fi
        if ((kept < keep_latest)); then
            kept=$((kept + 1))
            continue
        fi
        rm -rf -- "${dir}"
    done

    printf 'Kept latest %s checkpoints plus best_success if present under %s\n' \
        "${keep_latest}" "${checkpoint_root}" | tee -a "${MEGA_LOG_FILE}" >/dev/null || true
}

on_exit() {
    local status=$?
    cleanup_latest_checkpoints || true
    return "${status}"
}
trap on_exit EXIT

filter_training_log_stream() {
    "${PYTHON_BIN}" -u /dev/fd/3 3<<'PY'
import codecs
import os
import re
import sys


def enabled(name: str, default: str = "1") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value not in {"0", "false", "off", "no", "none", "disabled"}


strip_ansi = enabled("ROBOTWIN_LOG_FILTER_STRIP_ANSI")
drop_step_progress = enabled("ROBOTWIN_LOG_FILTER_DROP_STEP_PROGRESS")
drop_blank_lines = enabled("ROBOTWIN_LOG_FILTER_DROP_BLANK_LINES")
drop_empty_worker_lines = enabled("ROBOTWIN_LOG_FILTER_DROP_EMPTY_WORKER_LINES")

ansi_re = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
worker_prefix_re = re.compile(r"^\(.* pid=\d+\)\s*")
ray_repeated_re = re.compile(r"\[repeated \d+x across cluster\]")
step_progress_re = re.compile(r"step:\s*\d+\s*/\s*\d+")
inline_step_noise_re = re.compile(
    r"(?:\([^\n]*? pid=\d+\)\s*)?(?:step:\s*\d+\s*/\s*\d+\s*)+(?:\[repeated \d+x across cluster\]\s*)?"
)


def should_drop(line: str) -> bool:
    text = line.strip()
    if drop_blank_lines and not text:
        return True
    if not text:
        return False

    text_without_prefix = worker_prefix_re.sub("", text).strip()
    if drop_empty_worker_lines and not text_without_prefix:
        return True

    if not drop_step_progress:
        return False

    normalized = ray_repeated_re.sub("", text_without_prefix)
    normalized = step_progress_re.sub("", normalized)
    normalized = normalized.strip()
    return not normalized and bool(step_progress_re.search(text_without_prefix))


def sanitize_line(line: str) -> str:
    if drop_step_progress:
        line = inline_step_noise_re.sub("", line)
    return line


def emit(line: str) -> None:
    if strip_ansi:
        line = ansi_re.sub("", line)
    line = sanitize_line(line)
    if should_drop(line):
        return
    sys.stdout.write(line + "\n")


decoder = codecs.getincrementaldecoder("utf-8")("replace")
buffer = ""

while True:
    chunk = sys.stdin.buffer.read(8192)
    if not chunk:
        break
    buffer += decoder.decode(chunk)
    buffer = buffer.replace("\r", "\n")
    while "\n" in buffer:
        line, buffer = buffer.split("\n", 1)
        emit(line)

buffer += decoder.decode(b"", final=True).replace("\r", "\n")
if buffer:
    for line in buffer.split("\n"):
        if line:
            emit(line)
sys.stdout.flush()
PY
}

echo "Evaluation Mode: RoboTwin"
echo "Using ROBOT_PLATFORM=${ROBOT_PLATFORM}"
echo "Using ROBOTWIN_PATH=${ROBOTWIN_PATH}"
echo "Using ROBOTWIN_ASSETS_PATH=${ROBOTWIN_ASSETS_PATH}"
echo "Using CONFIG_NAME=${CONFIG_NAME}"
echo "Using TASK_NAME=${TASK_NAME:-<derived-by-config>}"
echo "Using TASK_SFT_MODEL=${TASK_SFT_MODEL}"
echo "Using TASK_RM_DESCRIPTION=${TASK_RM_DESCRIPTION:-<empty>}"
echo "Using PROCVLM_REWARD_BACKEND=${PROCVLM_REWARD_BACKEND:-value_head_subprocess}"
echo "Using PROCVLM_REWARD_ENABLE_VALUE_HEAD=${PROCVLM_REWARD_ENABLE_VALUE_HEAD:-False}"
echo "Using PROCVLM_REWARD_ROUTING_WEIGHTS=${RESOLVED_PROCVLM_REWARD_ROUTING_WEIGHTS:-<uniform>}"
echo "Using LOG_DIR=${LOG_DIR}"
echo "Using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Using Python at $(which python)"

CMD=(
    python "${SRC_FILE}"
    --config-path "${CONFIG_PATH}"
    --config-name "${CONFIG_NAME}"
    "runner.logger.log_path=${LOG_DIR}"
    "${DEFAULT_OVERRIDES[@]}"
    "${EXTRA_ARGS[@]}"
)

: > "${MEGA_LOG_FILE}"
printf '%q ' "${CMD[@]}" | tee -a "${MEGA_LOG_FILE}"
printf '\n' | tee -a "${MEGA_LOG_FILE}"
if [[ "${ROBOTWIN_LOG_FILTER_ENABLED}" == "0" || "${ROBOTWIN_LOG_FILTER_ENABLED,,}" == "false" || "${ROBOTWIN_LOG_FILTER_ENABLED,,}" == "off" ]]; then
    "${CMD[@]}" 2>&1 | tee -a "${MEGA_LOG_FILE}"
else
    "${CMD[@]}" 2>&1 | filter_training_log_stream | tee -a "${MEGA_LOG_FILE}"
fi
