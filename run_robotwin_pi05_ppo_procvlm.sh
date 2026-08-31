#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

source .venv/bin/activate

export EMBODIED_PATH="${SCRIPT_DIR}/examples/embodiment"
export REPO_PATH="${SCRIPT_DIR}"
export RLINF_TRAIN_ENTRY="${RLINF_TRAIN_ENTRY:-sync}"
case "${RLINF_TRAIN_ENTRY}" in
    sync)
        export SRC_FILE="${SRC_FILE:-${EMBODIED_PATH}/train_embodied_agent.py}"
        ;;
    async|async_ppo)
        export SRC_FILE="${SRC_FILE:-${EMBODIED_PATH}/train_async.py}"
        ;;
    *)
        export SRC_FILE="${SRC_FILE:-${RLINF_TRAIN_ENTRY}}"
        ;;
esac
export CONFIG_PATH="${CONFIG_PATH:-${EMBODIED_PATH}/config}"
export PROCVLM_REPO_PATH="${PROCVLM_REPO_PATH:-${SCRIPT_DIR}/third_party/ProcVLM}"
export ROBOTWIN_PATH="${ROBOTWIN_PATH:-${SCRIPT_DIR}/third_party/RoboTwin}"
export ROBOTWIN_ASSETS_PATH="${ROBOTWIN_ASSETS_PATH:-${ROBOTWIN_PATH}}"
export PI05_BASE_MODEL="${PI05_BASE_MODEL:-${SCRIPT_DIR}/models/pi05_base_vanilla_safetensors}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${SCRIPT_DIR}/.hf-cache}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${SCRIPT_DIR}/.cache/openpi}"
: "${PROCVLM_PYTHON:?Set PROCVLM_PYTHON to the Python executable of a ProcVLM-compatible environment}"
export PROCVLM_PYTHON
export PROCVLM_SERVER_SCRIPT="${PROCVLM_SERVER_SCRIPT:-${SCRIPT_DIR}/rlinf/models/embodiment/reward/procvlm_reward_server.py}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
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
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export VLLM_DO_NOT_TRACK="${VLLM_DO_NOT_TRACK:-1}"
export DO_NOT_TRACK="${DO_NOT_TRACK:-1}"
export ROBOT_PLATFORM="${ROBOT_PLATFORM:-ALOHA}"
export ROBOTWIN_VECTOR_ENV_MAX_WORKERS="${ROBOTWIN_VECTOR_ENV_MAX_WORKERS:-1}"
export ROBOTWIN_SERIALIZE_SCENE_STEP="${ROBOTWIN_SERIALIZE_SCENE_STEP:-1}"
export ROBOTWIN_ACTION_GUARD_ENABLED="${ROBOTWIN_ACTION_GUARD_ENABLED:-1}"
export ROBOTWIN_ACTION_STATS_PATH="${ROBOTWIN_ACTION_STATS_PATH:-${PI05_BASE_MODEL}/physical-intelligence/robotwin/norm_stats.json}"
export ROBOTWIN_ACTION_GUARD_DELTA_SCALE="${ROBOTWIN_ACTION_GUARD_DELTA_SCALE:-1.0}"
export ROBOTWIN_ACTION_GUARD_ABS_LIMIT="${ROBOTWIN_ACTION_GUARD_ABS_LIMIT:-3.2}"
export ROBOTWIN_ACTION_GUARD_LOG_EVERY_N="${ROBOTWIN_ACTION_GUARD_LOG_EVERY_N:-25}"
export PYTHONPATH="${REPO_PATH}:${ROBOTWIN_PATH}:${PROCVLM_REPO_PATH}:${PYTHONPATH:-}"

CONFIG_NAME="${1:-${CONFIG_NAME:-robotwin_put_bottles_dustbin_ppo_openpi_pi05_procvlm_reward}}"
if (($# > 1)); then
    EXTRA_ARGS=("${@:2}")
else
    EXTRA_ARGS=()
fi

case "${CONFIG_NAME}" in
    *handover_block*)
        TASK_NAME="${ROBOTWIN_TASK_NAME:-handover_block}"
        TASK_MAX_STEPS="${ROBOTWIN_TASK_MAX_STEPS:-800}"
        TASK_RM_DESCRIPTION="${ROBOTWIN_TASK_DESCRIPTION:-use the left arm to grasp the red block on the table, handover it to the right arm and place it on the blue pad}"
        ;;
    *beat_block_hammer*)
        TASK_NAME="${ROBOTWIN_TASK_NAME:-beat_block_hammer}"
        TASK_MAX_STEPS="${ROBOTWIN_TASK_MAX_STEPS:-400}"
        TASK_RM_DESCRIPTION="${ROBOTWIN_TASK_DESCRIPTION:-there is a hammer and a block on the table, use the arm to grab the hammer and beat the block}"
        ;;
    *pick_dual_bottles*)
        TASK_NAME="${ROBOTWIN_TASK_NAME:-pick_dual_bottles}"
        TASK_MAX_STEPS="${ROBOTWIN_TASK_MAX_STEPS:-400}"
        TASK_RM_DESCRIPTION="${ROBOTWIN_TASK_DESCRIPTION:-pick up one bottle with one arm, and pick up another bottle with the other arm}"
        ;;
    *place_container_plate*)
        TASK_NAME="${ROBOTWIN_TASK_NAME:-place_container_plate}"
        TASK_MAX_STEPS="${ROBOTWIN_TASK_MAX_STEPS:-150}"
        TASK_RM_DESCRIPTION="${ROBOTWIN_TASK_DESCRIPTION:-place the container onto the plate}"
        ;;
    *move_can_pot*)
        TASK_NAME="${ROBOTWIN_TASK_NAME:-move_can_pot}"
        TASK_MAX_STEPS="${ROBOTWIN_TASK_MAX_STEPS:-200}"
        TASK_RM_DESCRIPTION="${ROBOTWIN_TASK_DESCRIPTION:-there is a can and a pot on the table, use one arm to pick up the can and move it to beside the pot}"
        ;;
    *put_bottles_dustbin*)
        TASK_NAME="${ROBOTWIN_TASK_NAME:-put_bottles_dustbin}"
        TASK_MAX_STEPS="${ROBOTWIN_TASK_MAX_STEPS:-1700}"
        TASK_RM_DESCRIPTION="${ROBOTWIN_TASK_DESCRIPTION:-use arms to grab the bottles and put them into the dustbin to the left of the table}"
        ;;
    *blocks_ranking_rgb*)
        TASK_NAME="${ROBOTWIN_TASK_NAME:-blocks_ranking_rgb}"
        TASK_MAX_STEPS="${ROBOTWIN_TASK_MAX_STEPS:-800}"
        TASK_RM_DESCRIPTION="${ROBOTWIN_TASK_DESCRIPTION:-<Place> the red block, green block, and blue block <in the order> of red, green, and blue from left to right, <placing in a row>.}"
        ;;
    *stack_bowls_two*)
        TASK_NAME="${ROBOTWIN_TASK_NAME:-stack_bowls_two}"
        TASK_MAX_STEPS="${ROBOTWIN_TASK_MAX_STEPS:-900}"
        TASK_RM_DESCRIPTION="${ROBOTWIN_TASK_DESCRIPTION:-pick up two bowls and stack them one on top of the other}"
        ;;
    *)
        TASK_NAME="${ROBOTWIN_TASK_NAME:-}"
        TASK_MAX_STEPS="${ROBOTWIN_TASK_MAX_STEPS:-200}"
        TASK_RM_DESCRIPTION="${ROBOTWIN_TASK_DESCRIPTION:-}"
        ;;
esac

if [[ -z "${ROBOTWIN_TASK_MAX_STEPS:-}" ]]; then
    case "${TASK_NAME}" in
        handover_block) TASK_MAX_STEPS="800" ;;
        beat_block_hammer) TASK_MAX_STEPS="400" ;;
        pick_dual_bottles) TASK_MAX_STEPS="400" ;;
        place_container_plate) TASK_MAX_STEPS="150" ;;
        move_can_pot) TASK_MAX_STEPS="200" ;;
        put_bottles_dustbin) TASK_MAX_STEPS="1700" ;;
        blocks_ranking_rgb) TASK_MAX_STEPS="800" ;;
        stack_bowls_two) TASK_MAX_STEPS="900" ;;
    esac
fi

if [[ -z "${ROBOTWIN_TASK_DESCRIPTION:-}" ]]; then
    case "${TASK_NAME}" in
        handover_block)
            TASK_RM_DESCRIPTION="use the left arm to grasp the red block on the table, handover it to the right arm and place it on the blue pad"
            ;;
        beat_block_hammer)
            TASK_RM_DESCRIPTION="there is a hammer and a block on the table, use the arm to grab the hammer and beat the block"
            ;;
        pick_dual_bottles)
            TASK_RM_DESCRIPTION="pick up one bottle with one arm, and pick up another bottle with the other arm"
            ;;
        place_container_plate)
            TASK_RM_DESCRIPTION="place the container onto the plate"
            ;;
        move_can_pot)
            TASK_RM_DESCRIPTION="there is a can and a pot on the table, use one arm to pick up the can and move it to beside the pot"
            ;;
        put_bottles_dustbin)
            TASK_RM_DESCRIPTION="use arms to grab the bottles and put them into the dustbin to the left of the table"
            ;;
        blocks_ranking_rgb)
            TASK_RM_DESCRIPTION="<Place> the red block, green block, and blue block <in the order> of red, green, and blue from left to right, <placing in a row>."
            ;;
        stack_bowls_two)
            TASK_RM_DESCRIPTION="pick up two bowls and stack them one on top of the other"
            ;;
    esac
fi

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
fi

export LOG_DIR="${LOG_DIR:-/data/fengyouhe/rlinf_logs/robotwin_pi05_procvlm_active}"
GROUP_SIZE="${GROUP_SIZE:-1}"
case "${RLINF_TRAIN_ENTRY}" in
    async|async_ppo)
        # Async workers keep env/rollout/reward resident at the same time.  With
        # RoboTwin this is much more memory hungry than LIBERO, so keep the
        # default resident env count conservative and use the phased sync path
        # for the main high-throughput run.
        DEFAULT_TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-64}"
        DEFAULT_PIPELINE_STAGE_NUM="${ROLLOUT_PIPELINE_STAGE_NUM:-1}"
        DEFAULT_ROLLOUT_EPOCH="${ROLLOUT_EPOCH:-4}"
        DEFAULT_UPDATE_EPOCH="${UPDATE_EPOCH:-2}"
        DEFAULT_RUNNER_VAL_CHECK_INTERVAL="${RUNNER_VAL_CHECK_INTERVAL:-10}"
        DEFAULT_RUNNER_MAX_STEPS="${RUNNER_MAX_STEPS:-150}"
        DEFAULT_RUNNER_SAVE_INTERVAL="${RUNNER_SAVE_INTERVAL:-50}"
        DEFAULT_RLINF_LOSS_TYPE="${RLINF_LOSS_TYPE:-decoupled_actor_critic}"
        DEFAULT_ACTOR_GLOBAL_BATCH_SIZE="${ACTOR_GLOBAL_BATCH_SIZE:-4096}"
        DEFAULT_ACTOR_MICRO_BATCH_SIZE="${ACTOR_MICRO_BATCH_SIZE:-32}"
        DEFAULT_ACTOR_LR="${ACTOR_LR:-1.0e-5}"
        DEFAULT_ACTOR_SYNC_WEIGHT_NO_WAIT="${ACTOR_SYNC_WEIGHT_NO_WAIT:-True}"
        DEFAULT_ROLLOUT_ENABLE_OFFLOAD="${ROLLOUT_ENABLE_OFFLOAD:-False}"
        DEFAULT_REWARD_ENABLE_OFFLOAD="${REWARD_ENABLE_OFFLOAD:-False}"
        DEFAULT_ENV_TRAIN_OFFLOAD_BEFORE_ACTOR_HANDOFF="${ENV_TRAIN_OFFLOAD_BEFORE_ACTOR_HANDOFF:-False}"
        DEFAULT_ENV_TRAIN_SPOOL_TRAJECTORIES_BEFORE_ACTOR_HANDOFF="${ENV_TRAIN_SPOOL_TRAJECTORIES_BEFORE_ACTOR_HANDOFF:-False}"
        ;;
    *)
        DEFAULT_TRAIN_NUM_ENVS="${TRAIN_NUM_ENVS:-64}"
        DEFAULT_PIPELINE_STAGE_NUM="${ROLLOUT_PIPELINE_STAGE_NUM:-2}"
        DEFAULT_ROLLOUT_EPOCH="${ROLLOUT_EPOCH:-4}"
        DEFAULT_UPDATE_EPOCH="${UPDATE_EPOCH:-2}"
        DEFAULT_RUNNER_VAL_CHECK_INTERVAL="${RUNNER_VAL_CHECK_INTERVAL:-10}"
        DEFAULT_RUNNER_MAX_STEPS="${RUNNER_MAX_STEPS:-150}"
        DEFAULT_RUNNER_SAVE_INTERVAL="${RUNNER_SAVE_INTERVAL:-50}"
        DEFAULT_RLINF_LOSS_TYPE="${RLINF_LOSS_TYPE:-actor_critic}"
        DEFAULT_ACTOR_GLOBAL_BATCH_SIZE="${ACTOR_GLOBAL_BATCH_SIZE:-4096}"
        DEFAULT_ACTOR_MICRO_BATCH_SIZE="${ACTOR_MICRO_BATCH_SIZE:-32}"
        DEFAULT_ACTOR_LR="${ACTOR_LR:-1.0e-5}"
        DEFAULT_ACTOR_SYNC_WEIGHT_NO_WAIT="${ACTOR_SYNC_WEIGHT_NO_WAIT:-False}"
        DEFAULT_ROLLOUT_ENABLE_OFFLOAD="${ROLLOUT_ENABLE_OFFLOAD:-True}"
        DEFAULT_REWARD_ENABLE_OFFLOAD="${REWARD_ENABLE_OFFLOAD:-True}"
        DEFAULT_ENV_TRAIN_OFFLOAD_BEFORE_ACTOR_HANDOFF="${ENV_TRAIN_OFFLOAD_BEFORE_ACTOR_HANDOFF:-True}"
        DEFAULT_ENV_TRAIN_SPOOL_TRAJECTORIES_BEFORE_ACTOR_HANDOFF="${ENV_TRAIN_SPOOL_TRAJECTORIES_BEFORE_ACTOR_HANDOFF:-True}"
        ;;
esac
DEFAULT_EVAL_TOTAL_NUM_ENVS="${EVAL_TOTAL_NUM_ENVS:-${DEFAULT_TRAIN_NUM_ENVS}}"
DEFAULT_EVAL_NUM_PARALLEL_ENVS="${EVAL_NUM_PARALLEL_ENVS:-${DEFAULT_EVAL_TOTAL_NUM_ENVS}}"
DEFAULT_TRAJECTORY_VIDEO_FINALIZE_AFTER_POINTS="${PROCVLM_RM_TRAJECTORY_VIDEO_FINALIZE_AFTER_POINTS:-16}"
DEFAULT_REWARD_ASYNC_MAX_PENDING="${PROCVLM_REWARD_ASYNC_MAX_PENDING:-16}"
DEFAULT_REWARD_ASYNC_MERGE_MAX_REQUESTS="${PROCVLM_REWARD_ASYNC_MERGE_MAX_REQUESTS:-32}"
DEFAULT_REWARD_ASYNC_MERGE_WAIT_TIMEOUT_S="${PROCVLM_REWARD_ASYNC_MERGE_WAIT_TIMEOUT_S:-0.10}"
DEFAULT_RM_HISTORY_SIZE="${PROCVLM_RM_HISTORY_SIZE:-8}"
DEFAULT_RM_MIN_HISTORY_SIZE="${PROCVLM_RM_MIN_HISTORY_SIZE:-2}"
DEFAULT_RM_HISTORY_INPUT_INTERVAL="${PROCVLM_RM_HISTORY_INPUT_INTERVAL:-2}"

DEFAULT_OVERRIDES=(
    "runner.val_check_interval=${DEFAULT_RUNNER_VAL_CHECK_INTERVAL}"
    "runner.max_steps=${DEFAULT_RUNNER_MAX_STEPS}"
    "runner.save_interval=${DEFAULT_RUNNER_SAVE_INTERVAL}"
    "++runner.pause_reward_during_actor_training=${RUNNER_PAUSE_REWARD_DURING_ACTOR_TRAINING:-False}"
    "++runner.reward_pause_timeout_s=${RUNNER_REWARD_PAUSE_TIMEOUT_S:-2.0}"
    'runner.logger.logger_backends=["tensorboard","wandb"]'
    "algorithm.rollout_epoch=${DEFAULT_ROLLOUT_EPOCH}"
    "algorithm.eval_rollout_epoch=${EVAL_ROLLOUT_EPOCH:-1}"
    "algorithm.update_epoch=${DEFAULT_UPDATE_EPOCH}"
    "algorithm.group_size=${GROUP_SIZE}"
    "++algorithm.staleness_threshold=${RLINF_STALENESS_THRESHOLD:-1}"
    "++algorithm.behave_weight_threshold=${RLINF_BEHAVE_WEIGHT_THRESHOLD:-2.0}"
    "algorithm.loss_type=${DEFAULT_RLINF_LOSS_TYPE}"
    "actor.global_batch_size=${DEFAULT_ACTOR_GLOBAL_BATCH_SIZE}"
    "actor.micro_batch_size=${DEFAULT_ACTOR_MICRO_BATCH_SIZE}"
    "actor.optim.lr=${DEFAULT_ACTOR_LR}"
    "actor.enable_offload=${ACTOR_ENABLE_OFFLOAD:-True}"
    "actor.offload_optimizer=${ACTOR_OFFLOAD_OPTIMIZER:-True}"
    "++actor.sync_weight_no_wait=${DEFAULT_ACTOR_SYNC_WEIGHT_NO_WAIT}"
    "actor.fsdp_config.gradient_checkpointing=False"
    "actor.fsdp_config.enable_gradient_accumulation=${ACTOR_ENABLE_GRADIENT_ACCUMULATION:-False}"
    "actor.fsdp_config.limit_all_gathers=${ACTOR_LIMIT_ALL_GATHERS:-True}"
    "actor.model.model_path=${PI05_BASE_MODEL}"
    "rollout.model.model_path=${PI05_BASE_MODEL}"
    "rollout.enable_offload=${DEFAULT_ROLLOUT_ENABLE_OFFLOAD}"
    "rollout.pipeline_stage_num=${DEFAULT_PIPELINE_STAGE_NUM}"
    "++rollout.batch_pipeline_stages=${ROLLOUT_BATCH_PIPELINE_STAGES:-False}"
    "cluster.component_placement.actor=${COMPONENT_PLACEMENT_ACTOR:-0-3}"
    "cluster.component_placement.rollout=${COMPONENT_PLACEMENT_ROLLOUT:-0-3}"
    "cluster.component_placement.env=${COMPONENT_PLACEMENT_ENV:-0-3}"
    "cluster.component_placement.reward=${COMPONENT_PLACEMENT_REWARD:-0-1}"
    "env.enable_offload=${ENV_ENABLE_OFFLOAD:-False}"
    "env.train.total_num_envs=${DEFAULT_TRAIN_NUM_ENVS}"
    "env.train.group_size=${ENV_TRAIN_GROUP_SIZE:-${GROUP_SIZE}}"
    "env.train.assets_path=${ROBOTWIN_ASSETS_PATH}"
    "env.eval.assets_path=${ROBOTWIN_ASSETS_PATH}"
    "env.train.max_steps_per_rollout_epoch=${TASK_MAX_STEPS}"
    "env.train.max_episode_steps=${TASK_MAX_STEPS}"
    "env.eval.max_steps_per_rollout_epoch=${TASK_MAX_STEPS}"
    "env.eval.max_episode_steps=${TASK_MAX_STEPS}"
    "env.train.task_config.step_lim=${TASK_MAX_STEPS}"
    "env.eval.task_config.step_lim=${TASK_MAX_STEPS}"
    "env.train.task_config.planner_backend=${ROBOTWIN_PLANNER_BACKEND:-mplib}"
    "env.eval.task_config.planner_backend=${ROBOTWIN_PLANNER_BACKEND:-mplib}"
    "env.train.task_config.data_type.pointcloud=False"
    "env.eval.task_config.data_type.pointcloud=False"
    "++env.train.enable_offload=${ENV_TRAIN_ENABLE_OFFLOAD:-False}"
    "++env.eval.enable_offload=${ENV_EVAL_ENABLE_OFFLOAD:-False}"
    "++env.train.offload_before_actor_handoff=${DEFAULT_ENV_TRAIN_OFFLOAD_BEFORE_ACTOR_HANDOFF}"
    "++env.train.spool_trajectories_before_actor_handoff=${DEFAULT_ENV_TRAIN_SPOOL_TRAJECTORIES_BEFORE_ACTOR_HANDOFF}"
    "++env.train.trajectory_spool_dir=${ENV_TRAIN_TRAJECTORY_SPOOL_DIR:-${LOG_DIR}/tmp/actor_handoff}"
    "++env.train.trajectory_spool_min_free_gb=${ENV_TRAIN_TRAJECTORY_SPOOL_MIN_FREE_GB:-40}"
    "env.train.video_cfg.save_video=False"
    "env.eval.total_num_envs=${DEFAULT_EVAL_TOTAL_NUM_ENVS}"
    "++env.eval.num_parallel_envs=${DEFAULT_EVAL_NUM_PARALLEL_ENVS}"
    "env.eval.close_after_eval=${EVAL_CLOSE_AFTER_EVAL:-True}"
    "env.eval.video_cfg.save_video=False"
    "reward.enable_offload=${DEFAULT_REWARD_ENABLE_OFFLOAD}"
    "++reward.offload_during_actor_training=${REWARD_OFFLOAD_DURING_ACTOR_TRAINING:-True}"
    "reward.defer_init_until_after_first_weight_sync=${REWARD_DEFER_INIT_UNTIL_AFTER_FIRST_WEIGHT_SYNC:-True}"
    "reward.async_queue.enabled=${PROCVLM_REWARD_ASYNC_QUEUE_ENABLED:-True}"
    "reward.async_queue.max_pending=${DEFAULT_REWARD_ASYNC_MAX_PENDING}"
    "reward.async_queue.merge_max_requests=${DEFAULT_REWARD_ASYNC_MERGE_MAX_REQUESTS}"
    "++reward.async_queue.merge_wait_timeout_s=${DEFAULT_REWARD_ASYNC_MERGE_WAIT_TIMEOUT_S}"
    "++reward.async_queue.merge_wait_poll_s=${PROCVLM_REWARD_ASYNC_MERGE_WAIT_POLL_S:-0.002}"
    "reward.reward_weight=${PROCVLM_REWARD_WEIGHT:-1.0}"
    "reward.env_reward_weight=${PROCVLM_ENV_REWARD_WEIGHT:-0.1}"
    "reward.model.model_path=${PROCVLM_REWARD_MODEL_PATH:-/home/fengyouhe/models/frosty-moonshine-bf16}"
    "reward.model.procvlm_repo_path=${PROCVLM_REPO_PATH}"
    "reward.model.backend=${PROCVLM_REWARD_BACKEND:-value_head_subprocess}"
    "reward.model.enable_value_head=${PROCVLM_REWARD_ENABLE_VALUE_HEAD:-False}"
    "reward.model.procvlm_python=${PROCVLM_PYTHON}"
    "reward.model.procvlm_server_script=${PROCVLM_SERVER_SCRIPT}"
    "reward.model.attn_implementation=${PROCVLM_REWARD_ATTN_IMPLEMENTATION:-sdpa}"
    "reward.model.infer_micro_batch_size=${PROCVLM_REWARD_INFER_MICRO_BATCH_SIZE:-0}"
    "reward.model.eager_init=${PROCVLM_REWARD_EAGER_INIT:-True}"
    "reward.model.warmup_on_init=${PROCVLM_REWARD_WARMUP_ON_INIT:-True}"
    "reward.model.progress_smoothing.enabled=${PROCVLM_REWARD_PROGRESS_SMOOTHING_ENABLED:-True}"
    "reward.model.progress_smoothing.alpha=${PROCVLM_REWARD_PROGRESS_SMOOTHING_ALPHA:-0.6}"
    "reward.model.input_builder_params.default_task_description='${TASK_RM_DESCRIPTION}'"
    "reward.model.input_builder_params.force_default_task_description=${PROCVLM_RM_FORCE_DEFAULT_TASK_DESCRIPTION:-True}"
    "reward.model.history_buffers.history_window.history_size=${DEFAULT_RM_HISTORY_SIZE}"
    "reward.model.history_buffers.history_window.min_history_size=${DEFAULT_RM_MIN_HISTORY_SIZE}"
    "reward.model.history_buffers.history_window.input_interval=${DEFAULT_RM_HISTORY_INPUT_INTERVAL}"
    "reward.model.history_buffers.history_window.input_on_done=${PROCVLM_RM_HISTORY_INPUT_ON_DONE:-True}"
    "reward.model.debug_samples.enabled=${PROCVLM_RM_DEBUG_SAMPLES_ENABLED:-True}"
    "reward.model.debug_samples.every_n_steps=${PROCVLM_RM_DEBUG_SAMPLES_EVERY_N_STEPS:-10}"
    "reward.model.debug_samples.max_samples_per_step=${PROCVLM_RM_DEBUG_SAMPLES_MAX_PER_STEP:-1}"
    "reward.model.debug_samples.output_dir=${PROCVLM_RM_DEBUG_SAMPLES_OUTPUT_DIR:-${LOG_DIR}/rm_samples}"
    "reward.model.debug_samples.save_format=${PROCVLM_RM_DEBUG_SAMPLES_SAVE_FORMAT:-gif}"
    "reward.model.debug_samples.fps=${PROCVLM_RM_DEBUG_SAMPLES_FPS:-2}"
    "reward.model.debug_samples.include_prompt=${PROCVLM_RM_DEBUG_SAMPLES_INCLUDE_PROMPT:-True}"
    "reward.model.debug_samples.trajectory_video.enabled=${PROCVLM_RM_TRAJECTORY_VIDEO_ENABLED:-True}"
    "reward.model.debug_samples.trajectory_video.every_n_steps=${PROCVLM_RM_TRAJECTORY_VIDEO_EVERY_N_STEPS:-1}"
    "reward.model.debug_samples.trajectory_video.output_dir=${PROCVLM_RM_TRAJECTORY_VIDEO_OUTPUT_DIR:-${LOG_DIR}/rm_rollout_videos}"
    "reward.model.debug_samples.trajectory_video.target_reward_rank=${PROCVLM_RM_TRAJECTORY_VIDEO_TARGET_REWARD_RANK:-0}"
    "reward.model.debug_samples.trajectory_video.target_env_id=${PROCVLM_RM_TRAJECTORY_VIDEO_TARGET_ENV_ID:-0}"
    "reward.model.debug_samples.trajectory_video.fps=${PROCVLM_RM_TRAJECTORY_VIDEO_FPS:-4}"
    "reward.model.debug_samples.trajectory_video.max_frames=${PROCVLM_RM_TRAJECTORY_VIDEO_MAX_FRAMES:-${TASK_MAX_STEPS}}"
    "reward.model.debug_samples.trajectory_video.finalize_after_points=${DEFAULT_TRAJECTORY_VIDEO_FINALIZE_AFTER_POINTS}"
    "reward.model.debug_samples.trajectory_video.min_width=${PROCVLM_RM_TRAJECTORY_VIDEO_MIN_WIDTH:-960}"
    "reward.model.debug_samples.trajectory_video.min_height=${PROCVLM_RM_TRAJECTORY_VIDEO_MIN_HEIGHT:-540}"
    "reward.model.debug_samples.trajectory_video.include_prompt=${PROCVLM_RM_TRAJECTORY_VIDEO_INCLUDE_PROMPT:-True}"
    "reward.model.debug_samples.trajectory_video.include_raw_output=${PROCVLM_RM_TRAJECTORY_VIDEO_INCLUDE_RAW_OUTPUT:-True}"
)

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

cleanup_latest_checkpoint() {
    [[ "${KEEP_LATEST_CHECKPOINT:-1}" == "1" ]] || return 0

    local checkpoint_root="${CHECKPOINT_ROOT:-${LOG_DIR}/${EXPERIMENT_NAME}/checkpoints}"
    [[ -d "${checkpoint_root}" ]] || return 0

    local dirs=()
    local dir base step latest_dir=""
    local latest_step=-1

    shopt -s nullglob
    dirs=("${checkpoint_root}"/global_step_*)
    shopt -u nullglob

    ((${#dirs[@]} > 1)) || return 0

    for dir in "${dirs[@]}"; do
        [[ -d "${dir}" ]] || continue
        base="$(basename "${dir}")"
        step="${base#global_step_}"
        [[ "${step}" =~ ^[0-9]+$ ]] || continue
        if ((step > latest_step)); then
            latest_step="${step}"
            latest_dir="${dir}"
        fi
    done

    [[ -n "${latest_dir}" ]] || return 0

    for dir in "${dirs[@]}"; do
        [[ "${dir}" == "${latest_dir}" ]] && continue
        [[ -d "${dir}" ]] && rm -rf -- "${dir}"
    done

    printf 'Kept latest checkpoint: %s\n' "${latest_dir}" | tee -a "${MEGA_LOG_FILE}" >/dev/null || true
}

on_exit() {
    local status=$?
    cleanup_latest_checkpoint || true
    return "${status}"
}
trap on_exit EXIT

echo "Evaluation Mode: RoboTwin"
echo "Using ROBOT_PLATFORM=${ROBOT_PLATFORM}"
echo "Using ROBOTWIN_PATH=${ROBOTWIN_PATH}"
echo "Using ROBOTWIN_ASSETS_PATH=${ROBOTWIN_ASSETS_PATH}"
echo "Using ROBOTWIN_VECTOR_ENV_MAX_WORKERS=${ROBOTWIN_VECTOR_ENV_MAX_WORKERS}"
echo "Using ROBOTWIN_SERIALIZE_SCENE_STEP=${ROBOTWIN_SERIALIZE_SCENE_STEP}"
echo "Using ROBOTWIN_ACTION_GUARD_ENABLED=${ROBOTWIN_ACTION_GUARD_ENABLED}"
echo "Using ROBOTWIN_ACTION_STATS_PATH=${ROBOTWIN_ACTION_STATS_PATH}"
echo "Using ROBOTWIN_ACTION_GUARD_DELTA_SCALE=${ROBOTWIN_ACTION_GUARD_DELTA_SCALE}"
echo "Using ROBOTWIN_ACTION_GUARD_ABS_LIMIT=${ROBOTWIN_ACTION_GUARD_ABS_LIMIT}"
echo "Using ROBOTWIN_ACTION_GUARD_LOG_EVERY_N=${ROBOTWIN_ACTION_GUARD_LOG_EVERY_N}"
echo "Using CONFIG_NAME=${CONFIG_NAME}"
echo "Using RLINF_TRAIN_ENTRY=${RLINF_TRAIN_ENTRY}"
echo "Using SRC_FILE=${SRC_FILE}"
echo "Using TASK_NAME=${TASK_NAME:-<derived-by-config>}"
echo "Using TASK_RM_DESCRIPTION=${TASK_RM_DESCRIPTION:-<empty>}"
echo "Using PI05_BASE_MODEL=${PI05_BASE_MODEL}"
echo "Using PROCVLM_REWARD_BACKEND=${PROCVLM_REWARD_BACKEND:-value_head_subprocess}"
echo "Using PROCVLM_REWARD_MODEL_PATH=${PROCVLM_REWARD_MODEL_PATH:-/home/fengyouhe/models/frosty-moonshine-bf16}"
echo "Using PROCVLM_REWARD_ENABLE_VALUE_HEAD=${PROCVLM_REWARD_ENABLE_VALUE_HEAD:-False}"
echo "Using TRAIN_NUM_ENVS=${DEFAULT_TRAIN_NUM_ENVS}"
echo "Using ROLLOUT_PIPELINE_STAGE_NUM=${DEFAULT_PIPELINE_STAGE_NUM}"
echo "Using ROLLOUT_BATCH_PIPELINE_STAGES=${ROLLOUT_BATCH_PIPELINE_STAGES:-False}"
echo "Using ROLLOUT_EPOCH=${DEFAULT_ROLLOUT_EPOCH}"
echo "Using UPDATE_EPOCH=${DEFAULT_UPDATE_EPOCH}"
echo "Using RUNNER_MAX_STEPS=${DEFAULT_RUNNER_MAX_STEPS}"
echo "Using RUNNER_VAL_CHECK_INTERVAL=${DEFAULT_RUNNER_VAL_CHECK_INTERVAL}"
echo "Using RUNNER_SAVE_INTERVAL=${DEFAULT_RUNNER_SAVE_INTERVAL}"
echo "Using ACTOR_GLOBAL_BATCH_SIZE=${DEFAULT_ACTOR_GLOBAL_BATCH_SIZE}"
echo "Using ACTOR_MICRO_BATCH_SIZE=${DEFAULT_ACTOR_MICRO_BATCH_SIZE}"
echo "Using ACTOR_LR=${DEFAULT_ACTOR_LR}"
echo "Using RLINF_LOSS_TYPE=${DEFAULT_RLINF_LOSS_TYPE}"
echo "Using RLINF_STALENESS_THRESHOLD=${RLINF_STALENESS_THRESHOLD:-1}"
echo "Using ACTOR_SYNC_WEIGHT_NO_WAIT=${DEFAULT_ACTOR_SYNC_WEIGHT_NO_WAIT}"
echo "Using ROLLOUT_ENABLE_OFFLOAD=${DEFAULT_ROLLOUT_ENABLE_OFFLOAD}"
echo "Using REWARD_ENABLE_OFFLOAD=${DEFAULT_REWARD_ENABLE_OFFLOAD}"
echo "Using RUNNER_PAUSE_REWARD_DURING_ACTOR_TRAINING=${RUNNER_PAUSE_REWARD_DURING_ACTOR_TRAINING:-False}"
echo "Using COMPONENT_PLACEMENT_ACTOR=${COMPONENT_PLACEMENT_ACTOR:-0-3}"
echo "Using COMPONENT_PLACEMENT_ROLLOUT=${COMPONENT_PLACEMENT_ROLLOUT:-0-3}"
echo "Using COMPONENT_PLACEMENT_ENV=${COMPONENT_PLACEMENT_ENV:-0-3}"
echo "Using COMPONENT_PLACEMENT_REWARD=${COMPONENT_PLACEMENT_REWARD:-0-1}"
echo "Using PROCVLM_REWARD_ASYNC_MAX_PENDING=${DEFAULT_REWARD_ASYNC_MAX_PENDING}"
echo "Using PROCVLM_REWARD_ASYNC_MERGE_MAX_REQUESTS=${DEFAULT_REWARD_ASYNC_MERGE_MAX_REQUESTS}"
echo "Using PROCVLM_REWARD_ASYNC_MERGE_WAIT_TIMEOUT_S=${DEFAULT_REWARD_ASYNC_MERGE_WAIT_TIMEOUT_S}"
echo "Using PROCVLM_RM_HISTORY_SIZE=${DEFAULT_RM_HISTORY_SIZE}"
echo "Using PROCVLM_RM_MIN_HISTORY_SIZE=${DEFAULT_RM_MIN_HISTORY_SIZE}"
echo "Using PROCVLM_RM_HISTORY_INPUT_INTERVAL=${DEFAULT_RM_HISTORY_INPUT_INTERVAL}"
echo "Using PROCVLM_RM_TRAJECTORY_VIDEO_FINALIZE_AFTER_POINTS=${DEFAULT_TRAJECTORY_VIDEO_FINALIZE_AFTER_POINTS}"
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
"${CMD[@]}" 2>&1 | tee -a "${MEGA_LOG_FILE}"
