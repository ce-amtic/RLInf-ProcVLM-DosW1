#!/usr/bin/env bash
# Validated launch for RoboTwin move_can_pot OpenVLA-OFT GRPO + ProcVLM RM
# on the original 4x A6000 deployment. Machine-specific paths remain
# configurable through environment variables.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R=${RLINF_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
CFG="$R/examples/embodiment/config/robotwin_move_can_pot_grpo_openvlaoft.yaml"
UTILS="$R/rlinf/envs/utils.py"
RM=${PROCVLM_REWARD_MODEL_PATH:-/home/fengyouhe/models/LoRA/procvlm/move_can_pot_merged}
RLINF_LOG_ROOT=${RLINF_LOG_ROOT:-/data/fengyouhe/rlinf_logs}
LOG="${LOG:-${RLINF_LOG_ROOT}/move_can_pot_run1.log}"
# UNIFIED log: every meta-run's stdout is APPENDED here (the per-launch $LOG is overwritten each launch) so
# there is ONE continuous log across all 5-step recycles. Prunable if it grows large.
UNIFIED="${UNIFIED:-${RLINF_LOG_ROOT}/move_can_pot_unified.log}"

echo "== pre-flight: in-repo fixes (a repo re-sync from the source box RESETS these) =="
fail=0
grep -qE 'actor:[[:space:]]*0-3' "$CFG"                  && echo "  [ok] placement 4-GPU (actor:0-3)"          || { echo "  [MISSING] placement: expect 'actor: 0-3' not 0-7 — re-apply"; fail=1; }
grep -q  'weight_syncer/bucket_syncer@weight_syncer' "$CFG" && echo "  [ok] bucket_syncer"                      || { echo "  [MISSING] weight_syncer: config shipped patch_syncer (crashes init_sender) — switch to bucket_syncer + 128MB/bf16 block"; fail=1; }
grep -q  'grid_sample' "$UTILS"                          && echo "  [ok] TF-free torch center_crop_image"        || { echo "  [MISSING] center_crop_image: TF version SIGSEGVs in env worker — re-apply torch grid_sample crop"; fail=1; }
grep -q  'robotwin_sapien_teardown.lock' "$R/third_party/RoboTwin/envs/_base_task.py" && echo "  [ok] close_env cross-process teardown lock (SAPIEN reset-crash fix)" || { echo "  [MISSING] close_env teardown flock in _base_task.py — re-apply (fixes SAPIEN reset SIGSEGV)"; fail=1; }
{ grep -q '_robotwin_reuse_scene_enabled' "$R/third_party/RoboTwin/envs/_base_task.py" && grep -q 'final=True' "$R/third_party/RoboTwin/robotwin/envs/vector_env.py"; } && echo "  [ok] scene-reuse optimization (ROBOTWIN_REUSE_SCENE; eval-validated accuracy-neutral, eliminates per-episode SAPIEN teardown + speeds rollout)" || { echo "  [MISSING] scene-reuse code (_robotwin_reuse_scene_enabled in _base_task.py + final= call sites in vector_env.py) — re-apply; reuse keeps the SAPIEN scene alive across resets so the per-episode teardown SIGSEGV is gone, destroying only at the update boundary"; fail=1; }
# NOTE: the start_step+8 OIDN crash is NOT fixed in-code (the apply-config-once attempt #18 failed +
# was reverted; the leak is in SAPIEN's .so). It is MITIGATED by the supervisor's PROACTIVE RECYCLE
# (run 5 steps -> clean-exit -> relaunch, never reaching +8). So there is intentionally NO OIDN
# pre-flight check here anymore.
{ [ -f "$RM/config.json" ] && [ -f "$RM/procvlm_extra/procvlm_head.pt" ]; } && echo "  [ok] ProcVLM RM present" || { echo "  [MISSING] ProcVLM RM at $RM"; fail=1; }
_sv=${RUNNER_SAVE_INTERVAL:-10}; _vc=${RUNNER_VAL_CHECK_INTERVAL:-10}; [ $((_sv % _vc)) -eq 0 ] && echo "  [ok] save_interval($_sv) is a multiple of val_check_interval($_vc)" || { echo "  [MISSING] save_interval=$_sv MUST be a multiple of val_check_interval=$_vc (else AssertionError crashes the run after step 1 — runner_utils.py:44)"; fail=1; }
if [ "$fail" = 1 ]; then echo "ABORT: re-apply the missing fix(es) before launching."; exit 1; fi

# wandb: ONE continuous ONLINE run owned by the standalone aggregator ~/wandb_monitor.py (tmux 'wandb_mon').
# It is the SOLE wandb writer: scrapes the live tensorboard (the only source carrying eval/*) and forwards ALL
# recycle cycles into ONE resumable online run (resume="allow", fixed id) with eval in its own section, fully
# decoupled from training (a wedged proxy can never block/crash training; all its wandb I/O is in a
# SIGKILL-able child). Training is tensorboard-only (WANDB_MODE=disabled + logger_backends=[tensorboard] on the
# run cmd below) so there is NO training wandb-core to collide with the aggregator's wandb-core cleanup.
# eval_history.{jsonl,txt} + metrics_history.jsonl are local fallbacks. Verified on-box 2026-06-21 (V1 + V4).
# IMPORTANT: do NOT `export WANDB_MODE` here -- the aggregator spawned just below MUST run ONLINE; a stray
# exported WANDB_MODE would be inherited by it and silence the cloud run.
ENABLE_WANDB_AGGREGATOR=${ENABLE_WANDB_AGGREGATOR:-0}
ENABLE_HANG_WATCHDOG=${ENABLE_HANG_WATCHDOG:-0}
WANDB_MONITOR_SCRIPT=${WANDB_MONITOR_SCRIPT:-/home/fengyouhe/wandb_monitor.py}
HANG_WATCHDOG_SCRIPT=${HANG_WATCHDOG_SCRIPT:-/home/fengyouhe/hang_watchdog.sh}
_HB=${RLINF_LOG_ROOT}/wandb_overall.heartbeat
# respawn the aggregator if running but its heartbeat is MISSING or stale (>15min): wedged, or a wrong/old build
if [ "$ENABLE_WANDB_AGGREGATOR" = 1 ] && pgrep -f '[w]andb_monitor' >/dev/null 2>&1 && { [ ! -f "$_HB" ] || [ $(( $(date +%s) - $(stat -c %Y "$_HB" 2>/dev/null || echo 0) )) -gt 900 ]; }; then
  echo "wandb aggregator heartbeat missing/stale -> respawn"; pkill -9 -f '[w]andb_monitor'; tmux kill-session -t wandb_mon 2>/dev/null; sleep 2
fi
# spawn if not running; kill any leftover dead session first so new-session can't fail with 'duplicate session'
if [ "$ENABLE_WANDB_AGGREGATOR" = 1 ]; then
  pgrep -f '[w]andb_monitor' >/dev/null 2>&1 || { tmux kill-session -t wandb_mon 2>/dev/null; tmux new-session -d -s wandb_mon "exec $R/.venv/bin/python $WANDB_MONITOR_SCRIPT >${RLINF_LOG_ROOT}/wandb_mon_console.log 2>&1"; }
fi
# hang-watchdog (ALERT-ONLY): warns on training DEADLOCK hangs; it no longer auto-kills (see hang_watchdog.sh)
if [ "$ENABLE_HANG_WATCHDOG" = 1 ]; then
  pgrep -f '[h]ang_watchdog' >/dev/null 2>&1 || tmux new-session -d -s hang_dog "exec bash $HANG_WATCHDOG_SCRIPT" 2>/dev/null
fi
echo "== training logger: tensorboard-only; optional external monitors are disabled by default =="

echo "== GPUs (want 4x idle ~20MiB) =="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

cd "$R" || exit 9
echo "== launch -> $LOG (+ appended to unified $UNIFIED) =="
printf '\n\n======== LAUNCH %s | resume=%s | max_steps=%s ========\n' "$(date '+%F %T')" "${*:-fresh}" "${RUNNER_MAX_STEPS:-?}" >> "$UNIFIED"
echo "   gbs=480 (4-GPU divisibility fix; launcher default 960 crashes), dispatch=all (official YAML"
echo "   default; launcher's non-official streaming_ready gave no speedup), RM=move_can_pot, envs=96, ALOHA"
# SEED-COVERAGE FIX (2026-06-20): the proactive-recycle relaunches with the SAME env seed (cfg.seed=0) and the
# env resets _current_seed_index=0 each launch (robotwin_env.py:49,721), so EVERY 5-step meta-run re-trained the
# SAME first-N train seeds -- the rollout-SO per-step pattern repeated each recycle (e.g. step-4-of-meta-run was
# always a relative dip) and only a tiny fixed subset of the 1000 train scenes was ever seen. Derive a
# per-meta-run train seed from the resume step so each recycle RESHUFFLES the 1000 -> a fresh random subset each
# time -> broad coverage over many recycles. Eval is unaffected (env.eval shuffle uses its own fixed cfg.seed +
# use_fixed_reset_state_ids). Override path is the same proven form as ++env.train.action_exec_horizon below.
_RS=$(printf '%s\n' "$@" | grep -aoE 'global_step_[0-9]+' | grep -aoE '[0-9]+' | head -1); TRAIN_SEED=${_RS:-0}
echo "   train-seed=$TRAIN_SEED (per-meta-run reshuffle of the 1000 train scenes; was a fixed repeat each recycle)"
PYTHONUNBUFFERED=1 \
WANDB_MODE=disabled \
ACTOR_GLOBAL_BATCH_SIZE=480 \
TRAIN_NUM_ENVS=48 \
ROLLOUT_EPOCH=10 \
COMPONENT_PLACEMENT_ENV=2-3:0-5 \
COMPONENT_PLACEMENT_REWARD=0-3 \
EVAL_TOTAL_NUM_ENVS=48 \
EVAL_NUM_PARALLEL_ENVS=48 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
ROBOTWIN_REUSE_SCENE=1 \
PROCVLM_REWARD_MODEL_PATH="$RM" \
PROCVLM_REWARD_ASYNC_MERGE_MAX_REQUESTS=8 \
PROCVLM_FRAME_MAX_BYTES=2147483648 \
RUNNER_KEEP_LATEST_CHECKPOINTS=4 \
RUNNER_KEEP_BEST_CHECKPOINT=True \
RUNNER_SAVE_INTERVAL=${RUNNER_SAVE_INTERVAL:-5} \
REWARD_ENABLE_OFFLOAD=False \
PROCVLM_REWARD_ASYNC_MAX_PENDING=8 \
PROCVLM_RM_DEBUG_SAMPLES_ENABLED=False \
PROCVLM_RM_TRAJECTORY_VIDEO_ENABLED=False \
bash run_robotwin_openvla_grpo_procvlm.sh robotwin_move_can_pot_grpo_openvlaoft ALOHA \
  rollout.env_rank_dispatch_mode=all \
  runner.logger.logger_backends="[tensorboard]" \
  algorithm.filter_rewards=False \
  ++reward.offload_before_actor_update=False \
  ++env.train.seed=$TRAIN_SEED \
  ++env.train.action_exec_horizon=25 ++env.train.task_config.rdt_step=25 ++env.train.execute_action_prefix_individually=False \
  ++env.eval.action_exec_horizon=25  ++env.eval.task_config.rdt_step=25  ++env.eval.execute_action_prefix_individually=False \
  ++env.eval.video_cfg.save_video=False ++env.train.video_cfg.save_video=False \
  "$@" \
  2>&1 | tee "$LOG" | tee -a "$UNIFIED"
# ^ action_exec_horizon=25 (=num_action_chunks, full-chunk open-loop like official).
#   The launcher default 8 truncates the SFT's 25-step chunk -> SFT eval 0% (vs 33% at 25) and crippled RL.
# OOM-safety (host RAM; box rebooted from host-OOM at step 2 before): the dominant per-step host churn was
#   the ProcVLM subprocess being KILLED+reloaded (4.88GB torch.load x2 ranks) every step via reward.enable_offload.
#   REWARD_ENABLE_OFFLOAD=False + ++reward.offload_before_actor_update=False keep it RESIDENT (no reload churn);
#   ProcVLM (~5GB) then stays on GPU0/1 during the actor update (which has headroom since rollout is destroyed).
#   PROCVLM_REWARD_ASYNC_MAX_PENDING=8 (was 16) trims the reward-queue host buffer. Debug video/sample dumps
#   OFF (PROCVLM_RM_*_ENABLED=False) -> they buffered frames + spawned never-joined encoder threads (leak/overhead).
#   All accuracy-neutral (same model/weights/rewards/trajectories; only memory residency & diagnostics change).
