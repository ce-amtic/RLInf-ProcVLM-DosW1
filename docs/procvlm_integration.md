# ProcVLM integration

## Dependency layout

ProcVLM is pinned as the public Git submodule at `third_party/ProcVLM`:

```bash
git submodule update --init --recursive
cd third_party/ProcVLM
uv sync
```

The launchers default `PROCVLM_REPO_PATH` to that checkout. The Python
environment is deliberately decoupled from the source checkout: callers must
set `PROCVLM_PYTHON` to a compatible interpreter.

## Runtime path

The integration sends rollout image history and task text from RLInf to a
standalone ProcVLM subprocess. The subprocess loads the public
`evqa.model.load_procvlm` API, performs batched value-head inference, parses the
predicted progress, and returns the configured progress-derived reward to RLInf.

The main adapter files are:

- `rlinf/models/embodiment/reward/procvlm_reward_model.py`
- `rlinf/models/embodiment/reward/procvlm_reward_backend.py`
- `rlinf/models/embodiment/reward/procvlm_reward_server.py`
- `rlinf/models/embodiment/reward/procvlm_input_builder.py`
- `rlinf/models/embodiment/reward/reward_model_worker.py`
- `run_robotwin_openvla_grpo_procvlm.sh`
- `run_robotwin_pi05_ppo_procvlm.sh`

## Checkpoint configuration

Checkpoints are intentionally external to Git. Supply the merged model at
launch time:

```bash
export PROCVLM_REWARD_MODEL_PATH=/path/to/merged/procvlm
export PROCVLM_REPO_PATH="$PWD/third_party/ProcVLM"
export PROCVLM_PYTHON=/path/to/procvlm/environment/bin/python
```

The validated checkpoint on the original 127 host is:

```text
/home/fengyouhe/models/LoRA/procvlm/move_can_pot_merged
```

## Validated RoboTwin entry point

The portable form of the validated launch command is:

```bash
PROCVLM_PYTHON=/path/to/procvlm/environment/bin/python \
PROCVLM_REWARD_MODEL_PATH=/path/to/merged/procvlm \
  bash scripts/robotwin/launch_move_can_pot_rl.sh /path/to/checkpoint
```

RoboTwin runtime fixes are archived as a small patch instead of vendoring its
large asset tree. Apply them to the documented base revision with:

```bash
bash scripts/apply_robotwin_patch.sh /path/to/RoboTwin
```

## DOS-W1 adaptation

The DOS-W1 integration reuses the official two-node SAC/RLPD flow. The robot
node owns hardware I/O and camera capture; the GPU node owns actor/rollout and
the existing external ProcVLM reward worker. ProcVLM shapes the reward but does
not replace robot termination, intervention, or safety logic.

Two modes are provided:

| Mode | Config | ProcVLM images | Actor images |
| --- | --- | --- | --- |
| Stage 1 | `dosw1_pick_sac_flow_procvlm.yaml` | 8-frame `cam_left` history | Unchanged official 128x128 views |
| Stage 2 | `dosw1_pick_sac_flow_procvlm_multiview.yaml` | 8-step, high-resolution front/left/right history | Unchanged official 128x128 views |

Stage 2 captures `reward_frames` in the DOS-W1 environment and packs them as
`reward_main_images` in the explicit front, left, right order. The existing
RLInf reward transport consumes that field and strips it before observations
are sent to the actor. Flattening is timestep-major, so each history step is
expanded as front, left, right before the next step.

Both modes use progress delta rewards, disable historical reward reassignment,
and blend ProcVLM/environment reward with weights 1.0/0.1. The default backend
is the stable external Transformers subprocess with deterministic decoding.

### Launch

Configure the standard DOS-W1 two-node Ray cluster and edit camera serials,
robot address, safety boxes, actor checkpoint, and demonstration buffer as
described by the official RLInf guide. Then set:

```bash
export RLINF_PYTHON=/path/to/rlinf/environment/bin/python
export PROCVLM_PYTHON=/path/to/procvlm/environment/bin/python
export PROCVLM_REWARD_MODEL_PATH=/path/to/dosw1-calibrated/merged/procvlm
export DOSW1_POLICY_MODEL_PATH=/path/to/dosw1/flow-policy
export DOSW1_DEMO_BUFFER_PATH=/path/to/dosw1/demo-buffer
export DOSW1_TASK_DESCRIPTION="Pick up the object with the left arm."
```

Stage 1:

```bash
bash run_dosw1_procvlm.sh single
```

Stage 2 (default reward resolution 336x336):

```bash
export DOSW1_PROCVLM_REWARD_IMAGE_SIZE=336
bash run_dosw1_procvlm.sh multiview
```

Extra arguments are passed through as Hydra overrides. W&B is not enabled by
these configs; logs default to a timestamped directory under `logs/`.

### Deployment boundary

The repository tests observation shapes, camera ordering, temporal flattening,
and configurable task text without touching robot hardware. Before online RL,
collect DOS-W1 one-shot demonstrations and calibrate/validate ProcVLM on the
real camera distribution. The RoboTwin `move_can_pot_merged` checkpoint can be
used only for interface smoke tests; it is not a validated DOS-W1 reward model.

References:

- ProcVLM: <https://github.com/RUCKBReasoning/ProcVLM>
- RLInf DOS-W1 guide: <https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/dosw1.html>
