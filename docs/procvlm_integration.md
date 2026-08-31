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

## DOS-W1 boundary

RLInf already contains the official DOS-W1 environment, robot scheduler,
collection configuration, synchronous/asynchronous SAC flow configurations,
and dummy smoke configuration. This archive preserves those files unchanged.
The next stage is to route DOS-W1 camera history and task text into the same
ProcVLM worker and configure its reward contribution while preserving the
official two-node robot/GPU deployment flow.

References:

- ProcVLM: <https://github.com/RUCKBReasoning/ProcVLM>
- RLInf DOS-W1 guide: <https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/dosw1.html>
