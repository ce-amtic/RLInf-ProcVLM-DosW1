# 右臂 + Top 相机：GPU 节点部署

GPU 节点运行 actor、rollout 和 ProcVLM reward；机器人节点运行环境和驱动。这里提供部署步骤，GPU 节点未在本机调试。机器人侧见 [机器人部署文档](dosw1_robot_node.md)。

## 1. 安装同一版本代码

```bash
git clone --recurse-submodules git@github.com:ce-amtic/RLInf-ProcVLM-DosW1.git
cd RLInf-ProcVLM-DosW1
# 两节点切换到同一包含 right_top 配置的分支/commit。
bash requirements/install.sh embodied --env dosw1 --no-root --no-flash-attn
source .venv/bin/activate
python -c 'import sys, torch, ray; print(sys.version, torch.__version__, ray.__version__, torch.cuda.is_available())'
```

需要 NVIDIA CUDA GPU；型号、显存是否满足取决于策略、ProcVLM checkpoint 和 batch size。GPU 机器没有 AirBot wheel/API 源码时安装器会提示跳过；GPU worker 不实例化机器人 SDK。不需要给 GPU 机器接相机。

Python 使用 3.11.14；本次机器人实测 Ray 为 2.58.0，GPU 上需固定同一版本（`uv pip install ray[default]==2.58.0`）。机器人安装后的 `toolkits/dosw1/robot_environment.txt` 记录实测版本，按其中 Ray 版本固定 GPU 环境。独立 ProcVLM 环境按 `third_party/ProcVLM` 的 README 安装，并使用其所支持的 merged checkpoint。它不应覆盖 RLInf `.venv` 的依赖。

ProcVLM 公共仓库自带独立 `pyproject.toml`。可在 GPU 机器按该子模块的版本创建环境：

```bash
"$PWD/.venv/bin/uv" sync --project third_party/ProcVLM --python 3.10
export PROCVLM_PYTHON="$PWD/third_party/ProcVLM/.venv/bin/python"
"$PROCVLM_PYTHON" -c 'import torch, transformers, PIL; print(torch.cuda.is_available(), transformers.__version__)'
```

该子模块默认还安装 vLLM 等依赖；本配置使用 `transformers_subprocess` + `sdpa`，不需要另外启动 vLLM HTTP 服务或手工运行 reward server。RLInf reward worker 会启动并管理子进程。不要把它切到系统 Python。

## 2. 准备模型和 demonstrations

```bash
export DOSW1_POLICY_MODEL_PATH=/absolute/path/to/flow-assets
export DOSW1_POLICY_CHECKPOINT=/absolute/path/to/flow-policy-state-dict.pt
export DOSW1_DEMO_BUFFER_PATH=/absolute/path/to/rlinf-demos
export PROCVLM_REPO_PATH="$PWD/third_party/ProcVLM"
export PROCVLM_PYTHON=/absolute/path/to/procvlm-env/bin/python
export PROCVLM_REWARD_MODEL_PATH=/absolute/path/to/merged-procvlm-checkpoint
export REPO_PATH="$PWD"
export EMBODIED_PATH="$PWD/examples/embodiment"
```

注意区别：`DOSW1_POLICY_MODEL_PATH` 是包含 `resnet10_pretrained.pt` 的编码器资产目录，**仅设置它不会加载完整 flow policy**。完整策略通过 `DOSW1_POLICY_CHECKPOINT` → `runner.ckpt_path` 同时加载到 actor 和 rollout，内容应为与配置匹配的完整 PyTorch state_dict（包括配置中的 Q heads），不是任意训练器的包装字典。

策略要求 `state_dim=7`、`action_dim=7`、`image_num=1`；状态和动作均按右臂 joint1–6 + gripper 排列。`main_images` 是 RGB、128×128、中心裁剪。不要直接使用旧双臂 14 维 checkpoint，也不要把归一化增量动作当成绝对关节角。

RLPD 路径必须是 RLInf replay buffer，包含 `metadata.json`、`trajectory_index.json` 及索引指向的完整轨迹文件；仅有 LeRobot/HDF5 视频数据不能直接加载。轨迹应具有同一 state/action/图像定义。预检检查基本资产存在，首次加载仍需验证 checkpoint 的完整键/形状和轨迹内容。

ProcVLM 使用 `config.json`、merged 权重及所需 processor/tokenizer 文件完整目录。输入是 Top 相机 336×336 单视角历史，不是三视角。默认环境关节空间奖励关闭（旧左臂目标没有经过右臂校准），`env_reward_weight=0`；不要将示例目标关节用作真实成功标签。

## 3. 设置节点参数并启动 Ray

```bash
export GPU_SERVER_IP=<GPU局域网IP>
export DOSW1_ROBOT_REPO=/home/ubuntu/RLInf-ProcVLM-DosW1
export DOSW1_ROBOT_PYTHON="$DOSW1_ROBOT_REPO/.venv/bin/python"
# 这个地址由机器人环境 worker 使用，驱动在同一机器人机器时保持回环地址。
export DOSW1_ROBOT_URL=127.0.0.1
export DOSW1_URDF_PATH=/home/ubuntu/dos_w1/airbot/airbot_api/config/airbot_urdf/play_g2/urdf/play_g2.urdf
bash toolkits/dosw1/node.sh head
```

然后在机器人上运行 `node.sh robot`，具体步骤见机器人文档。GPU 上 `ray status` 应看到两个节点；RLInf 的 `gpu`/`dosw1` 是配置中的 node group 标签，并非原生 Ray status 必然显示的资源名称。不要仅开放 head 的 6379：Ray worker/对象管理通信需要局域网双向连通。Ray 不负责分发本仓库或模型；两机必须分别有代码，机器人解释器/PYTHONPATH 由 node group 配置指定。

## 4. 生成配置（不启动实验）

将现场确认的校准文件复制到 GPU：

```bash
python toolkits/dosw1/prepare_experiment.py \
  --calibration /absolute/path/to/right-calibration.yaml
```

脚本解析 Hydra 配置、验证集群配置结构、检查必要模型/demo 资产，生成 `logs/dosw1-prepared/experiment.yaml`。默认生成只读配置，不启动 Ray 或训练。保存下来的文件可检查 actor/rollout/reward 都在 GPU、env 在机器人节点，以及路径、7 维动作、单相机和 reset 参数是否正确。

## 5. 模型及现场准备完成后启动

先在机器人完成 `preflight.py --hardware`，停止其他机械臂控制器，现场验证 home/reset 和安全盒。校准文件中需有 `motion_validated: true` 和 `env.reset_to_home: true`。

```bash
python toolkits/dosw1/prepare_experiment.py \
  --calibration /absolute/path/to/right-calibration.yaml \
  --execute
```

此命令会真实启动训练并允许环境发送右臂动作。当前调试不执行它。若不用 ProcVLM，可选择 `--config dosw1_right_top_sac_flow`，但须另行设计/校准奖励；当前该配置的任务奖励关闭，不能期待有效学习。

当前本机 CPU 检查不能证明 CUDA/FSDP、GPU 显存、ProcVLM 推理或跨节点通信已通过。首次 GPU 部署应先验证权重加载和 reward 服务，再在现场进行短时受控实验。
