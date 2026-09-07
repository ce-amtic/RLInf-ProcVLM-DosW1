# DOS-W1 真机部署：RLInf + ProcVLM

本文说明如何在 Dexmal DOS-W1 双臂机器人上运行本仓库的默认 SAC-Flow + RLPD 真机流程。流程采用 RLinf 官方的双节点结构：GPU 节点运行 actor、rollout 和可选的 ProcVLM 奖励服务；机器人控制节点运行 DOS-W1 环境、AirBot gRPC 服务和 RealSense 相机。

## 1. 运行前提

需要准备：

- DOS-W1 双臂本体、leader/follower 控制服务和安全操作员；
- 机器人节点与 GPU 节点处于同一局域网；
- GPU 节点至少有一张 CUDA GPU；
- 三个 RealSense 相机，或明确配置实际使用的相机数量；
- 一个与 DOS-W1 观测分布匹配的 flow policy checkpoint；
- RLPD demonstration buffer。没有演示数据时，可以移除 `demo_buffer` 配置，但这会变成纯在线 SAC 冷启动；
- 如果使用 ProcVLM，准备已校准的 merged checkpoint、ProcVLM 公共代码和独立 Python 环境。

默认动作是 14 维：左臂 6 个关节角 + 夹爪宽度，右臂 6 个关节角 + 夹爪宽度。默认任务是左臂抓取并抬升目标物体。

## 2. 安装环境

在 GPU 节点和机器人节点分别安装 RLinf。机器人节点需要 AirBot SDK，GPU 节点只通过 gRPC 连接机器人：

```bash
git clone --recurse-submodules git@github.com:ce-amtic/RLInf-ProcVLM-DosW1.git
cd RLInf-ProcVLM-DosW1
bash requirements/install.sh embodied --env dosw1
source .venv/bin/activate
```

机器人节点如果 SDK 不在默认位置，先指定：

```bash
export DOSW1_SDK_WHEEL=/path/to/airbot_py-5.1.6-py3-none-any.whl
export DOSW1_API_PATH=/path/to/airbot_api
bash requirements/install.sh embodied --env dosw1
```

不要只运行 `uv pip install -e .`，因为真机流程还需要 embodied extra、evdev、OpenCV、RealSense 等依赖。

## 3. 启动机器人服务

在机器人控制节点确认 AirBot 服务已经启动：

```bash
sh ~/dos_w1/airbot/whole_start.sh
```

默认端口如下：

| 服务 | 默认端口 |
| --- | ---: |
| 左 follower | 50051 |
| 左 leader | 50050 |
| 右 follower | 50053 |
| 右 leader | 50052 |

如果服务运行在独立控制机上，把配置中的 `robot_url` 改成该机器地址。GPU 节点应能连通四个端口：

```bash
nc -vz <ROBOT_IP> 50050
nc -vz <ROBOT_IP> 50051
nc -vz <ROBOT_IP> 50052
nc -vz <ROBOT_IP> 50053
```

在机器人节点检查相机和序列号：

```bash
rs-enumerate-devices
```

## 4. 配置 GPU 节点和机器人节点

从完整模板开始：

```text
examples/embodiment/config/dosw1_pick_sac_flow.yaml
```

如果使用本仓库的 ProcVLM 接入：

```text
examples/embodiment/config/dosw1_pick_sac_flow_procvlm.yaml
```

配置中必须检查：

```yaml
cluster:
  num_nodes: 2
  component_placement:
    actor:   {node_group: gpu, placement: 0}
    rollout: {node_group: gpu, placement: 0}
    env:     {node_group: dosw1, placement: 0}
```

在 `cluster.node_groups` 中填入真实的 `robot_url`、四个端口和 RealSense 序列号。`target_grasp_joint`、`target_lift_joint`、关节限制和末端安全盒必须根据当前机器人和工作台实测校准，不能直接照搬示例值。

actor 和 rollout 使用同一个 flow policy：

```yaml
actor:
  model:
    model_path: ${oc.env:DOSW1_POLICY_MODEL_PATH}
rollout:
  model:
    model_path: ${oc.env:DOSW1_POLICY_MODEL_PATH}
```

RLPD demonstration buffer 使用：

```yaml
algorithm:
  demo_buffer:
    load_path: ${oc.env:DOSW1_DEMO_BUFFER_PATH}
```

## 5. 启动 Ray 双节点集群

Ray 启动前必须先激活同一个 Python 环境并设置节点 rank；Ray 会在启动时固定解释器和环境变量。

在 GPU 节点（node rank 0，作为 head）执行：

```bash
source .venv/bin/activate
export RLINF_NODE_RANK=0
export GPU_SERVER_IP=<GPU_NODE_IP>
ray start --head --port=6379 --node-ip-address="$GPU_SERVER_IP"
```

在机器人节点（node rank 1）执行：

```bash
source .venv/bin/activate
export RLINF_NODE_RANK=1
ray start --address=<GPU_NODE_IP>:6379
```

在 GPU 节点确认：

```bash
ray status
```

应看到一个 `gpu` 节点和一个 `dosw1` 节点。两节点都必须使用仓库对应的 `.venv`，不要在 Ray 启动后才切换 Python 或 `PYTHONPATH`。

## 6. 先做无硬件 smoke test

在接入真机前可以运行 dummy 配置验证 Hydra、Ray 和 worker 连接：

```bash
export REPO_PATH="$PWD"
ray start --head
python examples/embodiment/train_embodied_agent.py \
  --config-path tests/e2e_tests/embodied \
  --config-name dosw1_dummy_sac_mlp_pick \
  runner.max_epochs=1
```

该测试不会访问机器人或相机，只能验证软件连接，不代表真机安全性或策略效果。

## 7. 采集 RLPD 演示数据

可选地先用 leader arm 采集成功轨迹：

```bash
bash examples/embodiment/collect_data.sh dosw1_collect_data
```

典型按键流程：

1. 环境进入自由 teleop；
2. 用 leader arm 将 follower 放到起始姿态；
3. 按 `s` 开始 episode；
4. 完成抓取和抬升；
5. 按 `d` 保存成功演示，按 `r` 丢弃并重试。

把生成的 demos 路径设置为：

```bash
export DOSW1_DEMO_BUFFER_PATH=/path/to/logs/dosw1-collect/<run>/demos
```

建议先确认演示轨迹能被加载，再开始在线训练。

## 8. 配置环境变量并启动默认训练

普通 DOS-W1 SAC-Flow + RLPD：

```bash
export DOSW1_POLICY_MODEL_PATH=/path/to/dosw1-flow-policy
export DOSW1_DEMO_BUFFER_PATH=/path/to/dosw1-demo-buffer

bash examples/embodiment/run_realworld.sh dosw1_pick_sac_flow
```

本仓库的 ProcVLM reward 版本：

```bash
export REPO_PATH="$PWD"
export EMBODIED_PATH="$PWD/examples/embodiment"
export RLINF_PYTHON="$PWD/.venv/bin/python"
export DOSW1_POLICY_MODEL_PATH=/path/to/dosw1-flow-policy
export DOSW1_DEMO_BUFFER_PATH=/path/to/dosw1-demo-buffer
export PROCVLM_REPO_PATH="$PWD/third_party/ProcVLM"
export PROCVLM_PYTHON=/path/to/procvlm-environment/bin/python
export PROCVLM_REWARD_MODEL_PATH=/path/to/dosw1-calibrated-procvlm

bash run_dosw1_procvlm.sh single
```

多视角奖励版本：

```bash
export DOSW1_PROCVLM_REWARD_IMAGE_SIZE=336
bash run_dosw1_procvlm.sh multiview
```

ProcVLM 配置使用 8 步历史窗口和 progress delta reward；历史 reward assignment 关闭，避免将同一个预测重复写回历史 transition。actor 仍使用官方 128×128 观测，多视角高分辨率图像只进入奖励模型。

## 9. 首次真机运行的安全顺序

第一次运行时建议按以下顺序逐项确认：

1. 机器人处于可急停状态，工作空间清空，操作员在现场；
2. 只启动环境并确认状态、相机帧和 task description 正常；
3. 只执行 reset，检查 home pose、目标姿态和安全盒；
4. 用极小的 `max_joint_delta` 做短时动作测试；
5. 确认 teleop、pause、model 三种控制模式切换正常；
6. 再启动包含奖励模型的完整训练。

默认安全层包括每步关节增量限制、绝对关节限制和末端位姿安全盒。任何校准修改都应先在低速和人工介入下验证。

## 10. 监控与停止

日志默认写入：

```text
logs/<timestamp>-<config>/run_embodiment.log
```

TensorBoard：

```bash
tensorboard --logdir ./logs --port 6006
```

重点观察：

- `env/success_once`
- `env/episode_len`
- `env/return`
- `train/sac/critic_loss`
- `train/sac/actor_loss`
- `train/sac/alpha`
- `train/replay_buffer/size`
- `train/replay_buffer/mean_reward`

出现危险动作、相机异常、奖励服务阻塞或 Ray 节点丢失时，先按机器人安全流程急停，再停止训练进程和 Ray worker。不要通过修改 reward 权重来掩盖硬件、相机或通信故障。

## 11. 常见故障

**AirBot SDK 导入失败**：在机器人节点设置 `DOSW1_SDK_WHEEL` 和 `DOSW1_API_PATH` 后重新安装 `embodied --env dosw1`。

**DOSW1 状态超时**：检查 AirBot 服务、`robot_url`、50050–50053 端口和两节点网络连通性。

**没有相机帧**：运行 `rs-enumerate-devices`，核对序列号和 USB 连接；无显示器时设置 `enable_camera_player: false`。

**奖励持续为 0**：检查 `is_dummy`、目标关节校准、ProcVLM checkpoint、task description 和奖励服务日志；先单独验证 environment observation，再验证 reward worker。

**Ray worker 使用了错误环境**：停止 Ray，在每个节点重新激活 `.venv`、设置 `RLINF_NODE_RANK` 和相关路径后再启动。环境变量必须在 `ray start` 之前设置。

**RLPD 无法加载演示数据**：检查 `DOSW1_DEMO_BUFFER_PATH` 是否指向完整 demos 目录，并确认该目录可被 Ray 所在节点访问。

## 参考

- [RLinf DOS-W1 官方文档](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/dosw1.html)
- [RLinf SAC-Flow 官方文档](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/sac_flow.html)
- [RLinf RLPD 参考](https://rlinf.readthedocs.io/en/latest/rst_source/reference/algorithms/rlpd.html)
- [ProcVLM 公共仓库](https://github.com/RUCKBReasoning/ProcVLM)
