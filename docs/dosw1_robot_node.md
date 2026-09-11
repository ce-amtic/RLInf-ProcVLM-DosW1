# 右臂 + Top 相机：机器人节点

本机部署配置为 `dosw1_right_top_sac_flow_procvlm`。只连接右 follower `127.0.0.1:50053`，不连接左臂和 leader。Top 指现有视频服务的 `front` 相机，序列号 `420222072569`。

## 接口约定

- action/state：7 维，依次为右臂 joint1–6（绝对关节角，rad）和夹爪宽度（m）。不是关节增量，也不是双臂 14 维。
- actor 图像：`main_images`，RGB，128×128，来自 Top 图像的中心正方形裁剪。
- ProcVLM 图像：`reward_main_images`，RGB，1 个视角，336×336；8 步历史窗口、interval=3、delta reward，无历史回填。
- 默认关闭 leader teleop 和 evdev，因此本部署不要求左臂、leader 或键盘设备权限。现有双臂演示采集脚本不适用于此配置。
- 无硬件预检不会产生有效训练样本。模型和演示数据必须按上述 7 维、单相机、相同裁剪与单位准备。

## 环境安装

本机仓库路径 `/home/ubuntu/RLInf-ProcVLM-DosW1`，独立环境 `.venv`，Python 3.11.14。不要修改已有 `dos-w1` conda 环境。

```bash
cd /home/ubuntu/RLInf-ProcVLM-DosW1
bash requirements/install.sh embodied --env dosw1 --no-root --no-flash-attn
source .venv/bin/activate
export PYTHONPATH="$PWD"  # 隔离系统 ROS Python 3.10 路径
```

安装器默认找到 `~/dos_w1/airbot/5.1.6/airbot_py-5.1.6-py3-none-any.whl`。本配置使用底层 `airbot_py`，绕过本机旧版 `airbot_sdk.AirbotRobot` 的双臂构造器。FK 读取 URDF，通过 NumPy/SciPy 计算，不依赖 PyKDL。默认 URDF 路径为：

```
/home/ubuntu/dos_w1/airbot/airbot_api/config/airbot_urdf/play_g2/urdf/play_g2.urdf
```

## 驱动与相机

```bash
bash toolkits/dosw1/driver.sh start
bash toolkits/dosw1/driver.sh status
bash toolkits/dosw1/driver.sh logs
```

使用本机现有 AirBot 5.1.6 Docker 镜像、`can_right` 和独立容器 `rlinf-dosw1-right`；不自动启用开机启动或重启策略。驱动启动会启用硬件，现场应保持急停可用。预检连接不切换控制模式，也不发送关节或夹爪命令。

Top 相机由现有 `ruc-video-camera@front.service` 持有：

```bash
systemctl --user status ruc-video-camera@front.service
```

RLInf 读取 `/dev/shm/ruc-video/front/meta.json` 和 `color.bgr`，校验序列号、write/ready 状态、前后元数据一致性和帧龄；帧超过 0.5 秒或 1 秒内拿不到完整新鲜图像即报错。无需抢占 USB 或停止控制台。若换到没有视频网关的控制机，可把 `camera_backend` 改为 `realsense`，并确保相机没有其他 owner。

## 只读预检

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD"  # 隔离系统 ROS Python 3.10 路径
python toolkits/dosw1/preflight.py
python toolkits/dosw1/preflight.py --hardware --output /tmp/dosw1-hardware.json
python toolkits/dosw1/ray_smoke.py --hardware
```

检查真实状态、相机、RLInf tensor 封装以及 URDF FK 与驱动末端坐标的一致性。预检强制 `read_only=true`、禁止构造时移动和 home reset。Ray smoke 只创建临时 CPU Ray 节点并运行预检，结束后关闭自己启动的 Ray；不启动训练，不加入 GPU 集群。已有正式 Ray 运行时不要同时运行此 smoke。

## 加入 GPU 集群

先按 [GPU 部署文档](dosw1_gpu_node.md) 启动 head。两节点必须能互通，使用相同仓库版本、Python 小版本和 Ray 版本。

```bash
export GPU_SERVER_IP=<GPU局域网IP>
export ROBOT_NODE_IP=<本机局域网IP>
bash toolkits/dosw1/node.sh robot
```

本机 `RLINF_NODE_RANK=1`，GPU 为 0。脚本会先激活 `.venv`，设置 rank/PYTHONPATH 后再启动 Ray。SSH 地址只用于登录或同步代码，Ray 使用可双向连通的 IP 和端口；不要填写 HTTP 代理/TUN 虚拟 IP。模型和 ProcVLM 只需部署在 GPU 机器。

## 首次动作前的现场校准

`toolkits/dosw1/calibration.example.yaml` 只记录零位附近约 2 cm 的检查区域，**不是可用的抓取工作空间**。本次预检不会擅自把它标记为完成校准。

复制此文件，测量并填写右臂 home、夹爪、绝对关节限制和末端安全盒（右臂 base_link 坐标）。在现场验证慢速 home/reset 和工作台间隙后，才设置 `reset_to_home: true`、`motion_validated: true`。将同一份校准文件交给 GPU 启动脚本。默认 `max_joint_delta=0.01 rad/step`，reset 同样分步限幅并检查安全盒；30 秒内未到目标会报错。

部署默认只读，GPU 启动器的 `--execute` 在校准和资源检查通过后才切换为可执行动作。没有 leader，人工接管依靠现场急停；不要把现有遥操作栈与 RL 控制器同时打开。真实运动、任务成功率、模型效果和跨机器 Ray 联调均不包含在只读预检结论中。

停止驱动：

```bash
bash toolkits/dosw1/driver.sh stop
```
