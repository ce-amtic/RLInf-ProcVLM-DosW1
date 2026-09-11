# 右臂 + Top 本机调试结果（2026-09-11）

范围：机器人节点、软件依赖、单臂 SDK、观测与配置。没有执行机械臂运动或启动训练；GPU 部署按用户要求分离为文档。

## 已通过

- AirBot 5.1.6 右臂驱动启动，can_right → 50053；读取 6 关节、夹爪、末端状态。
- 本机重启后，使用 `driver.sh start` 重新创建驱动并再次通过只读预检；重复 start 成功。
- Top 实时图像，序列号 420222072569；共享帧读取不抢占现有 USB owner。
- RLInf 单臂状态 `(1,7)`、actor RGB `(1,128,128,3)`、奖励 RGB `(1,1,336,336,3)`。
- URDF FK 与本次驱动反馈末端位置相差约 1.04 mm；该检查只覆盖当前姿态。
- 20 项回归测试通过，包含只调用右臂、只读禁止动作、新鲜反馈超时、相机陈旧/错误序列号、关节限幅、空间边界、配置组合、vector env、CPU FlowPolicy 前向和 checkpoint round-trip、配置生成与未校准执行拦截。
- CPU FlowPolicy 使用临时随机 encoder/policy fixture，输出 `(1,1,7)`、有限值、完整 state_dict 严格加载成功；没有把该 fixture 保存为实验模型。
- 本机 CPU Ray 远程任务中运行同一 `.venv` 的真机只读预检通过；重启恢复后再次通过，测试 Ray 已退出。
- 安装器复跑与 `pip check` 通过；Ruff lint/format、shell syntax、git diff whitespace 检查通过。

## 交付

- [机器人节点](dosw1_robot_node.md)
- [GPU 节点](dosw1_gpu_node.md)
- 配置：`dosw1_right_top_sac_flow` / `dosw1_right_top_sac_flow_procvlm`
- 工具：`toolkits/dosw1/{driver.sh,node.sh,preflight.py,ray_smoke.py,prepare_experiment.py}`
- 版本记录：`toolkits/dosw1/robot_environment.txt`
- 本地运行证据：`logs/dosw1-commissioning/`（不提交硬件运行日志）

## 尚需后续完成

1. 模型、奖励模型、符合 7 维单相机约定的 RLPD demos。
2. GPU 侧安装与模型加载，以及真正的两机器 Ray 联调；本机 CPU 测试不覆盖 CUDA/FSDP/显存。
3. 现场工作区、关节限制、home/reset 校准及低速动作验证。示例校准文件保留 `motion_validated: false`，实验默认只读；不把当前姿态检查当成运动安全性验证。

驱动容器保持运行用于后续检查；没有开启自动重启/开机启动，没有修改现有 conda 环境或控制台相机服务。
