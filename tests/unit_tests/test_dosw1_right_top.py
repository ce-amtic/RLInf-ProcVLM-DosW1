# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single follower commissioning regressions: no hardware required."""

import json
import time
from unittest.mock import Mock

import numpy as np
import pytest

from rlinf.envs.realworld.common.camera.shared_frame_camera import SharedFrameCamera
from rlinf.envs.realworld.dosw1.airbot_native import NativeAirbot
from rlinf.envs.realworld.dosw1.dosw1_env import DOSW1Config, DOSW1Env
from rlinf.envs.realworld.dosw1.tasks.pick import PickEnv
from rlinf.envs.realworld.realworld_env import RealWorldEnv


def env_config(**kwargs):
    return DOSW1Config(
        is_dummy=True,
        active_arms=["right"],
        camera_names=["cam_front"],
        enable_camera_player=False,
        **kwargs,
    )


def test_right_state_action_and_wrapper_order():
    env = DOSW1Env(env_config(), None, None, 0)
    assert env.action_space.shape == (7,)
    assert list(env.observation_space["state"].spaces) == ["right"]
    env.config.is_dummy = False
    env.robot_state.right_joint_positions = np.arange(6, dtype=float)
    env.robot_state.right_gripper = 0.04
    env._get_camera_frames = lambda: (
        {"cam_front": np.zeros((128, 128, 3), np.uint8)},
        {},
    )
    obs = env._get_observation()
    wrapper = object.__new__(RealWorldEnv)
    wrapper.main_image_key = "cam_front"
    wrapper.reward_image_keys = []
    wrapper.task_descriptions = ["right pick"]
    batched = {
        key: {k: v[None] for k, v in values.items()} for key, values in obs.items()
    }
    wrapped = wrapper._wrap_obs(batched)
    np.testing.assert_allclose(wrapped["states"][0].numpy(), [0, 1, 2, 3, 4, 5, 0.04])


def test_only_right_receives_bounded_action():
    env = DOSW1Env(env_config(max_joint_delta=0.01), None, None, 0)
    env.sdk = Mock()
    actual = env._execute_model_action(np.ones(7))
    env.sdk.left_go_joint.assert_not_called()
    env.sdk.right_go_joint.assert_called_once()
    np.testing.assert_allclose(actual, [0.01] * 6 + [0.07])


def test_read_only_dispatch_and_nonfinite_actions():
    env = DOSW1Env(env_config(read_only=True), None, None, 0)
    env.sdk = Mock()
    env._dispatch_action(np.ones(7))
    assert env.sdk.mock_calls == []
    with pytest.raises(ValueError, match="NaN"):
        env.step(np.full(7, np.nan))


def test_native_read_only_guard():
    backend = object.__new__(NativeAirbot)
    backend.config = env_config(read_only=True)
    backend.arms = {"right": Mock()}
    with pytest.raises(RuntimeError, match="Motion disabled"):
        backend.command("right", [0] * 6, 0)
    assert backend.arms["right"].mock_calls == []


def test_no_leader_in_single_arm_mode():
    with pytest.raises(ValueError, match="leader"):
        DOSW1Env(env_config(enable_human_in_loop=True), None, None, 0)


def test_right_task_reward_uses_right_arm_and_reports_success():
    env = PickEnv(
        {
            "is_dummy": True,
            "active_arms": ["right"],
            "task_arm": "right",
            "camera_names": ["cam_front"],
            "enable_camera_player": False,
            "target_lift_joint": [0] * 6,
        }
    )
    env.config.is_dummy = False
    env.phase = "lift"
    env.robot_state.left_joint_positions = np.full(6, 2.0)
    assert env._calc_step_reward({}) == 1.0
    assert env.task_success


def publish(directory, **updates):
    frame = np.full((4, 6, 3), [1, 2, 3], dtype=np.uint8)
    frame.tofile(directory / "color.bgr")
    meta = {
        "state": "ready",
        "serial": "test",
        "frame_id": 1,
        "captured_at": time.time(),
        "color_height": 4,
        "color_width": 6,
    }
    meta.update(updates)
    (directory / "meta.json").write_text(json.dumps(meta))
    return frame


def test_shared_camera_fresh_frame_and_stale_rejection(tmp_path):
    expected = publish(tmp_path)
    camera = SharedFrameCamera("cam_front", tmp_path, "test", timeout_s=0.03)
    np.testing.assert_array_equal(camera.get_frame(), expected)
    publish(tmp_path, captured_at=time.time() - 5)
    with pytest.raises(TimeoutError):
        camera.get_frame()


def test_shared_camera_rejects_wrong_serial_and_incomplete_write(tmp_path):
    publish(tmp_path, serial="wrong")
    camera = SharedFrameCamera("cam_front", tmp_path, "test", timeout_s=0.03)
    with pytest.raises(ValueError, match="serial"):
        camera.get_frame()
    publish(tmp_path, state="writing")
    with pytest.raises(TimeoutError):
        camera.get_frame()


def test_composed_right_top_config_and_vector_environment(monkeypatch):
    from pathlib import Path

    from hydra import compose, initialize_config_dir

    from rlinf.scheduler.cluster.config import ClusterConfig

    repo = Path(__file__).resolve().parents[2]
    with initialize_config_dir(
        version_base=None, config_dir=str(repo / "examples/embodiment/config")
    ):
        cfg = compose(config_name="dosw1_right_top_sac_flow_procvlm")
    ClusterConfig(**cfg.cluster)
    assert cfg.reward.use_reward_model
    assert cfg.reward.model.input_builder_params.image_keys == ["reward_main_images"]
    assert cfg.actor.model.state_dim == cfg.actor.model.action_dim == 7
    assert cfg.env.eval.override_cfg.active_arms == ["right"]
    cfg.env.train.override_cfg.is_dummy = True
    env = RealWorldEnv(cfg.env.train, 1, 0, 1, None)
    try:
        obs, _ = env.reset()
        assert tuple(obs["states"].shape) == (1, 7)
        assert tuple(obs["reward_main_images"].shape) == (1, 1, 336, 336, 3)
        env.step(np.zeros((1, 7)))
    finally:
        env.close()


def test_native_command_rechecks_fresh_state_before_sending():
    from airbot_py.arm import RobotMode

    backend = object.__new__(NativeAirbot)
    backend.config = env_config(max_joint_delta=0.01)
    arm = Mock()
    arm.get_control_mode.return_value = RobotMode.SERVO_JOINT_POS
    backend.arms = {"right": arm}
    backend.joint = lambda side: np.zeros(7)
    backend.kinematics = lambda joints: np.zeros(7)
    actual = backend.command("right", [1] * 6, 0.07)
    np.testing.assert_allclose(actual, [0.01] * 6 + [0.005])
    arm.servo_joint_pos.assert_called_once_with([0.01] * 6)
    arm.servo_eef_pos.assert_called_once_with([0.005])
    arm.switch_mode.assert_not_called()


def test_native_rejects_outside_workspace_without_sending():
    backend = object.__new__(NativeAirbot)
    backend.config = env_config(
        max_joint_delta=0.01,
        right_ee_pose_limit_min=np.zeros(3),
        right_ee_pose_limit_max=np.ones(3),
    )
    backend.arms = {"right": Mock()}
    backend.joint = lambda side: np.zeros(7)
    backend.kinematics = lambda joints: np.full(7, 2.0)
    with pytest.raises(ValueError, match="safety box"):
        backend.command("right", [0] * 6, 0)
    assert backend.arms["right"].mock_calls == []


def test_native_rejects_stale_feedback():
    backend = object.__new__(NativeAirbot)
    backend.config = env_config(state_timeout_s=0.01)
    arm = Mock()
    arm._feedback_jointstates = object()
    arm._feedbacking = True
    backend.arms = {"right": arm}
    with pytest.raises(TimeoutError, match="fresh"):
        backend.joint("right")
    arm.get_joint_pos.assert_not_called()


def test_flow_policy_cpu_forward_and_checkpoint_shape(tmp_path):
    from pathlib import Path

    import torch
    from hydra import compose, initialize_config_dir

    from rlinf.models.embodiment.flow_policy import get_model
    from rlinf.models.embodiment.modules.resnet_utils import ResNet10

    repo = Path(__file__).resolve().parents[2]
    with initialize_config_dir(
        version_base=None, config_dir=str(repo / "examples/embodiment/config")
    ):
        cfg = compose(config_name="dosw1_right_top_sac_flow")
    # A random fixture validates software only, never a usable robot policy.
    torch.save(ResNet10().state_dict(), tmp_path / "resnet10_pretrained.pt")
    cfg.actor.model.model_path = str(tmp_path)
    model = get_model(cfg.actor.model, torch_dtype=torch.float32)
    with torch.no_grad():
        actions, _ = model.predict_action_batch(
            {
                "states": torch.zeros(1, 7),
                "main_images": torch.zeros(1, 128, 128, 3, dtype=torch.uint8),
            }
        )
    assert actions.shape == (1, 1, 7)
    assert torch.isfinite(actions).all()
    torch.save(model.state_dict(), tmp_path / "policy.pt")
    model.load_state_dict(
        torch.load(tmp_path / "policy.pt", weights_only=True), strict=True
    )


def test_prepare_writes_read_only_config_and_blocks_uncalibrated_execute(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    from omegaconf import OmegaConf

    repo = Path(__file__).resolve().parents[2]
    for name in [
        "resnet10_pretrained.pt",
        "policy.pt",
        "metadata.json",
        "trajectory_index.json",
        "config.json",
    ]:
        (tmp_path / name).write_text("{}")
    environment = {
        **os.environ,
        "PYTHONPATH": str(repo),
        "DOSW1_POLICY_MODEL_PATH": str(tmp_path),
        "DOSW1_POLICY_CHECKPOINT": str(tmp_path / "policy.pt"),
        "DOSW1_DEMO_BUFFER_PATH": str(tmp_path),
        "PROCVLM_REPO_PATH": str(tmp_path),
        "PROCVLM_PYTHON": sys.executable,
        "PROCVLM_REWARD_MODEL_PATH": str(tmp_path),
    }
    command = [
        sys.executable,
        str(repo / "toolkits/dosw1/prepare_experiment.py"),
        "--calibration",
        str(repo / "toolkits/dosw1/calibration.example.yaml"),
        "--output-dir",
        str(tmp_path / "prepared"),
    ]
    result = subprocess.run(
        command, env=environment, text=True, capture_output=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    config = OmegaConf.load(tmp_path / "prepared/experiment.yaml")
    assert config.env.train.override_cfg.read_only
    assert config.env.eval.override_cfg.read_only
    assert config.reward.use_reward_model
    blocked = subprocess.run(
        command + ["--execute"],
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert blocked.returncode != 0
    assert "operator-validated" in blocked.stderr
