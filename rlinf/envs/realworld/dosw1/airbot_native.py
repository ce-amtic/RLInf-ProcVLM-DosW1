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

"""AIRBOT 5.1.6 backend without the site-specific dual-arm AirbotRobot class."""

import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

if TYPE_CHECKING:
    from .dosw1_env import DOSW1Config


class URDFKinematics:
    """Evaluate a serial URDF chain without ROS or PyKDL."""

    def __init__(self, path: str, tip: str = "end_link") -> None:
        root = ET.parse(Path(path)).getroot()
        by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
        chain = []
        while tip in by_child:
            joint = by_child[tip]
            chain.append(joint)
            tip = joint.find("parent").get("link")
        if tip != "base_link":
            raise ValueError("URDF chain must connect base_link to end_link")
        self.chain = list(reversed(chain))
        if sum(j.get("type") != "fixed" for j in self.chain) != 6:
            raise ValueError("Expected six movable joints in AIRBOT URDF")

    def __call__(self, joints: list[float] | np.ndarray) -> np.ndarray:
        joints = np.asarray(joints, dtype=float)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise ValueError("Expected six finite joint angles")
        transform = np.eye(4)
        index = 0
        for joint in self.chain:
            origin = joint.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            fixed = np.eye(4)
            fixed[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
            fixed[:3, 3] = xyz
            transform = transform @ fixed
            if joint.get("type") != "fixed":
                if joint.get("type") not in ("revolute", "continuous"):
                    raise ValueError("Only revolute AIRBOT joints are supported")
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                moving = np.eye(4)
                moving[:3, :3] = Rotation.from_rotvec(axis * joints[index]).as_matrix()
                transform = transform @ moving
                index += 1
        return np.r_[
            transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_quat()
        ]


class NativeAirbot:
    """Connect only configured followers; connecting never changes control mode."""

    def __init__(self, config: "DOSW1Config") -> None:
        from airbot_py.arm import AIRBOTPlay

        self.config = config
        self.arms = {}
        self.kinematics = URDFKinematics(config.urdf_path)
        if config.enable_human_in_loop:
            raise ValueError(
                "airbot_native has no leader arms; disable enable_human_in_loop"
            )
        try:
            for side in config.active_arms:
                arm = AIRBOTPlay(
                    url=config.robot_url, port=getattr(config, f"{side}_arm_port")
                )
                self.arms[side] = arm
                self._connect(arm, side)
                self.joint(side)
        except Exception:
            self.close()
            raise

    def _connect(self, arm, side):
        result = []

        def connect():
            try:
                result.append(arm.connect())
            except Exception as exc:
                result.append(exc)

        thread = threading.Thread(target=connect, daemon=True)
        thread.start()
        thread.join(3.0)
        if thread.is_alive():
            channel = arm._channel
            if channel is not None:
                channel.close()
            thread.join(1.0)
            raise TimeoutError(f"Timed out connecting {side} AIRBOT follower")
        if not result or result[0] is not True:
            raise ConnectionError(f"Cannot connect to {side} AIRBOT follower: {result}")

    def joint(self, side: str) -> np.ndarray:
        if side not in self.arms:
            raise ValueError(f"Inactive arm: {side}")
        arm = self.arms[side]
        previous = arm._feedback_jointstates
        deadline = time.monotonic() + self.config.state_timeout_s
        while arm._feedback_jointstates is previous:
            if time.monotonic() >= deadline or not arm._feedbacking:
                raise TimeoutError(f"No fresh {side} AIRBOT joint feedback")
            time.sleep(0.002)
        value = np.r_[arm.get_joint_pos(), arm.get_eef_pos()]
        if value.shape != (7,) or not np.isfinite(value).all():
            raise RuntimeError(f"Invalid {side} AIRBOT feedback: {value}")
        return value

    def pose(self, side: str) -> np.ndarray:
        self.joint(side)
        value = np.concatenate(self.arms[side].get_end_pose())
        if value.shape != (7,) or not np.isfinite(value).all():
            raise RuntimeError(f"Invalid {side} AIRBOT pose")
        return value

    def command(
        self, side: str, joints: list[float], gripper: float, *, interp: bool = False
    ) -> np.ndarray:
        if self.config.read_only:
            raise RuntimeError("Motion disabled: read_only=True")
        if side not in self.arms:
            raise ValueError(f"Inactive arm: {side}")
        target = np.r_[joints, gripper]
        if target.shape != (7,) or not np.isfinite(target).all():
            raise ValueError("Expected seven finite action values")
        current = self.joint(side)
        cfg = self.config
        if not np.isfinite(cfg.max_joint_delta) or cfg.max_joint_delta <= 0:
            raise ValueError("Native motion requires a finite positive max_joint_delta")
        # Recheck against fresh feedback; the arm may have moved since env.step.
        target[:6] = np.clip(
            target[:6],
            current[:6] - cfg.max_joint_delta,
            current[:6] + cfg.max_joint_delta,
        )
        if np.any(target[:6] < cfg.joint_limit_min) or np.any(
            target[:6] > cfg.joint_limit_max
        ):
            raise ValueError("Target outside absolute joint limits")
        if not cfg.gripper_width_min <= gripper <= cfg.gripper_width_max:
            raise ValueError("Target outside gripper limits")
        target[6] = np.clip(
            gripper,
            current[6] - cfg.max_gripper_delta,
            current[6] + cfg.max_gripper_delta,
        )
        xyz = self.kinematics(target[:6])[:3]
        if np.any(xyz < getattr(cfg, f"{side}_ee_pose_limit_min")) or np.any(
            xyz > getattr(cfg, f"{side}_ee_pose_limit_max")
        ):
            raise ValueError("Target outside end-effector safety box")
        from airbot_py.arm import RobotMode

        arm = self.arms[side]
        if arm.get_control_mode() != RobotMode.SERVO_JOINT_POS:
            if not arm.switch_mode(RobotMode.SERVO_JOINT_POS):
                raise RuntimeError("Failed to enter SERVO_JOINT_POS")
        arm.servo_joint_pos(target[:6].tolist())
        arm.servo_eef_pos([float(target[6])])
        return target

    def close(self) -> None:
        for arm in self.arms.values():
            # Close the gRPC stream before join: vendor disconnect otherwise
            # waits forever if the feedback server disappears.
            if arm._channel is not None:
                arm._channel.close()
            arm.disconnect()
        self.arms.clear()
