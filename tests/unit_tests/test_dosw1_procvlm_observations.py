# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
import torch

from rlinf.envs.realworld.dosw1.dosw1_env import DOSW1Config, DOSW1Env
from rlinf.envs.realworld.dosw1.tasks.pick import PickEnv
from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.models.embodiment.reward.vlm_reward_utils.input_builder import (
    ProcVLMProgressInputBuilder,
)

CAMERAS = ["cam_front", "cam_left", "cam_right"]


def _constant_image(value: int, size: int) -> np.ndarray:
    return np.full((1, size, size, 3), value, dtype=np.uint8)


def test_dosw1_dummy_keeps_actor_and_reward_image_streams_separate():
    env = DOSW1Env(
        DOSW1Config(
            is_dummy=True,
            enable_camera_player=False,
            reward_camera_names=CAMERAS,
            reward_image_size=24,
        ),
        worker_info=None,
        hardware_info=None,
        env_idx=0,
    )

    obs, _ = env.reset()

    assert list(obs["frames"]) == CAMERAS
    assert list(obs["reward_frames"]) == CAMERAS
    assert all(frame.shape == (128, 128, 3) for frame in obs["frames"].values())
    assert all(frame.shape == (24, 24, 3) for frame in obs["reward_frames"].values())


def test_dosw1_rejects_unavailable_reward_camera():
    with pytest.raises(ValueError, match="unavailable cameras"):
        DOSW1Env(
            DOSW1Config(
                is_dummy=True,
                enable_camera_player=False,
                reward_camera_names=["cam_missing"],
            ),
            worker_info=None,
            hardware_info=None,
            env_idx=0,
        )


def test_dosw1_camera_capture_reuses_frame_for_actor_and_reward_stream():
    class FakeCamera:
        name = "cam_front"

        @staticmethod
        def get_frame():
            # Camera backends provide BGR. Use a rectangular frame to exercise
            # the shared square crop as well as both resize paths.
            return np.full((6, 10, 3), [1, 2, 3], dtype=np.uint8)

    class FakePlayer:
        received = None

        def put_frame(self, frames):
            self.received = frames

    env = DOSW1Env(
        DOSW1Config(
            is_dummy=True,
            enable_camera_player=False,
            camera_names=["cam_front"],
            reward_camera_names=["cam_front"],
            reward_image_size=24,
        ),
        worker_info=None,
        hardware_info=None,
        env_idx=0,
    )
    env._cameras = [FakeCamera()]
    env._camera_player = FakePlayer()

    frames, reward_frames = env._get_camera_frames()

    assert frames["cam_front"].shape == (128, 128, 3)
    assert reward_frames["cam_front"].shape == (24, 24, 3)
    assert frames["cam_front"][0, 0].tolist() == [3, 2, 1]
    assert reward_frames["cam_front"][0, 0].tolist() == [3, 2, 1]
    assert env._camera_player.received["cam_front"][0, 0].tolist() == [1, 2, 3]


def test_realworld_wrapper_preserves_explicit_reward_camera_order():
    env = object.__new__(RealWorldEnv)
    env.main_image_key = "cam_left"
    env.reward_image_keys = CAMERAS
    env.task_descriptions = ["Pick up the object with the left arm."]
    raw_obs = {
        "state": {
            "left": np.zeros((1, 7), dtype=np.float32),
            "right": np.ones((1, 7), dtype=np.float32),
        },
        "frames": {
            "cam_front": _constant_image(10, 12),
            "cam_left": _constant_image(20, 12),
            "cam_right": _constant_image(30, 12),
        },
        "reward_frames": {
            "cam_right": _constant_image(103, 24),
            "cam_front": _constant_image(101, 24),
            "cam_left": _constant_image(102, 24),
        },
    }

    obs = env._wrap_obs(raw_obs)

    assert tuple(obs["main_images"].shape) == (1, 12, 12, 3)
    assert tuple(obs["reward_main_images"].shape) == (1, 3, 24, 24, 3)
    assert obs["main_images"][0, 0, 0, 0].item() == 20
    assert obs["reward_main_images"][0, :, 0, 0, 0].tolist() == [101, 102, 103]


def test_procvlm_multiview_history_is_timestep_major_then_camera_major():
    builder = ProcVLMProgressInputBuilder(
        history_buffer_names=["history_window"],
        history_buffer_name="history_window",
        image_keys=["reward_main_images"],
    )
    # One environment, two timesteps, three views per timestep. The wrapper has
    # already ordered the views as front, left, right.
    history = {
        "reward_main_images": [
            [
                torch.stack(
                    [
                        torch.full((4, 4, 3), value, dtype=torch.uint8)
                        for value in timestep_values
                    ]
                )
                for timestep_values in ((11, 12, 13), (21, 22, 23))
            ]
        ]
    }

    images = builder.extract_history_images(history, ["reward_main_images"])

    assert len(images) == 1
    assert len(images[0]) == 6
    assert [np.asarray(image)[0, 0, 0] for image in images[0]] == [
        11,
        12,
        13,
        21,
        22,
        23,
    ]


def test_pick_task_description_is_configurable():
    env = PickEnv(
        override_cfg={
            "is_dummy": True,
            "enable_camera_player": False,
            "task_description": "Lift the red object with the left arm.",
        }
    )

    assert env.task_description == "Lift the red object with the left arm."
