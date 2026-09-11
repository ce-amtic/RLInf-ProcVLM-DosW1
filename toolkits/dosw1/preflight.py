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

"""Check the right/top environment without moving hardware or starting training."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.setdefault("REPO_PATH", str(REPO))
os.environ.setdefault("EMBODIED_PATH", str(REPO / "examples/embodiment"))


def compose_config(name):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(
        version_base=None, config_dir=str(REPO / "examples/embodiment/config")
    ):
        cfg = compose(config_name=name)
    return cfg


def main():
    from omegaconf import OmegaConf

    from rlinf.envs.realworld.dosw1.tasks.pick import PickEnv
    from rlinf.envs.realworld.realworld_env import RealWorldEnv

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--config", default="dosw1_right_top_sac_flow_procvlm")
    parser.add_argument("--output")
    args = parser.parse_args()
    cfg = compose_config(args.config)
    params = OmegaConf.to_container(cfg.env.train.override_cfg, resolve=True)
    params.update(
        is_dummy=not args.hardware,
        read_only=True,
        move_on_init=False,
        reset_to_home=False,
        enable_human_in_loop=False,
        enable_camera_player=False,
    )
    env = PickEnv(params)
    try:
        obs, _ = env.reset()
        samples = []
        for _ in range(5):
            raw, reward, terminated, truncated, info = env.step(
                env.action_space.sample()
            )
            samples.append(raw["state"]["right"].tolist())
        assert env.action_space.shape == (7,)
        assert raw["frames"]["cam_front"].shape == (128, 128, 3)
        assert raw["reward_frames"]["cam_front"].shape == (336, 336, 3)
        # Check the exact tensor boundary used by the worker, including state order.
        import numpy as np

        wrapper = object.__new__(RealWorldEnv)
        wrapper.main_image_key = "cam_front"
        wrapper.reward_image_keys = ["cam_front"]
        wrapper.task_descriptions = [env.task_description]
        batched = {
            k: {n: np.expand_dims(v, 0) for n, v in values.items()}
            for k, values in raw.items()
        }
        wrapped = wrapper._wrap_obs(batched)
        assert tuple(wrapped["states"].shape) == (1, 7)
        np.testing.assert_allclose(
            wrapped["states"][0].numpy(), raw["state"]["right"], rtol=1e-6, atol=1e-8
        )
        report = {
            "hardware": args.hardware,
            "read_only": True,
            "config": args.config,
            "action_dim": 7,
            "state_order": "right joint1..6 radians, gripper metres",
            "tensor_shapes": {
                k: list(v.shape) for k, v in wrapped.items() if hasattr(v, "shape")
            },
            "samples": samples,
            "timestamp": time.time(),
        }
        if args.hardware:
            joint = env.sdk.get_right_joint()
            fk = env.sdk.forward_kinematics(joint[:6])
            actual = env.sdk.get_right_pose()
            error = float(np.linalg.norm(fk[:3] - actual[:3]))
            report.update(fk_position_error_m=error, right_pose=actual.tolist())
            if error > 0.005:
                raise RuntimeError(f"URDF does not match driver FK: {error:.4f} m")
        print(json.dumps(report, indent=2))
        if args.output:
            Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    finally:
        env.close()


if __name__ == "__main__":
    main()
