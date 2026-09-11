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

"""Resolve the two-node experiment and check GPU-local assets without starting Ray."""

import argparse
import subprocess
import sys
from pathlib import Path

from preflight import REPO, compose_config


def main():
    import numpy as np
    from omegaconf import OmegaConf

    from rlinf.scheduler.cluster.config import ClusterConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="dosw1_right_top_sac_flow_procvlm")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO / "logs/dosw1-prepared")
    parser.add_argument(
        "--execute", action="store_true", help="Actually start training on the GPU head"
    )
    args = parser.parse_args()
    cfg = compose_config(args.config)
    calibration = OmegaConf.load(args.calibration)
    allowed = {
        "right_reset_joint",
        "right_reset_gripper",
        "joint_limit_min",
        "joint_limit_max",
        "right_ee_pose_limit_min",
        "right_ee_pose_limit_max",
        "target_grasp_joint",
        "target_lift_joint",
        "max_joint_delta",
        "reset_to_home",
    }
    changes = dict(calibration.get("env", {}))
    unknown = set(changes) - allowed
    if unknown:
        raise ValueError(f"Unknown calibration keys: {sorted(unknown)}")
    cfg.env.train.override_cfg = OmegaConf.merge(cfg.env.train.override_cfg, changes)
    cfg.env.train.override_cfg.read_only = not args.execute
    params = cfg.env.train.override_cfg
    for key, size in (
        ("right_reset_joint", 6),
        ("joint_limit_min", 6),
        ("joint_limit_max", 6),
        ("right_ee_pose_limit_min", 3),
        ("right_ee_pose_limit_max", 3),
    ):
        value = np.asarray(params[key], dtype=float)
        if value.shape != (size,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid {key}")
    if not 0 < params.max_joint_delta <= 0.05:
        raise ValueError("Commissioning max_joint_delta must be in (0, 0.05]")
    for lo, hi in (
        ("joint_limit_min", "joint_limit_max"),
        ("right_ee_pose_limit_min", "right_ee_pose_limit_max"),
    ):
        if not np.all(np.asarray(params[lo]) < np.asarray(params[hi])):
            raise ValueError(f"Invalid interval {lo}/{hi}")
    resolved = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(resolved)
    ClusterConfig(**cfg.cluster)  # schema only; never creates a Ray cluster
    required = [
        Path(cfg.actor.model.model_path) / cfg.actor.model.encoder_config.ckpt_name,
        Path(cfg.runner.ckpt_path),
        Path(cfg.algorithm.demo_buffer.load_path) / "metadata.json",
        Path(cfg.algorithm.demo_buffer.load_path) / "trajectory_index.json",
    ]
    if cfg.reward.use_reward_model:
        required.extend(
            [
                Path(cfg.reward.model.model_path) / "config.json",
                Path(cfg.reward.model.procvlm_python),
                Path(cfg.reward.model.procvlm_repo_path),
            ]
        )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing experiment assets:\n" + "\n".join(missing))
    if args.execute and not calibration.get("motion_validated", False):
        raise ValueError(
            "Live execution requires an operator-validated motion calibration"
        )
    if args.execute and not params.reset_to_home:
        raise ValueError("Live episodic training requires a validated home reset")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir.resolve() / "experiment.yaml"
    OmegaConf.save(cfg, output)
    print(f"Prepared {output}; read_only={not args.execute}")
    if args.execute:
        subprocess.run(
            [
                sys.executable,
                str(REPO / "examples/embodiment/train_embodied_agent.py"),
                "--config-path",
                str(output.parent),
                "--config-name",
                output.stem,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
