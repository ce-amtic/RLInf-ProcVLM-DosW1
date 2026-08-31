# Copyright 2025 The RLinf Authors.
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

import json
import os
from typing import Optional, Union

import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from PIL import Image

from rlinf.envs.utils import center_crop_image, list_of_dict_to_dict_of_list

__all__ = ["RoboTwinEnv"]


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class RoboTwinEnv(gym.Env):
    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
        record_metrics=True,
    ):
        env_seed = cfg.seed
        self.seed = env_seed + seed_offset
        self.num_envs = num_envs
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.auto_reset = cfg.auto_reset
        self.use_rel_reward = cfg.use_rel_reward
        self.ignore_terminations = cfg.ignore_terminations

        self.group_size = cfg.group_size
        self.num_group = self.num_envs // self.group_size
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.use_custom_reward = cfg.use_custom_reward

        self.video_cfg = cfg.video_cfg

        self.cfg = cfg
        self.record_metrics = record_metrics
        self._is_start = True
        self._trace_nonfinite_obs = _env_flag("ROBOTWIN_OBS_TRACE_NONFINITE")
        self._trace_all_obs = _env_flag("ROBOTWIN_OBS_TRACE_ALL_STATS")
        self._trace_obs_count = 0
        self._trace_obs_limit = int(os.getenv("ROBOTWIN_OBS_TRACE_LIMIT", "40"))
        self._trace_rollout = _env_flag("ROBOTWIN_TRACE_ROLLOUT_STATS")
        self._trace_rollout_count = 0
        self._trace_rollout_limit = int(os.getenv("ROBOTWIN_TRACE_ROLLOUT_LIMIT", "80"))
        self._trace_rollout_every_n = max(
            1, int(os.getenv("ROBOTWIN_TRACE_ROLLOUT_EVERY_N", "25"))
        )
        self.action_exec_horizon = cfg.get("action_exec_horizon", None)
        if self.action_exec_horizon is not None:
            self.action_exec_horizon = int(self.action_exec_horizon)
        self.execute_action_prefix_individually = bool(
            cfg.get("execute_action_prefix_individually", False)
        )

        self.task_name = cfg.task_config.task_name

        self.center_crop = cfg.get("center_crop", False)
        self._init_reset_state_ids()

        self._init_env()

        self.prev_step_reward = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        if self.record_metrics:
            self._init_metrics()
            self._elapsed_steps = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )

    def _init_env(self):
        mp.set_start_method("spawn", force=True)
        os.environ["ASSETS_PATH"] = self.cfg.assets_path

        from robotwin.envs.vector_env import VectorEnv

        env_seeds = self.reset_state_ids.tolist()

        task_config = OmegaConf.to_container(self.cfg.task_config, resolve=True)
        task_config["execute_action_prefix_individually"] = (
            self.execute_action_prefix_individually
        )

        self.venv = VectorEnv(
            task_config=task_config,
            n_envs=self.num_envs,
            env_seeds=env_seeds,
        )

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    @staticmethod
    def _normalize_env_idx(env_idx):
        if env_idx is None:
            return None
        if isinstance(env_idx, torch.Tensor):
            return [int(idx) for idx in env_idx.detach().cpu().reshape(-1).tolist()]
        if isinstance(env_idx, np.ndarray):
            return [int(idx) for idx in env_idx.reshape(-1).tolist()]
        if isinstance(env_idx, (list, tuple)):
            return [int(idx) for idx in env_idx]
        return [int(env_idx)]

    def _init_metrics(self):
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.fail_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool, device=self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            if self.record_metrics:
                self.success_once[mask] = False
                self.fail_once[mask] = False
                self.returns[mask] = 0
                self._elapsed_steps[env_idx] = 0
        else:
            self.prev_step_reward[:] = 0
            if self.record_metrics:
                self.success_once[:] = False
                self.fail_once[:] = False
                self.returns[:] = 0.0
                self._elapsed_steps[:] = 0

    def _record_metrics(self, step_reward, infos):
        episode_info = {}
        self.returns += step_reward
        if "success" in infos:
            if isinstance(infos["success"], list):
                infos["success"] = torch.as_tensor(
                    np.array(infos["success"]).reshape(-1), device=self.device
                )
            self.success_once = self.success_once | infos["success"]
            episode_info["success_once"] = self.success_once.clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = self.elapsed_steps.clone()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        infos["episode"] = episode_info
        return infos

    def center_and_crop(self, image, center_crop=False):
        image = np.array(image)

        image = Image.fromarray(image).convert("RGB")
        if center_crop:
            image = center_crop_image(image)
        return np.array(image)

    @staticmethod
    def _tensor_range(tensor, finite_mask):
        if tensor.numel() == 0:
            return "empty"
        if not tensor.dtype.is_floating_point:
            tensor_float = tensor.to(torch.float32)
            return f"{float(tensor_float.min().item()):.4g}..{float(tensor_float.max().item()):.4g}"
        finite_values = tensor[finite_mask]
        if finite_values.numel() == 0:
            return "all-nonfinite"
        finite_values = finite_values.to(torch.float32)
        return f"{float(finite_values.min().item()):.4g}..{float(finite_values.max().item()):.4g}"

    def _trace_obs_tensor(self, name, tensor, *, force=False):
        if not (self._trace_nonfinite_obs or force):
            return
        if not torch.is_tensor(tensor):
            return
        if tensor.dtype.is_floating_point:
            finite_mask = torch.isfinite(tensor)
            invalid_count = int((~finite_mask).sum().item())
        else:
            finite_mask = torch.ones_like(tensor, dtype=torch.bool)
            invalid_count = 0
        should_log = force or invalid_count > 0
        if should_log and self._trace_obs_count < self._trace_obs_limit:
            self._trace_obs_count += 1
            print(
                "RoboTwin obs trace "
                f"rank_info={self.worker_info} {name}: shape={tuple(tensor.shape)} "
                f"dtype={tensor.dtype} invalid={invalid_count}/{tensor.numel()} "
                f"range={self._tensor_range(tensor, finite_mask)}",
                flush=True,
            )

    @staticmethod
    def _to_numpy(values):
        if isinstance(values, torch.Tensor):
            return values.detach().cpu().numpy()
        return np.asarray(values)

    @staticmethod
    def _array_range(values):
        arr = RoboTwinEnv._to_numpy(values)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return "all-nonfinite"
        return f"{float(finite.min()):.4g}..{float(finite.max()):.4g}"

    @staticmethod
    def _array_dim_ranges(values, max_dims=14):
        arr = RoboTwinEnv._to_numpy(values)
        if arr.size == 0:
            return "empty"
        arr = arr.reshape(-1, arr.shape[-1])
        ranges = []
        for idx in range(min(arr.shape[-1], max_dims)):
            dim = arr[:, idx]
            finite = dim[np.isfinite(dim)]
            if finite.size == 0:
                ranges.append(f"{idx}:nonfinite")
            else:
                ranges.append(
                    f"{idx}:{float(finite.min()):.3g}..{float(finite.max()):.3g}"
                )
        return ",".join(ranges)

    def _trace_rollout_step(
        self,
        chunk_actions,
        step_reward,
        terminations,
        truncations,
        infos,
        *,
        chunk_step: int,
    ):
        if not self._trace_rollout:
            return
        if self._trace_rollout_count >= self._trace_rollout_limit:
            return
        should_log = (
            self._trace_rollout_count < 3
            or self._trace_rollout_count % self._trace_rollout_every_n == 0
        )
        self._trace_rollout_count += 1
        if not should_log:
            return

        success = infos.get("success") if isinstance(infos, dict) else None
        success_arr = self._to_numpy(success).reshape(-1) if success is not None else None
        success_sum = int(success_arr.astype(bool).sum()) if success_arr is not None else 0
        term_sum = int(torch.as_tensor(terminations).bool().sum().item())
        trunc_sum = int(torch.as_tensor(truncations).bool().sum().item())
        reward_sum = float(torch.as_tensor(step_reward).float().sum().item())
        actual_exec_steps = infos.get("actual_exec_steps") if isinstance(infos, dict) else None
        if actual_exec_steps is not None:
            actual_exec_steps = self._to_numpy(actual_exec_steps).reshape(-1)
            actual_exec_range = (
                f"{int(actual_exec_steps.min())}..{int(actual_exec_steps.max())}"
                if actual_exec_steps.size
                else "empty"
            )
        else:
            actual_exec_range = "n/a"
        print(
            "RoboTwin rollout trace "
            f"rank_info={self.worker_info} count={self._trace_rollout_count} "
            f"task={self.task_name} chunk_step={chunk_step} "
            f"elapsed={int(self._elapsed_steps.min().item())}..{int(self._elapsed_steps.max().item())} "
            f"actions_shape={tuple(chunk_actions.shape)} "
            f"actions_range={self._array_range(chunk_actions)} "
            f"actions_dim_ranges={self._array_dim_ranges(chunk_actions)} "
            f"first_action_range={self._array_range(chunk_actions[:, 0, :])} "
            f"last_action_range={self._array_range(chunk_actions[:, -1, :])} "
            f"actual_exec_steps={actual_exec_range} "
            f"success_sum={success_sum}/{self.num_envs} "
            f"term_sum={term_sum} trunc_sum={trunc_sum} reward_sum={reward_sum:.4g}",
            flush=True,
        )

    def _extract_obs_image(self, raw_obs):
        batch_images = []
        batch_wrist_images = []
        batch_left_wrist_images = []
        batch_right_wrist_images = []
        batch_states = []
        batch_instructions = []
        for obs in raw_obs:
            batch_images.append(
                self.center_and_crop(obs["full_image"], center_crop=self.center_crop)
            )
            wrist_images = []
            left_wrist_image = None
            right_wrist_image = None
            if "left_wrist_image" in obs and obs["left_wrist_image"] is not None:
                left_wrist_image = self.center_and_crop(
                    obs["left_wrist_image"], center_crop=self.center_crop
                )
                wrist_images.append(left_wrist_image)
            if "right_wrist_image" in obs and obs["right_wrist_image"] is not None:
                right_wrist_image = self.center_and_crop(
                    obs["right_wrist_image"], center_crop=self.center_crop
                )
                wrist_images.append(right_wrist_image)
            batch_wrist_images.append(
                torch.stack([torch.from_numpy(img) for img in wrist_images])
                if wrist_images
                else None
            )
            batch_states.append(obs["state"])
            batch_instructions.append(obs["instruction"])
            batch_left_wrist_images.append(
                torch.from_numpy(left_wrist_image) if left_wrist_image is not None else None
            )
            batch_right_wrist_images.append(
                torch.from_numpy(right_wrist_image) if right_wrist_image is not None else None
            )

        batch_images = torch.stack([torch.from_numpy(img) for img in batch_images])
        if batch_wrist_images and all(image is not None for image in batch_wrist_images):
            batch_wrist_images = torch.stack(batch_wrist_images)
        else:
            batch_wrist_images = None
        batch_left_wrist_images = (
            torch.stack(batch_left_wrist_images)
            if all(image is not None for image in batch_left_wrist_images)
            else None
        )
        batch_right_wrist_images = (
            torch.stack(batch_right_wrist_images)
            if all(image is not None for image in batch_right_wrist_images)
            else None
        )
        batch_states = torch.stack([torch.from_numpy(state) for state in batch_states])
        self._trace_obs_tensor(
            "main_images", batch_images, force=self._trace_all_obs and self._trace_obs_count < self._trace_obs_limit
        )
        if batch_wrist_images is not None:
            self._trace_obs_tensor(
                "wrist_images",
                batch_wrist_images,
                force=self._trace_all_obs and self._trace_obs_count < self._trace_obs_limit,
            )
        if batch_left_wrist_images is not None:
            self._trace_obs_tensor(
                "left_wrist_images",
                batch_left_wrist_images,
                force=self._trace_all_obs and self._trace_obs_count < self._trace_obs_limit,
            )
        if batch_right_wrist_images is not None:
            self._trace_obs_tensor(
                "right_wrist_images",
                batch_right_wrist_images,
                force=self._trace_all_obs and self._trace_obs_count < self._trace_obs_limit,
            )
        self._trace_obs_tensor(
            "states", batch_states, force=self._trace_all_obs and self._trace_obs_count < self._trace_obs_limit
        )
        if self._trace_all_obs and self._trace_obs_count < self._trace_obs_limit and batch_instructions:
            self._trace_obs_count += 1
            print(
                "RoboTwin obs trace "
                f"rank_info={self.worker_info} instruction0={batch_instructions[0]!r}",
                flush=True,
            )

        extracted_obs = {
            "main_images": batch_images,
            "wrist_images": batch_wrist_images,
            "left_wrist_images": batch_left_wrist_images,
            "right_wrist_images": batch_right_wrist_images,
            "states": batch_states,
            "task_descriptions": batch_instructions,
        }

        return extracted_obs

    def _extract_reward_step_images(self, infos, extracted_obs, target_steps: int):
        target_steps = max(1, int(target_steps))
        step_observations = (
            infos.get("step_observations") if isinstance(infos, dict) else None
        )
        fallback_images = extracted_obs["main_images"]
        batch_images = []

        for env_id in range(self.num_envs):
            env_step_observations = []
            if isinstance(step_observations, (list, tuple)) and env_id < len(
                step_observations
            ):
                env_step_observations = step_observations[env_id] or []

            frames = []
            for step_obs in env_step_observations[:target_steps]:
                if not isinstance(step_obs, dict) or "full_image" not in step_obs:
                    continue
                frame = self.center_and_crop(
                    step_obs["full_image"], center_crop=self.center_crop
                )
                frames.append(torch.from_numpy(frame))

            if not frames:
                frames = [fallback_images[env_id].clone()]

            while len(frames) < target_steps:
                frames.append(frames[-1].clone())
            frames = frames[:target_steps]
            batch_images.append(torch.stack(frames))

        return torch.stack(batch_images)

    def _calc_step_reward(self, terminations):
        reward = self.cfg.reward_coef * terminations

        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward

        if self.use_rel_reward:
            return reward_diff
        else:
            return reward

    def _cal_chunk_rewards(self, step_reward, chunk_step, terminations, infos):
        if isinstance(infos, dict) and "actual_exec_steps" in infos:
            actual_exec_steps = np.asarray(infos["actual_exec_steps"]).reshape(-1)
            n_steps_to_run = np.maximum(0, chunk_step - actual_exec_steps)
        else:
            n_steps_to_run = np.array(
                [[0] for i in range(self.num_envs)]
            )  # infos.get("n_steps_to_run", np.array([[0] for i in range(self.num_envs)]))

        n_steps_to_run = torch.as_tensor(
            np.array(n_steps_to_run).reshape(-1), device=self.device
        )
        chunk_rewards = torch.zeros(self.num_envs, chunk_step, device=self.device)
        for env_id in range(self.num_envs):
            steps_left = n_steps_to_run[env_id]
            reward = step_reward[env_id]
            start_idx = chunk_step - steps_left - 1

            if terminations[env_id] and start_idx >= 0:
                if self.use_rel_reward:
                    chunk_rewards[env_id, start_idx] = reward
                else:
                    chunk_rewards[env_id, start_idx:] = reward

        return chunk_rewards

    def reset(
        self,
        env_idx: Optional[Union[int, list[int]]] = None,
        env_seeds=None,
    ):
        if self._is_start:
            self._is_start = False

        env_idx_list = self._normalize_env_idx(env_idx)
        if env_seeds is None:
            if env_idx_list is None:
                env_seeds = self.reset_state_ids.tolist()
            else:
                env_seeds = self.reset_state_ids[env_idx_list].tolist()

        self.venv.reset(env_idx=env_idx, env_seeds=env_seeds)
        raw_obs = self.venv.get_obs()
        infos = {}

        self._reset_metrics(env_idx)

        extracted_obs = self._extract_obs_image(raw_obs)

        return extracted_obs, infos

    def step(
        self, actions: Union[torch.Tensor, np.ndarray, dict] = None, auto_reset=True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if actions is None:
            assert self._is_start, "Actions must be provided after the first reset."

        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        elif isinstance(actions, dict):
            actions = actions.get("actions", actions)

        # [n_envs, horizon, action_dim]
        if len(actions.shape) == 2:
            # [n_envs, action_dim] -> [n_envs, 1, action_dim]
            actions = actions[:, None, :]

        raw_obs, step_reward, terminations, truncations, info_list = self.venv.step(
            actions
        )
        extracted_obs = self._extract_obs_image(raw_obs)
        infos = list_of_dict_to_dict_of_list(info_list)

        if isinstance(terminations, list):
            terminations = torch.as_tensor(
                np.array(terminations).reshape(-1), device=self.device
            )
        if isinstance(truncations, list):
            truncations = torch.as_tensor(
                np.array(truncations).reshape(-1), device=self.device
            )

        if self.use_custom_reward:
            step_reward = self._calc_step_reward(terminations)
        else:
            if isinstance(step_reward, list):
                step_reward = torch.as_tensor(
                    np.array(step_reward, dtype=np.float32).reshape(-1),
                    device=self.device,
                )

        self._elapsed_steps += actions.shape[1]
        truncated = self._elapsed_steps >= self.cfg.max_episode_steps
        if truncated.any():
            truncations = torch.logical_or(truncated, truncations)

        infos = self._record_metrics(step_reward, infos)

        if self.ignore_terminations:
            terminations[:] = False
            if self.record_metrics:
                if "success" in infos:
                    infos["episode"]["success_at_end"] = infos["success"].clone()

        dones = torch.logical_or(terminations, truncations)

        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            extracted_obs, infos = self._handle_auto_reset(dones, extracted_obs, infos)

        return extracted_obs, step_reward, terminations, truncations, infos

    def chunk_step(self, chunk_actions):
        if isinstance(chunk_actions, torch.Tensor):
            chunk_actions = chunk_actions.cpu().numpy()

        # chunk_actions: [num_envs, model_chunk_step, action_dim]
        model_chunk_step = chunk_actions.shape[1]
        if (
            self.action_exec_horizon is not None
            and self.action_exec_horizon > 0
            and self.action_exec_horizon < chunk_actions.shape[1]
        ):
            chunk_actions = chunk_actions[:, : self.action_exec_horizon, :]
        num_envs = chunk_actions.shape[0]
        chunk_step = chunk_actions.shape[1]
        obs_list = []
        infos_list = []

        raw_obs, step_reward, terminations, truncations, info_list = self.venv.step(
            chunk_actions
        )
        extracted_obs = self._extract_obs_image(raw_obs)
        infos = list_of_dict_to_dict_of_list(info_list)
        extracted_obs["reward_main_images"] = self._extract_reward_step_images(
            infos, extracted_obs, target_steps=chunk_step
        )
        infos.pop("step_observations", None)
        obs_list.append(extracted_obs)
        infos_list.append(infos)
        if isinstance(terminations, list):
            terminations = torch.as_tensor(
                np.array(terminations).reshape(-1), device=self.device
            )
        if isinstance(truncations, list):
            truncations = torch.as_tensor(
                np.array(truncations).reshape(-1), device=self.device
            )

        if self.use_custom_reward:
            step_reward = self._calc_step_reward(terminations)
        else:
            if isinstance(step_reward, list):
                step_reward = torch.as_tensor(
                    np.array(step_reward, dtype=np.float32).reshape(-1),
                    device=self.device,
                )

        chunk_rewards = self._cal_chunk_rewards(
            step_reward, chunk_step, terminations, infos
        )
        if model_chunk_step > chunk_step:
            padded_rewards = torch.zeros(
                self.num_envs, model_chunk_step, device=self.device
            )
            padded_rewards[:, :chunk_step] = chunk_rewards
            chunk_rewards = padded_rewards

        actual_exec_steps = infos.get("actual_exec_steps") if isinstance(infos, dict) else None
        valid_action_steps = chunk_step
        if actual_exec_steps is not None:
            elapsed_delta = torch.as_tensor(
                np.array(actual_exec_steps).reshape(-1),
                dtype=self._elapsed_steps.dtype,
                device=self._elapsed_steps.device,
            )
            self._elapsed_steps += elapsed_delta
            valid_action_steps = elapsed_delta.detach().cpu()
        else:
            self._elapsed_steps += chunk_actions.shape[1]
            valid_action_steps = torch.full((num_envs,), chunk_step, dtype=torch.long)
        valid_action_steps = torch.as_tensor(valid_action_steps, dtype=torch.long).clamp(
            min=0, max=model_chunk_step
        )
        valid_action_mask = torch.zeros((num_envs, model_chunk_step), dtype=torch.bool)
        for env_id, valid_steps in enumerate(valid_action_steps.tolist()):
            valid_action_mask[env_id, :valid_steps] = True
        infos["valid_action_mask"] = valid_action_mask
        truncated = self._elapsed_steps >= self.cfg.max_episode_steps
        if truncated.any():
            truncations = torch.logical_or(truncated, truncations)

        infos = self._record_metrics(step_reward, infos)
        self._trace_rollout_step(
            chunk_actions,
            step_reward,
            terminations,
            truncations,
            infos,
            chunk_step=chunk_step,
        )

        if self.ignore_terminations:
            terminations[:] = False
            if self.record_metrics:
                if "success" in infos:
                    infos["episode"]["success_at_end"] = infos["success"].clone()

        past_dones = torch.logical_or(terminations, truncations)
        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones, obs_list[-1], infos_list[-1]
            )

        done_step_idx = torch.full((num_envs,), chunk_step - 1, dtype=torch.long)
        if actual_exec_steps is not None:
            actual_exec_steps = torch.as_tensor(
                np.array(actual_exec_steps).reshape(-1), dtype=torch.long
            )
            done_step_idx = torch.clamp(actual_exec_steps, min=1, max=chunk_step) - 1

        row_idx = torch.arange(num_envs)
        chunk_terminations = torch.zeros((num_envs, model_chunk_step), dtype=bool)
        chunk_terminations[row_idx, done_step_idx] = terminations.cpu().to(torch.bool)

        chunk_truncations = torch.zeros((num_envs, model_chunk_step), dtype=bool)
        chunk_truncations[row_idx, done_step_idx] = truncations.cpu().to(torch.bool)

        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _handle_auto_reset(self, dones, extracted_obs, infos):
        final_obs = extracted_obs.copy()
        env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]
        final_info = infos.copy()
        if self.cfg.is_eval:
            self.update_reset_state_ids(env_idx=env_idx)

        extracted_obs, infos = self.reset(env_idx=env_idx.tolist())
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        for key in ("actual_exec_steps", "valid_action_mask"):
            if key in final_info:
                infos[key] = final_info[key]
        return extracted_obs, infos

    def close(self, clear_cache=True):
        if hasattr(self, "venv"):
            self.venv.close(clear_cache)

    def sample_action_space(self):
        return np.random.randn(self.num_envs, self.horizon, 14)

    def _init_reset_state_ids(self):
        if self.cfg.get("seeds_path", None) is not None and os.path.exists(
            self.cfg.seeds_path
        ):
            with open(self.cfg.seeds_path, "r") as f:
                data = json.load(f)
            success_seeds = data[self.task_name].get("success_seeds", None)
            if success_seeds is not None:
                success_seeds = torch.as_tensor(success_seeds, dtype=torch.long)
                self._generator = torch.Generator()
                self._generator.manual_seed(self.seed)
                shuffle_generator = self._generator
                if self.cfg.is_eval:
                    shuffle_generator = torch.Generator()
                    shuffle_generator.manual_seed(int(self.cfg.seed))
                shuffled_indices = torch.randperm(
                    success_seeds.numel(), generator=shuffle_generator
                )
                shuffled_seeds = success_seeds[shuffled_indices]
                # Drop last to make total divisible by all env ranks and groups,
                # then give each rank a disjoint deterministic slice.
                total_seeds = shuffled_seeds.numel()
                partition_size = max(1, self.total_num_processes * self.num_group)
                keep_count = (total_seeds // partition_size) * partition_size
                shuffled_seeds = shuffled_seeds[:keep_count]
                self.success_seeds = shuffled_seeds.reshape(
                    self.total_num_processes, -1
                )[self.seed_offset % self.total_num_processes]
                self._current_seed_index = 0
            else:
                self.success_seeds = None
                self._current_seed_index = 0
        else:
            self.success_seeds = None
            self._current_seed_index = 0

        if not hasattr(self, "_generator"):
            self._generator = torch.Generator()
            self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

    def update_reset_state_ids(self, env_idx=None, *, force: bool = False):
        env_idx_list = self._normalize_env_idx(env_idx)
        if (
            self.use_fixed_reset_state_ids
            and hasattr(self, "reset_state_ids")
            and env_idx_list is None
            and not force
        ):
            return

        if env_idx_list is not None and hasattr(self, "reset_state_ids"):
            if self.success_seeds is not None:
                total_seeds = self.success_seeds.numel()
                indices = (
                    torch.arange(len(env_idx_list), device=self.success_seeds.device)
                    + self._current_seed_index
                ) % total_seeds
                reset_state_ids = self.success_seeds[indices]
                self._current_seed_index = (
                    self._current_seed_index + len(env_idx_list)
                ) % total_seeds
            else:
                reset_state_ids = torch.randint(
                    low=10000,
                    high=200000,
                    size=(len(env_idx_list),),
                    generator=self._generator,
                )
            for seed_pos, idx in enumerate(env_idx_list):
                self.reset_state_ids[idx] = reset_state_ids[seed_pos]
        else:
            if self.success_seeds is not None:
                total_seeds = self.success_seeds.numel()
                indices = (
                    torch.arange(self.num_group, device=self.success_seeds.device)
                    + self._current_seed_index
                ) % total_seeds
                reset_state_ids = self.success_seeds[indices]
                reset_state_ids = reset_state_ids.repeat_interleave(
                    repeats=self.group_size
                )
                self._current_seed_index = (
                    self._current_seed_index + self.num_group
                ) % total_seeds
            else:
                reset_state_ids = torch.randint(
                    low=10000,
                    high=200000,
                    size=(self.num_group,),
                    generator=self._generator,
                )
                reset_state_ids = reset_state_ids.repeat_interleave(
                    repeats=self.group_size
                )
            self.reset_state_ids = reset_state_ids

    def check_seeds(self, seeds):
        resutls = self.venv.check_seeds(seeds)

        return resutls
