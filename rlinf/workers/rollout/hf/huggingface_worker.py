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

import asyncio
import copy
import gc
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from rlinf.config import SupportedModel
from rlinf.data.embodied_io_struct import (
    RolloutResult,
)
from rlinf.hybrid_engines.weight_syncer import WeightSyncer
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.scheduler import Channel, Cluster, CollectiveGroupOptions, Worker
from rlinf.utils.comm_mapping import CommMapper
from rlinf.utils.nested_dict_process import split_dict
from rlinf.utils.placement import HybridComponentPlacement


class MultiStepRolloutWorker(Worker):
    @staticmethod
    def _resolve_action_exec_horizon(
        env_cfg: DictConfig | None, model_num_action_chunks: int
    ) -> int:
        if env_cfg is None:
            return model_num_action_chunks
        explicit_horizon = env_cfg.get("action_exec_horizon", None)
        if explicit_horizon is not None:
            return int(explicit_horizon)
        return model_num_action_chunks

    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.should_stop = False

        self.actor_group_name = cfg.actor.group_name
        self.device = self.torch_platform.current_device()

        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.rollout.get("enable_offload", False)
        self.offload_strategy = str(
            self.cfg.rollout.get("offload_strategy", "cpu")
        ).lower()
        if self.offload_strategy not in {"cpu", "destroy"}:
            raise ValueError(
                "rollout.offload_strategy must be one of {'cpu', 'destroy'}, "
                f"got {self.offload_strategy!r}"
            )
        weight_syncer_type = OmegaConf.select(cfg, "weight_syncer.type", default=None)
        if self.offload_strategy == "destroy" and weight_syncer_type != "bucket":
            raise ValueError(
                "rollout.offload_strategy=destroy requires full/bucket weight sync; "
                f"got weight_syncer.type={weight_syncer_type!r}."
            )

        self.placement = HybridComponentPlacement(cfg, Cluster())

        actor_world_size = self.placement.get_world_size("actor")
        self.actor_weight_src_rank = self._rank % actor_world_size
        self.rollout_epoch = cfg.algorithm.get("rollout_epoch", 1)
        self.collect_transitions = self.cfg.rollout.get("collect_transitions", False)
        self.expert_model = None
        self.batch_pipeline_stages = bool(
            self.cfg.rollout.get("batch_pipeline_stages", False)
        )
        self.env_rank_dispatch_mode = str(
            self.cfg.rollout.get("env_rank_dispatch_mode", "all")
        ).lower()
        if self.env_rank_dispatch_mode not in {"all", "ready", "streaming_ready"}:
            raise ValueError(
                "rollout.env_rank_dispatch_mode must be one of "
                "{'all', 'ready', 'streaming_ready'}, "
                f"got {self.env_rank_dispatch_mode!r}."
            )
        self.env_rank_ready_min_batch_ranks = max(
            1, int(self.cfg.rollout.get("env_rank_ready_min_batch_ranks", 1))
        )
        self.env_rank_ready_max_wait_s = max(
            0.0, float(self.cfg.rollout.get("env_rank_ready_max_wait_s", 0.0))
        )
        self.infer_micro_batch_size = int(
            self.cfg.rollout.get("infer_micro_batch_size", 0) or 0
        )
        if self.infer_micro_batch_size < 0:
            raise ValueError(
                "rollout.infer_micro_batch_size must be non-negative, "
                f"got {self.infer_micro_batch_size}."
            )
        profiling_cfg = self.cfg.runner.get("profiling", {})
        self.rollout_profile_enabled = bool(
            profiling_cfg.get(
                "rollout_waits",
                os.environ.get("RLINF_PROFILE_ROLLOUT_WAITS", "0")
                .strip()
                .lower()
                in {"1", "true", "yes", "on"},
            )
        )
        self.rollout_profile_log_every = max(
            1,
            int(
                profiling_cfg.get(
                    "rollout_wait_log_every",
                    os.environ.get("RLINF_PROFILE_ROLLOUT_LOG_EVERY", 64),
                )
            ),
        )
        self._rollout_profile_counters: dict[str, float] = defaultdict(float)
        self.rollout_profile_output_dir = str(
            profiling_cfg.get(
                "output_dir",
                os.environ.get("RLINF_PROFILE_OUTPUT_DIR", ""),
            )
            or ""
        )

        # Sync weight comm options
        max_ctas = cfg.rollout.get("sync_weight_nccl_max_ctas", None)
        min_ctas = cfg.rollout.get("sync_weight_nccl_min_ctas", None)
        self._sync_weight_comm_options = CollectiveGroupOptions(
            accel_max_ctas=max_ctas, accel_min_ctas=min_ctas
        )
        self.total_num_train_envs = cfg.env.train.total_num_envs
        self.total_num_eval_envs = cfg.env.eval.total_num_envs
        self.eval_parallel_total_num_envs = int(
            cfg.env.eval.get("num_parallel_envs", self.total_num_eval_envs)
        )
        if self.total_num_eval_envs % self.eval_parallel_total_num_envs != 0:
            raise ValueError(
                "env.eval.total_num_envs must be divisible by "
                "env.eval.num_parallel_envs for batched evaluation."
            )
        self.eval_batch_count = (
            self.total_num_eval_envs // self.eval_parallel_total_num_envs
        )
        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num

        self.train_batch_size = (
            self.total_num_train_envs // self._world_size // self.num_pipeline_stages
        )
        self.eval_batch_size = (
            self.eval_parallel_total_num_envs
            // self._world_size
            // self.num_pipeline_stages
        )
        self.enable_cuda_graph = cfg.rollout.get("enable_cuda_graph", False)
        self.enable_eval = cfg.runner.val_check_interval > 0 or cfg.runner.only_eval

        self.train_action_exec_horizon = self._resolve_action_exec_horizon(
            cfg.env.train,
            cfg.actor.model.num_action_chunks,
        )
        self.eval_action_exec_horizon = self._resolve_action_exec_horizon(
            cfg.env.eval,
            cfg.actor.model.num_action_chunks,
        )
        if self.train_action_exec_horizon <= 0 or self.eval_action_exec_horizon <= 0:
            raise ValueError(
                "RoboTwin action_exec_horizon must be positive; "
                f"got train={self.train_action_exec_horizon}, "
                f"eval={self.eval_action_exec_horizon}."
            )
        self.n_train_chunk_steps = (
            cfg.env.train.max_steps_per_rollout_epoch
            // self.train_action_exec_horizon
        )
        self.n_eval_chunk_steps = (
            cfg.env.eval.max_steps_per_rollout_epoch
            // self.eval_action_exec_horizon
        )
        self.collect_prev_infos = self.cfg.rollout.get("collect_prev_infos", True)
        self.version = 0
        self.finished_episodes = None

        weight_syncer_cfg = OmegaConf.select(cfg, "weight_syncer", default=None)
        assert weight_syncer_cfg is not None, (
            "rollout.weight_syncer config must be provided"
        )
        self.weight_syncer = WeightSyncer.create(weight_syncer_cfg)
        self._rollout_model_config = None
        self._expert_model_config = None
        self.hf_model = None
        self.expert_model = None
        self._nonfinite_rollout_warning_count = 0

    def _warn_nonfinite_rollout(self, name: str, count: int, total: int) -> None:
        self._nonfinite_rollout_warning_count += 1
        if self._nonfinite_rollout_warning_count <= 20 or (
            self._nonfinite_rollout_warning_count % 100 == 0
        ):
            self.log_warning(
                "Sanitized non-finite rollout tensor "
                f"{name}: count={count}/{total}, "
                f"warnings={self._nonfinite_rollout_warning_count}."
            )

    def _profile_add(self, key: str, value: float) -> None:
        if self.rollout_profile_enabled:
            self._rollout_profile_counters[key] += float(value)

    def _profile_log_rollout_waits(
        self,
        *,
        chunk_step_idx: int,
        force: bool = False,
    ) -> None:
        if not self.rollout_profile_enabled:
            return
        counters = self._rollout_profile_counters
        steps = int(counters.get("steps", 0))
        if steps <= 0:
            return
        if not force and steps % self.rollout_profile_log_every != 0:
            return

        elapsed_s = counters.get("elapsed_s", 0.0)
        recv_s = counters.get("recv_env_s", 0.0)
        predict_s = counters.get("predict_s", 0.0)
        send_s = counters.get("send_rollout_s", 0.0)
        other_s = max(0.0, elapsed_s - recv_s - predict_s - send_s)
        ready_batches = int(counters.get("ready_batches", 0))
        ready_env_ranks = int(counters.get("ready_env_ranks", 0))
        ready_envs = int(counters.get("ready_envs", 0))
        ready_suffix = ""
        if ready_batches:
            ready_suffix = (
                f" ready_batches={ready_batches} "
                f"ready_avg_ranks={ready_env_ranks / ready_batches:.2f} "
                f"ready_avg_envs={ready_envs / ready_batches:.2f}"
            )
        self.log_info(
            "[rollout-profile][rollout] "
            f"rank={self._rank} chunk={chunk_step_idx} steps={steps} "
            f"elapsed={elapsed_s:.2f}s recv_env={recv_s:.2f}s "
            f"predict={predict_s:.2f}s send_rollout={send_s:.2f}s "
            f"other={other_s:.2f}s dispatch={self.env_rank_dispatch_mode}"
            f"{ready_suffix}"
        )
        self._profile_write_jsonl(
            {
                "component": "rollout",
                "rank": int(self._rank),
                "chunk_step_idx": int(chunk_step_idx),
                "steps": steps,
                "elapsed_s": elapsed_s,
                "recv_env_s": recv_s,
                "predict_s": predict_s,
                "send_rollout_s": send_s,
                "other_s": other_s,
                "env_rank_dispatch_mode": self.env_rank_dispatch_mode,
                "ready_batches": ready_batches,
                "ready_env_ranks": ready_env_ranks,
                "ready_envs": ready_envs,
                "force": bool(force),
                "time": time.time(),
            }
        )

    def _profile_write_jsonl(self, record: dict[str, Any]) -> None:
        if not self.rollout_profile_enabled or not self.rollout_profile_output_dir:
            return
        try:
            output_dir = Path(self.rollout_profile_output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f"rollout_rank_{self._rank}.jsonl"
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, sort_keys=True) + "\n")
        except Exception as exc:
            self.log_warning(f"Failed to write rollout profile jsonl: {exc}")

    def _sanitize_rollout_tensor(
        self,
        value: torch.Tensor,
        name: str,
        *,
        nan: float = 0.0,
        posinf: float = 0.0,
        neginf: float = 0.0,
        clamp: tuple[float, float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        finite_mask = torch.isfinite(value)
        invalid_mask = None if finite_mask.all() else ~finite_mask
        if invalid_mask is not None:
            self._warn_nonfinite_rollout(name, int(invalid_mask.sum().item()), value.numel())
            value = torch.nan_to_num(value, nan=nan, posinf=posinf, neginf=neginf)
        if clamp is not None:
            value = torch.clamp(value, min=clamp[0], max=clamp[1])
        return value, invalid_mask

    def _sanitize_nested_rollout_tensors(
        self,
        value: Any,
        name: str,
    ) -> tuple[Any, torch.Tensor | None]:
        if isinstance(value, torch.Tensor):
            if value.dtype.is_floating_point:
                return self._sanitize_rollout_tensor(value, name)
            return value, None
        if isinstance(value, dict):
            sanitized = {}
            invalid_mask = None
            for key, nested_value in value.items():
                sanitized_value, nested_invalid_mask = (
                    self._sanitize_nested_rollout_tensors(
                        nested_value, f"{name}.{key}"
                    )
                )
                sanitized[key] = sanitized_value
                invalid_mask = self._merge_invalid_masks(
                    invalid_mask, nested_invalid_mask
                )
            return sanitized, invalid_mask
        return value, None

    @staticmethod
    def _merge_invalid_masks(
        lhs: torch.Tensor | None, rhs: torch.Tensor | None
    ) -> torch.Tensor | None:
        def _to_batch_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
            if mask is None:
                return None
            return mask.reshape(mask.shape[0], -1).any(dim=-1)

        lhs_batch = _to_batch_mask(lhs)
        rhs_batch = _to_batch_mask(rhs)
        if rhs_batch is None:
            return lhs_batch
        if lhs_batch is None:
            return rhs_batch
        return lhs_batch | rhs_batch

    @staticmethod
    def _batch_valid_mask_from_invalid(
        invalid_mask: torch.Tensor | None, batch_size: int, num_action_chunks: int
    ) -> torch.Tensor:
        if invalid_mask is None:
            return torch.ones(batch_size, num_action_chunks, dtype=torch.bool)
        invalid_mask = invalid_mask.detach().cpu()
        if invalid_mask.shape[0] != batch_size:
            invalid_mask = invalid_mask.reshape(batch_size, -1)
        if invalid_mask.dim() == 1:
            invalid_per_chunk = invalid_mask[:, None].expand(-1, num_action_chunks)
        else:
            invalid_flat = invalid_mask.reshape(batch_size, -1)
            if invalid_flat.shape[1] % num_action_chunks == 0:
                invalid_per_chunk = invalid_flat.reshape(
                    batch_size, num_action_chunks, -1
                ).any(dim=-1)
            else:
                invalid_per_chunk = invalid_flat.any(dim=-1, keepdim=True).expand(
                    -1, num_action_chunks
                )
        return ~invalid_per_chunk

    @staticmethod
    def _prefix_valid_mask(
        batch_size: int,
        num_action_chunks: int,
        action_exec_horizon: int,
    ) -> torch.Tensor | None:
        if action_exec_horizon <= 0 or action_exec_horizon >= num_action_chunks:
            return None
        valid_mask = torch.zeros(batch_size, num_action_chunks, dtype=torch.bool)
        valid_mask[:, :action_exec_horizon] = True
        return valid_mask

    def _build_hf_model(self) -> BasePolicy:
        rollout_model_config = copy.deepcopy(self.cfg.actor.model)
        with open_dict(rollout_model_config):
            rollout_model_config.precision = self.cfg.rollout.model.precision
            rollout_model_config.model_path = self.cfg.rollout.model.model_path
        self._rollout_model_config = rollout_model_config

        hf_model: BasePolicy = get_model(rollout_model_config)

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            hf_model.load_state_dict(model_dict)

        hf_model.eval()
        if self.cfg.rollout.get("enable_torch_compile", False):
            mode = self.cfg.rollout.get(
                "torch_compile_mode", "max-autotune-no-cudagraphs"
            )
            hf_model.enable_torch_compile(mode=mode)
        return hf_model

    def _build_expert_model(self) -> BasePolicy | None:
        if self.cfg.rollout.get("expert_model", None):
            expert_model_config = copy.deepcopy(self.cfg.actor.model)
            with open_dict(expert_model_config):
                expert_model_config.precision = self.cfg.rollout.expert_model.precision
                expert_model_config.model_path = (
                    self.cfg.rollout.expert_model.model_path
                )
            self._expert_model_config = expert_model_config
            expert_model = get_model(expert_model_config)

            if self.cfg.runner.get("expert_ckpt_path", None):
                expert_model_dict = torch.load(self.cfg.runner.expert_ckpt_path)
                expert_model.load_state_dict(expert_model_dict)

            expert_model.eval()
            return expert_model
        return None

    def init_worker(self):
        self.hf_model = self._build_hf_model()
        self.expert_model = self._build_expert_model()

        if self.enable_cuda_graph and not self.enable_offload:
            self.hf_model.capture_cuda_graph(
                train_batch_size=self.train_batch_size,
                eval_batch_size=self.eval_batch_size,
            )

        self.dst_ranks = {}
        self.src_ranks = {}
        if not self.cfg.runner.only_eval:
            self.dst_ranks = {
                "train": self._setup_dst_ranks(
                    self.total_num_train_envs // self.num_pipeline_stages
                ),
            }
            self.src_ranks = {
                "train": self._setup_src_ranks(
                    self.total_num_train_envs // self.num_pipeline_stages
                ),
            }
        if self.enable_eval:
            self.dst_ranks["eval"] = self._setup_dst_ranks(
                self.eval_parallel_total_num_envs // self.num_pipeline_stages
            )
            self.src_ranks["eval"] = self._setup_src_ranks(
                self.eval_parallel_total_num_envs // self.num_pipeline_stages
            )

        self.log_info(f"Rollout worker initialized with dst_ranks: {self.dst_ranks}")
        self.log_info(f"Rollout worker initialized with src_ranks: {self.src_ranks}")
        self.setup_sample_params()
        if self.enable_offload:
            self.offload_model()

    def setup_sample_params(self):
        # length parameters for rollout
        self._length_params = OmegaConf.to_container(
            self.cfg.algorithm.length_params, resolve=True
        )
        # sampling parameters for rollout
        self._sampling_params = OmegaConf.to_container(
            self.cfg.algorithm.sampling_params, resolve=True
        )
        self._train_sampling_params = {
            "do_sample": self._sampling_params["do_sample"],
            "temperature": self._sampling_params["temperature_train"]
            if self._sampling_params["do_sample"]
            else 1.0,
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

        self._eval_sampling_params = {
            "do_sample": True
            if self._sampling_params.get("temperature_eval", -1) > 0
            else False,
            "temperature": self._sampling_params["temperature_eval"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

        if self.expert_model is not None:
            self._dagger_sampling_params = {
                "beta": self.cfg.algorithm.get("dagger", {}).get("init_beta", 0.5),
                "beta_schedule": self.cfg.algorithm.get("dagger", {}).get(
                    "beta_schedule", "exponential"
                ),
                "beta_min": self.cfg.algorithm.get("dagger", {}).get("beta_min", 0.05),
                "beta_decay": self.cfg.algorithm.get("dagger", {}).get(
                    "beta_decay", 0.99
                ),
            }

    def update_dagger_beta(self):
        if self.expert_model is None:
            return

        if self._dagger_sampling_params["beta_schedule"] == "exponential":
            self._dagger_sampling_params["beta"] = max(
                self._dagger_sampling_params["beta_min"],
                self._dagger_sampling_params["beta"]
                * self._dagger_sampling_params["beta_decay"],
            )
        else:
            raise NotImplementedError(
                f"Beta schedule {self._dagger_sampling_params['beta_schedule']} is not implemented"
            )

    def _setup_dst_ranks(self, batch_size: int) -> list[tuple[int, int]]:
        """Compute env peer ranks for this rollout worker.

        This mapping supports both one-to-many and many-to-one env/rollout layouts.
        The returned ranks are used as communication counterparts for receiving env
        outputs and sending action chunks.

        Args:
            batch_size: Total env batch size per pipeline stage across all workers.

        Returns:
            Ordered ``(env_rank, batch_size)`` tuples this rollout worker should
            send action chunks to.
        """
        env_world_size = self.placement.get_world_size("env")
        rollout_world_size = self.placement.get_world_size("rollout")
        return CommMapper.get_dst_ranks(
            batch_size=batch_size,
            src_world_size=rollout_world_size,
            dst_world_size=env_world_size,
            src_rank=self._rank,
        )

    def _setup_src_ranks(self, batch_size: int) -> list[tuple[int, int]]:
        """Compute env source ranks and sizes for receiving env outputs."""
        env_world_size = self.placement.get_world_size("env")
        rollout_world_size = self.placement.get_world_size("rollout")
        return CommMapper.get_src_ranks(
            batch_size=batch_size,
            src_world_size=env_world_size,
            dst_world_size=rollout_world_size,
            dst_rank=self._rank,
        )

    @Worker.timer("predict")
    def predict(
        self, env_obs: dict[str, Any], mode: Literal["train", "eval"] = "train"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        kwargs = (
            self._train_sampling_params
            if mode == "train"
            else self._eval_sampling_params
        )

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENPI,
            SupportedModel.MLP_POLICY,
            SupportedModel.GR00T,
            SupportedModel.DREAMZERO,
            SupportedModel.CNN_POLICY,
            SupportedModel.CFG_MODEL,
        ]:
            if self.cfg.algorithm.loss_type == "embodied_dagger":
                kwargs = {"mode": "eval"}
            else:
                kwargs = {"mode": mode}

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.CNN_POLICY,
            SupportedModel.FLOW_POLICY,
            SupportedModel.MLP_POLICY,
        ]:
            kwargs["return_obs"] = not hasattr(self.hf_model, "q_head")

        only_save_expert = self.cfg.algorithm.get("dagger", {}).get(
            "only_save_expert", True
        )

        if mode == "train" and self.expert_model is not None:
            # training with expert model. Beta-probability acting.
            use_expert = torch.rand(1).item() < self._dagger_sampling_params["beta"]
        else:
            use_expert = False

        with torch.no_grad():
            expert_label_flag = False
            # Decide which model to act via use_expert
            if use_expert:
                actions, result = self._predict_action_batch_microbatched(
                    self.expert_model, env_obs, kwargs
                )
                expert_label_flag = True
            else:
                actions, result = self._predict_action_batch_microbatched(
                    self.hf_model, env_obs, kwargs
                )

            # Decide re-label or not
            if (
                not only_save_expert  # only re-label in classic dagger mode
                and not use_expert  # only re-label if not using expert
                and self.expert_model is not None  # only re-label if expert exists
                and mode == "train"  # only re-label in train mode
            ):
                _, expert_result = self._predict_action_batch_microbatched(
                    self.expert_model, env_obs, kwargs
                )
                expert_forward_inputs = expert_result["forward_inputs"]
                expert_target = expert_forward_inputs.get(
                    "model_action", expert_forward_inputs.get("action")
                )
                if expert_target is not None:
                    result["forward_inputs"]["model_action"] = expert_target
                expert_label_flag = True

        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)

        invalid_mask = None
        actions, actions_invalid_mask = self._sanitize_rollout_tensor(
            actions, "actions"
        )
        invalid_mask = self._merge_invalid_masks(invalid_mask, actions_invalid_mask)
        if "prev_logprobs" in result and result["prev_logprobs"] is not None:
            result["prev_logprobs"], prev_logprobs_invalid_mask = (
                self._sanitize_rollout_tensor(
                    result["prev_logprobs"],
                    "prev_logprobs",
                    posinf=100.0,
                    neginf=-100.0,
                    clamp=(-100.0, 100.0),
                )
            )
            invalid_mask = self._merge_invalid_masks(
                invalid_mask, prev_logprobs_invalid_mask
            )
        if "prev_values" in result and result["prev_values"] is not None:
            result["prev_values"], prev_values_invalid_mask = (
                self._sanitize_rollout_tensor(
                    result["prev_values"],
                    "prev_values",
                    posinf=1e4,
                    neginf=-1e4,
                    clamp=(-1e4, 1e4),
                )
            )
            invalid_mask = self._merge_invalid_masks(invalid_mask, prev_values_invalid_mask)
        if "forward_inputs" in result and result["forward_inputs"]:
            existing_valid_action_mask = result["forward_inputs"].get(
                "valid_action_mask", None
            )
            result["forward_inputs"], forward_inputs_invalid_mask = (
                self._sanitize_nested_rollout_tensors(
                    result["forward_inputs"], "forward_inputs"
                )
            )
            invalid_mask = self._merge_invalid_masks(
                invalid_mask, forward_inputs_invalid_mask
            )
            valid_action_mask = self._batch_valid_mask_from_invalid(
                invalid_mask,
                actions.shape[0],
                self.cfg.actor.model.num_action_chunks,
            )
            if existing_valid_action_mask is not None:
                valid_action_mask = valid_action_mask & existing_valid_action_mask.cpu().to(
                    torch.bool
                )
            result["forward_inputs"]["valid_action_mask"] = valid_action_mask

        result["expert_label_flag"] = bool(expert_label_flag)
        return actions, result

    def _predict_action_batch_microbatched(
        self,
        model: BasePolicy,
        env_obs: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor | np.ndarray, dict[str, Any]]:
        batch_size = self._infer_env_batch_size(env_obs)
        micro_batch_size = self.infer_micro_batch_size
        if micro_batch_size <= 0 or batch_size <= micro_batch_size:
            return model.predict_action_batch(env_obs=env_obs, **kwargs)

        split_sizes: list[int] = []
        remaining = batch_size
        while remaining > 0:
            size = min(micro_batch_size, remaining)
            split_sizes.append(size)
            remaining -= size

        action_chunks: list[torch.Tensor | np.ndarray] = []
        result_chunks: list[dict[str, Any]] = []
        for obs_chunk in split_dict(env_obs, split_sizes):
            actions, result = model.predict_action_batch(env_obs=obs_chunk, **kwargs)
            action_chunks.append(actions)
            result_chunks.append(result)

        if isinstance(action_chunks[0], np.ndarray):
            merged_actions = np.concatenate(action_chunks, axis=0)
        else:
            merged_actions = torch.cat(action_chunks, dim=0)
        return merged_actions, self._concat_result_dicts(result_chunks)

    @classmethod
    def _concat_result_dicts(cls, results: list[dict[str, Any]]) -> dict[str, Any]:
        if not results:
            return {}
        merged: dict[str, Any] = {}
        for key in results[0].keys():
            merged[key] = cls._concat_batch_values([result.get(key) for result in results])
        return merged

    @classmethod
    def _concat_batch_values(cls, values: list[Any]) -> Any:
        first_non_none = next((value for value in values if value is not None), None)
        if first_non_none is None:
            return None
        if isinstance(first_non_none, torch.Tensor):
            return torch.cat([value for value in values if value is not None], dim=0)
        if isinstance(first_non_none, np.ndarray):
            return np.concatenate([value for value in values if value is not None], axis=0)
        if isinstance(first_non_none, list):
            return [item for value in values if value is not None for item in value]
        if isinstance(first_non_none, dict):
            merged: dict[str, Any] = {}
            for key in first_non_none.keys():
                merged[key] = cls._concat_batch_values(
                    [
                        value.get(key)
                        for value in values
                        if isinstance(value, dict) and key in value
                    ]
                )
            return merged
        return first_non_none

    def get_bootstrap_values(
        self, final_obs: dict[str, Any] | None
    ) -> torch.Tensor | None:
        if final_obs is None:
            return None
        if not (
            hasattr(self.hf_model, "value_head") or hasattr(self.hf_model, "q_head")
        ):
            return None
        with torch.no_grad():
            actions, result = self.predict(final_obs)
            if "prev_values" in result and result["prev_values"] is not None:
                final_values = result["prev_values"]
            else:
                final_values = torch.zeros_like(actions[:, :1], dtype=torch.float32)
        return final_values[:, :1].cpu().contiguous()

    def _build_rollout_result(
        self,
        env_output: dict[str, Any],
        actions: torch.Tensor | np.ndarray,
        result: dict[str, Any],
        *,
        include_logprobs: bool,
    ) -> RolloutResult:
        save_flags = None
        if include_logprobs and result.get("expert_label_flag", False):
            save_flags = torch.full(
                (actions.shape[0], self.cfg.actor.model.num_action_chunks),
                True,
                dtype=torch.bool,
                device=actions.device,
            )

        forward_inputs = result["forward_inputs"] if include_logprobs else {}
        if include_logprobs:
            prefix_valid_mask = self._prefix_valid_mask(
                actions.shape[0],
                self.cfg.actor.model.num_action_chunks,
                self.train_action_exec_horizon,
            )
            if prefix_valid_mask is not None:
                existing_valid_mask = forward_inputs.get("valid_action_mask", None)
                if existing_valid_mask is not None:
                    prefix_valid_mask = prefix_valid_mask & existing_valid_mask.cpu().to(
                        torch.bool
                    )
                forward_inputs["valid_action_mask"] = prefix_valid_mask

        return RolloutResult(
            actions=actions,
            prev_logprobs=result["prev_logprobs"]
            if include_logprobs and self.collect_prev_infos
            else None,
            prev_values=result["prev_values"] if self.collect_prev_infos else None,
            bootstrap_values=self.get_bootstrap_values(
                env_output.get("final_obs", None)
            ),
            save_flags=save_flags,
            forward_inputs=forward_inputs,
            versions=torch.full_like(
                result["prev_logprobs"],
                float(self.version),
                dtype=torch.float32,
            )
            if include_logprobs
            else None,
        )

    @staticmethod
    def _split_result_dict(
        result: dict[str, Any], split_sizes: list[int]
    ) -> list[dict[str, Any]]:
        split_results = [{} for _ in split_sizes]
        for key, value in result.items():
            if key == "expert_label_flag":
                for split_result in split_results:
                    split_result[key] = value
            elif isinstance(value, torch.Tensor):
                values = torch.split(value, split_sizes, dim=0)
                for idx, value_i in enumerate(values):
                    split_results[idx][key] = value_i.contiguous()
            elif isinstance(value, np.ndarray):
                split_indices = np.cumsum(split_sizes[:-1]).tolist()
                values = np.split(value, split_indices, axis=0)
                for idx, value_i in enumerate(values):
                    split_results[idx][key] = value_i
            elif isinstance(value, dict):
                values = split_dict(value, split_sizes)
                for idx, value_i in enumerate(values):
                    split_results[idx][key] = value_i
            elif value is None:
                for split_result in split_results:
                    split_result[key] = None
            else:
                for split_result in split_results:
                    split_result[key] = value
        return split_results

    def _run_batched_stage_prediction(
        self,
        env_outputs: list[dict[str, Any]],
        *,
        include_logprobs: bool,
    ) -> list[RolloutResult]:
        split_sizes = [
            self._infer_env_batch_size(env_output) for env_output in env_outputs
        ]
        merged_obs = self._merge_obs_batches(env_outputs)["obs"]
        actions, result = self.predict(merged_obs)
        action_splits = self._split_actions(actions, split_sizes)
        result_splits = self._split_result_dict(result, split_sizes)

        return [
            self._build_rollout_result(
                env_output,
                actions_i,
                result_i,
                include_logprobs=include_logprobs,
            )
            for env_output, actions_i, result_i in zip(
                env_outputs, action_splits, result_splits
            )
        ]

    async def sync_model_from_actor(self):
        """Sync model parameters from the actor worker."""
        if self.enable_offload:
            self.reload_model()

        async def recv_func() -> Any:
            data = await self.recv(
                src_group_name=self.actor_group_name,
                src_rank=self.actor_weight_src_rank,
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()
            return data

        async def send_func(data: Any) -> None:
            await self.send(
                data,
                dst_group_name=self.actor_group_name,
                dst_rank=self.actor_weight_src_rank,
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()

        if not self.weight_syncer.receiver_initialized():
            await self.weight_syncer.init_receiver(
                state_dict=self.hf_model.state_dict(),
                recv=recv_func,
                send=send_func,
            )

        applied_version = await self.weight_syncer.apply(self.hf_model, recv_func)
        self.version = applied_version
        if self.finished_episodes is None:
            self.finished_episodes = (
                self.version * self.total_num_train_envs * self.rollout_epoch
            )
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(applied_version)

        gc.collect()
        self.torch_platform.empty_cache()

    @Worker.timer("generate_one_epoch")
    async def generate_one_epoch(self, input_channel: Channel, output_channel: Channel):
        self.update_dagger_beta()

        if (
            self.batch_pipeline_stages
            and self.num_pipeline_stages > 1
            and self.expert_model is None
        ):
            for chunk_step_idx in range(self.n_train_chunk_steps):
                profile_step_start = (
                    time.perf_counter() if self.rollout_profile_enabled else 0.0
                )
                profile_t0 = (
                    time.perf_counter() if self.rollout_profile_enabled else 0.0
                )
                env_outputs = [
                    await self.recv_env_output(input_channel)
                    for _ in range(self.num_pipeline_stages)
                ]
                self._profile_add("recv_env_s", time.perf_counter() - profile_t0)

                profile_t0 = (
                    time.perf_counter() if self.rollout_profile_enabled else 0.0
                )
                rollout_results = list(
                    self._run_batched_stage_prediction(
                        env_outputs, include_logprobs=True
                    )
                )
                self._profile_add("predict_s", time.perf_counter() - profile_t0)

                profile_t0 = (
                    time.perf_counter() if self.rollout_profile_enabled else 0.0
                )
                for rollout_result in rollout_results:
                    self.send_rollout_result(
                        output_channel, rollout_result, mode="train"
                    )
                self._profile_add("send_rollout_s", time.perf_counter() - profile_t0)

                if self.rollout_profile_enabled:
                    self._profile_add("steps", 1)
                    self._profile_add(
                        "elapsed_s", time.perf_counter() - profile_step_start
                    )
                    self._profile_log_rollout_waits(chunk_step_idx=chunk_step_idx)

            profile_step_start = (
                time.perf_counter() if self.rollout_profile_enabled else 0.0
            )
            profile_t0 = time.perf_counter() if self.rollout_profile_enabled else 0.0
            env_outputs = [
                await self.recv_env_output(input_channel)
                for _ in range(self.num_pipeline_stages)
            ]
            self._profile_add("recv_env_s", time.perf_counter() - profile_t0)

            profile_t0 = time.perf_counter() if self.rollout_profile_enabled else 0.0
            rollout_results = list(
                self._run_batched_stage_prediction(
                    env_outputs, include_logprobs=False
                )
            )
            self._profile_add("predict_s", time.perf_counter() - profile_t0)

            profile_t0 = time.perf_counter() if self.rollout_profile_enabled else 0.0
            for rollout_result in rollout_results:
                self.send_rollout_result(output_channel, rollout_result, mode="train")
            self._profile_add("send_rollout_s", time.perf_counter() - profile_t0)

            if self.rollout_profile_enabled:
                self._profile_add("steps", 1)
                self._profile_add("elapsed_s", time.perf_counter() - profile_step_start)
                self._profile_log_rollout_waits(
                    chunk_step_idx=self.n_train_chunk_steps,
                    force=True,
            )
            return

        use_ready_dispatch = (
            self.env_rank_dispatch_mode in {"ready", "streaming_ready"}
            and self.num_pipeline_stages == 1
            and self.expert_model is None
        )
        if use_ready_dispatch:
            if self.env_rank_dispatch_mode == "streaming_ready":
                await self._generate_one_epoch_streaming_ready(
                    input_channel, output_channel
                )
                return

            for chunk_step_idx in range(self.n_train_chunk_steps):
                profile_step_start = (
                    time.perf_counter() if self.rollout_profile_enabled else 0.0
                )
                async for src_ranks, split_sizes, env_output in (
                    self._iter_ready_env_output_batches(
                        input_channel,
                        mode="train",
                        chunk_step_idx=chunk_step_idx,
                    )
                ):
                    profile_t0 = (
                        time.perf_counter() if self.rollout_profile_enabled else 0.0
                    )
                    actions, result = self.predict(env_output["obs"])
                    self._profile_add("predict_s", time.perf_counter() - profile_t0)

                    rollout_result = self._build_rollout_result(
                        env_output,
                        actions,
                        result,
                        include_logprobs=True,
                    )

                    profile_t0 = (
                        time.perf_counter() if self.rollout_profile_enabled else 0.0
                    )
                    self.send_rollout_result_to_ranks(
                        output_channel,
                        rollout_result,
                        dst_ranks=src_ranks,
                        sizes=split_sizes,
                        mode="train",
                    )
                    self._profile_add("send_rollout_s", time.perf_counter() - profile_t0)

                if self.rollout_profile_enabled:
                    self._profile_add("steps", 1)
                    self._profile_add(
                        "elapsed_s", time.perf_counter() - profile_step_start
                    )
                    self._profile_log_rollout_waits(chunk_step_idx=chunk_step_idx)

            profile_step_start = (
                time.perf_counter() if self.rollout_profile_enabled else 0.0
            )
            async for src_ranks, split_sizes, env_output in (
                self._iter_ready_env_output_batches(
                    input_channel,
                    mode="train",
                    chunk_step_idx=self.n_train_chunk_steps,
                )
            ):
                profile_t0 = time.perf_counter() if self.rollout_profile_enabled else 0.0
                actions, result = self.predict(env_output["obs"])
                self._profile_add("predict_s", time.perf_counter() - profile_t0)

                rollout_result = self._build_rollout_result(
                    env_output,
                    actions,
                    result,
                    include_logprobs=False,
                )

                profile_t0 = time.perf_counter() if self.rollout_profile_enabled else 0.0
                self.send_rollout_result_to_ranks(
                    output_channel,
                    rollout_result,
                    dst_ranks=src_ranks,
                    sizes=split_sizes,
                    mode="train",
                )
                self._profile_add("send_rollout_s", time.perf_counter() - profile_t0)

            if self.rollout_profile_enabled:
                self._profile_add("steps", 1)
                self._profile_add("elapsed_s", time.perf_counter() - profile_step_start)
                self._profile_log_rollout_waits(
                    chunk_step_idx=self.n_train_chunk_steps,
                    force=True,
                )
            return

        for chunk_step_idx in range(self.n_train_chunk_steps):
            for _ in range(self.num_pipeline_stages):
                profile_step_start = (
                    time.perf_counter() if self.rollout_profile_enabled else 0.0
                )
                profile_t0 = (
                    time.perf_counter() if self.rollout_profile_enabled else 0.0
                )
                env_output = await self.recv_env_output(input_channel)
                self._profile_add("recv_env_s", time.perf_counter() - profile_t0)

                profile_t0 = (
                    time.perf_counter() if self.rollout_profile_enabled else 0.0
                )
                actions, result = self.predict(env_output["obs"])
                self._profile_add("predict_s", time.perf_counter() - profile_t0)

                rollout_result = self._build_rollout_result(
                    env_output,
                    actions,
                    result,
                    include_logprobs=True,
                )

                profile_t0 = (
                    time.perf_counter() if self.rollout_profile_enabled else 0.0
                )
                self.send_rollout_result(output_channel, rollout_result, mode="train")
                self._profile_add("send_rollout_s", time.perf_counter() - profile_t0)

                if self.rollout_profile_enabled:
                    self._profile_add("steps", 1)
                    self._profile_add(
                        "elapsed_s", time.perf_counter() - profile_step_start
                    )
                    self._profile_log_rollout_waits(chunk_step_idx=chunk_step_idx)

        for _ in range(self.num_pipeline_stages):
            profile_step_start = (
                time.perf_counter() if self.rollout_profile_enabled else 0.0
            )
            profile_t0 = time.perf_counter() if self.rollout_profile_enabled else 0.0
            env_output = await self.recv_env_output(input_channel)
            self._profile_add("recv_env_s", time.perf_counter() - profile_t0)

            profile_t0 = time.perf_counter() if self.rollout_profile_enabled else 0.0
            actions, result = self.predict(env_output["obs"])
            self._profile_add("predict_s", time.perf_counter() - profile_t0)

            rollout_result = RolloutResult(
                actions=actions,
                prev_values=result["prev_values"] if self.collect_prev_infos else None,
                bootstrap_values=self.get_bootstrap_values(
                    env_output.get("final_obs", None)
                ),
            )

            profile_t0 = time.perf_counter() if self.rollout_profile_enabled else 0.0
            self.send_rollout_result(output_channel, rollout_result, mode="train")
            self._profile_add("send_rollout_s", time.perf_counter() - profile_t0)

            if self.rollout_profile_enabled:
                self._profile_add("steps", 1)
                self._profile_add("elapsed_s", time.perf_counter() - profile_step_start)

        self._profile_log_rollout_waits(
            chunk_step_idx=self.n_train_chunk_steps,
            force=True,
        )

    async def _generate_one_epoch_streaming_ready(
        self, input_channel: Channel, output_channel: Channel
    ) -> None:
        """Run rollout with per-env-rank streaming instead of chunk-level barriers."""
        pending: dict[asyncio.Task, tuple[int, int, int]] = {}
        ready_items: list[tuple[int, int, dict[str, Any], int]] = []
        max_obs_step_idx = self.n_train_chunk_steps

        def _spawn_recv(src_rank: int, expected_size: int, obs_step_idx: int) -> None:
            task = asyncio.create_task(
                self._recv_one_env_output(
                    input_channel,
                    src_rank,
                    expected_size,
                    mode="train",
                )
            )
            pending[task] = (src_rank, expected_size, obs_step_idx)

        def _add_done_tasks(done_tasks: set[asyncio.Task]) -> None:
            for task in done_tasks:
                expected_src_rank, expected_size, obs_step_idx = pending.pop(task)
                src_rank, actual_size, obs_batch = task.result()
                assert src_rank == expected_src_rank, (
                    f"Expected env rank {expected_src_rank}, got {src_rank}."
                )
                assert actual_size == expected_size, (
                    f"Expected env output size {expected_size} from env rank "
                    f"{src_rank}, got {actual_size}."
                )
                ready_items.append((src_rank, actual_size, obs_batch, obs_step_idx))

        def _is_regular_item(item: tuple[int, int, dict[str, Any], int]) -> bool:
            return item[3] < max_obs_step_idx

        def _ready_kind_count(regular: bool) -> int:
            return sum(1 for item in ready_items if _is_regular_item(item) == regular)

        async def _wait_for_ready_items() -> None:
            if not pending and ready_items:
                return

            profile_t0 = time.perf_counter() if self.rollout_profile_enabled else 0.0
            if not ready_items:
                done, _ = await asyncio.wait(
                    set(pending.keys()), return_when=asyncio.FIRST_COMPLETED
                )
                _add_done_tasks(done)

            if pending and self.env_rank_ready_max_wait_s > 0 and ready_items:
                regular = _is_regular_item(ready_items[0])
                min_batch = min(
                    self.env_rank_ready_min_batch_ranks,
                    _ready_kind_count(regular) + len(pending),
                )
                deadline = time.perf_counter() + self.env_rank_ready_max_wait_s
                while _ready_kind_count(regular) < min_batch and pending:
                    timeout = max(0.0, deadline - time.perf_counter())
                    if timeout <= 0:
                        break
                    more_done, _ = await asyncio.wait(
                        set(pending.keys()),
                        timeout=timeout,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if not more_done:
                        break
                    _add_done_tasks(more_done)

            self._profile_add("recv_env_s", time.perf_counter() - profile_t0)

        def _pop_ready_batch() -> list[tuple[int, int, dict[str, Any], int]]:
            if any(_is_regular_item(item) for item in ready_items):
                regular = True
            else:
                regular = False
            selected = [item for item in ready_items if _is_regular_item(item) == regular]
            ready_items[:] = [
                item for item in ready_items if _is_regular_item(item) != regular
            ]
            selected.sort(key=lambda item: (item[3], item[0]))
            return selected

        for src_rank, expected_size in self.src_ranks["train"]:
            _spawn_recv(src_rank, expected_size, obs_step_idx=0)

        while pending or ready_items:
            await _wait_for_ready_items()
            selected_items = _pop_ready_batch()
            if not selected_items:
                continue

            profile_step_start = (
                time.perf_counter() if self.rollout_profile_enabled else 0.0
            )
            src_ranks = [src_rank for src_rank, _, _, _ in selected_items]
            split_sizes = [size for _, size, _, _ in selected_items]
            obs_batches = [obs_batch for _, _, obs_batch, _ in selected_items]
            obs_step_indices = [
                obs_step_idx for _, _, _, obs_step_idx in selected_items
            ]
            include_logprobs = obs_step_indices[0] < max_obs_step_idx
            assert all(
                (obs_step_idx < max_obs_step_idx) == include_logprobs
                for obs_step_idx in obs_step_indices
            ), "Cannot mix train action and final bootstrap observations."

            env_count = sum(split_sizes)
            self._profile_add("ready_batches", 1)
            self._profile_add("ready_env_ranks", len(src_ranks))
            self._profile_add("ready_envs", env_count)
            if self.rollout_profile_enabled:
                self._profile_write_jsonl(
                    {
                        "component": "rollout",
                        "event": "ready_batch",
                        "rank": int(self._rank),
                        "mode": "train",
                        "dispatch": "streaming_ready",
                        "chunk_step_idx": int(min(obs_step_indices)),
                        "obs_step_indices": [
                            int(obs_step_idx) for obs_step_idx in obs_step_indices
                        ],
                        "ready_env_ranks": len(src_ranks),
                        "ready_envs": env_count,
                        "src_ranks": [int(src_rank) for src_rank in src_ranks],
                        "sizes": [int(size) for size in split_sizes],
                        "pending_after": len(pending),
                        "buffered_after": len(ready_items),
                        "time": time.time(),
                    }
                )

            env_output = self._merge_obs_batches(obs_batches)
            profile_t0 = time.perf_counter() if self.rollout_profile_enabled else 0.0
            actions, result = self.predict(env_output["obs"])
            self._profile_add("predict_s", time.perf_counter() - profile_t0)

            rollout_result = self._build_rollout_result(
                env_output,
                actions,
                result,
                include_logprobs=include_logprobs,
            )

            profile_t0 = time.perf_counter() if self.rollout_profile_enabled else 0.0
            self.send_rollout_result_to_ranks(
                output_channel,
                rollout_result,
                dst_ranks=src_ranks,
                sizes=split_sizes,
                mode="train",
            )
            self._profile_add("send_rollout_s", time.perf_counter() - profile_t0)

            for src_rank, expected_size, _, obs_step_idx in selected_items:
                if obs_step_idx < max_obs_step_idx:
                    _spawn_recv(src_rank, expected_size, obs_step_idx + 1)

            if self.rollout_profile_enabled:
                self._profile_add("steps", 1)
                self._profile_add("elapsed_s", time.perf_counter() - profile_step_start)
                force = not pending and not ready_items
                self._profile_log_rollout_waits(
                    chunk_step_idx=max(obs_step_indices),
                    force=force,
                )

    async def generate(
        self,
        input_channel: Channel,
        output_channel: Channel,
    ):
        if self.enable_offload:
            self.reload_model()

        for _ in tqdm(
            range(self.rollout_epoch),
            desc="Generating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            await self.generate_one_epoch(input_channel, output_channel)

        if self.enable_offload:
            self.offload_model()

    async def evaluate(self, input_channel: Channel, output_channel: Channel):
        if self.enable_offload:
            self.reload_model()
        for _ in tqdm(
            range(self.eval_batch_count),
            desc="Evaluating Batches",
            disable=(self._rank != 0),
        ):
            for _ in range(self.cfg.algorithm.eval_rollout_epoch):
                for _ in range(self.n_eval_chunk_steps):
                    for _ in range(self.num_pipeline_stages):
                        env_output = await self.recv_env_output(
                            input_channel, mode="eval"
                        )
                        actions, _ = self.predict(env_output["obs"], mode="eval")
                        self.send_chunk_actions(output_channel, actions, mode="eval")

        if self.enable_offload:
            self.offload_model()

    def offload_model(self):
        if self.hf_model is None:
            return
        if self.enable_cuda_graph:
            self.hf_model.release_cuda_graph()
        if self.offload_strategy == "destroy":
            del self.hf_model
            self.hf_model = None
            if self.expert_model is not None:
                del self.expert_model
                self.expert_model = None
            gc.collect()
            self.torch_platform.empty_cache()
            self.log_info("Destroyed rollout model before actor/policy phase.")
            return

        self.hf_model.to("cpu")
        if self.expert_model is not None:
            self.expert_model.to("cpu")
        gc.collect()
        self.torch_platform.empty_cache()

    def reload_model(self):
        if self.hf_model is None:
            self.hf_model = self._build_hf_model()
            self.expert_model = self._build_expert_model()
            self.log_info("Rebuilt rollout model for rollout/sync phase.")
        if self.expert_model is not None:
            self.expert_model.to(self.device)
        self.hf_model.to(self.device)
        if self.enable_cuda_graph:
            self.hf_model.capture_cuda_graph(
                train_batch_size=self.train_batch_size,
                eval_batch_size=self.eval_batch_size,
            )

    async def recv_env_output(
        self, input_channel: Channel, mode: Literal["train", "eval"] = "train"
    ) -> dict[str, Any]:
        """Receive env outputs from mapped env ranks and merge if needed.

        Args:
            input_channel: Channel carrying env->rollout outputs.
            mode: Rollout mode, either ``"train"`` or ``"eval"``.

        Returns:
            A single env output dict. When multiple env ranks are mapped to this
            rollout worker, outputs are merged on batch dimension.
        """
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        src_ranks_and_sizes = self.src_ranks[mode]
        obs_batches = []
        for src_rank, expected_size in src_ranks_and_sizes:
            obs_batch = await input_channel.get(
                key=CommMapper.build_channel_key(
                    src_rank, self._rank, extra=f"{mode}_obs"
                ),
                async_op=True,
            ).async_wait()
            actual_size = self._infer_env_batch_size(obs_batch)
            assert actual_size == expected_size, (
                f"Expected env output batch size {expected_size} from env rank {src_rank}, "
                f"got {actual_size}."
            )
            obs_batches.append(obs_batch)
        return self._merge_obs_batches(obs_batches)

    async def _recv_one_env_output(
        self,
        input_channel: Channel,
        src_rank: int,
        expected_size: int,
        *,
        mode: Literal["train", "eval"],
    ) -> tuple[int, int, dict[str, Any]]:
        obs_batch = await input_channel.get(
            key=CommMapper.build_channel_key(
                src_rank, self._rank, extra=f"{mode}_obs"
            ),
            async_op=True,
        ).async_wait()
        actual_size = self._infer_env_batch_size(obs_batch)
        assert actual_size == expected_size, (
            f"Expected env output batch size {expected_size} from env rank {src_rank}, "
            f"got {actual_size}."
        )
        return src_rank, expected_size, obs_batch

    async def _iter_ready_env_output_batches(
        self,
        input_channel: Channel,
        mode: Literal["train", "eval"] = "train",
        chunk_step_idx: int | None = None,
    ):
        """Yield env-output groups as their source ranks become ready.

        The default ``recv_env_output`` path waits for all source env ranks mapped
        to this rollout rank. This ready path keeps the same data contract but
        releases fast env ranks earlier, which reduces chunk-level lockstep stalls.
        """
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        tasks = [
            asyncio.create_task(
                self._recv_one_env_output(
                    input_channel,
                    src_rank,
                    expected_size,
                    mode=mode,
                )
            )
            for src_rank, expected_size in self.src_ranks[mode]
        ]
        pending = set(tasks)

        while pending:
            profile_t0 = time.perf_counter() if self.rollout_profile_enabled else 0.0
            done, pending = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED
            )

            if pending and self.env_rank_ready_max_wait_s > 0:
                min_batch = min(
                    self.env_rank_ready_min_batch_ranks, len(done) + len(pending)
                )
                deadline = time.perf_counter() + self.env_rank_ready_max_wait_s
                while len(done) < min_batch and pending:
                    timeout = max(0.0, deadline - time.perf_counter())
                    if timeout <= 0:
                        break
                    more_done, pending = await asyncio.wait(
                        pending,
                        timeout=timeout,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if not more_done:
                        break
                    done.update(more_done)
            self._profile_add("recv_env_s", time.perf_counter() - profile_t0)
            recv_elapsed_s = (
                time.perf_counter() - profile_t0 if self.rollout_profile_enabled else 0.0
            )

            ready_items = [task.result() for task in done]
            ready_items.sort(key=lambda item: item[0])
            src_ranks = [src_rank for src_rank, _, _ in ready_items]
            sizes = [expected_size for _, expected_size, _ in ready_items]
            obs_batches = [obs_batch for _, _, obs_batch in ready_items]
            env_count = sum(sizes)
            self._profile_add("ready_batches", 1)
            self._profile_add("ready_env_ranks", len(src_ranks))
            self._profile_add("ready_envs", env_count)
            if self.rollout_profile_enabled:
                self._profile_write_jsonl(
                    {
                        "component": "rollout",
                        "event": "ready_batch",
                        "rank": int(self._rank),
                        "mode": mode,
                        "chunk_step_idx": (
                            None
                            if chunk_step_idx is None
                            else int(chunk_step_idx)
                        ),
                        "ready_env_ranks": len(src_ranks),
                        "ready_envs": env_count,
                        "src_ranks": [int(src_rank) for src_rank in src_ranks],
                        "sizes": [int(size) for size in sizes],
                        "pending_after": len(pending),
                        "recv_wait_s": float(recv_elapsed_s),
                        "time": time.time(),
                    }
                )
            yield src_ranks, sizes, self._merge_obs_batches(obs_batches)

    def _split_actions(
        self, actions: torch.Tensor | np.ndarray, sizes: list[int]
    ) -> list[torch.Tensor | np.ndarray]:
        """Split rollout actions into size-specified shards along dim-0.

        Args:
            actions: Model-predicted action chunk batch (tensor or ndarray).
            sizes: Batch sizes for each destination env rank.

        Returns:
            A list of action shards aligned with destination rank order.
        """
        assert sum(sizes) == actions.shape[0], (
            f"Number of actions ({actions.shape[0]}) must equal split sizes sum ({sum(sizes)})."
        )
        if isinstance(actions, np.ndarray):
            split_indices = np.cumsum(sizes[:-1]).tolist()
            return list(np.split(actions, split_indices, axis=0))
        return list(torch.split(actions, sizes, dim=0))

    @staticmethod
    def _infer_env_batch_size(obs_batch: dict[str, Any]) -> int:
        obs = obs_batch["obs"] if "obs" in obs_batch else obs_batch
        for key in ("states", "main_images", "task_descriptions"):
            value = obs.get(key)
            if isinstance(value, torch.Tensor):
                return value.shape[0]
            if isinstance(value, list):
                return len(value)
        raise ValueError("Cannot infer batch size from env obs.")

    @staticmethod
    def _merge_obs_batches(obs_batches: list[dict[str, Any]]) -> dict[str, Any]:
        if not obs_batches:
            return {}
        obs_dicts = [
            obs_batch["obs"] if "obs" in obs_batch else obs_batch
            for obs_batch in obs_batches
        ]
        final_obs_list = [obs_batch.get("final_obs", None) for obs_batch in obs_batches]

        def _merge_obs_dicts(dicts: list[dict[str, Any]]) -> dict[str, Any]:
            merged: dict[str, Any] = {}
            for key in dicts[0].keys():
                values = [obs_dict[key] for obs_dict in dicts]
                first_non_none = next(
                    (value for value in values if value is not None), None
                )
                if first_non_none is None:
                    merged[key] = None
                elif isinstance(first_non_none, torch.Tensor):
                    merged[key] = torch.cat(values, dim=0)
                elif isinstance(first_non_none, list):
                    merged[key] = [item for sublist in values for item in sublist]
                else:
                    merged[key] = values
            return merged

        merged_obs = _merge_obs_dicts(obs_dicts)
        merged_final_obs = None
        if any(final_obs is not None for final_obs in final_obs_list):
            final_obs_or_obs = [
                final_obs if final_obs is not None else obs_dict
                for obs_dict, final_obs in zip(obs_dicts, final_obs_list)
            ]
            merged_final_obs = _merge_obs_dicts(final_obs_or_obs)

        return {"obs": merged_obs, "final_obs": merged_final_obs}

    def send_chunk_actions(
        self,
        output_channel: Channel,
        chunk_actions: torch.Tensor | np.ndarray,
        mode: Literal["train", "eval"] = "train",
    ):
        """Send action shards to mapped env ranks.

        Args:
            output_channel: Channel carrying rollout->env action chunks.
            chunk_actions: Predicted action chunk batch (tensor or ndarray).
            mode: Rollout mode, either ``"train"`` or ``"eval"``.
        """
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        dst_ranks_and_sizes = self.dst_ranks[mode]
        split_sizes = [size for _, size in dst_ranks_and_sizes]
        chunk_actions_split = self._split_actions(chunk_actions, split_sizes)
        for (dst_rank, _), chunk_action_i in zip(
            dst_ranks_and_sizes, chunk_actions_split
        ):
            if isinstance(chunk_action_i, torch.Tensor):
                chunk_action_i = (
                    chunk_action_i.detach().cpu().contiguous()
                )  # for evaluation
            output_channel.put(
                chunk_action_i,
                key=CommMapper.build_channel_key(
                    self._rank, dst_rank, extra=f"{mode}_actions"
                ),
                async_op=True,
            )

    def send_chunk_actions_to_ranks(
        self,
        output_channel: Channel,
        chunk_actions: torch.Tensor | np.ndarray,
        dst_ranks: list[int],
        sizes: list[int],
        mode: Literal["train", "eval"] = "train",
    ):
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        chunk_actions_split = self._split_actions(chunk_actions, sizes)
        for dst_rank, chunk_action_i in zip(dst_ranks, chunk_actions_split):
            if isinstance(chunk_action_i, torch.Tensor):
                chunk_action_i = chunk_action_i.detach().cpu().contiguous()
            output_channel.put(
                chunk_action_i,
                key=CommMapper.build_channel_key(
                    self._rank, dst_rank, extra=f"{mode}_actions"
                ),
                async_op=True,
            )

    def _split_rollout_result(
        self, rollout_result: RolloutResult, sizes: list[int]
    ) -> list[RolloutResult]:
        def _split_optional_tensor(
            tensor: torch.Tensor | None,
        ) -> tuple[torch.Tensor | None, ...]:
            if tensor is None:
                return tuple(None for _ in sizes)
            return tuple(torch.split(tensor, sizes, dim=0))

        split_actions = _split_optional_tensor(rollout_result.actions)
        split_prev_logprobs = _split_optional_tensor(rollout_result.prev_logprobs)
        split_prev_values = _split_optional_tensor(rollout_result.prev_values)
        split_bootstrap_values = _split_optional_tensor(rollout_result.bootstrap_values)
        split_save_flags = _split_optional_tensor(rollout_result.save_flags)
        split_versions = _split_optional_tensor(rollout_result.versions)
        split_forward_inputs = (
            [{} for _ in sizes]
            if not rollout_result.forward_inputs
            else [
                {
                    key: torch.split(value, sizes, dim=0)[idx]
                    for key, value in rollout_result.forward_inputs.items()
                }
                for idx in range(len(sizes))
            ]
        )

        return [
            RolloutResult(
                actions=split_actions[idx],
                prev_logprobs=split_prev_logprobs[idx],
                prev_values=split_prev_values[idx],
                bootstrap_values=split_bootstrap_values[idx],
                save_flags=split_save_flags[idx],
                forward_inputs=split_forward_inputs[idx],
                versions=split_versions[idx],
            )
            for idx in range(len(sizes))
        ]

    def send_rollout_result(
        self,
        output_channel: Channel,
        rollout_result: RolloutResult,
        mode: Literal["train", "eval"] = "train",
    ):
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        dst_ranks_and_sizes = self.dst_ranks[mode]
        split_sizes = [size for _, size in dst_ranks_and_sizes]
        split_rollout_results = self._split_rollout_result(rollout_result, split_sizes)
        for (dst_rank, _), rollout_result_i in zip(
            dst_ranks_and_sizes, split_rollout_results
        ):
            output_channel.put(
                rollout_result_i,
                key=CommMapper.build_channel_key(
                    self._rank, dst_rank, extra=f"{mode}_rollout_results"
                ),
                async_op=True,
            )

    def send_rollout_result_to_ranks(
        self,
        output_channel: Channel,
        rollout_result: RolloutResult,
        dst_ranks: list[int],
        sizes: list[int],
        mode: Literal["train", "eval"] = "train",
    ):
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        split_rollout_results = self._split_rollout_result(rollout_result, sizes)
        for dst_rank, rollout_result_i in zip(dst_ranks, split_rollout_results):
            output_channel.put(
                rollout_result_i,
                key=CommMapper.build_channel_key(
                    self._rank, dst_rank, extra=f"{mode}_rollout_results"
                ),
                async_op=True,
            )

    def set_global_step(self, global_step: int):
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(global_step)
