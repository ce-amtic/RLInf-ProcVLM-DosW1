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

import gc
import json
import logging
import os
import queue
import re
import shutil
import threading
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Union

from omegaconf.dictconfig import DictConfig

from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics, print_metrics_table
from rlinf.utils.runner_utils import check_progress
from rlinf.utils.timers import Timer

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from rlinf.workers.actor.async_fsdp_sac_policy_worker import (
        AsyncEmbodiedSACFSDPPolicy,
    )
    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
    from rlinf.workers.actor.fsdp_nft_policy_worker import EmbodiedNFTFSDPPolicy
    from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy
    from rlinf.workers.env.async_env_worker import AsyncEnvWorker
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker
    from rlinf.workers.rollout.hf.async_huggingface_worker import (
        AsyncMultiStepRolloutWorker,
    )
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class EmbodiedRunner:
    def __init__(
        self,
        cfg: DictConfig,
        actor: Union[
            "EmbodiedFSDPActor",
            "EmbodiedNFTFSDPPolicy",
            "EmbodiedSACFSDPPolicy",
            "AsyncEmbodiedSACFSDPPolicy",
        ],
        rollout: Union["MultiStepRolloutWorker", "AsyncMultiStepRolloutWorker"],
        env: Union["EnvWorker", "AsyncEnvWorker"],
        reward: Union["EmbodiedRewardWorker"] = None,
        critic=None,
    ):
        self.cfg = cfg
        self.actor = actor
        self.rollout = rollout
        self.env = env
        self.critic = critic
        self.reward = reward
        self.weight_sync_interval = self.cfg.runner.weight_sync_interval
        self.defer_reward_init_until_after_first_weight_sync = bool(
            self.cfg.reward.get("defer_init_until_after_first_weight_sync", False)
        )
        self.reward_initialized = False
        self.overlap_env_bootstrap = bool(
            self.cfg.runner.get("overlap_env_bootstrap", False)
        )
        # Data channels
        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")
        self.actor_channel = Channel.create("Actor")
        if self.reward is not None:
            self.reward_channel = Channel.create("Reward")
        else:
            self.reward_channel = None

        # this timer checks if we should stop training
        self.run_timer = Timer(None)  # Timer that checks if we should stop training

        self.consumed_samples = 0
        # the step here is GRPO step
        self.global_step = 0

        # compute `max_steps`
        self.set_max_steps()

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)

        self.logger = get_logger()
        self.metric_logger = MetricLogger(cfg)
        self.enable_per_worker_metric_log = bool(
            self.cfg.runner.get("per_worker_log", False)
        )
        self.keep_latest_checkpoints = int(
            self.cfg.runner.get("keep_latest_checkpoints", 3) or 0
        )
        self.keep_best_checkpoint = bool(
            self.cfg.runner.get("keep_best_checkpoint", True)
        )
        self.best_checkpoint_metric = str(
            self.cfg.runner.get("best_checkpoint_metric", "eval/success_once")
        )
        self.best_checkpoint_fallback_metric = str(
            self.cfg.runner.get("best_checkpoint_fallback_metric", "env/success_once")
        )
        self.best_checkpoint_score = float("-inf")
        self.best_checkpoint_step: int | None = None
        self._load_best_checkpoint_metadata()

        # Async logging setup
        self.stop_logging = False
        self.log_queue = queue.Queue()
        self.log_thread = threading.Thread(target=self._log_worker, daemon=True)
        self.log_thread.start()

    def _log_worker(self):
        """Background thread for processing log messages."""
        while not self.stop_logging:
            try:
                # Wait for log message with timeout
                log_func, args = self.log_queue.get(timeout=0.1)
                log_func(*args)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Logging error: {e}")
                continue

    def print_metrics_table_async(
        self,
        step: int,
        total_steps: int,
        start_time: float,
        metrics: dict,
        start_step: int = 0,
    ):
        """Async version that puts table printing in queue."""
        self.log_queue.put(
            (print_metrics_table, (step, total_steps, start_time, metrics, start_step))
        )

    def init_workers(self):
        # create worker in order to decrease the maximum memory usage
        rollout_handle = self.rollout.init_worker()
        env_handle = self.env.init_worker()
        if self.reward is not None and not self.defer_reward_init_until_after_first_weight_sync:
            self.reward.init_worker().wait()
            self.reward_initialized = True

        rollout_handle.wait()
        env_handle.wait()
        self.actor.init_worker().wait()

        resume_dir = self.cfg.runner.get("resume_dir", None)
        if resume_dir is None:
            return

        self.logger.info(f"Resuming training from checkpoint directory {resume_dir}.")
        actor_checkpoint_path = os.path.join(resume_dir, "actor")
        assert os.path.exists(actor_checkpoint_path), (
            f"resume_dir {actor_checkpoint_path} does not exist."
        )
        self.actor.load_checkpoint(actor_checkpoint_path).wait()
        self.global_step = int(resume_dir.split("global_step_")[-1])

    def ensure_reward_initialized(self):
        if self.reward is None or self.reward_initialized:
            return
        self.reward.init_worker().wait()
        self.reward_initialized = True

    def update_rollout_weights(self):
        rollout_handle: Handle = self.rollout.sync_model_from_actor()
        actor_handle: Handle = self.actor.sync_model_to_rollout()
        actor_handle.wait()
        rollout_handle.wait()

    def evaluate(self):
        env_handle: Handle = self.env.evaluate(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
        )
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        env_results = env_handle.wait()
        rollout_handle.wait()
        eval_metrics_list = [results for results in env_results if results is not None]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    def _log_ranked_metrics(
        self,
        metrics_list: list[dict] | None,
        step: int,
        prefix: str,
        worker_group_name: str,
        add_prefix: bool = True,
    ):
        if not self.enable_per_worker_metric_log or not metrics_list:
            return
        for rank, metrics in enumerate(metrics_list):
            if not metrics:
                continue
            metrics_to_log = (
                {f"{prefix}/{k}": v for k, v in metrics.items()}
                if add_prefix
                else metrics
            )
            self.metric_logger.log(
                data=metrics_to_log,
                step=step,
                worker_group_name=worker_group_name,
                rank=rank,
            )

    def _aggregate_numeric_metrics(self, metrics_list: list[dict] | None) -> dict:
        if not metrics_list:
            return {}
        merged_metrics = defaultdict(list)
        for metrics in metrics_list:
            if not metrics:
                continue
            for key, value in metrics.items():
                merged_metrics[key].append(value)
        return {
            key: (sum(values) / len(values))
            for key, values in merged_metrics.items()
            if values
        }

    def _process_ranked_numeric_results(
        self, results: list[dict], metric_field: str
    ) -> tuple[dict, list[dict]]:
        metric_list: list[dict] = []
        per_rank_metrics: dict[int, list[dict]] = defaultdict(list)
        for result in results:
            metrics = result.get(metric_field, None)
            if not metrics:
                continue
            metric_list.append(metrics)
            rank = result.get("rank", None)
            if rank is not None:
                per_rank_metrics[int(rank)].append(metrics)

        aggregated_metrics = self._aggregate_numeric_metrics(metric_list)
        ranked_metrics_list: list[dict] = []
        if per_rank_metrics:
            max_rank = max(per_rank_metrics.keys())
            ranked_metrics_list = [{} for _ in range(max_rank + 1)]
            for rank, metrics_list in per_rank_metrics.items():
                ranked_metrics_list[rank] = self._aggregate_numeric_metrics(
                    metrics_list
                )
        return aggregated_metrics, ranked_metrics_list

    def _process_ranked_eval_results(
        self, results: list[dict], metric_field: str
    ) -> tuple[dict, list[dict]]:
        metric_list: list[dict] = []
        per_rank_metrics: dict[int, list[dict]] = defaultdict(list)
        for result in results:
            metrics = result.get(metric_field, None)
            if not metrics:
                continue
            metric_list.append(metrics)
            rank = result.get("rank", None)
            if rank is not None:
                per_rank_metrics[int(rank)].append(metrics)

        aggregated_metrics = (
            compute_evaluate_metrics(metric_list) if metric_list else {}
        )
        ranked_metrics_list: list[dict] = []
        if per_rank_metrics:
            max_rank = max(per_rank_metrics.keys())
            ranked_metrics_list = [{} for _ in range(max_rank + 1)]
            for rank, metrics_list in per_rank_metrics.items():
                ranked_metrics_list[rank] = compute_evaluate_metrics(metrics_list)
        return aggregated_metrics, ranked_metrics_list

    def _log_training_step_metrics(
        self,
        step: int,
        start_time: float,
        start_step: int,
        env_handle: Handle,
        rollout_handle: Handle,
        reward_handle: Handle | None,
        actor_training_handle: Handle,
        actor_rollout_metrics: list[dict] | None,
        actor_training_metrics: list[dict] | None,
    ) -> dict:
        time_metrics = self.timer.consume_durations()
        time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
        env_time_metrics, env_time_metrics_per_rank = env_handle.consume_durations(
            return_per_rank=True
        )
        rollout_time_metrics, rollout_time_metrics_per_rank = (
            rollout_handle.consume_durations(return_per_rank=True)
        )
        actor_time_metrics, actor_time_metrics_per_rank = (
            actor_training_handle.consume_durations(return_per_rank=True)
        )
        time_metrics.update({f"time/env/{k}": v for k, v in env_time_metrics.items()})
        time_metrics.update(
            {f"time/rollout/{k}": v for k, v in rollout_time_metrics.items()}
        )
        time_metrics.update(
            {f"time/actor/{k}": v for k, v in actor_time_metrics.items()}
        )
        reward_time_metrics_per_rank = None
        if self.reward is not None and reward_handle is not None:
            reward_time_metrics, reward_time_metrics_per_rank = (
                reward_handle.consume_durations(return_per_rank=True)
            )
            time_metrics.update(
                {f"time/reward/{k}": v for k, v in reward_time_metrics.items()}
            )

        env_results = env_handle.wait()
        env_results_list = [results for results in env_results if results is not None]
        env_metrics = compute_evaluate_metrics(env_results_list)
        env_metrics = {f"env/{k}": v for k, v in env_metrics.items()}
        ranked_env_results = [
            {"rank": rank, "env": rank_metrics}
            for rank, rank_metrics in enumerate(env_results)
            if rank_metrics is not None
        ]
        _, env_metrics_per_rank = self._process_ranked_eval_results(
            ranked_env_results, metric_field="env"
        )

        rollout_metrics = {
            f"rollout/{k}": v
            for k, v in self._aggregate_numeric_metrics(actor_rollout_metrics).items()
        }
        training_metrics = {
            f"train/{k}": v
            for k, v in self._aggregate_numeric_metrics(actor_training_metrics).items()
        }

        logging_metrics = {}
        logging_metrics.update(time_metrics)
        logging_metrics.update(env_metrics)
        logging_metrics.update(rollout_metrics)
        logging_metrics.update(training_metrics)
        self.metric_logger.log(logging_metrics, step, commit=True)
        self._log_ranked_metrics(
            metrics_list=actor_rollout_metrics,
            step=step,
            prefix="rollout",
            worker_group_name=self.actor.worker_group_name,
        )
        self._log_ranked_metrics(
            metrics_list=actor_training_metrics,
            step=step,
            prefix="train",
            worker_group_name=self.actor.worker_group_name,
        )
        self._log_ranked_metrics(
            metrics_list=actor_time_metrics_per_rank,
            step=step,
            prefix="time/actor",
            worker_group_name=self.actor.worker_group_name,
        )
        self._log_ranked_metrics(
            metrics_list=rollout_time_metrics_per_rank,
            step=step,
            prefix="time/rollout",
            worker_group_name=self.rollout.worker_group_name,
        )
        self._log_ranked_metrics(
            metrics_list=env_time_metrics_per_rank,
            step=step,
            prefix="time/env",
            worker_group_name=self.env.worker_group_name,
        )
        self._log_ranked_metrics(
            metrics_list=env_metrics_per_rank,
            step=step,
            prefix="env",
            worker_group_name=self.env.worker_group_name,
        )
        if self.reward is not None:
            self._log_ranked_metrics(
                metrics_list=reward_time_metrics_per_rank,
                step=step,
                prefix="time/reward",
                worker_group_name=self.reward.worker_group_name,
            )

        self.print_metrics_table_async(
            step, self.max_steps, start_time, logging_metrics, start_step
        )
        return logging_metrics

    def _log_eval_metrics(
        self,
        step: int,
        start_time: float,
        start_step: int,
        eval_metrics: dict,
    ) -> dict:
        time_metrics = self.timer.consume_durations()
        logging_metrics = {f"time/{k}": v for k, v in time_metrics.items()}
        logging_metrics.update({f"eval/{k}": v for k, v in eval_metrics.items()})
        self.metric_logger.log(logging_metrics, step, commit=True)
        self.print_metrics_table_async(
            step, self.max_steps, start_time, logging_metrics, start_step
        )
        return logging_metrics

    def run(self):
        start_step = self.global_step
        start_time = time.time()
        if self.cfg.runner.get("eval_before_training", False):
            with self.timer("eval"):
                self.update_rollout_weights()
                eval_metrics = self.evaluate()
            eval_log_step = start_step - 1 if start_step > 0 else 0
            self._log_eval_metrics(
                eval_log_step,
                start_time,
                eval_log_step,
                eval_metrics,
            )
            del eval_metrics
            gc.collect()

        for _step in range(start_step, self.max_steps):
            # set global step
            self.actor.set_global_step(self.global_step)
            self.rollout.set_global_step(self.global_step)
            if self.reward is not None:
                self.reward.set_global_step(self.global_step)

            with self.timer("step"):
                with self.timer("sync_weights"):
                    if _step % self.weight_sync_interval == 0:
                        self.update_rollout_weights()
                with self.timer("generate_rollouts"):
                    self.ensure_reward_initialized()
                    env_handle: Handle = self.env.interact(
                        input_channel=self.env_channel,
                        rollout_channel=self.rollout_channel,
                        reward_channel=self.reward_channel,
                        actor_channel=self.actor_channel,
                    )
                    rollout_handle: Handle = self.rollout.generate(
                        input_channel=self.rollout_channel,
                        output_channel=self.env_channel,
                    )
                    reward_handle = None
                    if self.reward is not None:
                        reward_handle: Handle = self.reward.compute_rewards(
                            input_channel=self.reward_channel,
                            output_channel=self.env_channel,
                        )
                    self.actor.recv_rollout_trajectories(
                        input_channel=self.actor_channel
                    ).wait()
                    rollout_handle.wait()
                    if self.reward is not None:
                        reward_handle.wait()
                    env_handle.wait()

                # compute advantages and returns.
                with self.timer("cal_adv_and_returns"):
                    actor_rollout_metrics = (
                        self.actor.compute_advantages_and_returns().wait()
                    )

                # actor training.
                if (
                    self.reward is not None
                    and self.cfg.reward.get("offload_before_actor_update", True)
                ):
                    with self.timer("reward_offload_before_actor_update"):
                        self.reward.offload_for_actor_update().wait()
                actor_training_handle: Handle = self.actor.run_training()
                env_bootstrap_handle: Handle | None = None
                if self.overlap_env_bootstrap and _step + 1 < self.max_steps:
                    env_bootstrap_handle = self.env.prefetch_train_bootstrap(
                        rollout_channel=self.rollout_channel
                    )

                actor_training_metrics = actor_training_handle.wait()
                if env_bootstrap_handle is not None:
                    env_bootstrap_handle.wait()

                self.global_step += 1

                run_val, save_model, is_train_end = check_progress(
                    self.global_step,
                    self.max_steps,
                    self.cfg.runner.val_check_interval,
                    self.cfg.runner.save_interval,
                    1.0,
                    run_time_exceeded=False,
                )

            training_logging_metrics = self._log_training_step_metrics(
                step=_step,
                start_time=start_time,
                start_step=start_step,
                env_handle=env_handle,
                rollout_handle=rollout_handle,
                reward_handle=reward_handle,
                actor_training_handle=actor_training_handle,
                actor_rollout_metrics=actor_rollout_metrics,
                actor_training_metrics=actor_training_metrics,
            )
            del (
                env_handle,
                rollout_handle,
                reward_handle,
                actor_training_handle,
                actor_rollout_metrics,
                actor_training_metrics,
            )
            gc.collect()

            eval_logging_metrics = {}
            if run_val:
                with self.timer("eval"):
                    self.update_rollout_weights()
                    eval_metrics = self.evaluate()
                eval_logging_metrics = self._log_eval_metrics(
                    _step, start_time, start_step, eval_metrics
                )
                del eval_metrics
                gc.collect()

            if save_model:
                with self.timer("save_checkpoint"):
                    checkpoint_dir = self._save_checkpoint()
                    checkpoint_metrics = {}
                    checkpoint_metrics.update(training_logging_metrics)
                    checkpoint_metrics.update(eval_logging_metrics)
                    self._update_best_checkpoint(checkpoint_metrics, checkpoint_dir)
                    self._prune_checkpoints()
                save_time_metrics = {
                    f"time/{k}": v for k, v in self.timer.consume_durations().items()
                }
                self.metric_logger.log(save_time_metrics, _step, commit=True)

        self.metric_logger.finish()

        # Stop logging thread
        self.stop_logging = True
        self.log_queue.join()  # Wait for all queued logs to be processed
        self.log_thread.join(timeout=1.0)

    def _checkpoint_root(self) -> str:
        return os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
            "checkpoints",
        )

    def _load_best_checkpoint_metadata(self) -> None:
        metadata_path = os.path.join(self._checkpoint_root(), "best_success.json")
        if not os.path.exists(metadata_path):
            return
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            self.best_checkpoint_score = float(metadata.get("score", float("-inf")))
            step = metadata.get("step", None)
            self.best_checkpoint_step = int(step) if step is not None else None
        except Exception as exc:
            self.logger.warning(
                f"Failed to load best checkpoint metadata from {metadata_path}: {exc}"
            )

    @staticmethod
    def _metric_to_float(value) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _select_checkpoint_metric(self, metrics: dict) -> tuple[str, float] | None:
        for metric_name in (
            self.best_checkpoint_metric,
            self.best_checkpoint_fallback_metric,
            "eval/success_once",
            "env/success_once",
        ):
            score = self._metric_to_float(metrics.get(metric_name))
            if score is not None:
                return metric_name, score
        return None

    def _update_best_checkpoint(self, metrics: dict, checkpoint_dir: str) -> None:
        if not self.keep_best_checkpoint:
            return
        selected = self._select_checkpoint_metric(metrics)
        if selected is None:
            return
        metric_name, score = selected
        if score < self.best_checkpoint_score:
            return

        checkpoint_root = self._checkpoint_root()
        checkpoint_name = os.path.basename(checkpoint_dir.rstrip(os.sep))
        best_link_path = os.path.join(checkpoint_root, "best_success")
        metadata_path = os.path.join(checkpoint_root, "best_success.json")

        os.makedirs(checkpoint_root, exist_ok=True)
        if os.path.lexists(best_link_path):
            if os.path.islink(best_link_path) or os.path.isfile(best_link_path):
                os.unlink(best_link_path)
            else:
                shutil.rmtree(best_link_path)
        os.symlink(checkpoint_name, best_link_path)

        self.best_checkpoint_score = score
        self.best_checkpoint_step = self.global_step
        metadata = {
            "step": self.global_step,
            "score": score,
            "metric": metric_name,
            "checkpoint": checkpoint_name,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
        self.logger.info(
            "Updated best-success checkpoint: "
            f"step={self.global_step}, {metric_name}={score:.6f}"
        )

    def _prune_checkpoints(self) -> None:
        if self.keep_latest_checkpoints <= 0:
            return

        checkpoint_root = self._checkpoint_root()
        if not os.path.isdir(checkpoint_root):
            return

        step_dirs: list[tuple[int, str]] = []
        for name in os.listdir(checkpoint_root):
            match = re.fullmatch(r"global_step_(\d+)", name)
            if match is None:
                continue
            path = os.path.join(checkpoint_root, name)
            if os.path.isdir(path):
                step_dirs.append((int(match.group(1)), path))

        if len(step_dirs) <= self.keep_latest_checkpoints:
            return

        step_dirs.sort(key=lambda item: item[0], reverse=True)
        keep_steps = {step for step, _ in step_dirs[: self.keep_latest_checkpoints]}
        if self.best_checkpoint_step is not None:
            keep_steps.add(self.best_checkpoint_step)

        for step, path in step_dirs:
            if step in keep_steps:
                continue
            shutil.rmtree(path)
            self.logger.info(f"Pruned old checkpoint: {path}")

    def _save_checkpoint(self) -> str:
        self.logger.info(f"Saving checkpoint at step {self.global_step}.")
        base_output_dir = os.path.join(
            self._checkpoint_root(),
            f"global_step_{self.global_step}",
        )
        actor_save_path = os.path.join(base_output_dir, "actor")
        os.makedirs(actor_save_path, exist_ok=True)
        self.actor.save_checkpoint(actor_save_path, self.global_step).wait()
        return base_output_dir

    def set_max_steps(self):
        self.num_steps_per_epoch = 1
        self.max_steps = self.num_steps_per_epoch * self.cfg.runner.max_epochs

        if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    @property
    def epoch(self):
        return self.global_step // self.num_steps_per_epoch
