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

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw, ImageFont

from rlinf.config import torch_dtype_from_precision
from rlinf.models.embodiment.reward.procvlm_reward_backend import (
    ProcVLMSubprocessClient,
)
from rlinf.models.embodiment.reward.vlm_reward_model import HistoryVLMRewardModel

logger = logging.getLogger(__name__)


class ProcVLMHistoryRewardModel(HistoryVLMRewardModel):
    """History-buffer reward model backed by ProcVLM progress inference.

    The input builder creates ProcVLM ``batch_items`` from RLinf history windows.
    This model generates ``<progress>XX%</progress>`` text, parses it through the
    configured reward parser, and by default returns clipped progress deltas.
    """

    def __init__(self, cfg: DictConfig):
        self.backend = str(cfg.get("backend", "direct")).lower()
        self.enable_value_head = bool(cfg.get("enable_value_head", False))
        self.reward_form = str(cfg.get("reward_form", "delta")).lower()
        self.pos_clip = float(cfg.get("pos_clip", 0.2))
        self.neg_clip = float(cfg.get("neg_clip", 0.2))
        self.invalid_delta_reward = float(cfg.get("invalid_delta_reward", 0.0))
        self.reset_progress_on_done = bool(cfg.get("reset_progress_on_done", True))
        progress_smoothing_cfg = self._as_plain_dict(cfg.get("progress_smoothing", {}))
        self.progress_smoothing_enabled = self._as_bool(
            progress_smoothing_cfg.get("enabled", False)
        )
        self.progress_smoothing_alpha = min(
            1.0, max(0.0, float(progress_smoothing_cfg.get("alpha", 0.6)))
        )
        self.procvlm_repo_path = cfg.get("procvlm_repo_path", None)
        self.procvlm_python = str(cfg.get("procvlm_python", sys.executable))
        self.device_map = cfg.get("device_map", None)
        self.torch_dtype = cfg.get("torch_dtype", None)
        self.attn_implementation = cfg.get("attn_implementation", "sdpa")
        self.vllm_tp = int(cfg.get("vllm_tp", 1))
        self.vllm_engine_kwargs = self._as_plain_dict(
            cfg.get("vllm_engine_kwargs", {})
        )
        self.base_cost_mb = cfg.get("base_cost_mb", 200)
        self.image_cost_mb = cfg.get("image_cost_mb", 150)
        self.show_progress = bool(cfg.get("show_progress", False))
        self.eager_init = self._as_bool(cfg.get("eager_init", True))
        self.warmup_on_init = self._as_bool(cfg.get("warmup_on_init", True))
        self._warmup_done = False
        self._setup_debug_samples(cfg.get("debug_samples", {}))
        self.global_step = int(cfg.get("global_step", 0))
        self.reward_rank = int(cfg.get("reward_rank", 0))
        self._debug_sample_step: int | None = None
        self._debug_samples_written = 0
        self._trajectory_step: int | None = None
        self._trajectory_records: list[dict[str, Any]] = []
        self._trajectory_selected_env_id: int | None = None
        self._trajectory_selected_key: tuple[int | None, int, int] | None = None
        self._trajectory_written_steps: set[int] = set()
        self._trajectory_writer_threads: list[threading.Thread] = []
        self._previous_progress: torch.Tensor | None = None
        self._evqa_load_error: Exception | None = None
        self._subprocess_client: ProcVLMSubprocessClient | None = None
        self._subprocess_client_kwargs: dict[str, Any] | None = None

        super().__init__(cfg)

    @staticmethod
    def _as_plain_dict(value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, DictConfig):
            return dict(OmegaConf.to_container(value, resolve=True))
        return dict(value)

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def _setup_debug_samples(self, debug_cfg: Any) -> None:
        debug_cfg = self._as_plain_dict(debug_cfg)
        self.debug_samples_enabled = self._as_bool(debug_cfg.get("enabled", False))
        self.debug_samples_every_n_steps = max(
            1, int(debug_cfg.get("every_n_steps", 1))
        )
        self.debug_samples_max_per_step = max(
            0, int(debug_cfg.get("max_samples_per_step", 1))
        )
        self.debug_samples_output_dir = Path(
            str(debug_cfg.get("output_dir", "rm_samples"))
        ).expanduser()
        self.debug_samples_save_format = str(debug_cfg.get("save_format", "gif")).lower()
        self.debug_samples_fps = max(1, int(debug_cfg.get("fps", 2)))
        self.debug_samples_include_prompt = self._as_bool(
            debug_cfg.get("include_prompt", True)
        )
        trajectory_cfg = self._as_plain_dict(debug_cfg.get("trajectory_video", {}))
        self.trajectory_video_enabled = self._as_bool(
            trajectory_cfg.get("enabled", False)
        )
        self.trajectory_video_every_n_steps = max(
            1,
            int(
                trajectory_cfg.get(
                    "every_n_steps", self.debug_samples_every_n_steps
                )
            ),
        )
        self.trajectory_video_output_dir = Path(
            str(trajectory_cfg.get("output_dir", self.debug_samples_output_dir))
        ).expanduser()
        self.trajectory_video_target_reward_rank = int(
            trajectory_cfg.get("target_reward_rank", 0)
        )
        self.trajectory_video_target_env_id = int(
            trajectory_cfg.get("target_env_id", 0)
        )
        self.trajectory_video_fps = max(1, int(trajectory_cfg.get("fps", 4)))
        self.trajectory_video_max_frames = max(
            1, int(trajectory_cfg.get("max_frames", 512))
        )
        self.trajectory_video_finalize_after_points = int(
            trajectory_cfg.get("finalize_after_points", 0)
        )
        self.trajectory_video_min_width = max(
            256, int(trajectory_cfg.get("min_width", 960))
        )
        self.trajectory_video_min_height = max(
            256, int(trajectory_cfg.get("min_height", 540))
        )
        self.trajectory_video_include_prompt = self._as_bool(
            trajectory_cfg.get("include_prompt", self.debug_samples_include_prompt)
        )
        self.trajectory_video_include_raw_output = self._as_bool(
            trajectory_cfg.get("include_raw_output", True)
        )

    def _uses_subprocess_backend(self) -> bool:
        return self.backend in {
            "subprocess",
            "transformers_subprocess",
            "value_head_subprocess",
            "vllm_subprocess",
        }

    def set_global_step(self, global_step: int) -> None:
        if self._trajectory_step is not None and int(global_step) != self._trajectory_step:
            self._finalize_trajectory_video()
        self.global_step = int(global_step)

    def set_worker_info(self, reward_rank: int) -> None:
        self.reward_rank = int(reward_rank)

    def setup_processor(self) -> None:
        if self._uses_subprocess_backend():
            self._processor = None
            return
        super().setup_processor()

    def setup_model(self) -> None:
        if self._uses_subprocess_backend():
            procvlm_python = self.procvlm_python
            procvlm_server_script = self.cfg.get(
                "procvlm_server_script",
                str(Path(__file__).with_name("procvlm_reward_server.py").resolve()),
            )
            torch_dtype = self.torch_dtype or self._precision_to_procvlm_dtype()
            device_map = self.device_map or self._default_device_map()
            if self.backend == "vllm_subprocess" and self.enable_value_head:
                logger.warning(
                    "ProcVLM vLLM reward backend does not support value-head "
                    "generation; disabling reward.model.enable_value_head."
                )
                self.enable_value_head = False
            self._subprocess_client_kwargs = {
                "python_executable": str(procvlm_python),
                "server_script": str(procvlm_server_script),
                "model_path": self.model_path,
                "procvlm_repo_path": self.procvlm_repo_path,
                "device_map": device_map,
                "torch_dtype": torch_dtype,
                "attn_implementation": self.attn_implementation,
            }
            if self.backend == "vllm_subprocess":
                self._subprocess_client_kwargs.update(
                    {
                        "server_backend": "vllm",
                        "vllm_tp": self.vllm_tp,
                        "vllm_engine_kwargs": self.vllm_engine_kwargs,
                    }
                )
            elif self.backend in {"transformers_subprocess", "value_head_subprocess"}:
                self._subprocess_client_kwargs.update(
                    {
                        "server_backend": self.backend,
                    }
                )
            self._model = None
            return

        load_procvlm = self._import_load_procvlm()
        torch_dtype = self.torch_dtype or self._precision_to_procvlm_dtype()
        device_map = self.device_map or self._default_device_map()
        self._model, self._processor = load_procvlm(
            self.model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=self.attn_implementation,
        )
        self._model.eval()

    def _import_load_procvlm(self):
        if self.procvlm_repo_path:
            repo_path = Path(str(self.procvlm_repo_path)).expanduser().resolve()
            if str(repo_path) not in sys.path:
                sys.path.insert(0, str(repo_path))
        try:
            from evqa.model import load_procvlm
        except Exception as exc:
            self._evqa_load_error = exc
            raise ImportError(
                "ProcVLMHistoryRewardModel requires the ProcVLM repo on PYTHONPATH "
                "and its inference dependencies. Set reward.model.procvlm_repo_path "
                "or launch RLInf with PYTHONPATH including third_party/ProcVLM."
            ) from exc
        return load_procvlm

    def _precision_to_procvlm_dtype(self) -> str | torch.dtype:
        try:
            dtype = torch_dtype_from_precision(self.cfg.precision)
        except Exception:
            return "auto"
        if dtype is torch.bfloat16:
            return "bf16"
        if dtype is torch.float16:
            return "fp16"
        if dtype is torch.float32:
            return "fp32"
        return dtype

    def _default_device_map(self) -> str:
        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        return "cpu"

    def _ensure_subprocess_client(self) -> ProcVLMSubprocessClient:
        if self._subprocess_client is None:
            if self._subprocess_client_kwargs is None:
                raise RuntimeError("ProcVLM subprocess backend is not configured.")
            self._subprocess_client = ProcVLMSubprocessClient(
                **self._subprocess_client_kwargs
            )
        return self._subprocess_client

    def eager_initialize(self) -> None:
        if not self._uses_subprocess_backend() or not self.eager_init:
            return

        client = self._ensure_subprocess_client()
        if not self.warmup_on_init or self._warmup_done:
            return

        warmup_item = {
            "image": [Image.new("RGB", (16, 16), color=(0, 0, 0))],
            "conversations": [
                {
                    "from": "human",
                    "value": (
                        "<image>\nEstimate task progress as a percentage. "
                        "Answer with <progress>0%</progress> only."
                    ),
                }
            ],
        }
        logger.info("Warming up ProcVLM reward subprocess backend.")
        client.batch_infer(
            batch_items=[warmup_item],
            max_new_tokens=16,
            temperature=0.0,
            enable_value_head=False,
            base_cost_mb=self.base_cost_mb,
            image_cost_mb=self.image_cost_mb,
            show_progress=False,
        )
        self._warmup_done = True
        logger.info("ProcVLM reward subprocess backend warmup complete.")

    def _close_subprocess_client(self) -> None:
        client = self._subprocess_client
        self._subprocess_client = None
        if client is not None:
            client.close()

    @staticmethod
    def _is_cpu_target(target: Any) -> bool:
        try:
            return torch.device(target).type == "cpu"
        except Exception:
            return str(target).lower().startswith("cpu")

    @property
    def model_device(self) -> torch.device:
        if self._uses_subprocess_backend():
            return torch.device("cpu")
        try:
            return next(self._model.parameters()).device
        except Exception:
            if torch.cuda.is_available():
                return torch.device(f"cuda:{torch.cuda.current_device()}")
            return torch.device("cpu")

    def to(self, *args, **kwargs):
        # ProcVLM is already placed by load_procvlm(device_map=...). Avoid moving
        # an accelerate-dispatched or cached model a second time in RewardWorker.
        if self._uses_subprocess_backend():
            target = args[0] if args else kwargs.get("device", None)
            if target is not None and self._is_cpu_target(target):
                self._finalize_trajectory_video()
                self._close_subprocess_client()
        return self

    @torch.no_grad()
    def _generate_outputs(self, batch_items: list[dict[str, Any]]) -> list[str]:
        if not batch_items:
            return []

        generate_kwargs = {
            "max_new_tokens": int(self.cfg.get("max_new_tokens", 128)),
            "temperature": float(self.cfg.get("temperature", 0.0)),
            "enable_value_head": self.enable_value_head,
            "base_cost_mb": self.base_cost_mb,
            "image_cost_mb": self.image_cost_mb,
            "show_progress": self.show_progress,
        }

        top_p = self.cfg.get("top_p", None)
        if top_p is not None:
            generate_kwargs["top_p"] = float(top_p)
        do_sample = self.cfg.get("do_sample", None)
        if do_sample is not None and not bool(do_sample):
            generate_kwargs["temperature"] = 0.0

        if self._uses_subprocess_backend():
            client = self._ensure_subprocess_client()
            return client.batch_infer(
                batch_items=batch_items,
                **generate_kwargs,
            )

        return self._model.batch_infer(
            batch_items=batch_items,
            processor=self._processor,
            **generate_kwargs,
        )

    def _debug_samples_remaining(self) -> int:
        if (
            not self.debug_samples_enabled
            or self.debug_samples_max_per_step <= 0
            or self.global_step % self.debug_samples_every_n_steps != 0
        ):
            return 0
        if self._debug_sample_step != self.global_step:
            self._debug_sample_step = self.global_step
            self._debug_samples_written = 0
        return max(0, self.debug_samples_max_per_step - self._debug_samples_written)

    @staticmethod
    def _select_debug_value(value: Any, index: int) -> Any:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            item = value.detach().cpu()
            if item.ndim > 0 and item.shape[0] > index:
                item = item[index]
            return item.item() if item.numel() == 1 else item.tolist()
        if isinstance(value, (list, tuple)):
            return value[index] if len(value) > index else None
        try:
            if hasattr(value, "shape") and value.shape and value.shape[0] > index:
                item = value[index]
                return item.item() if hasattr(item, "item") else item
        except Exception:
            return None
        return value

    def _debug_int_value(self, observations: dict[str, Any], key: str, index: int) -> int | None:
        value = self._select_debug_value(observations.get(key), index)
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            return None

    def _save_debug_media(self, images: list[Any], output_base: Path) -> Path | None:
        frames = [
            image.convert("RGB")
            for image in images
            if isinstance(image, Image.Image)
        ]
        if not frames:
            return None

        media_path = output_base.with_suffix(".gif")
        duration_ms = int(1000 / self.debug_samples_fps)
        frames[0].save(
            media_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
        )
        return media_path

    def _write_debug_sample(
        self,
        batch_item: dict[str, Any],
        output: str,
        progress_value: float,
        reward_value: float,
        micro_observations: dict[str, Any],
        local_env_id: int,
        batch_env_id: int,
    ) -> None:
        sample_dir = (
            self.debug_samples_output_dir
            / f"global_step_{self.global_step:06d}"
            / f"reward_rank_{self.reward_rank:03d}"
        )
        sample_dir.mkdir(parents=True, exist_ok=True)

        sample_id = self._debug_samples_written
        output_base = sample_dir / f"sample_{sample_id:03d}_env_{batch_env_id:03d}"
        media_path = self._save_debug_media(batch_item.get("image", []), output_base)

        conversations = batch_item.get("conversations", [])
        prompt = ""
        if conversations and isinstance(conversations[0], dict):
            prompt = str(conversations[0].get("value", ""))

        metadata = {
            "global_step": self.global_step,
            "reward_rank": self.reward_rank,
            "sample_id": sample_id,
            "batch_env_id": batch_env_id,
            "local_env_id": local_env_id,
            "env_rank": self._debug_int_value(
                micro_observations, "reward_debug_env_rank", local_env_id
            ),
            "stage_id": self._debug_int_value(
                micro_observations, "reward_debug_stage_id", local_env_id
            ),
            "global_env_id": self._debug_int_value(
                micro_observations, "reward_debug_global_env_id", local_env_id
            ),
            "task_description": self._select_debug_value(
                micro_observations.get("task_descriptions"), local_env_id
            ),
            "done": self._select_debug_value(
                micro_observations.get("dones"), local_env_id
            ),
            "env_success": batch_item.get("env_success"),
            "env_success_once": batch_item.get("env_success_once"),
            "parsed_progress": progress_value,
            "reward": reward_value,
            "raw_output": output,
            "prompt": prompt if self.debug_samples_include_prompt else "",
            "media_path": str(media_path) if media_path is not None else None,
        }

        json_path = output_base.with_suffix(".json")
        json_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        md_path = output_base.with_suffix(".md")
        media_line = (
            f"![reward sample]({media_path.name})\n\n"
            if media_path is not None
            else ""
        )
        md_path.write_text(
            "\n".join(
                [
                    "# ProcVLM Reward Sample",
                    "",
                    media_line.rstrip(),
                    f"- global_step: {self.global_step}",
                    f"- reward_rank: {self.reward_rank}",
                    f"- batch_env_id: {batch_env_id}",
                    f"- task: {metadata['task_description']}",
                    f"- done: {metadata['done']}",
                    f"- parsed_progress: {progress_value:.6f}",
                    f"- reward: {reward_value:.6f}",
                    "",
                    "## Raw Output",
                    "",
                    "```text",
                    output,
                    "```",
                    "",
                    "## Prompt",
                    "",
                    "```text",
                    prompt if self.debug_samples_include_prompt else "",
                    "```",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        self._debug_samples_written += 1

    def _maybe_dump_debug_samples(
        self,
        batch_items: list[dict[str, Any]],
        outputs: list[str],
        progress: torch.Tensor,
        reward_chunk: torch.Tensor,
        valid_input_ids: list[int],
        micro_observations: dict[str, Any],
        batch_start: int,
    ) -> None:
        remaining = self._debug_samples_remaining()
        if remaining <= 0:
            return

        try:
            progress_cpu = progress.detach().cpu().reshape(-1)
            reward_cpu = reward_chunk.detach().cpu().reshape(-1)
            for output_idx, local_env_id in enumerate(valid_input_ids):
                if remaining <= 0:
                    break
                batch_env_id = batch_start + int(local_env_id)
                progress_value = float(progress_cpu[output_idx].item())
                reward_value = float(reward_cpu[int(local_env_id)].item())
                self._write_debug_sample(
                    batch_item=batch_items[output_idx],
                    output=str(outputs[output_idx]),
                    progress_value=progress_value,
                    reward_value=reward_value,
                    micro_observations=micro_observations,
                    local_env_id=int(local_env_id),
                    batch_env_id=batch_env_id,
                )
                remaining -= 1
        except Exception:
            logger.exception("Failed to dump ProcVLM reward debug samples.")

    def _should_collect_trajectory_video(self) -> bool:
        return (
            self.trajectory_video_enabled
            and self.reward_rank == self.trajectory_video_target_reward_rank
            and self.global_step % self.trajectory_video_every_n_steps == 0
            and self.global_step not in self._trajectory_written_steps
        )

    def _ensure_trajectory_state(self) -> None:
        if self._trajectory_step == self.global_step:
            return
        if self._trajectory_step is not None:
            self._finalize_trajectory_video()
        self._trajectory_step = self.global_step
        self._trajectory_records = []
        self._trajectory_selected_env_id = None

    @staticmethod
    def _coerce_rgb_image(image: Any) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=-1)
        if arr.ndim != 3:
            raise ValueError(f"Invalid image shape for RGB conversion: {arr.shape}")

        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] >= 3:
            arr = arr[..., :3]
        else:
            raise ValueError(f"Invalid image channel count: {arr.shape[-1]}")

        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32, copy=False)
            if arr.size and np.nanmax(arr) <= 1.0:
                arr = arr * 255.0
            arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(np.ascontiguousarray(arr), mode="RGB")

    @staticmethod
    def _last_pil_image(images: Any) -> Image.Image | None:
        if not isinstance(images, (list, tuple)) or not images:
            return None
        image = images[-1]
        try:
            return ProcVLMHistoryRewardModel._coerce_rgb_image(image)
        except Exception:
            return None

    @staticmethod
    def _extract_prompt(batch_item: dict[str, Any]) -> str:
        conversations = batch_item.get("conversations", [])
        if conversations and isinstance(conversations[0], dict):
            return str(conversations[0].get("value", ""))
        return ""

    @staticmethod
    def _load_font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
        candidates = [
            Path(
                "/usr/share/fonts/truetype/dejavu/"
                + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
            ),
            Path(
                "/usr/share/fonts/truetype/liberation2/"
                + ("LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf")
            ),
        ]
        for path in candidates:
            try:
                if path.exists():
                    return ImageFont.truetype(str(path), size=size)
            except Exception:
                continue
        return ImageFont.load_default()

    @staticmethod
    def _wrap_text(
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.ImageFont,
        max_width: int,
        max_lines: int,
    ) -> list[str]:
        lines: list[str] = []
        for paragraph in str(text or "").splitlines() or [""]:
            words = paragraph.split()
            if not words:
                if len(lines) < max_lines:
                    lines.append("")
                continue
            current = ""
            for word in words:
                candidate = word if not current else f"{current} {word}"
                if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
                    current = candidate
                    continue
                if current:
                    lines.append(current)
                    current = word
                else:
                    lines.append(word)
                if len(lines) >= max_lines:
                    lines[-1] = lines[-1].rstrip(".") + "..."
                    return lines
            if current:
                lines.append(current)
                if len(lines) >= max_lines:
                    lines[-1] = lines[-1].rstrip(".") + "..."
                    return lines
        return lines[:max_lines]

    @staticmethod
    def _resize_keep_aspect_min(
        image: Image.Image,
        min_width: int,
        min_height: int,
    ) -> Image.Image:
        image = ProcVLMHistoryRewardModel._coerce_rgb_image(image)
        width, height = image.size
        scale = max(
            min_width / max(width, 1),
            min_height / max(height, 1),
            1.0,
        )
        if scale <= 1.001:
            return image
        resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        return image.resize(
            (int(round(width * scale)), int(round(height * scale))),
            resample,
        )

    def _render_trajectory_frame(
        self,
        record: dict[str, Any],
        records_so_far: list[dict[str, Any]],
    ) -> Image.Image:
        frame = self._resize_keep_aspect_min(
            record["frame"],
            self.trajectory_video_min_width,
            self.trajectory_video_min_height,
        )
        width, height = frame.size
        panel_h = max(230, int(height * 0.34))
        output = Image.new("RGB", (width, height + panel_h), (250, 252, 250))
        output.paste(frame, (0, 0))

        draw = ImageDraw.Draw(output)
        panel_top = height
        draw.rectangle((0, panel_top, width, height + panel_h), fill=(250, 252, 250))
        draw.line((0, panel_top, width, panel_top), fill=(226, 232, 226), width=1)

        title_font = self._load_font(24, bold=True)
        body_font = self._load_font(16)
        small_font = self._load_font(14)

        margin = max(24, int(width * 0.025))
        gap = max(24, int(width * 0.025))
        plot_w = int((width - 2 * margin - gap) * 0.62)
        plot_left = margin
        plot_right = plot_left + plot_w
        text_left = plot_right + gap
        text_right = width - margin
        top = panel_top + 18
        bottom = height + panel_h - 28

        progress_pct = float(record["progress"]) * 100.0
        reward_value = float(record["reward"])
        header = (
            f"RM progress {progress_pct:.2f}%   "
            f"reward {reward_value:+.4f}   "
            f"point {record['point_index']}   "
            f"window {record['window_size']} frames"
        )
        draw.text((plot_left, top), header, font=title_font, fill=(26, 32, 28))

        chart_left = plot_left + 58
        chart_top = top + 52
        chart_right = plot_right - 14
        chart_bottom = bottom - 30

        for tick in [0, 25, 50, 75, 100]:
            y = chart_bottom - (tick / 100.0) * (chart_bottom - chart_top)
            draw.line((chart_left, y, chart_right, y), fill=(224, 229, 224), width=1)
            label = str(tick)
            bbox = draw.textbbox((0, 0), label, font=small_font)
            draw.text(
                (chart_left - 10 - (bbox[2] - bbox[0]), y - 8),
                label,
                font=small_font,
                fill=(105, 112, 105),
            )

        n = max(1, len(records_so_far) - 1)
        points: list[tuple[float, float]] = []
        for idx, item in enumerate(records_so_far):
            x = chart_left + (idx / n) * (chart_right - chart_left)
            y = chart_bottom - (
                max(0.0, min(1.0, float(item["progress"])))
                * (chart_bottom - chart_top)
            )
            points.append((x, y))

        if len(points) >= 2:
            fill_poly = [(points[0][0], chart_bottom), *points, (points[-1][0], chart_bottom)]
            draw.polygon(fill_poly, fill=(224, 241, 230))
            draw.line(points, fill=(92, 154, 113), width=4)
        elif points:
            x, y = points[0]
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(92, 154, 113))
        if points:
            x, y = points[-1]
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(82, 142, 100))
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=(255, 255, 255), width=2)

        draw.text((chart_left, chart_bottom + 8), "rollout RM inference point", font=small_font, fill=(105, 112, 105))
        done_text = f"done={record['done']}"
        draw.text((plot_left, bottom - 18), done_text, font=small_font, fill=(105, 112, 105))

        draw.line(
            (text_left - gap // 2, top, text_left - gap // 2, bottom),
            fill=(226, 232, 226),
            width=1,
        )
        draw.text((text_left, top), "RM output", font=title_font, fill=(26, 32, 28))
        output_text = record["raw_output"] if self.trajectory_video_include_raw_output else ""
        if not output_text:
            output_text = "[raw output hidden]"
        lines = self._wrap_text(
            draw,
            output_text,
            body_font,
            max(1, text_right - text_left),
            max(1, (bottom - top - 45) // 22),
        )
        y = top + 42
        for line in lines:
            draw.text((text_left, y), line, font=body_font, fill=(36, 42, 36))
            y += 22
        return output

    @staticmethod
    def _prepare_video_frames(frames: list[Image.Image]) -> list[Image.Image]:
        if not frames:
            return []

        rgb_frames = [
            ProcVLMHistoryRewardModel._coerce_rgb_image(frame)
            for frame in frames
        ]
        width = max(frame.width for frame in rgb_frames)
        height = max(frame.height for frame in rgb_frames)
        width += width % 2
        height += height % 2

        prepared: list[Image.Image] = []
        for frame in rgb_frames:
            if frame.size == (width, height):
                prepared.append(frame)
                continue
            canvas = Image.new("RGB", (width, height), (250, 252, 250))
            canvas.paste(frame, (0, 0))
            prepared.append(canvas)
        return prepared

    @staticmethod
    def _write_moviepy_mp4_in_process(
        path: Path,
        frames: list[Image.Image],
        fps: int,
    ) -> None:
        from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

        arrays = [np.asarray(frame, dtype=np.uint8) for frame in frames]
        clip = ImageSequenceClip(arrays, fps=max(1, int(round(float(fps)))))
        try:
            clip.write_videofile(
                str(path),
                codec="libx264",
                audio=False,
                logger=None,
                verbose=False,
                ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
            )
        finally:
            close = getattr(clip, "close", None)
            if callable(close):
                close()

    def _write_moviepy_mp4_subprocess(
        self,
        path: Path,
        frames: list[Image.Image],
        fps: int,
    ) -> None:
        python_path = Path(self.procvlm_python).expanduser()
        if not python_path.exists():
            raise FileNotFoundError(
                f"ProcVLM moviepy encoder Python not found: {python_path}"
            )

        script = r"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

manifest_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
fps = max(1, int(round(float(sys.argv[3]))))
frame_paths = json.loads(manifest_path.read_text(encoding="utf-8"))
frames = [
    np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.uint8)
    for frame_path in frame_paths
]
clip = ImageSequenceClip(frames, fps=fps)
try:
    clip.write_videofile(
        str(output_path),
        codec="libx264",
        audio=False,
        logger=None,
        verbose=False,
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
finally:
    close = getattr(clip, "close", None)
    if callable(close):
        close()
"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f"{path.stem}_frames_",
            dir=str(path.parent),
        ) as tmp_dir:
            tmp_path = Path(tmp_dir)
            frame_paths: list[str] = []
            for idx, frame in enumerate(frames):
                frame_path = tmp_path / f"frame_{idx:06d}.png"
                frame.save(frame_path, format="PNG", compress_level=1)
                frame_paths.append(str(frame_path))
            manifest_path = tmp_path / "frames.json"
            manifest_path.write_text(
                json.dumps(frame_paths, ensure_ascii=False),
                encoding="utf-8",
            )

            env = os.environ.copy()
            if self.procvlm_repo_path:
                env["PYTHONPATH"] = (
                    f"{self.procvlm_repo_path}:"
                    f"{env.get('PYTHONPATH', '')}"
                )
            result = subprocess.run(
                [
                    str(python_path),
                    "-c",
                    script,
                    str(manifest_path),
                    str(path),
                    str(fps),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()[-4000:]
            stdout = (result.stdout or "").strip()[-1000:]
            raise RuntimeError(
                "ProcVLM moviepy encoder failed with "
                f"exit code {result.returncode}. stdout={stdout!r} stderr={stderr!r}"
            )

    def _write_mp4(self, path: Path, frames: list[Image.Image], fps: int) -> None:
        video_frames = self._prepare_video_frames(frames)
        if not video_frames:
            return

        try:
            self._write_moviepy_mp4_in_process(path, video_frames, fps)
            return
        except ModuleNotFoundError as exc:
            if exc.name != "moviepy":
                raise
            logger.debug(
                "moviepy is not available in the RLinf reward worker; "
                "encoding rollout reward video with procvlm_python=%s",
                self.procvlm_python,
            )
        except Exception:
            logger.warning(
                "In-process moviepy encoding failed; retrying with "
                "procvlm_python=%s",
                self.procvlm_python,
                exc_info=True,
            )

        self._write_moviepy_mp4_subprocess(path, video_frames, fps)

    def _metadata_from_trajectory_records(self) -> list[dict[str, Any]]:
        metadata = []
        for record in self._trajectory_records:
            metadata.append(
                {
                    key: value
                    for key, value in record.items()
                    if key not in {"frame"}
                }
            )
        return metadata

    def _cleanup_trajectory_writer_threads(self, join: bool = False) -> None:
        alive_threads: list[threading.Thread] = []
        for thread in self._trajectory_writer_threads:
            if join:
                thread.join()
            if thread.is_alive():
                alive_threads.append(thread)
        self._trajectory_writer_threads = alive_threads

    def _write_trajectory_video_artifacts(
        self,
        records: list[dict[str, Any]],
        metadata: list[dict[str, Any]],
        step: int,
        reward_rank: int,
        env_id: int | None,
    ) -> None:
        try:
            out_dir = (
                self.trajectory_video_output_dir
                / f"global_step_{step:06d}"
                / f"reward_rank_{reward_rank:03d}"
            )
            out_dir.mkdir(parents=True, exist_ok=True)
            base = out_dir / f"trajectory_env_{env_id if env_id is not None else 0:03d}"
            video_path = base.with_suffix(".mp4")
            json_path = base.with_suffix(".json")
            md_path = base.with_suffix(".md")

            rendered_frames = [
                self._render_trajectory_frame(record, records[: idx + 1])
                for idx, record in enumerate(records)
            ]
            self._write_mp4(video_path, rendered_frames, self.trajectory_video_fps)

            payload = {
                "global_step": step,
                "reward_rank": reward_rank,
                "batch_env_id": env_id,
                "num_points": len(metadata),
                "num_video_frames": len(records),
                "fps": self.trajectory_video_fps,
                "video_path": str(video_path),
                "records": metadata,
            }
            json_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            prompt = str(records[0].get("prompt", "")) if records else ""
            md_path.write_text(
                "\n".join(
                    [
                        "# ProcVLM Rollout Reward Visualization",
                        "",
                        f"- global_step: {step}",
                        f"- reward_rank: {reward_rank}",
                        f"- batch_env_id: {env_id}",
                        f"- num_points: {len(metadata)}",
                        f"- video: `{video_path.name}`",
                        "",
                        f'<video src="{video_path.name}" controls width="960"></video>',
                        "",
                        "## Prompt",
                        "",
                        "```text",
                        prompt,
                        "```",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            logger.info("Wrote ProcVLM rollout reward video to %s", video_path)
        except Exception:
            logger.exception("Failed to write ProcVLM rollout reward video.")

    def _finalize_trajectory_video(self) -> None:
        if (
            self._trajectory_step is None
            or not self.trajectory_video_enabled
            or not self._trajectory_records
            or self._trajectory_step in self._trajectory_written_steps
        ):
            return

        try:
            if len(self._trajectory_records) > self.trajectory_video_max_frames:
                indices = np.linspace(
                    0,
                    len(self._trajectory_records) - 1,
                    num=self.trajectory_video_max_frames,
                )
                indices = np.round(indices).astype(np.int64).tolist()
                records = [self._trajectory_records[idx] for idx in indices]
            else:
                records = list(self._trajectory_records)

            metadata = self._metadata_from_trajectory_records()
            step = self._trajectory_step
            reward_rank = self.reward_rank
            env_id = self._trajectory_selected_env_id
            self._trajectory_written_steps.add(step)

            thread = threading.Thread(
                target=self._write_trajectory_video_artifacts,
                args=(records, metadata, step, reward_rank, env_id),
                daemon=True,
            )
            thread.start()
            self._trajectory_writer_threads.append(thread)
            self._cleanup_trajectory_writer_threads(join=False)
        except Exception:
            logger.exception("Failed to write ProcVLM rollout reward video.")
        finally:
            self._trajectory_records = []
            self._trajectory_selected_env_id = None
            self._trajectory_selected_key = None

    def _maybe_record_trajectory_video(
        self,
        batch_items: list[dict[str, Any]],
        outputs: list[str],
        progress: torch.Tensor,
        reward_chunk: torch.Tensor,
        valid_input_ids: list[int],
        micro_observations: dict[str, Any],
        batch_start: int,
    ) -> None:
        if not self._should_collect_trajectory_video():
            return
        if not batch_items or not valid_input_ids:
            return

        try:
            self._ensure_trajectory_state()
            progress_cpu = progress.detach().cpu().reshape(-1)
            reward_cpu = reward_chunk.detach().cpu().reshape(-1)

            candidates: list[tuple[int, int, int, int | None, tuple[int | None, int, int]]] = []
            for output_idx, local_env_id in enumerate(valid_input_ids):
                batch_env_id = batch_start + int(local_env_id)
                global_env_id = self._debug_int_value(
                    micro_observations,
                    "reward_debug_global_env_id",
                    int(local_env_id),
                )
                stable_env_id = (
                    int(global_env_id)
                    if global_env_id is not None
                    else int(batch_env_id)
                )
                stage_id = self._debug_int_value(
                    micro_observations, "reward_debug_stage_id", int(local_env_id)
                )
                source_key = (stage_id, stable_env_id, int(local_env_id))
                candidates.append(
                    (
                        output_idx,
                        int(local_env_id),
                        batch_env_id,
                        global_env_id,
                        source_key,
                    )
                )

            chosen: tuple[int, int, int, int | None, tuple[int | None, int, int]] | None = None
            if self._trajectory_selected_key is not None:
                for candidate in candidates:
                    if candidate[4] == self._trajectory_selected_key:
                        chosen = candidate
                        break
            else:
                for candidate in candidates:
                    candidate_env_id = (
                        candidate[3]
                        if candidate[3] is not None
                        else candidate[2]
                    )
                    if candidate_env_id == self.trajectory_video_target_env_id:
                        chosen = candidate
                        break
                if chosen is None and self.trajectory_video_target_env_id < 0:
                    chosen = candidates[0]
                if chosen is not None:
                    self._trajectory_selected_env_id = (
                        chosen[3] if chosen[3] is not None else chosen[2]
                    )
                    self._trajectory_selected_key = chosen[4]

            if chosen is None:
                return

            output_idx, local_env_id, batch_env_id, global_env_id, source_key = chosen
            batch_item = batch_items[output_idx]
            current_frame = self._last_pil_image(batch_item.get("image", []))
            if current_frame is None:
                return

            images = batch_item.get("image", [])
            record = {
                "point_index": len(self._trajectory_records),
                "batch_env_id": batch_env_id,
                "local_env_id": local_env_id,
                "global_env_id": global_env_id,
                "trajectory_source_key": list(source_key),
                "env_rank": self._debug_int_value(
                    micro_observations, "reward_debug_env_rank", local_env_id
                ),
                "stage_id": self._debug_int_value(
                    micro_observations, "reward_debug_stage_id", local_env_id
                ),
                "task_description": self._select_debug_value(
                    micro_observations.get("task_descriptions"), local_env_id
                ),
                "done": self._select_debug_value(
                    micro_observations.get("dones"), local_env_id
                ),
                "progress": float(progress_cpu[output_idx].item()),
                "reward": float(reward_cpu[local_env_id].item()),
                "raw_output": str(outputs[output_idx]),
                "prompt": (
                    self._extract_prompt(batch_item)
                    if self.trajectory_video_include_prompt
                    else ""
                ),
                "window_size": len(images) if isinstance(images, list) else None,
                "frame": current_frame,
            }
            self._trajectory_records.append(record)
            done_value = record.get("done")
            if done_value is True or str(done_value).lower() == "true":
                self._finalize_trajectory_video()
            elif (
                self.trajectory_video_finalize_after_points > 0
                and len(self._trajectory_records)
                >= self.trajectory_video_finalize_after_points
            ):
                self._finalize_trajectory_video()
        except Exception:
            logger.exception("Failed to record ProcVLM rollout reward video point.")

    def _ensure_previous_progress(self, batch_size: int) -> torch.Tensor:
        if self._previous_progress is None or self._previous_progress.shape[0] != batch_size:
            self._previous_progress = torch.zeros(batch_size, dtype=torch.float32)
        return self._previous_progress

    def _compute_reward_from_progress(
        self,
        progress: torch.Tensor,
        valid_input_ids: list[int],
        micro_batch_size: int,
        micro_observations: dict[str, Any],
    ) -> torch.Tensor:
        reward_chunk = torch.full(
            (micro_batch_size,),
            fill_value=self.interval_reward,
            dtype=torch.float32,
        )
        if len(valid_input_ids) == 0:
            if self._previous_progress is not None:
                self._reset_done_progress(self._previous_progress, micro_observations)
            return reward_chunk

        valid_progress = progress.to(dtype=torch.float32).reshape(-1)
        previous = None
        valid_ids_tensor = torch.as_tensor(valid_input_ids, dtype=torch.long)
        if self.progress_smoothing_enabled:
            previous = self._ensure_previous_progress(micro_batch_size)
            valid_progress = (
                self.progress_smoothing_alpha * valid_progress
                + (1.0 - self.progress_smoothing_alpha) * previous[valid_ids_tensor]
            )

        if self.reward_form == "raw":
            reward_chunk[valid_input_ids] = valid_progress
            if self.progress_smoothing_enabled:
                assert previous is not None
                previous[valid_ids_tensor] = valid_progress
                self._reset_done_progress(previous, micro_observations)
        elif self.reward_form == "delta":
            if previous is None:
                previous = self._ensure_previous_progress(micro_batch_size)
            delta = valid_progress - previous[valid_ids_tensor]
            delta = torch.clamp(delta, min=-self.neg_clip, max=self.pos_clip)
            invalid_mask = ~torch.isfinite(delta)
            if invalid_mask.any():
                delta[invalid_mask] = self.invalid_delta_reward
            reward_chunk[valid_input_ids] = delta
            previous[valid_ids_tensor] = valid_progress
            self._reset_done_progress(previous, micro_observations)
        else:
            raise ValueError(
                f"Unsupported ProcVLM reward_form '{self.reward_form}'. "
                "Expected 'delta' or 'raw'."
            )
        return reward_chunk

    def _reset_done_progress(
        self,
        previous: torch.Tensor,
        observations: dict[str, Any],
    ) -> None:
        if not self.reset_progress_on_done:
            return
        dones = observations.get("dones", None)
        if dones is None:
            return
        dones = torch.as_tensor(dones).reshape(-1).bool().cpu()
        if dones.shape[0] == previous.shape[0] and dones.any():
            previous[dones] = 0.0

    def _prepare_reward_micro_batches(
        self,
        reward_input: dict[str, Any],
    ) -> list[dict[str, Any]]:
        history_input: dict[str, dict[str, list[list[Any]]]] = reward_input[
            "history_input"
        ]
        input_batch_size = len(next(iter(next(iter(history_input.values())).values())))
        observations = {
            key: value for key, value in reward_input.items() if key != "history_input"
        }

        infer_micro_batch_size = self.infer_micro_batch_size or input_batch_size
        micro_batches: list[dict[str, Any]] = []
        for start in range(0, input_batch_size, infer_micro_batch_size):
            end = min(start + infer_micro_batch_size, input_batch_size)
            micro_observations = self.slice_observations(observations, start, end)
            micro_history_input = self.slice_history_input(history_input, start, end)
            micro_batch_size = end - start

            prepared_inputs, valid_input_ids = self.input_builder.build_inputs(
                micro_observations,
                self.model_device,
                micro_history_input,
            )
            micro_batches.append(
                {
                    "start": start,
                    "end": end,
                    "micro_observations": micro_observations,
                    "micro_batch_size": micro_batch_size,
                    "prepared_inputs": prepared_inputs,
                    "valid_input_ids": valid_input_ids,
                }
            )
        return micro_batches

    def compute_reward(
        self,
        reward_input: dict[str, Any],
    ) -> torch.Tensor:
        observations = {
            key: value for key, value in reward_input.items() if key != "history_input"
        }

        reward_chunks: list[torch.Tensor] = []
        for micro_batch in self._prepare_reward_micro_batches(reward_input):
            micro_batch_size = micro_batch["micro_batch_size"]
            prepared_inputs = micro_batch["prepared_inputs"]
            valid_input_ids = micro_batch["valid_input_ids"]
            micro_observations = micro_batch["micro_observations"]
            if len(valid_input_ids) == 0:
                reward_chunks.append(
                    self._compute_reward_from_progress(
                        progress=torch.empty(0, dtype=torch.float32),
                        valid_input_ids=[],
                        micro_batch_size=micro_batch_size,
                        micro_observations=micro_observations,
                    )
                )
                continue

            outputs = self._generate_outputs(prepared_inputs["batch_items"])
            progress = self.reward_parser.parse_rewards(outputs)
            reward_chunk = self._compute_reward_from_progress(
                progress=progress,
                valid_input_ids=valid_input_ids,
                micro_batch_size=micro_batch_size,
                micro_observations=micro_observations,
            )
            self._maybe_dump_debug_samples(
                batch_items=prepared_inputs["batch_items"],
                outputs=outputs,
                progress=progress,
                reward_chunk=reward_chunk,
                valid_input_ids=valid_input_ids,
                micro_observations=micro_observations,
                batch_start=micro_batch["start"],
            )
            self._maybe_record_trajectory_video(
                batch_items=prepared_inputs["batch_items"],
                outputs=outputs,
                progress=progress,
                reward_chunk=reward_chunk,
                valid_input_ids=valid_input_ids,
                micro_observations=micro_observations,
                batch_start=micro_batch["start"],
            )
            reward_chunks.append(reward_chunk)

        rewards = torch.cat(reward_chunks, dim=0)
        return self.apply_gt_success_bonus(rewards, observations)

    def compute_reward_sequence(
        self,
        reward_inputs: list[dict[str, Any]],
    ) -> list[torch.Tensor]:
        if not reward_inputs:
            return []

        prepared_sequences = [
            self._prepare_reward_micro_batches(reward_input)
            for reward_input in reward_inputs
        ]
        merged_batch_items: list[dict[str, Any]] = []
        item_slices: dict[tuple[int, int], tuple[int, int]] = {}
        for input_idx, micro_batches in enumerate(prepared_sequences):
            for micro_idx, micro_batch in enumerate(micro_batches):
                valid_input_ids = micro_batch["valid_input_ids"]
                if len(valid_input_ids) == 0:
                    continue
                batch_items = micro_batch["prepared_inputs"]["batch_items"]
                start = len(merged_batch_items)
                merged_batch_items.extend(batch_items)
                item_slices[(input_idx, micro_idx)] = (
                    start,
                    start + len(batch_items),
                )

        merged_outputs = (
            self._generate_outputs(merged_batch_items) if merged_batch_items else []
        )

        rewards_per_input: list[torch.Tensor] = []
        for input_idx, (reward_input, micro_batches) in enumerate(
            zip(reward_inputs, prepared_sequences)
        ):
            observations = {
                key: value
                for key, value in reward_input.items()
                if key != "history_input"
            }
            reward_chunks: list[torch.Tensor] = []
            for micro_idx, micro_batch in enumerate(micro_batches):
                micro_batch_size = micro_batch["micro_batch_size"]
                valid_input_ids = micro_batch["valid_input_ids"]
                if len(valid_input_ids) == 0:
                    reward_chunks.append(
                        self._compute_reward_from_progress(
                            progress=torch.empty(0, dtype=torch.float32),
                            valid_input_ids=[],
                            micro_batch_size=micro_batch_size,
                            micro_observations=micro_batch["micro_observations"],
                        )
                    )
                    continue

                output_start, output_end = item_slices[(input_idx, micro_idx)]
                outputs = merged_outputs[output_start:output_end]
                progress = self.reward_parser.parse_rewards(outputs)
                reward_chunk = self._compute_reward_from_progress(
                    progress=progress,
                    valid_input_ids=valid_input_ids,
                    micro_batch_size=micro_batch_size,
                    micro_observations=micro_batch["micro_observations"],
                )
                self._maybe_dump_debug_samples(
                    batch_items=micro_batch["prepared_inputs"]["batch_items"],
                    outputs=outputs,
                    progress=progress,
                    reward_chunk=reward_chunk,
                    valid_input_ids=valid_input_ids,
                    micro_observations=micro_batch["micro_observations"],
                    batch_start=micro_batch["start"],
                )
                self._maybe_record_trajectory_video(
                    batch_items=micro_batch["prepared_inputs"]["batch_items"],
                    outputs=outputs,
                    progress=progress,
                    reward_chunk=reward_chunk,
                    valid_input_ids=valid_input_ids,
                    micro_observations=micro_batch["micro_observations"],
                    batch_start=micro_batch["start"],
                )
                reward_chunks.append(reward_chunk)

            rewards = torch.cat(reward_chunks, dim=0)
            rewards_per_input.append(self.apply_gt_success_bonus(rewards, observations))

        return rewards_per_input

    def forward(
        self,
        input_data: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "ProcVLMHistoryRewardModel is an inference-time reward model."
        )

    def __del__(self):
        try:
            self._close_subprocess_client()
        except Exception:
            pass
