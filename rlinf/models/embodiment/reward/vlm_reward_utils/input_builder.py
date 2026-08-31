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

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from rlinf.data.datasets.vlm import (
    QwenTrendProgressSFTDataset,
    VLMBaseDataset,
)

logger = logging.getLogger(__name__)


def _to_pil_images(
    images: Union[torch.Tensor, list[torch.Tensor]],
) -> list[Image.Image]:
    """Convert EnvOutput image tensors to per-sample PIL image lists.

    Expected EnvOutput image formats: [B, H, W, C] or [B, T, H, W, C].
    """
    if isinstance(images, torch.Tensor):
        arr = images.detach().cpu().numpy()
    elif isinstance(images, list):
        if len(images) == 0:
            return []
        arr = torch.stack(images).cpu().numpy()
    else:
        raise TypeError(f"Unsupported image input type: {type(images)}")

    arr = arr[None, ...] if arr.ndim == 3 else arr
    if arr.ndim == 5:
        arr = arr.reshape(arr.shape[0] * arr.shape[1], *arr.shape[2:])
    if arr.ndim != 4:
        raise ValueError(f"Invalid image batch shape for PIL conversion: {arr.shape}")

    return [Image.fromarray(frame[..., :3]).convert("RGB") for frame in arr]


def extract_images(
    observations: dict[str, Any],
    image_keys: list[str],
) -> list[list[Any]]:
    """
    Args:
        observations: dict[str, Any], shape = [num_envs, ...]
        image_keys: list[str], shape = [num_image_keys]

    Returns:
        list[list[Any]]: images array with shape [num_envs, num_image_keys]
    """
    image_keys = image_keys or ["main_images"]
    batch_size = observations[image_keys[0]].shape[0]
    images: list[list[Any]] = [[] for _ in range(batch_size)]

    for image_key in image_keys:
        images_all_env = observations[image_key]
        if images_all_env is None:
            continue
        images_all_env = _to_pil_images(images_all_env)
        for i in range(batch_size):
            images[i].append(images_all_env[i])

    return images


INPUT_BUILDER_REGISTRY: dict[str, type] = {}


def register_input_builder(name: str):
    def decorator(cls: type):
        INPUT_BUILDER_REGISTRY[name.lower()] = cls
        return cls

    return decorator


def get_input_builder(name: str) -> type:
    name_lower = name.lower()
    if name_lower not in INPUT_BUILDER_REGISTRY:
        raise ValueError(f"InputBuilder '{name}' not registered")
    return INPUT_BUILDER_REGISTRY[name_lower]


@register_input_builder("base_input_builder")
@dataclass
class BaseInputBuilder:
    system_prompt: Optional[str] = None
    use_chat_template: bool = True
    image_keys: list[str] = field(default_factory=lambda: ["main_images"])
    _processor: Optional[AutoProcessor] = field(default=None)

    def get_valid_input_ids(self, observations: dict[str, Any]) -> list[int]:
        return list(range(len(observations[self.image_keys[0]])))

    def prepare_inputs(
        self, observations: dict[str, Any], valid_input_ids: list[int]
    ) -> torch.Tensor:
        return {"images_list": None, "videos_list": None, "prompt_texts_list": None}

    def process_inputs(self, prepared_inputs: dict[str, Any]):
        return prepared_inputs

    def build_inputs(self, observations: dict[str, Any], device: torch.device):
        valid_input_ids = self.get_valid_input_ids(observations)
        prepared_inputs = self.prepare_inputs(observations, valid_input_ids)
        processed_inputs = self.process_inputs(prepared_inputs)
        processed_inputs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in processed_inputs.items()
        }
        return processed_inputs


@register_input_builder("base_vlm_input_builder")
@dataclass
class BaseVLMInputBuilder(BaseInputBuilder):
    def prepare_inputs(self, observations: dict[str, Any], valid_input_ids: list[int]):
        images = extract_images(observations, self.image_keys)
        images_list = [images[env_idx] for env_idx in valid_input_ids]
        task_descriptions = [
            str(observations["task_descriptions"][env_idx] or "")
            for env_idx in valid_input_ids
        ]

        prompt_texts_list: list[str] = []
        for task_description in task_descriptions:
            task_description = task_description.strip()
            prompt_texts = [
                # One prompt text
                f"Task: {task_description}\n\n"
                "Evaluate the task and return a reward score between 0 and 1."
            ]
            prompt_texts_list.append(prompt_texts)
        return {
            "images_list": images_list,
            "videos_list": None,
            "prompt_texts_list": prompt_texts_list,
        }

    def process_inputs(self, prepared_inputs: dict[str, Any]):
        prompt_texts_list = prepared_inputs.get("prompt_texts_list")
        images_list = prepared_inputs.get("images_list")

        processed_inputs: dict[str, Any] = {}
        for prompt_texts, images in zip(prompt_texts_list, images_list):
            _, processed_input = VLMBaseDataset.process_inputs(
                self._processor,
                self.system_prompt,
                self.use_chat_template,
                prompt_texts=prompt_texts,
                images=images,
            )
            for key, value in processed_input.items():
                if isinstance(value, torch.Tensor):
                    processed_inputs[key] = (
                        value
                        if key not in processed_inputs
                        else torch.cat([processed_inputs[key], value], dim=0)
                    )
                else:
                    processed_inputs[key] = value
        return processed_inputs


@register_input_builder("history_vlm_input_builder")
@dataclass(kw_only=True)
class HistoryVLMInputBuilder(BaseVLMInputBuilder):
    history_buffer_names: list[str]

    def get_valid_input_ids(
        self,
        observations: dict[str, Any],
        history_input: dict[str, dict[str, list[list[Any]]]],
    ) -> list[int]:
        histories = tuple(
            history
            for history_buffer in history_input.values()
            for history in history_buffer.values()
        )
        valid_ids = range(len(histories[0]))
        return [
            env_id
            for env_id in valid_ids
            if all(history[env_id] for history in histories)
        ]

    def prepare_inputs(
        self,
        observations: dict[str, Any],
        history_input: dict[str, dict[str, list[list[Any]]]],
        valid_input_ids: list[int],
    ):
        del history_input
        return {"images_list": None, "videos_list": None, "prompt_texts_list": None}

    def build_inputs(
        self,
        observations: dict[str, Any],
        device: torch.device,
        history_input: dict[str, dict[str, list[list[Any]]]],
    ):
        valid_input_ids = self.get_valid_input_ids(observations, history_input)
        if len(valid_input_ids) == 0:
            return {}, valid_input_ids

        prepared_inputs = self.prepare_inputs(
            observations, history_input, valid_input_ids
        )
        processed_inputs = self.process_inputs(prepared_inputs)
        processed_inputs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in processed_inputs.items()
        }
        return processed_inputs, valid_input_ids


@register_input_builder("video_vlm_input_builder")
@dataclass
class VideoVLMInputBuilder(HistoryVLMInputBuilder):
    video_keys: list[str] = field(default_factory=lambda: ["main_images"])

    def extract_videos(
        self,
        history_buffer: dict[str, list[list[Any]]],
        video_keys: Optional[list[str]] = None,
    ) -> list[list[Any]]:
        """
        Convert one named history buffer payload into processor-ready videos.
        """
        video_keys = video_keys or self.video_keys
        if not video_keys:
            return []

        first_video_key = video_keys[0]
        batch_size = len(history_buffer.get(first_video_key, []))

        if batch_size == 0:
            return []

        videos: list[list[list[Image.Image]]] = [
            [[] for _ in video_keys] for _ in range(batch_size)
        ]

        for batch_idx in range(batch_size):
            for video_idx, video_key in enumerate(video_keys):
                video_frames = _to_pil_images(history_buffer[video_key][batch_idx])
                videos[batch_idx][video_idx].extend(video_frames)

        return videos


@register_input_builder("qwentrend_input_builder")
@dataclass
class QwentrendInputBuilder(VideoVLMInputBuilder):
    video_keys: list[str] = field(
        default_factory=lambda: ["main_images", "extra_view_images"]
    )
    default_task_description: str = ""

    def prepare_inputs(
        self,
        observations: dict[str, Any],
        history_input: dict[str, dict[str, list[list[Any]]]],
        valid_input_ids: list[int],
    ):
        history_window = history_input.get("history_window", {})
        videos_clip = self.extract_videos(history_window, self.video_keys)
        videos_list = [videos_clip[env_id] for env_id in valid_input_ids]
        task_descriptions = observations.get(
            "task_descriptions",
            [self.default_task_description] * len(videos_clip),
        )

        prompt_texts_list: list[list[str]] = []
        for env_id in valid_input_ids:
            prompt_texts_list.append(
                [
                    f"You are currently performing the task: {task_descriptions[env_id]}. "
                    "You are given two synchronized 5-frame videos from different camera "
                    "views (main view and third-person view) of the same robot action "
                    "window. Judge whether the action trend is positive, negative, or "
                    "unclear. Answer with exactly one word: positive, negative, or unclear."
                ]
            )

        return {
            "images_list": None,
            "videos_list": videos_list,
            "prompt_texts_list": prompt_texts_list,
        }

    def process_inputs(self, prepared_inputs: dict[str, Any]):
        prompt_texts_list = prepared_inputs.get("prompt_texts_list")
        videos_list = prepared_inputs.get("videos_list")

        _, processed_inputs, _ = QwenTrendProgressSFTDataset.process_inputs(
            processor=self._processor,
            system_prompt=self.system_prompt,
            use_chat_template=self.use_chat_template,
            prompt_texts=prompt_texts_list,
            videos=videos_list,
            answer_text=None,
        )
        return processed_inputs


DEFAULT_PROCVLM_PROMPT_TEMPLATE = (
    'Given the recent observation and the task "{task}", first infer the '
    "remaining atomic actions required to complete the task. Then estimate the "
    "current completion percentage and output it as a float wrapped by "
    "<progress> tags."
)


@register_input_builder("procvlm_progress_input_builder")
@dataclass
class ProcVLMProgressInputBuilder(HistoryVLMInputBuilder):
    history_buffer_name: str = "history_window"
    image_keys: list[str] = field(default_factory=lambda: ["main_images"])
    default_task_description: str = ""
    force_default_task_description: bool = False
    prompt_template: str = DEFAULT_PROCVLM_PROMPT_TEMPLATE
    include_image_placeholders: bool = False
    max_frames_per_key: Optional[int] = None
    # Keep only the last N FLATTENED frames per image key (applied after history
    # entries are expanded to per-step frames). This expresses the official robometer
    # eval windowing: score a timestep from the window of the last N consecutive
    # frames ending at it (example_inference_local.py window_size=N sliding window).
    # NOTE max_frames_per_key operates on history ENTRIES (chunks), not frames — it
    # cannot express this.
    last_frames_per_key: Optional[int] = None
    allowed_image_keys: list[str] = field(
        default_factory=lambda: ["main_images", "extra_view_images", "reward_main_images"]
    )

    def prepare_inputs(
        self,
        observations: dict[str, Any],
        history_input: dict[str, dict[str, list[list[Any]]]],
        valid_input_ids: list[int],
    ):
        history_buffer = history_input.get(self.history_buffer_name)
        if history_buffer is None:
            raise ValueError(
                f"Missing ProcVLM history buffer '{self.history_buffer_name}'. "
                f"Available buffers: {list(history_input.keys())}"
            )

        images_by_env = self.extract_history_images(history_buffer, self.image_keys)
        if self.force_default_task_description:
            task_descriptions = [self.default_task_description] * len(images_by_env)
        else:
            task_descriptions = observations.get(
                "task_descriptions",
                [self.default_task_description] * len(images_by_env),
            )

        batch_items: list[dict[str, Any]] = []
        for env_id in valid_input_ids:
            task_description = self._get_task_description(task_descriptions, env_id)
            images = images_by_env[env_id]
            prompt = self.build_prompt(task_description, len(images))
            batch_items.append(
                {
                    "image": images,
                    "conversations": [{"from": "human", "value": prompt}],
                }
            )

        return {
            "batch_items": batch_items,
            "images_list": None,
            "videos_list": None,
            "prompt_texts_list": None,
        }

    def process_inputs(self, prepared_inputs: dict[str, Any]):
        batch_items = prepared_inputs.get("batch_items")
        if batch_items is None:
            raise ValueError("ProcVLM input builder did not produce batch_items.")
        return {"batch_items": batch_items}

    def build_prompt(self, task_description: str, num_images: int) -> str:
        task_description = str(task_description or self.default_task_description).strip()
        prompt = self.prompt_template.format(
            task=task_description,
            num_images=num_images,
        )
        if self.include_image_placeholders and num_images > 0:
            placeholders = "\n".join("<image>" for _ in range(num_images))
            prompt = f"Recent observations:\n{placeholders}\n{prompt}"
        return prompt

    def extract_history_images(
        self,
        history_buffer: dict[str, list[list[Any]]],
        image_keys: list[str],
    ) -> list[list[Image.Image]]:
        image_keys = image_keys or ["main_images"]
        disallowed_keys = sorted(set(image_keys) - set(self.allowed_image_keys))
        if disallowed_keys:
            raise ValueError(
                "ProcVLM reward images must use third-person camera views only. "
                f"Disallowed image keys: {disallowed_keys}. "
                f"Allowed image keys: {self.allowed_image_keys}."
            )
        first_key = image_keys[0]
        batch_size = len(history_buffer.get(first_key, []))
        images: list[list[Image.Image]] = [[] for _ in range(batch_size)]

        for env_id in range(batch_size):
            for image_key in image_keys:
                key_history = history_buffer.get(image_key)
                if not key_history:
                    continue
                frames = self._select_frames(key_history[env_id])
                pil_frames = _to_pil_images(frames)
                if self.last_frames_per_key is not None:
                    last_n = int(self.last_frames_per_key)
                    if last_n > 0:
                        pil_frames = pil_frames[-last_n:]
                images[env_id].extend(pil_frames)
        return images

    def _select_frames(self, frames: Any) -> Any:
        if self.max_frames_per_key is None:
            return frames
        max_frames = int(self.max_frames_per_key)
        if max_frames <= 0 or len(frames) <= max_frames:
            return frames
        indices = np.linspace(0, len(frames) - 1, num=max_frames)
        indices = np.round(indices).astype(np.int64).tolist()
        if isinstance(frames, torch.Tensor):
            return frames[indices]
        if isinstance(frames, np.ndarray):
            return frames[indices]
        return [frames[i] for i in indices]

    def _get_task_description(self, task_descriptions: Any, env_id: int) -> str:
        try:
            if isinstance(task_descriptions, torch.Tensor):
                value = task_descriptions[env_id]
                return str(value.item() if value.numel() == 1 else value)
            return str(task_descriptions[env_id])
        except Exception:
            return str(self.default_task_description)


@register_input_builder("robodopamine_progress_input_builder")
@dataclass
class RoboDopamineProgressInputBuilder(HistoryVLMInputBuilder):
    """Preserve synchronized camera identities for Robo-Dopamine.

    The generic ProcVLM builder concatenates every configured image key into one
    flat list, which makes it impossible to distinguish front/left/right views.
    This builder emits one temporal list per view and includes stable environment
    metadata required by the stateful incremental protocol.
    """

    history_buffer_name: str = "history_window"
    # The shared RLinf launcher injects image_keys into every progress builder.
    # View identities are controlled by front_key/wrist_key below, so retain
    # this field only for launcher/config compatibility.
    image_keys: Optional[list[str]] = None
    front_key: str = "main_images"
    wrist_key: str = "wrist_images"
    left_wrist_key: str = "left_wrist_images"
    right_wrist_key: str = "right_wrist_images"
    default_task_description: str = ""
    force_default_task_description: bool = True
    max_history_points: Optional[int] = 8
    # Accepted for compatibility with the existing launcher.  Temporal
    # downsampling is controlled by max_history_points in this builder.
    max_frames_per_key: Optional[int] = None
    require_true_multiview: bool = False

    @staticmethod
    def _scalar_at(values: Any, env_id: int, default: Any = None) -> Any:
        if values is None:
            return default
        try:
            value = values[env_id]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu()
                if value.numel() == 1:
                    return value.item()
                return value.reshape(-1).tolist()
            if isinstance(value, np.ndarray):
                if value.size == 1:
                    return value.reshape(-1)[0].item()
                return value.reshape(-1).tolist()
            return value
        except Exception:
            return default

    @staticmethod
    def _entry_images(entry: Any) -> list[Image.Image]:
        return _to_pil_images(entry)

    def _extract_views(
        self,
        history_buffer: dict[str, list[list[Any]]],
        env_id: int,
    ) -> tuple[list[Image.Image], list[Image.Image], list[Image.Image], bool]:
        front_history = history_buffer.get(self.front_key, [])
        if env_id >= len(front_history):
            return [], [], [], False
        front_entries = list(front_history[env_id])
        left_history = history_buffer.get(self.left_wrist_key, [])
        right_history = history_buffer.get(self.right_wrist_key, [])
        has_separate_views = (
            env_id < len(left_history)
            and env_id < len(right_history)
            and bool(left_history[env_id])
            and bool(right_history[env_id])
        )
        left_entries = list(left_history[env_id]) if has_separate_views else []
        right_entries = list(right_history[env_id]) if has_separate_views else []
        wrist_history = history_buffer.get(self.wrist_key, [])
        wrist_entries = (
            list(wrist_history[env_id]) if env_id < len(wrist_history) else []
        )

        if self.max_history_points is not None and int(self.max_history_points) > 0:
            keep = int(self.max_history_points)
            front_entries = front_entries[-keep:]
            left_entries = left_entries[-keep:]
            right_entries = right_entries[-keep:]
            wrist_entries = wrist_entries[-keep:]

        fronts: list[Image.Image] = []
        lefts: list[Image.Image] = []
        rights: list[Image.Image] = []
        true_multiview = bool(left_entries and right_entries) if has_separate_views else bool(wrist_entries)
        for index, front_entry in enumerate(front_entries):
            front_images = self._entry_images(front_entry)
            if not front_images:
                continue
            front = front_images[-1]
            if has_separate_views:
                left_images = (
                    self._entry_images(left_entries[index])
                    if index < len(left_entries)
                    else []
                )
                right_images = (
                    self._entry_images(right_entries[index])
                    if index < len(right_entries)
                    else []
                )
                if left_images and right_images:
                    left, right = left_images[-1], right_images[-1]
                else:
                    true_multiview = False
                    left = right = front
            else:
                wrist_images = (
                    self._entry_images(wrist_entries[index])
                    if index < len(wrist_entries)
                    else []
                )
                if len(wrist_images) >= 2:
                    left, right = wrist_images[0], wrist_images[1]
                else:
                    true_multiview = False
                    left = right = front
            fronts.append(front)
            lefts.append(left)
            rights.append(right)

        if has_separate_views and not (
            len(front_entries) == len(left_entries) == len(right_entries)
        ):
            true_multiview = False

        if self.require_true_multiview and not true_multiview:
            raise ValueError(
                "Robo-Dopamine true multiview was requested but synchronized "
                f"'{self.wrist_key}' observations are unavailable."
            )
        return fronts, lefts, rights, true_multiview

    def prepare_inputs(
        self,
        observations: dict[str, Any],
        history_input: dict[str, dict[str, list[list[Any]]]],
        valid_input_ids: list[int],
    ):
        history_buffer = history_input.get(self.history_buffer_name)
        if history_buffer is None:
            raise ValueError(
                f"Missing Robo-Dopamine history buffer '{self.history_buffer_name}'."
            )

        task_descriptions = observations.get("task_descriptions")
        env_infos = observations.get("env_infos")
        env_infos = env_infos if isinstance(env_infos, dict) else {}
        episode_infos = env_infos.get("episode")
        episode_infos = episode_infos if isinstance(episode_infos, dict) else {}
        batch_items: list[dict[str, Any]] = []
        for env_id in valid_input_ids:
            fronts, lefts, rights, true_multiview = self._extract_views(
                history_buffer, env_id
            )
            if self.force_default_task_description or task_descriptions is None:
                task = self.default_task_description
            else:
                task = self._scalar_at(
                    task_descriptions, env_id, self.default_task_description
                )
            batch_items.append(
                {
                    # Compatibility for RLinf's existing debug-sample and
                    # trajectory renderer.  The Robo-Dopamine server always
                    # prefers the separated view histories below.
                    "image": fronts,
                    "front_images": fronts,
                    "left_wrist_images": lefts,
                    "right_wrist_images": rights,
                    "true_multiview": true_multiview,
                    "task": str(task or self.default_task_description),
                    "stable_env_id": self._scalar_at(
                        observations.get("reward_debug_global_env_id"), env_id
                    ),
                    "done": bool(
                        self._scalar_at(observations.get("dones"), env_id, False)
                    ),
                    # Audit-only labels.  They are not included in the prompt
                    # and the reward server ignores them.
                    "env_success": self._scalar_at(
                        env_infos.get("success"), env_id
                    ),
                    "env_success_once": self._scalar_at(
                        episode_infos.get("success_once"), env_id
                    ),
                }
            )

        return {
            "batch_items": batch_items,
            "images_list": None,
            "videos_list": None,
            "prompt_texts_list": None,
        }

    def process_inputs(self, prepared_inputs: dict[str, Any]):
        batch_items = prepared_inputs.get("batch_items")
        if batch_items is None:
            raise ValueError("Robo-Dopamine input builder produced no batch items.")
        return {"batch_items": batch_items}
