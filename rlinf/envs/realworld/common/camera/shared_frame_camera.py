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

"""Read coherent, fresh RGB frames from the local RUC video gateway."""

import json
import time
from pathlib import Path

import numpy as np


class SharedFrameCamera:
    """Use the gateway's write/ready metadata protocol without owning USB."""

    def __init__(self, name, directory, serial, timeout_s=1.0, max_age_s=0.5):
        self.name = name
        self.directory = Path(directory)
        self.serial = serial
        self.timeout_s = timeout_s
        self.max_age_s = max_age_s

    def open(self) -> None:
        self.get_frame()

    def close(self) -> None:
        pass

    def get_frame(self) -> np.ndarray:
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            try:
                before = json.loads((self.directory / "meta.json").read_text())
                age = time.time() - before.get("captured_at", 0)
                if before.get("state") != "ready" or not 0 <= age <= self.max_age_s:
                    time.sleep(0.005)
                    continue
                if before.get("serial") != self.serial:
                    raise ValueError(f"Camera serial mismatch in {self.directory}")
                h, w = before["color_height"], before["color_width"]
                if not (0 < h <= 4096 and 0 < w <= 4096):
                    raise ValueError("Invalid shared camera dimensions")
                frame = np.fromfile(self.directory / "color.bgr", dtype=np.uint8)
                after = json.loads((self.directory / "meta.json").read_text())
                if before == after and frame.size == h * w * 3:
                    return frame.reshape(h, w, 3)
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            time.sleep(0.002)
        raise TimeoutError(f"No coherent fresh camera frame in {self.directory}")
