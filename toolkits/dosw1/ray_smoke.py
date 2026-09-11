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

"""Exercise a local CPU Ray environment worker, without actor/reward/training."""

import argparse
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main():
    import ray

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="dosw1-ray-", dir="/tmp") as tmp:
        ray.init(
            address="local",
            num_cpus=2,
            num_gpus=0,
            include_dashboard=False,
            _node_ip_address="127.0.0.1",
            _temp_dir=tmp,
            object_store_memory=128 * 1024 * 1024,
            runtime_env={
                "env_vars": {"PYTHONPATH": str(REPO), "RAY_USAGE_STATS_ENABLED": "0"}
            },
        )
        try:

            @ray.remote(num_cpus=1)
            def inspect_env(hardware):
                import subprocess

                command = [sys.executable, str(REPO / "toolkits/dosw1/preflight.py")]
                if hardware:
                    command.append("--hardware")
                result = subprocess.run(
                    command, text=True, capture_output=True, timeout=60
                )
                if result.returncode:
                    raise RuntimeError(result.stdout + result.stderr)
                return {"python": sys.executable, "stdout": result.stdout}

            print(
                json.dumps(
                    ray.get(inspect_env.remote(args.hardware), timeout=90), indent=2
                )
            )
        finally:
            ray.shutdown()


if __name__ == "__main__":
    main()
