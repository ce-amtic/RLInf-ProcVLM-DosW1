from __future__ import annotations

import atexit
import base64
import contextlib
import json
import logging
import os
import struct
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any, Deque, Iterator, Optional

from PIL import Image

logger = logging.getLogger(__name__)

_IMAGE_SENTINEL = "__rlinf_image__"
_MAX_FRAME_BYTES = int(os.getenv("PROCVLM_FRAME_MAX_BYTES", str(512 * 1024 * 1024)))
# Reward-image PNG encoding is the dominant client-side cost when the reward window
# (history_size) is large (e.g. robometer's 8-frame windows -> 8x the images of a
# 1-frame ProcVLM window). It is embarrassingly parallel (PIL's C encoder releases the
# GIL), so encode across a thread pool. compress_level=1 (fast zlib) keeps payloads
# close to the default (level 6) size while cutting per-image CPU several-fold; level 0
# would triple the payload and risk the frame-size limit.
_PNG_COMPRESS_LEVEL = int(os.getenv("PROCVLM_PNG_COMPRESS_LEVEL", "1"))
_ENCODE_WORKERS = int(os.getenv("PROCVLM_ENCODE_WORKERS", str(min(32, (os.cpu_count() or 8)))))


def _get_nonnegative_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return max(0, int(default))
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return max(0, int(default))


_REQUEST_LOG_INTERVAL = _get_nonnegative_int_env("PROCVLM_LOG_EVERY_N_REQUESTS", 50)


def _should_log_request(request_id: Any, interval: int = _REQUEST_LOG_INTERVAL) -> bool:
    try:
        request_id_int = int(request_id)
    except (TypeError, ValueError):
        return False
    return request_id_int == 1 or (interval > 0 and request_id_int % interval == 0)


def _read_exact(stream, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8")
        chunks.append(chunk)
        remaining -= len(chunk)
    data = b"".join(chunks)
    if len(data) != size:
        raise EOFError(f"Expected {size} bytes, received {len(data)} bytes")
    return data


def _write_all(stream, data: bytes) -> None:
    view = memoryview(data)
    offset = 0
    while offset < len(view):
        written = stream.write(view[offset:])
        if written is None:
            offset = len(view)
        elif written <= 0:
            raise BrokenPipeError("ProcVLM stream write returned no progress")
        else:
            offset += written


def write_frame(stream, payload: dict[str, Any]) -> None:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    _write_all(stream, struct.pack("!I", len(data)))
    _write_all(stream, data)
    stream.flush()


def read_frame(stream) -> dict[str, Any] | None:
    header = stream.read(4)
    if not header:
        return None
    if isinstance(header, str):
        header = header.encode("utf-8")
    if len(header) != 4:
        header += _read_exact(stream, 4 - len(header))
    length = struct.unpack("!I", header)[0]
    if length > _MAX_FRAME_BYTES:
        raise ValueError(
            f"ProcVLM frame length {length} exceeds limit {_MAX_FRAME_BYTES}; "
            "stdout may have been polluted by non-frame output"
        )
    payload = _read_exact(stream, length)
    return json.loads(payload.decode("utf-8"))


def _encode_one_image(image: Image.Image) -> dict[str, Any]:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG", compress_level=_PNG_COMPRESS_LEVEL)
    return {
        _IMAGE_SENTINEL: {
            "format": "PNG",
            "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }
    }


def _rebuild_value(value: Any, encoded: "Iterator[Any]") -> Any:
    if isinstance(value, Image.Image):
        return next(encoded)
    if isinstance(value, dict):
        return {key: _rebuild_value(item, encoded) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rebuild_value(item, encoded) for item in value]
    return value


def _encode_value(value: Any) -> Any:
    # Two-pass parallel image encoding. Pass 1 collects every PIL image in traversal
    # order; the images are PNG-encoded in a thread pool (order preserved by map); pass 2
    # rebuilds the exact same structure, pulling encoded images in order. Byte-identical
    # to the previous serial recursion (tuples still become lists), just parallel.
    images: list[Image.Image] = []

    def _collect(item: Any) -> None:
        if isinstance(item, Image.Image):
            images.append(item)
        elif isinstance(item, dict):
            for sub in item.values():
                _collect(sub)
        elif isinstance(item, (list, tuple)):
            for sub in item:
                _collect(sub)

    _collect(value)

    if not images:
        return _rebuild_value(value, iter(()))

    workers = max(1, min(_ENCODE_WORKERS, len(images)))
    if workers == 1:
        encoded = [_encode_one_image(img) for img in images]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            encoded = list(pool.map(_encode_one_image, images))
    return _rebuild_value(value, iter(encoded))


def _decode_value(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value.keys()) == {_IMAGE_SENTINEL}:
            payload = value[_IMAGE_SENTINEL]
            image_bytes = base64.b64decode(payload["data"])
            return Image.open(BytesIO(image_bytes)).convert("RGB")
        return {key: _decode_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_value(item) for item in value]
    return value


def encode_batch_items(batch_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _encode_value(batch_items)


def decode_batch_items(batch_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _decode_value(batch_items)


def _count_images(value: Any) -> int:
    if isinstance(value, Image.Image):
        return 1
    if isinstance(value, dict):
        return sum(_count_images(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_count_images(item) for item in value)
    return 0


class ProcVLMSubprocessClient:
    def __init__(
        self,
        python_executable: str,
        server_script: str,
        model_path: str,
        procvlm_repo_path: str | None = None,
        device_map: str | None = None,
        torch_dtype: str | None = None,
        attn_implementation: str | None = None,
        server_backend: str | None = None,
        vllm_tp: int | None = None,
        vllm_engine_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.python_executable = str(Path(python_executable).expanduser())
        self.server_script = str(Path(server_script).expanduser().resolve())
        self.model_path = str(model_path)
        self.procvlm_repo_path = (
            str(Path(procvlm_repo_path).expanduser().resolve())
            if procvlm_repo_path
            else None
        )
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.attn_implementation = attn_implementation
        self.server_backend = str(server_backend).lower() if server_backend else None
        self.vllm_tp = int(vllm_tp) if vllm_tp is not None else None
        self.vllm_engine_kwargs = dict(vllm_engine_kwargs or {})

        if not Path(self.python_executable).exists():
            raise FileNotFoundError(
                f"ProcVLM python executable not found: {self.python_executable}"
            )
        if not Path(self.server_script).exists():
            raise FileNotFoundError(
                f"ProcVLM server script not found: {self.server_script}"
            )

        self._process: subprocess.Popen[bytes] | None = None
        self._request_lock = threading.Lock()
        self._request_id = 0
        self._stderr_tail: Deque[str] = deque(maxlen=200)
        self._stderr_thread: threading.Thread | None = None
        self._closed = False

        self._start_process()
        atexit.register(self.close)

    def _build_command(self) -> list[str]:
        command = [self.python_executable, "-u", self.server_script]
        command.extend(["--model-path", self.model_path])
        if self.procvlm_repo_path:
            command.extend(["--procvlm-repo-path", self.procvlm_repo_path])
        if self.device_map is not None:
            command.extend(["--device-map", str(self.device_map)])
        if self.torch_dtype is not None:
            command.extend(["--torch-dtype", str(self.torch_dtype)])
        if self.attn_implementation is not None:
            command.extend(["--attn-implementation", str(self.attn_implementation)])
        if self.server_backend:
            command.extend(["--backend", self.server_backend])
        if self.vllm_tp is not None:
            command.extend(["--vllm-tp", str(self.vllm_tp)])
        if self.vllm_engine_kwargs:
            command.extend(
                [
                    "--vllm-engine-kwargs",
                    json.dumps(
                        self.vllm_engine_kwargs,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                ]
            )
        return command

    def _drain_stderr(self) -> None:
        process = self._process
        if process is None or process.stderr is None:
            return
        traceback_budget = 0
        for raw_line in iter(process.stderr.readline, b""):
            if not raw_line:
                break
            line = raw_line.decode("utf-8", errors="replace").rstrip("\n")
            self._stderr_tail.append(line)
            if (
                (
                    line.startswith("PROCVLM_SERVER:")
                    and not line.startswith("PROCVLM_SERVER_DEBUG:")
                )
                or "ERROR" in line
                or "Traceback" in line
                or traceback_budget > 0
            ):
                logger.info("ProcVLM server stderr: %s", line)
            if "Traceback" in line:
                traceback_budget = 80
            elif traceback_budget > 0:
                traceback_budget -= 1

    def _stderr_text(self) -> str:
        return "\n".join(self._stderr_tail)

    def _start_process(self) -> None:
        if self._closed:
            raise RuntimeError("ProcVLMSubprocessClient is closed")

        command = self._build_command()
        logger.info("Starting ProcVLM subprocess backend: %s", command)
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=os.environ.copy(),
        )
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()

        if self._process.stdout is None:
            raise RuntimeError("ProcVLM subprocess stdout is not available")

        ready = read_frame(self._process.stdout)
        if ready is None:
            self._raise_process_error("ProcVLM server exited before readiness")
        if ready.get("type") != "ready" or not ready.get("ok", False):
            self._raise_process_error(
                f"Unexpected ProcVLM startup response: {ready!r}"
            )
        logger.info("ProcVLM subprocess backend is ready.")

    def _ensure_process(self) -> subprocess.Popen[bytes]:
        if self._closed:
            raise RuntimeError("ProcVLMSubprocessClient is closed")
        if self._process is None or self._process.poll() is not None:
            self._start_process()
        assert self._process is not None
        return self._process

    def _raise_process_error(self, message: str) -> None:
        stderr_text = self._stderr_text()
        if stderr_text:
            message = f"{message}\nProcVLM stderr tail:\n{stderr_text}"
        process = self._process
        if process is not None and process.poll() is not None:
            message = f"{message}\nProcVLM exit code: {process.returncode}"
        raise RuntimeError(message)

    def batch_infer(
        self,
        batch_items: list[dict[str, Any]],
        **generate_kwargs: Any,
    ) -> list[str]:
        if not batch_items:
            return []

        with self._request_lock:
            process = self._ensure_process()
            if process.stdin is None or process.stdout is None:
                self._raise_process_error("ProcVLM subprocess streams are unavailable")

            self._request_id += 1
            request_id = self._request_id
            payload = {
                "id": request_id,
                "op": "batch_infer",
                "batch_items": encode_batch_items(batch_items),
                "generate_kwargs": generate_kwargs,
            }

            try:
                logger.debug(
                    "Sending ProcVLM request id=%s batch_size=%s",
                    request_id,
                    len(batch_items),
                )
                start_time = time.perf_counter()
                write_frame(process.stdin, payload)
                response = read_frame(process.stdout)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            except Exception as exc:
                self._raise_process_error(
                    f"Failed to communicate with ProcVLM server: {exc}"
                )

            if response is None:
                self._raise_process_error("ProcVLM server closed the stream")
            if response.get("id") != request_id:
                self._raise_process_error(
                    f"ProcVLM server returned mismatched response id: {response!r}"
                )
            if not response.get("ok", False):
                error = response.get("error", "ProcVLM server reported an error")
                traceback_text = response.get("traceback", "")
                if traceback_text:
                    error = f"{error}\n{traceback_text}"
                self._raise_process_error(error)

            outputs = response.get("outputs", [])
            if not isinstance(outputs, list):
                self._raise_process_error(
                    f"ProcVLM server returned invalid outputs: {response!r}"
                )
            logger.debug(
                "Received ProcVLM response id=%s output_count=%s",
                request_id,
                len(outputs),
            )
            if _should_log_request(request_id):
                logger.info(
                    "ProcVLM request id=%s finished batch_size=%s output_count=%s elapsed_ms=%.0f",
                    request_id,
                    len(batch_items),
                    len(outputs),
                    elapsed_ms,
                )
            return [str(item) for item in outputs]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        process = self._process
        self._process = None
        if process is None:
            return

        try:
            if process.poll() is None and process.stdin is not None:
                try:
                    write_frame(
                        process.stdin,
                        {
                            "id": self._request_id + 1,
                            "op": "shutdown",
                        },
                    )
                except Exception:
                    pass
                try:
                    process.wait(timeout=5)
                except Exception:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except Exception:
                        process.kill()
        finally:
            for stream in (process.stdin, process.stdout, process.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def run_procvlm_reward_server(
    model_path: str,
    procvlm_repo_path: str | None = None,
    device_map: str | None = None,
    torch_dtype: str | None = None,
    attn_implementation: str | None = "sdpa",
    backend: str = "hf",
    vllm_tp: int = 1,
    vllm_engine_kwargs: dict[str, Any] | None = None,
    input_stream: Any | None = None,
    output_stream: Any | None = None,
) -> None:
    input_stream = input_stream or sys.stdin.buffer
    output_stream = output_stream or sys.stdout.buffer

    if procvlm_repo_path:
        repo_path = Path(procvlm_repo_path).expanduser().resolve()
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

    backend = str(backend).lower()
    vllm_engine_kwargs = dict(vllm_engine_kwargs or {})
    server_log_interval = _get_nonnegative_int_env(
        "PROCVLM_SERVER_LOG_EVERY_N_REQUESTS", _REQUEST_LOG_INTERVAL
    )

    value_head_backends = {"value_head", "value_head_subprocess", "transformers_value_head"}
    transformer_backends = {"hf", "direct", "subprocess", "transformers", "transformers_subprocess"}

    if backend in {"vllm", "vllm_subprocess"}:
        with contextlib.redirect_stdout(sys.stderr):
            from evqa.model import batch_chat_with_vllm

        model = None
        processor = None
        infer_fn = batch_chat_with_vllm
    elif backend in value_head_backends:
        with contextlib.redirect_stdout(sys.stderr):
            from evqa.model import batch_chat_with_value_head

        model = None
        processor = None
        infer_fn = batch_chat_with_value_head
    elif backend in transformer_backends:
        with contextlib.redirect_stdout(sys.stderr):
            from evqa.model import load_procvlm

        with contextlib.redirect_stdout(sys.stderr):
            model, processor = load_procvlm(
                model_path,
                device_map=device_map or "auto",
                torch_dtype=torch_dtype or "auto",
                attn_implementation=attn_implementation or "sdpa",
            )
            model.eval()
        infer_fn = None
    else:
        raise ValueError(
            f"Unsupported ProcVLM server backend '{backend}'. "
            "Expected one of 'hf', 'value_head', or 'vllm'."
        )

    logger.info(
        "ProcVLM server ready backend=%s model_path=%s device_map=%s",
        backend,
        model_path,
        device_map,
    )
    write_frame(output_stream, {"type": "ready", "ok": True})

    while True:
        request = read_frame(input_stream)
        if request is None:
            break

        request_id = request.get("id")
        op = request.get("op")
        logger.debug("ProcVLM server received request id=%s op=%s", request_id, op)
        if op == "shutdown":
            write_frame(
                output_stream,
                {"id": request_id, "ok": True, "shutdown": True},
            )
            break
        if op != "batch_infer":
            write_frame(
                output_stream,
                {
                    "id": request_id,
                    "ok": False,
                    "error": f"Unsupported ProcVLM server op: {op!r}",
                },
            )
            continue

        try:
            batch_items = decode_batch_items(request.get("batch_items", []))
            generate_kwargs = dict(request.get("generate_kwargs", {}))
            logger.debug(
                "ProcVLM server decoded request id=%s batch_size=%s image_count=%s",
                request_id,
                len(batch_items),
                _count_images(batch_items),
            )
            with contextlib.redirect_stdout(sys.stderr):
                if backend in {"vllm", "vllm_subprocess"}:
                    sampling_kwargs = {}
                    for key in ("top_p", "top_k", "min_p", "repetition_penalty"):
                        if key in generate_kwargs:
                            sampling_kwargs[key] = generate_kwargs[key]
                    logger.debug(
                        "ProcVLM server entering vLLM request id=%s tp=%s engine_kwargs=%s",
                        request_id,
                        vllm_tp,
                        vllm_engine_kwargs,
                    )
                    outputs = infer_fn(
                        batch_items=batch_items,
                        model_path=model_path,
                        max_new_tokens=int(generate_kwargs.get("max_new_tokens", 128)),
                        temperature=float(generate_kwargs.get("temperature", 0.0)),
                        tp=int(vllm_tp),
                        sampling_kwargs=sampling_kwargs or None,
                        engine_kwargs=vllm_engine_kwargs,
                    )
                elif backend in value_head_backends:
                    logger.debug(
                        "ProcVLM server entering value-head transformers request id=%s",
                        request_id,
                    )
                    outputs = infer_fn(
                        batch_items=batch_items,
                        model_path=model_path,
                        max_new_tokens=int(generate_kwargs.pop("max_new_tokens", 128)),
                        temperature=float(generate_kwargs.pop("temperature", 0.0)),
                        device_map=device_map or "auto",
                        torch_dtype=torch_dtype or "auto",
                        attn_implementation=attn_implementation or "sdpa",
                        **generate_kwargs,
                    )
                else:
                    logger.debug(
                        "ProcVLM server entering transformers request id=%s",
                        request_id,
                    )
                    outputs = model.batch_infer(
                        batch_items=batch_items,
                        processor=processor,
                        **generate_kwargs,
                    )
            if _should_log_request(request_id, server_log_interval):
                logger.info(
                    "ProcVLM server finished request id=%s output_count=%s",
                    request_id,
                    len(outputs),
                )
            else:
                logger.debug(
                    "ProcVLM server finished request id=%s output_count=%s",
                    request_id,
                    len(outputs),
                )
            write_frame(
                output_stream,
                {"id": request_id, "ok": True, "outputs": outputs},
            )
        except Exception as exc:
            write_frame(
                output_stream,
                {
                    "id": request_id,
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )


def main(argv: Optional[list[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the ProcVLM reward server")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--procvlm-repo-path", default=None)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--torch-dtype", default=None)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--backend",
        default="hf",
        choices=[
            "hf",
            "direct",
            "subprocess",
            "transformers",
            "transformers_subprocess",
            "value_head",
            "value_head_subprocess",
            "transformers_value_head",
            "vllm",
            "vllm_subprocess",
        ],
    )
    parser.add_argument("--vllm-tp", type=int, default=1)
    parser.add_argument("--vllm-engine-kwargs", default=None)
    args = parser.parse_args(argv)

    log_level_name = os.getenv("PROCVLM_SERVER_LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    logging.basicConfig(
        level=log_level,
        stream=sys.stderr,
        format="PROCVLM_SERVER:%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    vllm_engine_kwargs = None
    if args.vllm_engine_kwargs:
        vllm_engine_kwargs = json.loads(args.vllm_engine_kwargs)

    try:
        frame_fd = os.dup(sys.stdout.buffer.fileno())
        os.set_inheritable(frame_fd, False)
        frame_stdout = os.fdopen(frame_fd, "wb", buffering=0)
        os.dup2(sys.stderr.buffer.fileno(), sys.stdout.buffer.fileno())
        sys.stdout = sys.stderr
        with contextlib.redirect_stdout(sys.stderr):
            run_procvlm_reward_server(
                model_path=args.model_path,
                procvlm_repo_path=args.procvlm_repo_path,
                device_map=args.device_map,
                torch_dtype=args.torch_dtype,
                attn_implementation=args.attn_implementation,
                backend=args.backend,
                vllm_tp=args.vllm_tp,
                vllm_engine_kwargs=vllm_engine_kwargs,
                input_stream=sys.stdin.buffer,
                output_stream=frame_stdout,
            )
    except Exception:
        logger.exception("ProcVLM reward server exited with an error")
        raise SystemExit(1)
