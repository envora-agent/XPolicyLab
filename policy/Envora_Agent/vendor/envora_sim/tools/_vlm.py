"""Shared single-image VLM client and blocking GPU worker; no idle inference."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from envora_sim.tools.process import ProcessClient


@dataclass(frozen=True)
class Config:
    model: str
    default_revision: ClassVar[str]
    backend: ClassVar[str]
    revision: str | None = None
    device: str = "cuda:0"

    def resolve_model(self) -> tuple[str, str | None]:
        model = self.model
        revision = self.revision
        if Path(model).expanduser().is_dir():
            model = str(Path(model).expanduser().absolute())
        else:
            if revision is None and model == type(self)().model:
                revision = self.default_revision
            if revision is None or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
                raise ValueError(
                    "a remote VLM model requires a full commit SHA revision"
                )
        return model, revision

    def command(self) -> list[str]:
        model, revision = self.resolve_model()
        command = [
            sys.executable,
            "-m",
            f"{type(self).__module__}.inference",
            "--model",
            model,
            "--device",
            self.device,
        ]
        if revision is not None:
            command.extend(["--revision", revision])
        return command

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Model-specific consumer flags and environment variables."""
        defaults = cls()
        prefix = cls.__module__.split(".")[-1]
        for name, default, description in (
            ("model", defaults.model, "Hugging Face model ID or local snapshot"),
            ("revision", None, "full model/code commit SHA (default: pinned model)"),
            ("device", defaults.device, "CUDA device; no CPU offload"),
            ("socket", None, "borrow a worker at this Unix socket"),
        ):
            parser.add_argument(
                f"--{prefix.replace('_', '-')}-{name}",
                type=Path if name == "socket" else str,
                default=os.environ.get(f"{prefix.upper()}_{name.upper()}", default),
                help=description,
            )


class Client(ProcessClient):
    config_type: ClassVar[type[Config]]

    def __init__(
        self,
        config: Config | None = None,
        socket_path: Path | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.config = config or self.config_type()
        super().__init__(self.config.model, self.config.command, socket_path, timeout)

    def start(self) -> Client:
        # Initial startup can download 34.65 GB; request timeouts stay unchanged.
        with self._lock:
            timeout = self.timeout
            self.timeout = max(timeout, 3600.0)
            try:
                return super().start()
            finally:
                self.timeout = timeout

    @classmethod
    def from_arguments(cls, arguments: argparse.Namespace) -> Client:
        prefix = cls.config_type.__module__.split(".")[-1]
        return cls(
            cls.config_type(
                getattr(arguments, f"{prefix}_model"),
                getattr(arguments, f"{prefix}_revision"),
                getattr(arguments, f"{prefix}_device"),
            ),
            getattr(arguments, f"{prefix}_socket"),
        )

    def query(self, rgb: Path, prompt: str, max_new_tokens: int = 512) -> dict:
        return self._call(
            {
                "operation": "query",
                "rgb": str(rgb.expanduser().absolute()),
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
            }
        )

    def point(self, rgb: Path, objects: list[str], max_new_tokens: int = 256) -> dict:
        """One generation per description; return all parsed pixel points."""
        return self._call(
            {
                "operation": "point",
                "rgb": str(rgb.expanduser().absolute()),
                "objects": objects,
                "max_new_tokens": max_new_tokens,
            }
        )

    def warmup(self) -> dict:
        """Explicit blank-image inference; never run periodically."""
        return self._call({"operation": "warmup"})


def main(config_type: type[Config]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=config_type().model)
    parser.add_argument("--revision")
    parser.add_argument("--device", default=config_type().device)
    parser.add_argument("--socket", type=Path)
    parser.add_argument("--rgb", type=Path)
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument("--prompt")
    prompts.add_argument("--objects", nargs="+")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    arguments = parser.parse_args()
    if arguments.socket is None and (
        arguments.rgb is None
        or (arguments.prompt is None and arguments.objects is None)
    ):
        parser.error(
            "--rgb and either --prompt or --objects are required without --socket"
        )
    arguments.model, revision = config_type(
        arguments.model, arguments.revision, arguments.device
    ).resolve_model()

    # Reserve fd 1 for JSON even if model/native libraries print diagnostics.
    sys.stdout.flush()
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    started = time.perf_counter()
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = torch.device(arguments.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "VLM inference requires a CUDA device; CPU offload is disabled"
        )
    torch.cuda.set_device(device)
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "VLM inference requires bfloat16 support on the selected GPU"
        )
    processor = AutoProcessor.from_pretrained(
        arguments.model,
        revision=revision,
        trust_remote_code=config_type.backend == "molmo2",
        token=False,
        use_fast=config_type.backend == "qwen3_vl",
    )
    model = AutoModelForImageTextToText.from_pretrained(
        arguments.model,
        revision=revision,
        trust_remote_code=config_type.backend == "molmo2",
        token=False,
        dtype=torch.bfloat16,
        device_map={"": str(device)},
    ).eval()
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - started

    def infer(request: dict) -> dict:
        nonlocal load_seconds
        started = time.perf_counter()
        operation = request.get("operation", "query")
        if operation not in ("query", "point", "warmup"):
            raise ValueError("operation must be query, point or warmup")
        max_tokens = request.get("max_new_tokens", 512)
        if type(max_tokens) is not int or not 1 <= max_tokens <= 2048:
            raise ValueError("max_new_tokens must be an integer in [1, 2048]")
        if operation == "point":
            objects = request.get("objects")
            if (
                not isinstance(objects, list)
                or not objects
                or any(not isinstance(obj, str) or not obj.strip() for obj in objects)
            ):
                raise ValueError("objects must be a nonempty list of nonempty strings")
            questions = [f"Point at {obj}" for obj in objects]
            if config_type.backend == "qwen3_vl":
                questions = [
                    question + ". Return only a JSON array of objects with "
                    '"point_2d": [x, y], using integer coordinates normalized to '
                    "0..1000. Return [] if no point is visible."
                    for question in questions
                ]
        elif operation == "warmup":
            questions, max_tokens = ["Describe this image."], 8
        else:
            prompt = request.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("prompt must be a nonempty string")
            questions = [prompt]
        if operation == "warmup":
            image = Image.new("RGB", (512, 384))
        else:
            rgb = Path(request["rgb"])
            if not rgb.is_absolute():
                raise ValueError("IPC rgb path must be absolute")
            with Image.open(rgb) as source:
                image = source.convert("RGB")
        width, height = image.size
        loaded = time.perf_counter()
        outputs = []
        for index, question in enumerate(questions):
            with torch.inference_mode():
                inputs = processor.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": question},
                                {"type": "image", "image": image},
                            ],
                        }
                    ],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                generated = model.generate(
                    **inputs, max_new_tokens=max_tokens, do_sample=False, use_cache=True
                )
                tokens = generated[0, inputs["input_ids"].shape[1] :]
                text = processor.tokenizer.decode(tokens, skip_special_tokens=True)
                count = len(tokens)
                eos = model.generation_config.eos_token_id
                eos_ids = eos if isinstance(eos, list) else [eos]
                truncated = count >= max_tokens and int(tokens[-1]) not in eos_ids
                torch.cuda.synchronize(device)
            result = {"text": text, "generated_tokens": count, "truncated": truncated}
            if operation == "point":
                points = []
                if config_type.backend == "qwen3_vl":
                    payload = text.strip()
                    if payload.startswith("```json\n") and payload.endswith("```"):
                        payload = payload[8:-3].strip()
                    status = "points"
                    try:
                        items = json.loads(payload)
                        if not isinstance(items, list):
                            raise ValueError("expected a JSON array")
                        for item in items:
                            pair = item["point_2d"]
                            if (
                                not isinstance(pair, list)
                                or len(pair) != 2
                                or any(
                                    type(v) is not int or not 0 <= v <= 1000
                                    for v in pair
                                )
                            ):
                                raise ValueError("invalid normalized point")
                            points.append(
                                [
                                    min(width - 1, int(pair[0] * width / 1000)),
                                    min(height - 1, int(pair[1] * height / 1000)),
                                ]
                            )
                        if not points:
                            status = "no_points"
                    except (ValueError, TypeError, KeyError):
                        status = "invalid_point_json"
                else:
                    # Single-image Molmo2: frame id followed by (object id, x, y) triplets.
                    # x/y are on a 0..1000 scale. Never interpret unrelated prose as points.
                    tags = re.findall(
                        r'<points\b[^>]*\bcoords=["\']([^"\']*)["\'][^>]*>', text
                    )
                    points = []
                    status = "points" if tags else "no_point_tag"
                    for tag in tags:
                        fields = tag.split()
                        if (
                            len(fields) < 4
                            or (len(fields) - 1) % 3
                            or fields[0] != "1"
                            or any(not field.isdigit() for field in fields)
                        ):
                            status = "invalid_point_tag"
                            break
                        for offset in range(1, len(fields), 3):
                            identifier, x, y = map(int, fields[offset : offset + 3])
                            if identifier < 1 or not (
                                0 <= x <= 1000 and 0 <= y <= 1000
                            ):
                                status = "invalid_point_tag"
                                break
                            points.append(
                                [
                                    min(width - 1, int(x * width / 1000)),
                                    min(height - 1, int(y * height / 1000)),
                                ]
                            )
                if truncated:
                    status = "truncated"
                result.update(
                    object=objects[index],
                    points_xy=points if status == "points" else [],
                    pointing_status=status,
                )
            outputs.append(result)
            del inputs, generated, tokens
        result = {
            "model": arguments.model,
            "revision": revision,
            "image_size": [width, height],
            "seconds": {
                "load_model": load_seconds,
                "read_input": loaded - started,
                "inference": time.perf_counter() - loaded,
            },
        }
        if operation == "point":
            result["objects"] = outputs
        else:
            result.update(outputs[0])
        load_seconds = 0.0
        return result

    if arguments.socket is not None:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(arguments.socket))
            arguments.socket.chmod(0o600)
            listener.listen()
            print(json.dumps({"ready": True}), file=protocol, flush=True)
            while True:
                connection, _ = listener.accept()
                try:
                    with (
                        connection,
                        connection.makefile("rw", encoding="utf-8") as stream,
                    ):
                        line = stream.readline()
                        if not line:
                            continue
                        try:
                            result = infer(json.loads(line))
                        except Exception as error:  # noqa: BLE001 - report failed RPC
                            result = {"error": f"{type(error).__name__}: {error}"}
                        stream.write(json.dumps(result) + "\n")
                        stream.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
    else:
        result = infer(
            {
                "operation": "point" if arguments.objects else "query",
                "rgb": str(arguments.rgb.expanduser().absolute()),
                "prompt": arguments.prompt,
                "objects": arguments.objects,
                "max_new_tokens": arguments.max_new_tokens,
            }
        )
        print(json.dumps(result), file=protocol, flush=True)
    return 0
