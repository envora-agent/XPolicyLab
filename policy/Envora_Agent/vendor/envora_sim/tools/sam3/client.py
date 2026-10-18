"""Lazy SAM 3 requests; lifecycle and transport are shared by model tools."""

from __future__ import annotations

import argparse
from pathlib import Path

from envora_sim.tools.process import ProcessClient
from envora_sim.tools.sam3.config import Config


class Client(ProcessClient):
    def __init__(
        self,
        config: Config | None = None,
        socket_path: Path | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.config = config or Config()
        super().__init__("SAM 3", self.config.command, socket_path, timeout)

    def start(self) -> Client:
        # First startup may download 3.45 GB; keep inference request timeouts unchanged.
        with self._lock:
            timeout = self.timeout
            self.timeout = max(timeout, 1800.0)
            try:
                return super().start()
            finally:
                self.timeout = timeout

    @classmethod
    def from_arguments(cls, arguments: argparse.Namespace) -> Client:
        return cls(
            Config(
                arguments.sam3_python, arguments.sam3_repo, arguments.sam3_checkpoint
            ),
            arguments.sam3_socket,
        )

    def segment(self, rgb: Path, output: Path, prompts: list[str]) -> dict:
        return self._call(
            {
                "operation": "segment",
                "rgb": str(rgb.expanduser().absolute()),
                "output": str(output.expanduser().absolute()),
                "prompts": prompts,
            }
        )

    def warmup(self, prompts: list[str], width: int = 1280, height: int = 720) -> dict:
        """One explicit blank-image pass inside the worker; no image file transfer."""
        return self._call(
            {
                "operation": "warmup",
                "width": width,
                "height": height,
                "prompts": prompts,
            }
        )
