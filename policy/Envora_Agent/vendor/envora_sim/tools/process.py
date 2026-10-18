"""Standard-library IPC and lifecycle for resident model workers."""

from __future__ import annotations

import json
import select
import socket
import subprocess
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Self


class ProcessClient:
    """Own one worker, or borrow one at ``socket_path``. Construction is CPU-only.

    Start explicitly or on the first request. Calls are serialized; an idle client
    has no background thread, polling, inference, or image transfers.
    """

    def __init__(
        self,
        name: str,
        command: Callable[[], list[str]],
        socket_path: Path | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._name = name
        self._command = command
        self._socket_path = (
            socket_path.expanduser().absolute() if socket_path is not None else None
        )
        self.timeout = timeout
        self._process: subprocess.Popen | None = None
        self._directory: tempfile.TemporaryDirectory | None = None
        self._lock = threading.RLock()

    @property
    def socket_path(self) -> Path:
        self.start()
        return self._socket_path

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    def start(self) -> Self:
        with self._lock:
            if self._process is not None:
                if self._process.poll() is not None:
                    raise RuntimeError(
                        f"the {self._name} worker exited; close and restart it"
                    )
                return self
            if self._socket_path is not None:
                return self
            command = self._command()
            self._directory = tempfile.TemporaryDirectory(prefix="envora-tool-")
            self._socket_path = Path(self._directory.name) / "worker.sock"
            try:
                self._process = subprocess.Popen(
                    [*command, "--socket", str(self._socket_path)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
                # stdout is reserved for the ready message; model logs use stderr.
                ready, _, _ = select.select(
                    [self._process.stdout], [], [], self.timeout
                )
                if not ready:
                    raise TimeoutError(f"{self._name} worker startup timed out")
                line = self._process.stdout.readline()
                if not line or not json.loads(line).get("ready"):
                    raise RuntimeError(
                        f"{self._name} worker failed to start; see its stderr"
                    )
            except BaseException:
                self.close()
                raise
            return self

    def _call(self, request: dict) -> dict:
        """One JSON request and response; paths refer to the local filesystem."""
        with self._lock:
            self.start()
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(self.timeout)
                connection.connect(str(self._socket_path))
                with connection.makefile("rw", encoding="utf-8") as stream:
                    stream.write(json.dumps(request) + "\n")
                    stream.flush()
                    line = stream.readline()
            if not line:
                raise RuntimeError(
                    f"{self._name} worker closed the connection without a result"
                )
            response = json.loads(line)
            if "error" in response:
                raise RuntimeError(f"{self._name}: {response['error']}")
            return response

    def close(self) -> None:
        """Stop only a worker owned by this client; borrowed workers keep running."""
        with self._lock:
            if self._process is not None:
                try:
                    self._process.terminate()
                    try:
                        self._process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        self._process.kill()
                        self._process.wait()
                finally:
                    self._process.stdout.close()
                    self._process = None
            if self._directory is not None:
                self._directory.cleanup()
                self._directory = None
                self._socket_path = None

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, exception_type, exception, traceback) -> None:
        self.close()
