"""SAM 3 runtime paths and shared command-line options; no model imports."""

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    python: Path | None = None
    repository: Path | None = None
    checkpoint: Path | None = None

    def command(self) -> list[str]:
        python = Path(self.python or sys.executable).expanduser().absolute()
        if not python.is_file():
            raise FileNotFoundError(python)
        # Keep the venv interpreter symlink: resolving it loses the venv packages.
        command = [
            str(python),
            str(Path(__file__).with_name("inference.py")),
        ]
        if self.repository is not None:
            repository = self.repository.expanduser().absolute()
            if not (repository / "sam3").is_dir():
                raise ValueError(f"not a SAM 3 source checkout: {repository}")
            command.extend(["--sam3-repo", str(repository)])
        if self.checkpoint is not None:
            checkpoint = self.checkpoint.expanduser().absolute()
            command.extend(["--checkpoint", str(checkpoint)])
        return command


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The same runtime configuration for every tool consumer."""
    for suffix, description in (
        ("python", "Python override (otherwise the current interpreter)"),
        ("repo", "source checkout override (otherwise installed SAM 3)"),
        ("checkpoint", "checkpoint override (otherwise local checkpoints/sam3.pt)"),
        ("socket", "connect to an existing SAM 3 worker instead of loading one"),
    ):
        variable = f"SAM3_{suffix.upper()}"
        parser.add_argument(
            f"--sam3-{suffix}",
            type=Path,
            default=os.environ.get(variable),
            help=f"{description} (default: ${variable})",
        )
