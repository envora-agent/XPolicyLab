"""Where the RoboDojo assets and task configs lie, for the offline `precompute.py`
scripts that write each task's `geometry.npz`: no agent reads them at run time."""

from pathlib import Path

ROBODOJO_ROOT = Path(__file__).resolve().parents[4] / "deps" / "RoboDojo"
TASK_CONFIG_DIR = ROBODOJO_ROOT / "task" / "RoboDojo" / "config"
ROBOT_DIR = ROBODOJO_ROOT / "Assets" / "Robots" / "x5"
