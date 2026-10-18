"""Envora_Agent: the unified camera-only agent of envora_sim as an XPolicyLab policy.

The agent (``vendor/envora_sim/environment/robodojo_physx/solutions``) reads the
task from the instruction and runs that task's scripted, camera-only agent over all
worlds of a batch. It sees what a RoboDojo policy sees: the three 480x640 RGB
cameras and the 14-D joint state. SAM 3 and Molmo2-4B run as worker processes on
the policy GPU. Top-level imports stay light: the environment client imports this
module too.
"""

import atexit
import os
import signal
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_POLICY_DIR = Path(__file__).resolve().parent
_IMPORTABLE_ROOT = _POLICY_DIR.parents[2]
_VENDOR_ROOT = _POLICY_DIR / "vendor"

if str(_IMPORTABLE_ROOT) not in sys.path:
    sys.path.insert(0, str(_IMPORTABLE_ROOT))

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)

CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
IMAGE_SHAPE = (480, 640, 3)


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        if model_cfg["action_type"] != "joint":
            raise ValueError(
                f"Envora_Agent supports action_type=joint only, got {model_cfg['action_type']!r}"
            )
        self.dim_info = get_robot_action_dim_info(model_cfg["env_cfg_type"])
        if list(self.dim_info["arm_dim"]) != [6, 6] or list(
            self.dim_info["ee_dim"]
        ) != [1, 1]:
            raise ValueError(
                f"Envora_Agent drives the dual 6-DoF X5 arms; {model_cfg['env_cfg_type']} "
                f"has {self.dim_info}"
            )

        # The vendored envora_sim, first on the path of this process and of the
        # tool workers it starts (Molmo2 runs as ``python -m envora_sim.tools...``).
        if str(_VENDOR_ROOT) not in sys.path:
            sys.path.insert(0, str(_VENDOR_ROOT))
        python_path = os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if str(_VENDOR_ROOT) not in python_path:
            os.environ["PYTHONPATH"] = os.pathsep.join(
                [str(_VENDOR_ROOT), *filter(None, python_path)]
            )
        from envora_sim.tools import molmo2_4b, sam3

        checkpoint = model_cfg.get("sam3_checkpoint")
        self.sam3 = sam3.Client(
            sam3.Config(checkpoint=None if checkpoint is None else Path(checkpoint))
        )
        self.molmo = molmo2_4b.Client(
            molmo2_4b.Config(revision=model_cfg.get("molmo_revision"))
        )
        atexit.register(self._close_tools)
        signal.signal(signal.SIGTERM, self._terminate)
        print("[Envora_Agent] starting SAM 3 and Molmo2-4B", flush=True)
        self.sam3.start()
        self.molmo.start()
        if model_cfg.get("warmup", True):
            self.sam3.warmup(["object"], width=IMAGE_SHAPE[1], height=IMAGE_SHAPE[0])
            self.molmo.warmup()

        workdir = model_cfg.get("workdir")
        self.workdir = Path(
            tempfile.mkdtemp(prefix="envora_agent_") if workdir is None else workdir
        )
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.agent = None
        self.agent_key = None
        self.observation = None
        self.instructions = None
        self.seen = None
        self.pending_reset = True
        self.crash = None
        print(
            f"[Envora_Agent] ready: task={model_cfg.get('task_name')}, workdir={self.workdir}",
            flush=True,
        )

    def _close_tools(self):
        self.sam3.close()
        self.molmo.close()

    def _terminate(self, signum, frame):
        # The tool workers run in their own sessions: stop them before exiting.
        self._close_tools()
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGTERM)

    def begin_batch(self, info):
        """The episode's worlds (sent by deploy.py after every reset): their count,
        the task config and the arms' mounts (``entity_origin_pose``, left then
        right). The agent reads them from this stand-in for the environment."""
        task = info.get("task_name") or self.model_cfg["task_name"]
        poses = info.get("robot_origin_poses") or self.model_cfg["robot_origin_poses"]
        num_envs = int(info["num_envs"])
        key = (num_envs, task, tuple(tuple(float(v) for v in pose) for pose in poses))
        if key != self.agent_key:
            from envora_sim.environment.robodojo_physx.solutions.unified.agent import (
                Agent,
            )

            env = SimpleNamespace(
                num_envs=num_envs,
                task=task,
                robots=[
                    SimpleNamespace(entity_origin_pose=list(pose)) for pose in key[2]
                ],
                sim_env=SimpleNamespace(
                    scene_manager=SimpleNamespace(layout_manager=None)
                ),
            )
            self.agent = Agent(env, self.sam3, self.molmo, self.workdir)
            self.agent_key = key
            print(f"[Envora_Agent] agent for {num_envs} worlds of {task}", flush=True)
        self.observation = {
            "observation.state": np.zeros((num_envs, 14), np.float32),
            **{
                f"observation.images.{camera}": np.zeros(
                    (num_envs, *IMAGE_SHAPE), np.uint8
                )
                for camera in CAMERAS
            },
        }
        self.instructions = [""] * num_envs
        self.seen = np.zeros(num_envs, bool)
        self.pending_reset = True

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        if self.agent is None:
            raise RuntimeError("begin_batch must follow reset (see deploy.py)")
        # Fresh arrays every step: agents keep earlier frames by reference.
        observation = {key: value.copy() for key, value in self.observation.items()}
        for obs in obs_list:
            world = int(obs["env_idx"])
            observation["observation.state"][world] = pack_robot_state(
                obs, "joint", self.dim_info
            )
            for camera in CAMERAS:
                observation[f"observation.images.{camera}"][world] = obs["vision"][
                    camera
                ]["color"]
            self.instructions[world] = obs["instruction"]
            self.seen[world] = True
        self.observation = observation
        if not self.pending_reset:
            return
        # The first observation of an episode: worlds that never run (no seed, or
        # an unstable layout) copy the first running one, as policy_eval pads.
        first = int(np.flatnonzero(self.seen)[0])
        for world in np.flatnonzero(~self.seen):
            for value in observation.values():
                value[world] = value[first]
            self.instructions[world] = self.instructions[first]
        self.pending_reset = False
        self.crash = None
        try:
            self.agent.reset(observation, list(self.instructions))
        except Exception:
            self._fail()

    def _fail(self):
        """An agent error fails the batch's episodes: the arms hold until they end."""
        self.crash = traceback.format_exc()
        log = self.workdir / "errors.log"
        with log.open("a") as file:
            file.write(self.crash + "\n")
        print(
            f"[Envora_Agent] agent error, the arms hold until the episodes end: "
            f"{self.crash.strip().splitlines()[-1]} (traceback in {log})",
            file=sys.stderr,
            flush=True,
        )

    def get_action(self):
        return self.get_action_batch([0])[0]

    def get_action_batch(self, env_idx_list=None):
        state = self.observation["observation.state"]
        if env_idx_list is None:
            env_idx_list = range(len(state))
        actions = state  # measured joints and commanded grippers: hold
        if self.crash is None:
            try:
                actions = np.asarray(self.agent.act(self.observation), np.float32)
                if actions.shape != state.shape:
                    raise ValueError(f"agent actions {actions.shape}, not {state.shape}")
            except Exception:
                self._fail()
                actions = state
        return [
            [unpack_robot_state(actions[int(world)].copy(), "joint", self.dim_info)]
            for world in env_idx_list
        ]

    def reset(self):
        self.pending_reset = True
