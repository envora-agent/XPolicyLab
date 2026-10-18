"""The base of camera-only agents: a task's scripted expert that sees only the
observation RoboDojo's evaluation protocol gives a policy."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from envora_sim.environment.robodojo_physx.solutions import cameras
from envora_sim.environment.robodojo_physx.solutions import x5_kinematics as kinematics
from envora_sim.environment.robodojo_physx.solutions.x5_kinematics import ARM_DOF


class CameraOnlyAgent:
    """Mixin that turns a task's scripted expert into a camera-only agent.

    ``class Agent(CameraOnlyAgent, TaskExpert)`` keeps the expert's motion, and
    overrides every simulator read the expert makes with perception of
    ``self.observation``: the three RGB cameras and ``observation.state``. Once
    built, the agent keeps only the environment's number of worlds, so a read
    left over fails instead of peeking at the simulator.

    ``sam3`` and ``molmo`` are started SAM 3 and Molmo2 clients; ``workdir``
    holds the images sent to them.
    """

    def __init__(self, env, sam3, molmo, workdir):
        super().__init__(env)  # the expert reads the arms' fixed mounts here
        self.env = SimpleNamespace(num_envs=env.num_envs)
        self.sam3, self.molmo = sam3, molmo
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.observation = None

    def act(self, observation):
        self.observation = observation
        return super().act()

    def _measured_joints(self, env_idx):
        state = self.observation["observation.state"][env_idx]
        return [
            np.asarray(
                state[arm * (ARM_DOF + 1) : arm * (ARM_DOF + 1) + ARM_DOF], dtype=float
            )
            for arm in range(2)
        ]

    def _camera(self, env_idx, camera):
        """(focal length, world pose) of ``camera``, the wrist ones from the measured joints."""
        if camera == "cam_head":
            return cameras.HEAD_FOCAL, cameras.HEAD_POSE
        arm = cameras.WRIST_CAMERAS.index(camera)
        link6, _ = kinematics.forward(
            self.chains[arm], self._measured_joints(env_idx)[arm]
        )
        return cameras.WRIST_FOCAL, link6 @ cameras.WRIST_IN_LINK6

    def _image(self, env_idx, camera):
        path = self.workdir / f"{env_idx}_{camera}.png"
        image = self.observation[f"observation.images.{camera}"][env_idx]
        Image.fromarray(image).save(path)
        return path

    def _segment(self, env_idx, camera, prompts):
        """SAM 3 on ``camera``'s current image: per prompt, its (mask, score) pairs."""
        output = self.workdir / f"{env_idx}_{camera}"
        result = self.sam3.segment(self._image(env_idx, camera), output, list(prompts))
        return [
            list(zip(np.load(output / f"{index}.npy"), prompt["scores"]))
            for index, prompt in enumerate(result["prompts"])
        ]

    def _ask(self, env_idx, camera, question):
        """Molmo2's yes or no to ``question`` about ``camera``'s current image."""
        answer = self.molmo.query(
            self._image(env_idx, camera),
            f"{question} Answer yes or no.",
            max_new_tokens=8,
        )
        return answer["text"].strip().lower().startswith("yes")

    def _point(self, env_idx, camera, descriptions):
        """Molmo2's pixel points, an (N, 2) array per description, in ``camera``'s current image."""
        result = self.molmo.point(self._image(env_idx, camera), list(descriptions))
        return [
            np.asarray(found["points_xy"]).reshape(-1, 2) for found in result["objects"]
        ]
