"""Calibration and geometry of RoboDojo's ``arx_x5`` evaluation cell for camera-only agents.

The official observation has RGB images but no depth and no camera matrices.
These are the fixed facts a real cell measures once: the camera specs and
mounts, the table height and the camera stand. Masks are lifted onto known
planes and fitted with an object's CAD footprint.
"""

import cv2
import numpy as np

from envora_sim.environment.robodojo_physx.solutions.assets import ROBODOJO_ROOT

# Many small image operations per sighting; OpenCV's thread pool only contends
# with the simulator and the other worlds' processes.
cv2.setNumThreads(1)

TABLE_HEIGHT = 0.765
IMAGE_SIZE = (640, 480)
# env_cfg/camera/template.py: the head camera is a Gemini 345Lg, the wrist ones D435s.
HEAD_FOCAL = IMAGE_SIZE[0] * 10.0 / 22.212
WRIST_FOCAL = IMAGE_SIZE[0] * 13.0 / 20.955
# USD camera frames (x right, y up, looking along -z): the head camera of
# camera_config.yml in the world, the wrist camera of the X5 in link6.
HEAD_POSE = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, np.cos(np.pi / 6), -np.sin(np.pi / 6), -0.41],
        [0.0, np.sin(np.pi / 6), np.cos(np.pi / 6), 1.308],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
WRIST_IN_LINK6 = np.array(
    [
        [0.0, np.sin(np.pi / 6), -np.cos(np.pi / 6), 0.0848],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, np.cos(np.pi / 6), np.sin(np.pi / 6), 0.0509],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
WRIST_CAMERAS = ("cam_left_wrist", "cam_right_wrist")  # per arm
# The head-camera stand, fixed in every layout.
STAND_USD = (
    ROBODOJO_ROOT / "Assets/Object/RoboDojo/Geometry/camera_stand/00000/object.usd"
)
STAND_POSE = (
    np.array([0.0, -0.47, TABLE_HEIGHT]),
    np.array([-0.707, 0.7072135, 0.0, 0.0]),
)
# The grid footprints are drawn and fitted on: 0.5 mm cells, a 48 mm half-width.
GRID = 0.0005
GRID_HALF = 96


def ray_plane(focal, pose, pixels, height):
    """World points where the rays of ``pixels`` (N, 2) of a camera at ``pose`` meet z = height."""
    pixels = np.asarray(pixels, dtype=float)
    directions = (
        np.stack(
            [
                (pixels[:, 0] - IMAGE_SIZE[0] / 2) / focal,
                -(pixels[:, 1] - IMAGE_SIZE[1] / 2) / focal,
                -np.ones(len(pixels)),
            ],
            axis=1,
        )
        @ pose[:3, :3].T
    )
    scale = (height - pose[2, 3]) / directions[:, 2]
    return pose[:3, 3] + scale[:, None] * directions


def project(focal, pose, points):
    """Pixels of world ``points`` (N, 3) in a camera at ``pose``."""
    local = (np.asarray(points, dtype=float) - pose[:3, 3]) @ pose[:3, :3]
    return np.stack(
        [
            IMAGE_SIZE[0] / 2 + focal * local[:, 0] / -local[:, 2],
            IMAGE_SIZE[1] / 2 - focal * local[:, 1] / -local[:, 2],
        ],
        axis=1,
    )


def mesh_footprint(usd_path, prim="/World/collision"):
    """Top-view occupancy of an object's collision mesh (at ``prim``) on the ``GRID``,
    centred on its origin."""
    from pxr import Usd, UsdGeom  # Kit's USD, available once the app runs

    stage = Usd.Stage.Open(str(usd_path))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(prim))
    points = np.array(mesh.GetPointsAttr().Get())[:, :2]
    counts = np.array(mesh.GetFaceVertexCountsAttr().Get())
    indices = np.array(mesh.GetFaceVertexIndicesAttr().Get())
    image = np.zeros((2 * GRID_HALF + 1,) * 2, np.uint8)
    cells = np.round(points / GRID + GRID_HALF).astype(np.int32)
    start = 0
    for count in counts:
        cv2.fillConvexPoly(image, cells[indices[start : start + count]], 1)
        start += count
    return image


def plane_mask(focal, pose, mask, height, centre, cell, half):
    """``mask`` lifted onto z = height, drawn on a grid of ``cell`` metres and
    ``half`` cells either side of world xy ``centre``."""
    contours, hierarchy = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )
    image = np.zeros((2 * half + 1,) * 2, np.uint8)
    for index, contour in enumerate(contours):
        lifted = ray_plane(focal, pose, contour[:, 0] + 0.5, height)[:, :2]
        cells = np.round((lifted - centre) / cell + half).astype(np.int32)
        # Outer boundaries fill, holes (contours with a parent) clear.
        cv2.fillPoly(image, [cells], 0 if hierarchy[0][index][3] >= 0 else 1)
    return image


def fit_footprint(focal, pose, mask, footprint, period, height):
    """(x, y, yaw, IoU) of the object whose ``mask`` a camera at ``pose`` sees.

    The mask is lifted onto z = ``height``. The ``mesh_footprint`` is turned in
    1° steps over its rotational symmetry ``period`` (None: round), then shifted
    by up to 2 mm, for the best overlap.
    """
    rows, columns = np.nonzero(mask)
    pixels = np.stack([columns, rows], axis=1) + 0.5
    centre = ray_plane(focal, pose, pixels, height)[:, :2].mean(axis=0)
    observed = plane_mask(focal, pose, mask, height, centre, GRID, GRID_HALF)
    observed = observed.astype(bool)

    def turned(yaw):
        return cv2.warpAffine(
            footprint,
            cv2.getRotationMatrix2D((GRID_HALF, GRID_HALF), -np.degrees(yaw), 1.0),
            footprint.shape[::-1],
            flags=cv2.INTER_NEAREST,
        ).astype(bool)

    def overlap(candidate):
        return (candidate & observed).sum() / max((candidate | observed).sum(), 1)

    yaws = [0.0] if period is None else np.arange(0.0, period, np.radians(1.0))
    yaw = max(yaws, key=lambda yaw: overlap(turned(yaw)))
    best = turned(yaw)
    shifts = [(dx, dy) for dx in range(-4, 5) for dy in range(-4, 5)]
    dx, dy = max(
        shifts, key=lambda shift: overlap(np.roll(best, shift[::-1], axis=(0, 1)))
    )
    iou = overlap(np.roll(best, (dy, dx), axis=(0, 1)))
    return centre[0] + dx * GRID, centre[1] + dy * GRID, yaw, iou
