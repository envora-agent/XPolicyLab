"""Forward and inverse kinematics of RoboDojo's ARX X5 arm (its URDF's arm joints as
constants)."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

ARM_DOF = 6
# The arm joints of Assets/Robots/x5/X5A.urdf: name, origin xyz (m) and rpy (rad), axis.
ARM_JOINTS = (
    ("joint1", [0.0, 0.0, 0.0605], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    ("joint2", [0.02, 0.0, 0.04], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    ("joint3", [-0.264, 0.0, 0.0], [3.1416, 0.0, 0.0], [0.0, 1.0, 0.0]),
    ("joint4", [0.245, 0.0, -0.056], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    ("joint5", [0.06775, 0.0005, -0.0865], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    ("joint6", [0.02895, 0.0, 0.0865], [-3.1416, 0.0, 0.0], [1.0, 0.0, 0.0]),
)
# link6 to the centre of the finger pads, along link6 +x (``gripper_bias`` in
# the embodiment configuration); the pads end 12.7 mm further out.
TCP_OFFSET = 0.145
FINGERTIP_OFFSET = 0.1577


@dataclass(frozen=True)
class ArmChain:
    origins: tuple  # 4x4 parent-to-joint transforms
    axes: tuple
    base: np.ndarray  # world (environment) pose of base_link


def pose_matrix(position, quaternion_wxyz):
    matrix = np.eye(4)
    w, x, y, z = quaternion_wxyz
    matrix[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
    matrix[:3, 3] = position
    return matrix


def load_chain(base_pose):
    """The X5 arm mounted at ``base_pose`` ``[x, y, z, qw, qx, qy, qz]``."""
    origins, axes = [], []
    for _, xyz, rpy, axis in ARM_JOINTS:
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
        transform[:3, 3] = xyz
        origins.append(transform)
        axes.append(np.array(axis))
    return ArmChain(
        tuple(origins), tuple(axes), pose_matrix(base_pose[:3], base_pose[3:])
    )


def axis_rotation(axis, angle):
    """Rotation matrix of ``angle`` about the unit vector ``axis`` (Rodrigues)."""
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * skew @ skew


def rotation_vector(matrix):
    """Axis times angle of a rotation matrix."""
    angle = np.arccos(np.clip((np.trace(matrix) - 1.0) / 2.0, -1.0, 1.0))
    vector = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ]
    )
    if angle < 1e-6:
        return vector / 2.0
    if np.pi - angle < 1e-3:
        return Rotation.from_matrix(matrix).as_rotvec()
    return angle / (2.0 * np.sin(angle)) * vector


def forward(chain, joints):
    """World pose of link6 and the world frames of all six joints."""
    transform = chain.base.copy()
    frames = []
    for origin, axis, angle in zip(chain.origins, chain.axes, joints):
        transform = transform @ origin
        frames.append(transform.copy())
        rotation = np.eye(4)
        rotation[:3, :3] = axis_rotation(axis, angle)
        transform = transform @ rotation
    return transform, frames


# Capsule radii of the links between consecutive joint frames (from the link
# meshes): upper arm, forearm, wrist, and wrist to link6.
LINK_RADII = (0.04, 0.045, 0.04, 0.035)


def link_spheres(chain, joints):
    """Spheres covering the links from the shoulder to link6: (centres, radii)."""
    link6, frames = forward(chain, joints)
    points = [frame[:3, 3] for frame in frames[1:]] + [link6[:3, 3]]
    centres, radii = [], []
    for start, end, radius in zip(points[:-1], points[1:], LINK_RADII):
        for fraction in np.linspace(0.0, 1.0, 8):
            centres.append(start + (end - start) * fraction)
            radii.append(radius)
    return np.array(centres), np.array(radii)


def clearance(chain, joints, boxes):
    """Smallest distance between the arm's links and axis-aligned ``(low, high)`` boxes."""
    centres, radii = link_spheres(chain, joints)
    nearest = np.inf
    for low, high in boxes:
        outside = np.maximum(np.maximum(low - centres, centres - high), 0.0)
        nearest = min(nearest, float(np.min(np.linalg.norm(outside, axis=1) - radii)))
    return nearest


def top_down_rotation(yaw):
    """link6 orientation with the fingers pointing down and closing along ``yaw``."""
    approach = np.array([0.0, 0.0, -1.0])
    closing = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    return np.column_stack([approach, closing, np.cross(approach, closing)])


def link6_for_tcp(tcp_position, rotation):
    return np.asarray(tcp_position) - TCP_OFFSET * rotation[:, 0]


@dataclass(frozen=True)
class SeedGrid:
    """Top-down IK solutions on one branch over a grid of link6 positions."""

    points: np.ndarray  # (n, 3)
    joints: np.ndarray  # (n, 6), solved for closing direction ``yaw``
    yaw: float

    def seed(self, position, yaw):
        index = int(np.argmin(np.linalg.norm(self.points - position, axis=1)))
        joints = self.joints[index].copy()
        joints[5] = np.angle(np.exp(1j * (joints[5] + yaw - self.yaw)))
        return joints


def seed_grid(chain, xs, ys, zs, start, yaw=np.pi / 2):
    """Solve a grid by continuation from ``start``, so every entry shares its branch."""
    rotation = top_down_rotation(yaw)
    grid = {
        (i, j, k): np.array([x, y, z])
        for i, x in enumerate(xs)
        for j, y in enumerate(ys)
        for k, z in enumerate(zs)
    }
    first = min(grid, key=lambda key: np.linalg.norm(grid[key] - start[0]))
    solved = {}
    frontier = [(first, start[1])]
    while frontier:
        key, seed = frontier.pop(0)
        if key in solved:
            continue
        solution = solve(chain, grid[key], rotation, seed, iterations=60)
        if solution is None or np.abs(solution - seed).max() > 2.0:
            continue
        solved[key] = solution
        i, j, k = key
        for neighbour in (
            (i + 1, j, k),
            (i - 1, j, k),
            (i, j + 1, k),
            (i, j - 1, k),
            (i, j, k + 1),
            (i, j, k - 1),
        ):
            if neighbour in grid and neighbour not in solved:
                frontier.append((neighbour, solution))
    keys = sorted(solved)
    return SeedGrid(
        np.array([grid[key] for key in keys]),
        np.array([solved[key] for key in keys]),
        yaw,
    )


def solve(chain, position, rotation, seed, iterations=200, tolerance=1e-5):
    """Damped least-squares IK for the link6 pose, starting from ``seed``.

    Returns the joint solution nearest the seed's branch, or ``None`` when the
    pose is unreachable within 0.1 mm and 0.1 degree.
    """
    joints = np.array(seed, dtype=float)
    for iteration in range(iterations):
        pose, frames = forward(chain, joints)
        position_error = position - pose[:3, 3]
        rotation_error = rotation_vector(rotation @ pose[:3, :3].T)
        error = np.concatenate([position_error, rotation_error])
        if (
            np.linalg.norm(position_error) < tolerance
            and np.linalg.norm(rotation_error) < tolerance * 10
        ):
            return joints
        if iteration == 50 and np.linalg.norm(error) > 0.02:
            return None  # still centimetres off: out of reach
        axes = np.array(
            [frame[:3, :3] @ axis for frame, axis in zip(frames, chain.axes)]
        )
        origins = np.array([frame[:3, 3] for frame in frames])
        jacobian = np.zeros((6, 6))
        jacobian[:3] = np.cross(axes, pose[:3, 3] - origins).T
        jacobian[3:] = axes.T
        damping = 1e-4
        step = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + damping * np.eye(6), error
        )
        joints += np.clip(step, -0.3, 0.3)
    pose, _ = forward(chain, joints)
    if np.linalg.norm(position - pose[:3, 3]) < 1e-4 and np.linalg.norm(
        rotation_vector(rotation @ pose[:3, :3].T)
    ) < np.radians(0.1):
        return joints
    return None
