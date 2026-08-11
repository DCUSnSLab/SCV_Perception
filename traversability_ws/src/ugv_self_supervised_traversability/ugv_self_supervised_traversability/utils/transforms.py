"""Small transform helpers using [x, y, z, w] quaternion order."""

from typing import Sequence, Tuple

import numpy as np


def quaternion_to_rotation_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """Convert a quaternion to a 3x3 rotation matrix."""
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError('quaternion must have four elements')
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError('zero-length quaternion')
    x, y, z, w = q / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def transform_matrix(translation: Sequence[float], quaternion: Sequence[float]) -> np.ndarray:
    """Build a homogeneous target_T_source matrix."""
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_to_rotation_matrix(quaternion)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def transform_points(points: np.ndarray, target_t_source: np.ndarray) -> np.ndarray:
    """Transform an Nx3 point array using a 4x4 matrix."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points must have shape Nx3')
    homogeneous = np.column_stack((points, np.ones(points.shape[0])))
    return (target_t_source @ homogeneous.T).T[:, :3]


def euler_from_quaternion(quaternion: Sequence[float]) -> Tuple[float, float, float]:
    """Return roll, pitch, yaw from [x, y, z, w]."""
    x, y, z, w = np.asarray(quaternion, dtype=np.float64)
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)
    sinp = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(sinp)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny, cosy)
    return float(roll), float(pitch), float(yaw)
