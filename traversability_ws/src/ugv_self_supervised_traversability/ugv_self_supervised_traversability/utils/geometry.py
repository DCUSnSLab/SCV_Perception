"""Robot footprint geometry independent of ROS message types."""

from typing import Sequence

import numpy as np

from .transforms import rotation_matrix_from_euler


def footprint_corners(
    x: float,
    y: float,
    z: float,
    yaw: float,
    robot_length: float,
    robot_width: float,
    margin: float = 0.0,
    roll: float = 0.0,
    pitch: float = 0.0,
) -> np.ndarray:
    """Return four CCW world-frame corners of a rectangular UGV footprint."""
    if robot_length <= 0.0 or robot_width <= 0.0 or margin < 0.0:
        raise ValueError('robot dimensions must be positive and margin non-negative')
    half_l = robot_length * 0.5 + margin
    half_w = robot_width * 0.5 + margin
    local = np.array(
        [[half_l, half_w, 0.0], [-half_l, half_w, 0.0], [-half_l, -half_w, 0.0],
         [half_l, -half_w, 0.0]], dtype=np.float64)
    rotation = rotation_matrix_from_euler(roll, pitch, yaw)
    return local @ rotation.T + np.array([x, y, z], dtype=np.float64)


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    """Compute the unsigned XY shoelace area of a polygon."""
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 3 or array.shape[1] < 2:
        raise ValueError('at least three 2-D points are required')
    x, y = array[:, 0], array[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) * 0.5)


def estimate_ground_height(
    points: np.ndarray,
    x: float,
    y: float,
    radius: float,
    min_points: int,
    percentile: float,
) -> float | None:
    """Estimate local ground height from nearby LiDAR points."""
    if radius <= 0.0:
        raise ValueError('radius must be positive')
    if min_points <= 0:
        raise ValueError('min_points must be positive')
    if percentile < 0.0 or percentile > 100.0:
        raise ValueError('percentile must be between 0 and 100')
    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        raise ValueError('points must have shape Nx3')
    finite = cloud[np.isfinite(cloud).all(axis=1)]
    if finite.size == 0:
        return None
    distances = np.hypot(finite[:, 0] - x, finite[:, 1] - y)
    local = finite[distances <= radius]
    if local.shape[0] < min_points:
        return None
    return float(np.percentile(local[:, 2], percentile))
