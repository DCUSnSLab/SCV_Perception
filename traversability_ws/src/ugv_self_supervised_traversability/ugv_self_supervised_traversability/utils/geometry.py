"""Robot footprint geometry independent of ROS message types."""

from typing import Sequence

import numpy as np


def footprint_corners(
    x: float,
    y: float,
    z: float,
    yaw: float,
    robot_length: float,
    robot_width: float,
    margin: float = 0.0,
) -> np.ndarray:
    """Return four CCW world-frame corners of a rectangular UGV footprint."""
    if robot_length <= 0.0 or robot_width <= 0.0 or margin < 0.0:
        raise ValueError('robot dimensions must be positive and margin non-negative')
    half_l = robot_length * 0.5 + margin
    half_w = robot_width * 0.5 + margin
    local = np.array(
        [[half_l, half_w], [-half_l, half_w], [-half_l, -half_w],
         [half_l, -half_w]], dtype=np.float64)
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    xy = local @ rotation.T + np.array([x, y])
    return np.column_stack((xy, np.full(4, z, dtype=np.float64)))


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    """Compute the unsigned XY shoelace area of a polygon."""
    array = np.asarray(points, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 3 or array.shape[1] < 2:
        raise ValueError('at least three 2-D points are required')
    x, y = array[:, 0], array[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) * 0.5)
