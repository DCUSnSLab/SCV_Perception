import math

import numpy as np

from ugv_self_supervised_traversability.utils.geometry import footprint_corners, polygon_area
from ugv_self_supervised_traversability.utils.transforms import (
    euler_from_quaternion,
    quaternion_from_euler,
)


def test_footprint_geometry() -> None:
    corners = footprint_corners(2.0, 3.0, 0.2, 0.0, 2.0, 1.0, 0.1)
    assert corners.shape == (4, 3)
    assert np.allclose(corners.mean(axis=0), [2.0, 3.0, 0.2])
    assert math.isclose(polygon_area(corners), 2.2 * 1.2, rel_tol=1e-9)


def test_footprint_rotation_by_yaw() -> None:
    unrotated = footprint_corners(0.0, 0.0, 0.0, 0.0, 2.0, 1.0)
    rotated = footprint_corners(0.0, 0.0, 0.0, math.pi / 2.0, 2.0, 1.0)
    expected_xy = np.column_stack((-unrotated[:, 1], unrotated[:, 0]))
    assert np.allclose(rotated[:, :2], expected_xy, atol=1e-9)


def test_footprint_roll_and_pitch_affect_z() -> None:
    corners = footprint_corners(0.0, 0.0, 0.0, 0.0, 2.0, 1.0, roll=0.2, pitch=-0.1)
    assert not np.allclose(corners[:, 2], 0.0)
    assert math.isclose(float(corners[:, 2].mean()), 0.0, abs_tol=1e-9)


def test_quaternion_round_trip() -> None:
    quaternion = quaternion_from_euler(0.2, -0.1, 0.5)
    roll, pitch, yaw = euler_from_quaternion(quaternion)
    assert np.allclose([roll, pitch, yaw], [0.2, -0.1, 0.5], atol=1e-9)
