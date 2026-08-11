import math

import numpy as np

from ugv_self_supervised_traversability.utils.geometry import footprint_corners, polygon_area


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
