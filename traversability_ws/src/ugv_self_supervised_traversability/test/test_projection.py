import numpy as np

from ugv_self_supervised_traversability.utils.camera_projection import (
    depth_visibility_mask,
    project_points,
)
from ugv_self_supervised_traversability.utils.transforms import (
    transform_matrix,
    transform_points,
)


K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]])


def test_world_to_camera_coordinate_transform() -> None:
    world_t_camera = transform_matrix([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0])
    world_point = np.array([[1.0, 2.0, 8.0]])
    camera_point = transform_points(world_point, np.linalg.inv(world_t_camera))
    assert np.allclose(camera_point, [[0.0, 0.0, 5.0]])


def test_pinhole_projection() -> None:
    pixels, depth, valid = project_points(np.array([[1.0, 2.0, 10.0]]), K)
    assert np.allclose(pixels, [[60.0, 60.0]])
    assert np.allclose(depth, [10.0])
    assert valid.tolist() == [True]


def test_image_boundary_and_behind_camera_rejection() -> None:
    points = np.array([[0.0, 0.0, -1.0], [100.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
    _, _, valid = project_points(points, K, image_shape=(80, 100))
    assert valid.tolist() == [False, False, True]


def test_depth_occlusion_filter() -> None:
    projected = np.array([[2.0, 2.0], [2.0, np.nan]], dtype=np.float32)
    measured = np.array([[2.1, 3.0], [1.7, 2.0]], dtype=np.float32)
    visible = depth_visibility_mask(projected, measured, tolerance=0.2)
    assert visible.tolist() == [[True, False], [False, False]]
