"""Tests for the ROS-independent ground segmentation geometry."""

import numpy as np
from std_msgs.msg import Header

from panorama_stitcher_py.ground_segmentation import (
    GroundSegmentationConfig,
    classify_points,
    fit_ground_plane,
    voxel_first_indices,
)
from panorama_stitcher_py.ground_segmentation_node import (
    arrays_to_cloud,
    cloud_to_arrays,
)


def _config() -> GroundSegmentationConfig:
    return GroundSegmentationConfig(
        expected_up=np.array([0.0, -1.0, 0.0]),
        ransac_iterations=80,
        min_ground_inliers=40,
        min_ground_inlier_ratio=0.2,
        min_plane_distance_from_origin_m=0.8,
        max_plane_distance_from_origin_m=1.2,
    ).normalized()


def test_ransac_finds_optical_frame_ground_and_obstacles() -> None:
    rng = np.random.default_rng(7)
    x = rng.uniform(-2.0, 2.0, 500)
    z = rng.uniform(0.5, 5.0, 500)
    ground = np.column_stack((
        x,
        1.0 + rng.normal(0.0, 0.008, 500),
        z,
    ))
    obstacles = np.array([
        [-0.5, 0.45, 2.0],
        [0.0, 0.30, 2.5],
        [0.5, 0.15, 3.0],
    ])
    points = np.vstack((ground, obstacles))

    plane = fit_ground_plane(points, _config(), np.random.default_rng(42))

    assert plane is not None
    assert np.allclose(plane.normal, [0.0, -1.0, 0.0], atol=0.03)
    assert abs(plane.offset - 1.0) < 0.03
    obstacle_mask, ground_mask = classify_points(points, plane, _config())
    assert obstacle_mask[-3:].all()
    assert ground_mask[:500].mean() > 0.98


def test_invalid_vertical_plane_is_rejected() -> None:
    y = np.linspace(0.2, 1.8, 200)
    z = np.linspace(0.5, 5.0, 200)
    points = np.column_stack((np.ones(200), y, z))
    assert fit_ground_plane(
        points, _config(), np.random.default_rng(42)) is None


def test_voxel_filter_preserves_first_source_point() -> None:
    points = np.array([
        [0.01, 0.01, 0.01],
        [0.04, 0.04, 0.04],
        [0.11, 0.01, 0.01],
    ])
    assert voxel_first_indices(points, 0.1).tolist() == [0, 2]


def test_pointcloud2_xyz_rgb_round_trip() -> None:
    xyz = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ], dtype=np.float32)
    rgb = np.array([0x00112233, 0x00AABBCC], dtype=np.uint32)

    message = arrays_to_cloud(Header(frame_id='camera'), xyz, rgb)
    actual_xyz, actual_rgb = cloud_to_arrays(message)

    assert message.header.frame_id == 'camera'
    assert np.array_equal(actual_xyz, xyz)
    assert np.array_equal(actual_rgb, rgb)
