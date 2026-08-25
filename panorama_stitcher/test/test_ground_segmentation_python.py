"""Tests for the ROS-independent ground segmentation geometry."""

from dataclasses import replace

import numpy as np
from std_msgs.msg import Header

from panorama_stitcher_py.ground_segmentation import (
    GroundSegmentationConfig,
    PlaneModel,
    classify_points,
    fit_ground_plane,
    fit_ground_planes,
    radius_outlier_indices,
    range_residual_summary,
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
        max_ransac_points=30000,
        ransac_distance_threshold_m=0.05,
        max_ground_tilt_deg=25.0,
        min_ground_inliers=40,
        min_ground_inlier_ratio=0.2,
        secondary_ransac_iterations=60,
        secondary_min_ground_inlier_ratio=0.20,
        secondary_min_normal_delta_deg=4.0,
        secondary_max_plane_distance_from_origin_m=2.5,
        ground_candidate_min_range_m=0.4,
        ground_candidate_max_range_m=8.0,
        ground_candidate_min_down_m=0.15,
        ground_candidate_max_down_m=2.5,
        min_plane_distance_from_origin_m=0.8,
        max_plane_distance_from_origin_m=1.2,
        obstacle_min_height_m=0.10,
        obstacle_max_height_m=2.0,
        obstacle_min_range_m=0.25,
        obstacle_max_range_m=8.0,
        obstacle_radius_filter_radius_m=0.0,
        obstacle_radius_filter_min_neighbors=0,
        obstacle_voxel_size_m=0.0,
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


def test_second_plane_removes_ramp_without_removing_obstacle() -> None:
    rng = np.random.default_rng(12)
    flat_x = rng.uniform(-2.0, 2.0, 700)
    flat_z = rng.uniform(0.5, 4.2, 700)
    flat = np.column_stack((
        flat_x,
        1.0 + rng.normal(0.0, 0.006, len(flat_x)),
        flat_z,
    ))

    ramp_x = rng.uniform(-2.0, 2.0, 550)
    ramp_z = rng.uniform(4.0, 7.0, 550)
    ramp_y = 1.0 - np.tan(np.deg2rad(9.0)) * (ramp_z - 4.0)
    ramp = np.column_stack((
        ramp_x,
        ramp_y + rng.normal(0.0, 0.006, len(ramp_x)),
        ramp_z,
    ))
    obstacle_z = rng.uniform(5.0, 5.5, 80)
    obstacle = np.column_stack((
        rng.uniform(-0.25, 0.25, len(obstacle_z)),
        1.0 - np.tan(np.deg2rad(9.0)) * (obstacle_z - 4.0) - 0.40,
        obstacle_z,
    ))
    points = np.vstack((flat, ramp, obstacle))
    config = replace(
        _config(),
        max_plane_distance_from_origin_m=1.4,
        secondary_max_plane_distance_from_origin_m=1.8,
        secondary_ransac_iterations=80,
        secondary_min_ground_inlier_ratio=0.20,
    ).normalized()

    planes = fit_ground_planes(points, config, np.random.default_rng(42))

    assert len(planes) == 2
    obstacle_mask, ground_mask = classify_points(points, planes, config)
    assert ground_mask[:len(flat)].mean() > 0.95
    assert ground_mask[len(flat):len(flat) + len(ramp)].mean() > 0.95
    assert obstacle_mask[len(flat):len(flat) + len(ramp)].mean() < 0.05
    assert obstacle_mask[-len(obstacle):].mean() > 0.95

    legacy_planes = fit_ground_planes(
        points,
        replace(config, secondary_ransac_iterations=0),
        np.random.default_rng(42),
    )
    assert len(legacy_planes) == 1


def test_parallel_residual_surface_stays_obstacle() -> None:
    rng = np.random.default_rng(19)
    ground = np.column_stack((
        rng.uniform(-2.0, 2.0, 700),
        1.0 + rng.normal(0.0, 0.006, 700),
        rng.uniform(0.5, 6.0, 700),
    ))
    platform = np.column_stack((
        rng.uniform(-1.0, 1.0, 400),
        0.82 + rng.normal(0.0, 0.004, 400),
        rng.uniform(2.0, 5.0, 400),
    ))
    points = np.vstack((ground, platform))
    config = _config()

    planes = fit_ground_planes(points, config, np.random.default_rng(42))

    assert len(planes) == 1
    obstacle_mask, _ = classify_points(points, planes, config)
    assert obstacle_mask[-len(platform):].mean() > 0.95


def test_voxel_filter_preserves_first_source_point() -> None:
    points = np.array([
        [0.01, 0.01, 0.01],
        [0.04, 0.04, 0.04],
        [0.11, 0.01, 0.01],
    ])
    assert voxel_first_indices(points, 0.1).tolist() == [0, 2]


def test_radius_filter_removes_isolated_points() -> None:
    cluster = np.array([
        [0.00, 0.00, 2.00],
        [0.02, 0.00, 2.00],
        [-0.02, 0.00, 2.00],
        [0.00, 0.02, 2.00],
        [0.00, -0.02, 2.00],
        [0.00, 0.00, 2.02],
    ])
    noise = np.array([[1.0, 0.0, 2.0], [1.02, 0.0, 2.0]])
    points = np.vstack((cluster, noise))

    assert radius_outlier_indices(points, 0.15, 5).tolist() == list(range(6))


def test_range_residual_summary_reports_distance_bins() -> None:
    plane = PlaneModel(
        normal=np.array([0.0, -1.0, 0.0]),
        offset=1.0,
        inliers=4,
        squared_error=0.0,
    )
    points = np.array([
        [0.0, 1.01, 1.0],
        [0.0, 0.97, 1.5],
        [0.0, 1.10, 3.0],
        [0.0, 0.80, 3.5],
    ])

    summary = range_residual_summary(points, (plane,), 4.0)

    assert '0-2m:n=2,p50=2.0cm' in summary
    assert '2-4m:n=2,p50=15.0cm' in summary


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


def test_organized_cloud_stride_samples_rows_and_columns() -> None:
    xyz = np.column_stack((
        np.arange(16, dtype=np.float32),
        np.zeros(16, dtype=np.float32),
        np.ones(16, dtype=np.float32),
    ))
    rgb = np.arange(16, dtype=np.uint32)
    message = arrays_to_cloud(Header(frame_id='camera'), xyz, rgb)
    message.height = 4
    message.width = 4
    message.row_step = message.point_step * message.width

    actual_xyz, actual_rgb = cloud_to_arrays(message, input_stride=2)

    assert np.array_equal(actual_xyz[:, 0], [0.0, 2.0, 8.0, 10.0])
    assert np.array_equal(actual_rgb, [0, 2, 8, 10])
