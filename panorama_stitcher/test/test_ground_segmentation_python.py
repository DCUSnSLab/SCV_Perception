"""Tests for the ROS-independent ground segmentation geometry."""

from dataclasses import replace

import numpy as np
from std_msgs.msg import Header

from panorama_stitcher_py.ground_segmentation import (
    GroundSegmentationConfig,
    PlaneModel,
    classify_points,
    expand_bev_keys,
    extrapolate_planar_odometry,
    fit_ground_plane,
    fit_ground_planes,
    grow_ground_region,
    horizontal_surface_filter_indices,
    obstacle_bev_keys,
    project_points_to_vehicle_bev,
    recover_ground_from_plane_history,
    radius_outlier_indices,
    range_adaptive_radius_outlier_indices,
    range_residual_summary,
    temporal_obstacle_persistence_indices,
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
        obstacle_far_radius_filter_start_m=5.0,
        obstacle_far_radius_filter_radius_m=0.0,
        obstacle_far_radius_filter_min_neighbors=0,
        obstacle_voxel_size_m=0.0,
        ground_region_filter_enabled=True,
        ground_region_grid_size_m=0.10,
        ground_region_max_step_m=0.01,
        ground_region_max_slope_deg=40.0,
        ground_region_max_plane_residual_m=0.30,
        ground_region_point_tolerance_m=0.06,
        ground_region_min_points=2,
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


def test_region_growing_removes_smooth_road_residual_but_keeps_obstacle() -> None:
    # Sample more densely than the 10 cm cells, as the live RGB-D cloud does.
    x_values = np.linspace(-1.0, 1.0, 41)
    z_values = np.linspace(0.6, 6.0, 109)
    x, z = np.meshgrid(x_values, z_values)
    # The far road rises smoothly by 20 cm. A single flat plane therefore
    # classifies the far part as an obstacle even though local slope is small.
    road_height = np.maximum(z - 2.0, 0.0) * 0.05
    road = np.column_stack((
        x.reshape(-1),
        (1.0 - road_height).reshape(-1),
        z.reshape(-1),
    ))
    plane = PlaneModel(
        normal=np.array([0.0, -1.0, 0.0]),
        offset=1.0,
        inliers=int(np.count_nonzero(z <= 2.0)),
        squared_error=0.0,
    )
    config = replace(
        _config(), secondary_ransac_iterations=0).normalized()
    obstacle_mask, ground_mask = classify_points(road, plane, config)
    assert obstacle_mask[z.reshape(-1) >= 4.5].mean() > 0.90

    grown = grow_ground_region(road, ground_mask, (plane,), config)

    assert grown.mean() > 0.95
    obstacle_mask &= ~grown
    assert not obstacle_mask.any()

    # A 20 cm vertical object on that road stays above the locally estimated
    # surface even when it shares horizontal cells with ground returns.
    obstacle = np.array([
        [0.0, 1.0 - 0.15 - height, 3.0]
        for height in (0.12, 0.20, 0.28, 0.36)
    ])
    combined = np.vstack((road, obstacle))
    obstacle_mask, ground_mask = classify_points(combined, plane, config)
    grown = grow_ground_region(combined, ground_mask, (plane,), config)
    obstacle_mask &= ~grown
    assert obstacle_mask[-len(obstacle):].all()


def test_region_growing_does_not_cross_curb_step() -> None:
    x_values = np.linspace(-0.8, 0.8, 17)
    near_z = np.linspace(0.6, 2.0, 15)
    far_z = np.linspace(2.2, 3.4, 13)
    near_x, near_range = np.meshgrid(x_values, near_z)
    far_x, far_range = np.meshgrid(x_values, far_z)
    road = np.column_stack((
        near_x.reshape(-1),
        np.ones(near_x.size),
        near_range.reshape(-1),
    ))
    curb_top = np.column_stack((
        far_x.reshape(-1),
        np.full(far_x.size, 0.88),
        far_range.reshape(-1),
    ))
    points = np.vstack((road, curb_top))
    plane = PlaneModel(
        normal=np.array([0.0, -1.0, 0.0]),
        offset=1.0,
        inliers=len(road),
        squared_error=0.0,
    )
    config = _config()
    obstacle_mask, ground_mask = classify_points(points, plane, config)

    grown = grow_ground_region(points, ground_mask, (plane,), config)
    obstacle_mask &= ~grown

    assert obstacle_mask[-len(curb_top):].mean() > 0.95


def test_temporal_ground_recovery_removes_wobble_without_losing_curb() -> None:
    current = PlaneModel(
        normal=np.array([0.0, -1.0, 0.0]),
        offset=1.0,
        inliers=500,
        squared_error=0.0,
    )
    previous = PlaneModel(
        normal=np.array([0.0, -1.0, 0.0]),
        offset=0.945,
        inliers=500,
        squared_error=0.0,
    )
    # The first return is 10 cm above the current, temporarily shifted plane,
    # but lies on the recent road. The second is a real 12 cm curb and remains
    # farther than the strict 5 cm historical-ground tolerance.
    points = np.array([
        [0.0, 0.896, 4.0],
        [0.4, 0.88, 4.0],
    ])
    obstacle_mask, ground_mask = classify_points(
        points, (current,), _config())
    assert obstacle_mask.all()

    recovered = recover_ground_from_plane_history(
        points,
        ground_mask,
        (current,),
        (previous,),
        distance_threshold_m=0.05,
        max_normal_delta_deg=2.0,
        max_offset_delta_m=0.06,
    )

    assert recovered.tolist() == [True, False]
    obstacle_mask &= ~recovered
    assert obstacle_mask.tolist() == [False, True]


def test_temporal_ground_recovery_rejects_incompatible_old_plane() -> None:
    current = PlaneModel(
        normal=np.array([0.0, -1.0, 0.0]), offset=1.0,
        inliers=500, squared_error=0.0)
    stale = PlaneModel(
        normal=np.array([0.0, -1.0, 0.0]), offset=0.90,
        inliers=500, squared_error=0.0)
    point = np.array([[0.0, 0.90, 4.0]])

    recovered = recover_ground_from_plane_history(
        point, np.array([False]), (current,), (stale,),
        distance_threshold_m=0.05,
        max_normal_delta_deg=2.0,
        max_offset_delta_m=0.06,
    )

    assert not recovered[0]


def test_temporal_obstacle_filter_compensates_vehicle_motion() -> None:
    history = []
    # One fixed world obstacle appears closer as the vehicle moves forward.
    for odom_x, sensor_z in ((0.0, 5.0), (0.1, 4.9)):
        point = np.array([[0.0, 0.7, sensor_z]], dtype=np.float32)
        selected, keys = temporal_obstacle_persistence_indices(
            point, odom_x, 0.0, 0.0, history,
            cell_size_m=0.10,
            min_previous_hits=2,
            near_bypass_range_m=3.0,
        )
        # The bounded history warms up without hiding new obstacles.
        assert selected.tolist() == [0]
        history.append(expand_bev_keys(keys, 1))

    points = np.array([
        [0.0, 0.7, 4.8],   # Same fixed obstacle at world x=5.0.
        [1.5, 0.9, 5.0],   # A new one-frame far road artefact.
        [0.2, 0.9, 2.0],   # New but immediately hazardous near return.
    ], dtype=np.float32)
    selected, _ = temporal_obstacle_persistence_indices(
        points, 0.2, 0.0, 0.0, history,
        cell_size_m=0.10,
        min_previous_hits=2,
        near_bypass_range_m=3.0,
    )

    assert selected.tolist() == [0, 2]


def test_planar_odometry_extrapolation_integrates_body_twist() -> None:
    x, y, yaw = extrapolate_planar_odometry(
        odom_x_m=1.0,
        odom_y_m=2.0,
        odom_yaw_rad=np.deg2rad(90.0),
        body_velocity_x_mps=2.0,
        body_velocity_y_mps=0.0,
        yaw_rate_radps=0.0,
        delta_time_sec=0.5,
    )
    assert np.allclose([x, y, yaw], [1.0, 3.0, np.deg2rad(90.0)])

    x, y, yaw = extrapolate_planar_odometry(
        odom_x_m=0.0,
        odom_y_m=0.0,
        odom_yaw_rad=0.0,
        body_velocity_x_mps=1.0,
        body_velocity_y_mps=0.0,
        yaw_rate_radps=1.0,
        delta_time_sec=0.5,
    )
    assert np.allclose([x, y, yaw], [np.sin(0.5), 1.0 - np.cos(0.5), 0.5])


def test_rear_four_frame_persistence_rejects_transient_far_noise() -> None:
    forward_axis, left_axis, _, origin = _rear_axes()
    fixed = _rear_sensor_point(-5.0 - origin[0], -origin[1])[None, :]
    history = []
    for _ in range(3):
        keys = obstacle_bev_keys(
            fixed, 0.0, 0.0, 0.0, 0.10,
            forward_axis, left_axis, origin)
        history.append(expand_bev_keys(keys, 1))
    points = np.vstack((
        fixed[0],
        _rear_sensor_point(-6.0 - origin[0], 1.5 - origin[1]),
        _rear_sensor_point(-2.0 - origin[0], 0.5 - origin[1]),
    ))

    selected, _ = temporal_obstacle_persistence_indices(
        points,
        0.0,
        0.0,
        0.0,
        history,
        cell_size_m=0.10,
        min_previous_hits=3,
        near_bypass_range_m=3.0,
        vehicle_forward_vector=forward_axis,
        vehicle_left_vector=left_axis,
        sensor_origin_vehicle_xy_m=origin,
    )

    assert selected.tolist() == [0, 2]


def test_temporal_obstacle_filter_compensates_vehicle_yaw() -> None:
    # Avoid exact cell boundaries so harmless float32 round-off cannot move
    # floor(5.0 / 0.1) to the adjacent cell.
    world_forward = 5.03
    world_left = -1.02
    world_point = np.array(
        [[-world_left, 0.8, world_forward]], dtype=np.float32)
    first_keys = obstacle_bev_keys(
        world_point, 0.0, 0.0, 0.0, cell_size_m=0.10)
    # Express the same point after the vehicle has yawed +10 degrees. Optical
    # x is right, hence it is the negative of vehicle-left.
    yaw = np.deg2rad(10.0)
    forward = (
        np.cos(yaw) * world_forward + np.sin(yaw) * world_left)
    left = (
        -np.sin(yaw) * world_forward + np.cos(yaw) * world_left)
    rotated_sensor_point = np.array(
        [[-left, 0.8, forward]], dtype=np.float32)
    rotated_keys = obstacle_bev_keys(
        rotated_sensor_point, 0.0, 0.0, yaw, cell_size_m=0.10)

    assert first_keys.tolist() == rotated_keys.tolist()


def _rear_axes():
    forward = np.array(
        [-0.000372073187, 0.501787490762, -0.864990852944])
    left = np.array(
        [0.999999831527, 0.000572091975, -0.000098272169])
    up = np.array(
        [0.000445542580, -0.864990743780, -0.501787619084])
    origin = np.array([-0.684779832417, -0.014809559712])
    return forward, left, up, origin


def _rear_sensor_point(forward_m, left_m, up_m=0.0):
    forward, left, up, _ = _rear_axes()
    return forward_m * forward + left_m * left + up_m * up


def test_rear_bev_projection_uses_calibrated_backward_axis() -> None:
    forward, left, _, _ = _rear_axes()
    points = np.vstack((
        _rear_sensor_point(-4.5, 1.2),
        _rear_sensor_point(-2.0, -0.7),
    ))

    projected = project_points_to_vehicle_bev(points, forward, left)

    assert np.allclose(projected, [[-4.5, 1.2], [-2.0, -0.7]])


def test_rear_temporal_filter_compensates_turn_and_camera_lever_arm() -> None:
    forward_axis, left_axis, _, origin = _rear_axes()
    world = np.array([-5.23, 1.27])

    def observed_point(odom_x, odom_y, yaw):
        delta = world - np.array([odom_x, odom_y])
        cosine = np.cos(yaw)
        sine = np.sin(yaw)
        vehicle_forward = cosine * delta[0] + sine * delta[1]
        vehicle_left = -sine * delta[0] + cosine * delta[1]
        return _rear_sensor_point(
            vehicle_forward - origin[0], vehicle_left - origin[1])

    first = observed_point(0.0, 0.0, 0.0)[None, :]
    first_keys = obstacle_bev_keys(
        first,
        0.0,
        0.0,
        0.0,
        0.10,
        forward_axis,
        left_axis,
        origin,
    )
    yaw = np.deg2rad(12.0)
    turned = observed_point(0.14, -0.03, yaw)[None, :]
    turned_keys = obstacle_bev_keys(
        turned,
        0.14,
        -0.03,
        yaw,
        0.10,
        forward_axis,
        left_axis,
        origin,
    )

    assert first_keys.tolist() == turned_keys.tolist()
    # Omitting the 68.5 cm rear-camera lever arm moves the history cell while
    # yawing, which was the previous rear-camera behavior.
    without_lever_first = obstacle_bev_keys(
        first,
        0.0,
        0.0,
        0.0,
        0.10,
        forward_axis,
        left_axis,
    )
    without_lever_turned = obstacle_bev_keys(
        turned,
        0.14,
        -0.03,
        yaw,
        0.10,
        forward_axis,
        left_axis,
    )
    assert without_lever_first.tolist() != without_lever_turned.tolist()


def _horizontal_filter(points: np.ndarray, source: np.ndarray) -> np.ndarray:
    return horizontal_surface_filter_indices(
        points,
        source,
        expected_up=np.array([0.0, -1.0, 0.0]),
        cell_size_m=0.10,
        min_component_points=8,
        min_up_alignment=0.98,
        max_plane_thickness_m=0.07,
        near_bypass_range_m=3.0,
        vertical_bin_size_m=0.02,
        vertical_min_points_per_bin=2,
        vertical_min_run_bins=5,
        vertical_min_occupied_bins=4,
        vertical_min_support_cells=2,
    )


def test_horizontal_filter_removes_far_paint_sheet() -> None:
    x, z = np.meshgrid(
        np.linspace(-0.35, 0.35, 15),
        np.linspace(5.5, 6.5, 21),
    )
    paint = np.column_stack((
        x.reshape(-1),
        np.full(x.size, 0.86),
        z.reshape(-1),
    ))
    road = np.column_stack((
        x.reshape(-1),
        np.ones(x.size),
        z.reshape(-1),
    ))

    selected = _horizontal_filter(paint, np.vstack((road, paint)))

    assert selected.size == 0


def test_horizontal_filter_keeps_curb_top_with_vertical_face() -> None:
    x_values = np.linspace(-0.45, 0.45, 19)
    z_values = np.linspace(4.0, 4.8, 17)
    x, z = np.meshgrid(x_values, z_values)
    curb_top = np.column_stack((
        x.reshape(-1),
        np.full(x.size, 0.88),
        z.reshape(-1),
    ))
    face_x, face_height = np.meshgrid(
        x_values,
        np.linspace(0.0, 0.12, 13),
    )
    curb_face = np.column_stack((
        face_x.reshape(-1),
        (1.0 - face_height).reshape(-1),
        np.full(face_x.size, 4.0),
    ))

    selected = _horizontal_filter(
        curb_top, np.vstack((curb_top, curb_face)))

    assert selected.tolist() == list(range(len(curb_top)))


def test_horizontal_filter_keeps_vertical_and_near_obstacles() -> None:
    x, y = np.meshgrid(
        np.linspace(-0.25, 0.25, 11),
        np.linspace(0.2, 0.9, 15),
    )
    wall = np.column_stack((
        x.reshape(-1), y.reshape(-1), np.full(x.size, 6.0)))
    near_x, near_z = np.meshgrid(
        np.linspace(-0.2, 0.2, 9), np.linspace(1.8, 2.2, 9))
    near_sheet = np.column_stack((
        near_x.reshape(-1),
        np.full(near_x.size, 0.85),
        near_z.reshape(-1),
    ))
    points = np.vstack((wall, near_sheet))

    selected = _horizontal_filter(points, points)

    assert selected.tolist() == list(range(len(points)))


def test_horizontal_filter_uses_rear_bev_axes_for_road_paint() -> None:
    forward_axis, left_axis, up_axis, _ = _rear_axes()
    forward_values, left_values = np.meshgrid(
        np.linspace(-6.5, -5.5, 21),
        np.linspace(-0.35, 0.35, 15),
    )

    def rear_surface(up_height):
        return (
            forward_values.reshape(-1, 1) * forward_axis
            + left_values.reshape(-1, 1) * left_axis
            + up_height * up_axis
        )

    paint = rear_surface(-0.86)
    road = rear_surface(-1.0)
    selected = horizontal_surface_filter_indices(
        paint,
        np.vstack((road, paint)),
        expected_up=up_axis,
        cell_size_m=0.10,
        min_component_points=8,
        min_up_alignment=0.98,
        max_plane_thickness_m=0.07,
        near_bypass_range_m=3.0,
        vertical_bin_size_m=0.02,
        vertical_min_points_per_bin=1,
        vertical_min_run_bins=5,
        vertical_min_occupied_bins=3,
        vertical_min_support_cells=2,
        vehicle_forward_vector=forward_axis,
        vehicle_left_vector=left_axis,
    )

    assert selected.size == 0


def test_ransac_accepts_40_degree_ramp() -> None:
    rng = np.random.default_rng(31)
    x = rng.uniform(-1.5, 1.5, 900)
    z = rng.uniform(0.6, 2.6, 900)
    ramp_start = 1.2
    y = 1.0 - (z - ramp_start) * np.tan(np.deg2rad(40.0))
    points = np.column_stack((
        x,
        y + rng.normal(0.0, 0.004, len(y)),
        z,
    ))
    config = replace(
        _config(),
        max_ground_tilt_deg=40.0,
        max_plane_distance_from_origin_m=1.8,
    ).normalized()

    plane = fit_ground_plane(points, config, np.random.default_rng(42))

    assert plane is not None
    tilt_deg = np.degrees(np.arccos(np.clip(
        plane.normal @ config.expected_up, -1.0, 1.0)))
    assert 39.0 <= tilt_deg <= 40.0
    _, ground_mask = classify_points(points, plane, config)
    assert ground_mask.mean() > 0.95


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


def test_range_adaptive_radius_filter_preserves_sparse_far_cluster() -> None:
    near = np.column_stack((
        np.linspace(0.01, 0.09, 9),
        np.zeros(9),
        np.full(9, 2.01),
    ))
    far = np.column_stack((
        np.linspace(1.01, 1.25, 9),
        np.zeros(9),
        np.full(9, 6.01),
    ))
    noise = np.array([[2.0, 0.0, 6.0], [2.5, 0.0, 6.0]])
    points = np.vstack((near, far, noise))

    selected = range_adaptive_radius_outlier_indices(
        points,
        near_radius_m=0.15,
        near_min_neighbors=8,
        far_start_range_m=5.0,
        far_radius_m=0.40,
        far_min_neighbors=8,
    )

    assert selected.tolist() == list(range(18))


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
