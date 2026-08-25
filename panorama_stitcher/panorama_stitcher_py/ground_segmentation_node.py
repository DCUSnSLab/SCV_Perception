"""ROS 2 Python node for camera ground/obstacle segmentation."""

from __future__ import annotations

import time
from typing import Tuple

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2

from panorama_stitcher_py.ground_segmentation import (
    GroundSegmentationConfig,
    classify_points,
    fit_ground_planes,
    radius_outlier_indices,
    range_residual_summary,
    select_ground_candidates,
    voxel_first_indices,
)


XYZRGB_FIELDS = (
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
)
XYZRGB_DTYPE = np.dtype([
    ('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<f4')])


def cloud_to_arrays(
    message: PointCloud2,
    input_stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract finite XYZ and packed RGB arrays without Python point loops."""
    names = {field.name for field in message.fields}
    color_name = 'rgb' if 'rgb' in names else 'rgba' if 'rgba' in names else ''
    requested = ['x', 'y', 'z'] + ([color_name] if color_name else [])
    cloud = point_cloud2.read_points(
        message, field_names=requested, skip_nans=False)
    stride = max(1, int(input_stride))
    if stride > 1:
        if message.height > 1 and len(cloud) == message.width * message.height:
            cloud = cloud.reshape(message.height, message.width)[
                ::stride, ::stride].reshape(-1)
        else:
            cloud = cloud[::stride]
    finite = np.isfinite(cloud['x'])
    finite &= np.isfinite(cloud['y'])
    finite &= np.isfinite(cloud['z'])
    if not finite.all():
        cloud = cloud[finite]
    if len(cloud) == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty(0, dtype=np.uint32),
        )
    xyz = np.empty((len(cloud), 3), dtype=np.float32)
    xyz[:, 0] = cloud['x']
    xyz[:, 1] = cloud['y']
    xyz[:, 2] = cloud['z']
    if color_name:
        color = np.asarray(cloud[color_name])
        if color.dtype.itemsize == 4:
            rgb = color.view(np.uint32).reshape(-1)
        else:
            rgb = color.astype(np.uint32, copy=False).reshape(-1)
    else:
        rgb = np.zeros(len(xyz), dtype=np.uint32)
    return xyz, rgb


def arrays_to_cloud(header, xyz: np.ndarray, rgb: np.ndarray) -> PointCloud2:
    """Create a compact XYZRGB PointCloud2 from NumPy arrays."""
    output = np.empty(len(xyz), dtype=XYZRGB_DTYPE)
    output['x'] = xyz[:, 0].astype(np.float32, copy=False)
    output['y'] = xyz[:, 1].astype(np.float32, copy=False)
    output['z'] = xyz[:, 2].astype(np.float32, copy=False)
    output['rgb'] = np.asarray(rgb, dtype=np.uint32).view(np.float32)
    message = point_cloud2.create_cloud(header, XYZRGB_FIELDS, output)
    message.is_dense = True
    return message


class PanoramaGroundSegmentationNode(Node):
    """Fit road surfaces and publish ground-relative obstacle returns."""

    def __init__(self) -> None:
        super().__init__(
            'panorama_ground_segmentation',
            automatically_declare_parameters_from_overrides=True,
        )
        if not self.has_parameter('input_topic'):
            raise RuntimeError(
                'No ground-segmentation parameters loaded; pass the matching '
                'config/*_ground_segmentation.yaml file')
        self.input_topic = str(self.get_parameter('input_topic').value)
        self.obstacle_topic = str(self.get_parameter('obstacle_topic').value)
        self.ground_topic = str(self.get_parameter('ground_topic').value)
        self.publish_ground_cloud = bool(
            self.get_parameter('publish_ground_cloud').value)
        self.skip_when_unsubscribed = bool(
            self.get_parameter('skip_when_unsubscribed').value)
        self.input_stride = max(
            1, int(self.get_parameter('input_stride').value))
        self.diagnostics_period_sec = max(
            0.2, float(self.get_parameter('diagnostics_period_sec').value))
        self.config = self._load_config()
        self.rng = np.random.default_rng(
            int(self.get_parameter('random_seed').value))
        self._last_diagnostic_time = -float('inf')
        self._idle_frames = 0

        output_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.obstacle_publisher = self.create_publisher(
            PointCloud2, self.obstacle_topic, output_qos)
        self.ground_publisher = (
            self.create_publisher(PointCloud2, self.ground_topic, output_qos)
            if self.publish_ground_cloud else None
        )
        self.subscription = self.create_subscription(
            PointCloud2,
            self.input_topic,
            self._pointcloud_callback,
            qos_profile_sensor_data,
        )
        up = self.config.expected_up
        self.get_logger().info(
            'Python ground segmentation: '
            f'{self.input_topic} -> {self.obstacle_topic}, '
            f'input_stride={self.input_stride}, '
            f'obstacle_radius_filter='
            f'{self.config.obstacle_radius_filter_radius_m:.2f}m/'
            f'{self.config.obstacle_radius_filter_min_neighbors}, '
            f'expected_up=[{up[0]:.2f} {up[1]:.2f} {up[2]:.2f}], '
            f'tilt<={self.config.max_ground_tilt_deg:.1f} deg')

    def _load_config(self) -> GroundSegmentationConfig:
        def value(name):
            return self.get_parameter(name).value

        return GroundSegmentationConfig(
            expected_up=np.asarray(
                value('expected_up_vector'), dtype=np.float32),
            ransac_iterations=int(value('ransac_iterations')),
            max_ransac_points=int(value('max_ransac_points')),
            ransac_distance_threshold_m=float(
                value('ransac_distance_threshold_m')),
            max_ground_tilt_deg=float(value('max_ground_tilt_deg')),
            min_ground_inliers=int(value('min_ground_inliers')),
            min_ground_inlier_ratio=float(value('min_ground_inlier_ratio')),
            secondary_ransac_iterations=int(
                value('secondary_ransac_iterations')),
            secondary_min_ground_inlier_ratio=float(
                value('secondary_min_ground_inlier_ratio')),
            secondary_min_normal_delta_deg=float(
                value('secondary_min_normal_delta_deg')),
            secondary_max_plane_distance_from_origin_m=float(
                value('secondary_max_plane_distance_from_origin_m')),
            ground_candidate_min_range_m=float(
                value('ground_candidate_min_range_m')),
            ground_candidate_max_range_m=float(
                value('ground_candidate_max_range_m')),
            ground_candidate_min_down_m=float(
                value('ground_candidate_min_down_m')),
            ground_candidate_max_down_m=float(
                value('ground_candidate_max_down_m')),
            min_plane_distance_from_origin_m=float(
                value('min_plane_distance_from_origin_m')),
            max_plane_distance_from_origin_m=float(
                value('max_plane_distance_from_origin_m')),
            obstacle_min_height_m=float(value('obstacle_min_height_m')),
            obstacle_max_height_m=float(value('obstacle_max_height_m')),
            obstacle_min_range_m=float(value('obstacle_min_range_m')),
            obstacle_max_range_m=float(value('obstacle_max_range_m')),
            obstacle_radius_filter_radius_m=float(
                value('obstacle_radius_filter_radius_m')),
            obstacle_radius_filter_min_neighbors=int(
                value('obstacle_radius_filter_min_neighbors')),
            obstacle_voxel_size_m=float(value('obstacle_voxel_size_m')),
        ).normalized()

    def _diagnostic(self, level: str, message: str) -> None:
        now = time.monotonic()
        if now - self._last_diagnostic_time < self.diagnostics_period_sec:
            return
        self._last_diagnostic_time = now
        getattr(self.get_logger(), level)(message)

    def _has_output_subscriber(self) -> bool:
        return (
            self.obstacle_publisher.get_subscription_count() > 0
            or (
                self.ground_publisher is not None
                and self.ground_publisher.get_subscription_count() > 0
            )
        )

    def _publish_empty(self, header) -> None:
        empty_xyz = np.empty((0, 3), dtype=np.float32)
        empty_rgb = np.empty(0, dtype=np.uint32)
        self.obstacle_publisher.publish(
            arrays_to_cloud(header, empty_xyz, empty_rgb))
        if self.ground_publisher is not None:
            self.ground_publisher.publish(
                arrays_to_cloud(header, empty_xyz, empty_rgb))

    def _pointcloud_callback(self, message: PointCloud2) -> None:
        started = time.perf_counter()
        process_cpu_started = time.process_time()
        if self.skip_when_unsubscribed and not self._has_output_subscriber():
            self._idle_frames += 1
            self._diagnostic(
                'info',
                f'idle: no subscriber on {self.obstacle_topic}, '
                f'{self._idle_frames} clouds skipped',
            )
            return
        try:
            xyz, rgb = cloud_to_arrays(message, self.input_stride)
            converted = time.perf_counter()
            candidate_indices = select_ground_candidates(xyz, self.config)
            candidates_selected = time.perf_counter()
            candidate_count = len(candidate_indices)
            planes = fit_ground_planes(
                xyz, self.config, self.rng, candidate_indices)
            planes_fitted = time.perf_counter()
            if not planes:
                self._publish_empty(message.header)
                self._diagnostic(
                    'warning',
                    f'No valid ground plane: input={len(xyz)} '
                    f'candidates={candidate_count}',
                )
                return

            obstacle_mask, ground_mask = classify_points(
                xyz, planes, self.config)
            classified = time.perf_counter()
            obstacle_xyz = xyz[obstacle_mask]
            obstacle_rgb = rgb[obstacle_mask]
            raw_obstacle_count = len(obstacle_xyz)
            radius_selected = radius_outlier_indices(
                obstacle_xyz,
                self.config.obstacle_radius_filter_radius_m,
                self.config.obstacle_radius_filter_min_neighbors,
            )
            obstacle_xyz = obstacle_xyz[radius_selected]
            obstacle_rgb = obstacle_rgb[radius_selected]
            radius_filtered_count = len(obstacle_xyz)
            if self.config.obstacle_voxel_size_m > 0.0:
                selected = voxel_first_indices(
                    obstacle_xyz, self.config.obstacle_voxel_size_m)
                published_obstacle_xyz = obstacle_xyz[selected]
                published_obstacle_rgb = obstacle_rgb[selected]
            else:
                published_obstacle_xyz = obstacle_xyz
                published_obstacle_rgb = obstacle_rgb
            self.obstacle_publisher.publish(arrays_to_cloud(
                message.header,
                published_obstacle_xyz,
                published_obstacle_rgb,
            ))
            if (
                self.ground_publisher is not None
                and self.ground_publisher.get_subscription_count() > 0
            ):
                self.ground_publisher.publish(arrays_to_cloud(
                    message.header, xyz[ground_mask], rgb[ground_mask]))

            published_obstacle_count = len(published_obstacle_xyz)
            primary = planes[0]
            ratio = (
                primary.inliers / candidate_count if candidate_count else 0.0)
            residual_text = ''
            if (
                time.monotonic() - self._last_diagnostic_time
                >= self.diagnostics_period_sec
            ):
                residual_text = range_residual_summary(
                    xyz[candidate_indices],
                    planes,
                    self.config.ground_candidate_max_range_m,
                )
            processing_ms = (time.perf_counter() - started) * 1000.0
            process_cpu_ms = (
                time.process_time() - process_cpu_started) * 1000.0
            plane_text = '; '.join(
                f'{model.normal[0]:.4f} {model.normal[1]:.4f} '
                f'{model.normal[2]:.4f} {model.offset:.4f} '
                f'n={model.inliers}'
                for model in planes
            )
            self._diagnostic(
                'info',
                f'ground_planes={len(planes)} [{plane_text}] input={len(xyz)} '
                f'candidates={candidate_count} inliers={primary.inliers}'
                f'({ratio * 100.0:.1f}%) obstacles='
                f'{raw_obstacle_count}->{radius_filtered_count}'
                f'->{published_obstacle_count} '
                f'ground_residual_by_range={residual_text} '
                f'timing(total/cpu/read/candidate/ransac/classify+publish)='
                f'{processing_ms:.1f}/{process_cpu_ms:.1f}/'
                f'{(converted - started) * 1000.0:.1f}/'
                f'{(candidates_selected - converted) * 1000.0:.1f}/'
                f'{(planes_fitted - candidates_selected) * 1000.0:.1f}/'
                f'{(time.perf_counter() - classified) * 1000.0:.1f} ms',
            )
        except Exception as error:  # Keep the sensor pipeline alive per frame.
            self._publish_empty(message.header)
            self._diagnostic(
                'error', f'Ground segmentation failed: {error}')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = PanoramaGroundSegmentationNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
