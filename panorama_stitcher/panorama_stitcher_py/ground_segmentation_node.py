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
    fit_ground_plane,
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


def cloud_to_arrays(message: PointCloud2) -> Tuple[np.ndarray, np.ndarray]:
    """Extract finite XYZ and packed RGB arrays without Python point loops."""
    names = {field.name for field in message.fields}
    color_name = 'rgb' if 'rgb' in names else 'rgba' if 'rgba' in names else ''
    requested = ['x', 'y', 'z'] + ([color_name] if color_name else [])
    cloud = point_cloud2.read_points(
        message, field_names=requested, skip_nans=True)
    if len(cloud) == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.uint32),
        )
    xyz = np.column_stack((cloud['x'], cloud['y'], cloud['z'])).astype(
        np.float64, copy=False)
    if color_name:
        color = np.asarray(cloud[color_name])
        if color.dtype.itemsize == 4:
            rgb = color.view(np.uint32).reshape(-1).copy()
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
    """Fit one road plane and publish ground-relative obstacle returns."""

    def __init__(self) -> None:
        super().__init__('panorama_ground_segmentation')
        self._declare_parameters()
        self.input_topic = str(self.get_parameter('input_topic').value)
        self.obstacle_topic = str(self.get_parameter('obstacle_topic').value)
        self.ground_topic = str(self.get_parameter('ground_topic').value)
        self.publish_ground_cloud = bool(
            self.get_parameter('publish_ground_cloud').value)
        self.skip_when_unsubscribed = bool(
            self.get_parameter('skip_when_unsubscribed').value)
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
            f'expected_up=[{up[0]:.2f} {up[1]:.2f} {up[2]:.2f}], '
            f'tilt<={self.config.max_ground_tilt_deg:.1f} deg')

    def _declare_parameters(self) -> None:
        defaults = {
            'input_topic': '/panorama/points',
            'obstacle_topic': '/panorama/obstacle_points',
            'ground_topic': '/panorama/ground_points',
            'publish_ground_cloud': True,
            'expected_up_vector': [0.0, -1.0, 0.0],
            'ransac_iterations': 160,
            'max_ransac_points': 30000,
            'ransac_distance_threshold_m': 0.05,
            'max_ground_tilt_deg': 25.0,
            'min_ground_inliers': 300,
            'min_ground_inlier_ratio': 0.03,
            'ground_candidate_min_range_m': 0.4,
            'ground_candidate_max_range_m': 8.0,
            'ground_candidate_min_down_m': 0.15,
            'ground_candidate_max_down_m': 2.5,
            'min_plane_distance_from_origin_m': 0.15,
            'max_plane_distance_from_origin_m': 2.5,
            'obstacle_min_height_m': 0.10,
            'obstacle_max_height_m': 2.0,
            'obstacle_min_range_m': 0.25,
            'obstacle_max_range_m': 8.0,
            'obstacle_voxel_size_m': 0.0,
            'skip_when_unsubscribed': True,
            'diagnostics_period_sec': 2.0,
            'random_seed': 42,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _load_config(self) -> GroundSegmentationConfig:
        def value(name):
            return self.get_parameter(name).value

        return GroundSegmentationConfig(
            expected_up=np.asarray(
                value('expected_up_vector'), dtype=np.float64),
            ransac_iterations=int(value('ransac_iterations')),
            max_ransac_points=int(value('max_ransac_points')),
            ransac_distance_threshold_m=float(
                value('ransac_distance_threshold_m')),
            max_ground_tilt_deg=float(value('max_ground_tilt_deg')),
            min_ground_inliers=int(value('min_ground_inliers')),
            min_ground_inlier_ratio=float(value('min_ground_inlier_ratio')),
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
        empty_xyz = np.empty((0, 3), dtype=np.float64)
        empty_rgb = np.empty(0, dtype=np.uint32)
        self.obstacle_publisher.publish(
            arrays_to_cloud(header, empty_xyz, empty_rgb))
        if self.ground_publisher is not None:
            self.ground_publisher.publish(
                arrays_to_cloud(header, empty_xyz, empty_rgb))

    def _pointcloud_callback(self, message: PointCloud2) -> None:
        started = time.perf_counter()
        if self.skip_when_unsubscribed and not self._has_output_subscriber():
            self._idle_frames += 1
            self._diagnostic(
                'info',
                f'idle: no subscriber on {self.obstacle_topic}, '
                f'{self._idle_frames} clouds skipped',
            )
            return
        try:
            xyz, rgb = cloud_to_arrays(message)
            candidate_count = len(select_ground_candidates(xyz, self.config))
            plane = fit_ground_plane(xyz, self.config, self.rng)
            if plane is None:
                self._publish_empty(message.header)
                self._diagnostic(
                    'warning',
                    f'No valid ground plane: input={len(xyz)} '
                    f'candidates={candidate_count}',
                )
                return

            obstacle_mask, ground_mask = classify_points(
                xyz, plane, self.config)
            obstacle_xyz = xyz[obstacle_mask]
            obstacle_rgb = rgb[obstacle_mask]
            raw_obstacle_count = len(obstacle_xyz)
            selected = voxel_first_indices(
                obstacle_xyz, self.config.obstacle_voxel_size_m)
            self.obstacle_publisher.publish(arrays_to_cloud(
                message.header,
                obstacle_xyz[selected],
                obstacle_rgb[selected],
            ))
            if (
                self.ground_publisher is not None
                and self.ground_publisher.get_subscription_count() > 0
            ):
                self.ground_publisher.publish(arrays_to_cloud(
                    message.header, xyz[ground_mask], rgb[ground_mask]))

            processing_ms = (time.perf_counter() - started) * 1000.0
            ratio = plane.inliers / candidate_count if candidate_count else 0.0
            normal = plane.normal
            self._diagnostic(
                'info',
                'ground=['
                f'{normal[0]:.4f} {normal[1]:.4f} {normal[2]:.4f} '
                f'{plane.offset:.4f}] input={len(xyz)} '
                f'candidates={candidate_count} inliers={plane.inliers}'
                f'({ratio * 100.0:.1f}%) obstacles={raw_obstacle_count}'
                f'->{len(selected)} processing={processing_ms:.1f} ms',
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
