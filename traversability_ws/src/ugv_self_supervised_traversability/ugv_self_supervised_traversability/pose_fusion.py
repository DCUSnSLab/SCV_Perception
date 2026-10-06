"""Shared IMU and LiDAR pose refinement for traversability labeling nodes."""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import Imu, PointCloud2, PointField
from tf2_ros import Buffer, TransformException

from .utils.geometry import estimate_ground_height
from .utils.transforms import (euler_from_quaternion, transform_matrix, transform_points,
                               quaternion_to_rotation_matrix)


@dataclass
class PoseEstimate:
    """Pose estimate used for footprint generation and delayed reprojection."""

    timestamp_ns: int
    x: float
    y: float
    z: float
    base_z: float
    roll: float
    pitch: float
    yaw: float
    orientation_source: str
    z_source: str
    imu_timestamp_ns: Optional[int] = None


def stamp_to_ns(stamp) -> int:
    """Convert builtin_interfaces/Time into integer nanoseconds."""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def pointcloud_to_xyz(point_cloud: PointCloud2) -> np.ndarray:
    """Convert PointCloud2 x/y/z float32 fields into an Nx3 array."""
    field_offsets = {field.name: field.offset for field in point_cloud.fields
                     if field.datatype == PointField.FLOAT32}
    if not {'x', 'y', 'z'} <= set(field_offsets):
        raise ValueError('PointCloud2 must contain float32 x, y, z fields')
    endian = np.dtype('>f4' if point_cloud.is_bigendian else '<f4')
    shape = (max(int(point_cloud.height), 1), int(point_cloud.width))
    strides = (int(point_cloud.row_step), int(point_cloud.point_step))

    def extract(offset: int) -> np.ndarray:
        view = np.ndarray(
            shape=shape, dtype=endian, buffer=point_cloud.data,
            offset=offset, strides=strides)
        return np.asarray(view, dtype=np.float64).reshape(-1)

    points = np.column_stack((
        extract(field_offsets['x']),
        extract(field_offsets['y']),
        extract(field_offsets['z']),
    ))
    return points[np.isfinite(points).all(axis=1)]


class PoseFusion:
    """Refine odometry using IMU orientation and LiDAR-derived ground height."""

    def __init__(
        self,
        tf_buffer: Buffer,
        world_frame: str,
        *,
        use_imu_orientation: bool,
        imu_use_yaw: bool,
        imu_timeout_seconds: float,
        use_lidar_ground_height: bool,
        pointcloud_timeout_seconds: float,
        lidar_ground_radius: float,
        lidar_ground_min_points: int,
        lidar_ground_percentile: float,
        base_frame: str = 'base_link',
    ) -> None:
        self.tf_buffer = tf_buffer
        self.world_frame = world_frame
        self.base_frame = base_frame
        self.imu_status = 'missing'
        self.use_imu_orientation = use_imu_orientation
        self.imu_use_yaw = imu_use_yaw
        self.use_lidar_ground_height = use_lidar_ground_height
        self.imu_timeout_ns = int(imu_timeout_seconds * 1e9)
        self.pointcloud_timeout_ns = int(pointcloud_timeout_seconds * 1e9)
        self.lidar_ground_radius = lidar_ground_radius
        self.lidar_ground_min_points = lidar_ground_min_points
        self.lidar_ground_percentile = lidar_ground_percentile
        self._imu_timestamp_ns: Optional[int] = None
        self._imu_quaternion: Optional[np.ndarray] = None
        self._pointcloud_timestamp_ns: Optional[int] = None
        self._world_points: Optional[np.ndarray] = None

    def clear_sensor_cache(self) -> None:
        """Discard sensor state when recorded time moves backwards."""
        self._imu_timestamp_ns = None
        self._imu_quaternion = None
        self._pointcloud_timestamp_ns = None
        self._world_points = None
        self.imu_status = 'missing'

    def update_imu(self, message: Imu) -> None:
        """Store the latest usable IMU orientation."""
        self._imu_quaternion = None
        self.imu_status = 'invalid'
        if message.orientation_covariance[0] == -1 or not message.header.frame_id:
            return
        quaternion = np.array([
            message.orientation.x, message.orientation.y,
            message.orientation.z, message.orientation.w], dtype=np.float64)
        if not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) < 1e-8:
            return
        rotation = quaternion_to_rotation_matrix(quaternion)
        if message.header.frame_id != self.base_frame:
            try:
                extrinsic = self.tf_buffer.lookup_transform(
                    self.base_frame, message.header.frame_id,
                    Time.from_msg(message.header.stamp), timeout=Duration(seconds=0.0))
            except TransformException:
                self.imu_status = 'mounting_tf_missing'
                return
            q = extrinsic.transform.rotation
            rotation = rotation @ quaternion_to_rotation_matrix([q.x,q.y,q.z,q.w]).T
        # Convert the corrected reference_T_base orientation back to quaternion.
        from .utils.transforms import quaternion_from_euler
        roll = np.arctan2(rotation[2,1], rotation[2,2])
        pitch = np.arcsin(np.clip(-rotation[2,0], -1.0, 1.0))
        yaw = np.arctan2(rotation[1,0], rotation[0,0])
        self._imu_timestamp_ns = stamp_to_ns(message.header.stamp)
        self._imu_quaternion = quaternion_from_euler(roll,pitch,yaw)
        self.imu_status = 'valid'

    def update_pointcloud(self, message: PointCloud2) -> None:
        """Store the latest point cloud transformed into world coordinates."""
        points = pointcloud_to_xyz(message)
        if points.size == 0:
            return
        if message.header.frame_id and message.header.frame_id != self.world_frame:
            transform = self.tf_buffer.lookup_transform(
                self.world_frame, message.header.frame_id,
                Time.from_msg(message.header.stamp), timeout=Duration(seconds=0.1))
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            world_t_cloud = transform_matrix(
                [translation.x, translation.y, translation.z],
                [rotation.x, rotation.y, rotation.z, rotation.w])
            points = transform_points(points, world_t_cloud)
        self._pointcloud_timestamp_ns = stamp_to_ns(message.header.stamp)
        self._world_points = points

    def fuse_odometry(self, message: Odometry) -> PoseEstimate:
        """Return the pose used by traversability nodes for footprint placement."""
        timestamp_ns = stamp_to_ns(message.header.stamp)
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        roll, pitch, yaw = euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w])
        orientation_source = 'odometry'
        if (self.use_imu_orientation and self._imu_quaternion is not None and
                self._imu_timestamp_ns is not None and
                abs(self._imu_timestamp_ns - timestamp_ns) <= self.imu_timeout_ns):
            imu_roll, imu_pitch, imu_yaw = euler_from_quaternion(self._imu_quaternion)
            roll = imu_roll
            pitch = imu_pitch
            if self.imu_use_yaw:
                yaw = imu_yaw
            orientation_source = 'imu' if self.imu_use_yaw else 'odometry+imu'
        z = position.z
        z_source = 'odometry'
        if (self.use_lidar_ground_height and self._world_points is not None and
                self._pointcloud_timestamp_ns is not None and
                abs(self._pointcloud_timestamp_ns - timestamp_ns) <= self.pointcloud_timeout_ns):
            ground_z = estimate_ground_height(
                self._world_points, position.x, position.y, self.lidar_ground_radius,
                self.lidar_ground_min_points, self.lidar_ground_percentile)
            if ground_z is not None:
                z = ground_z
                z_source = 'lidar'
        return PoseEstimate(
            timestamp_ns=timestamp_ns,
            x=position.x,
            y=position.y,
            z=z,
            base_z=position.z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            orientation_source=orientation_source,
            z_source=z_source,
            imu_timestamp_ns=self._imu_timestamp_ns if orientation_source != 'odometry' else None)
