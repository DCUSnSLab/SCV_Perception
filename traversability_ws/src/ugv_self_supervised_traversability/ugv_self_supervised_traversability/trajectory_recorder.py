"""Record odometry or TF poses as a bounded nav_msgs/Path."""

import math
from typing import Dict, List, Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Imu, PointCloud2
from tf2_ros import Buffer, TransformException, TransformListener

from .pose_fusion import PoseFusion
from .utils.transforms import euler_from_quaternion, quaternion_from_euler


class TrajectoryRecorder(Node):
    """Publish the time-ordered robot trajectory for inspection and reuse."""

    def __init__(self) -> None:
        super().__init__('trajectory_recorder')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('trajectory_topic', '/traversability/trajectory')
        self.declare_parameter('world_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('pose_source', 'odom')
        self.declare_parameter('trajectory_publish_rate', 5.0)
        self.declare_parameter('min_pose_distance', 0.03)
        self.declare_parameter('max_path_poses', 10000)
        self.declare_parameter('imu_topic', '/imu/data')
        self.declare_parameter('pointcloud_topic', '/velodyne_points')
        self.declare_parameter('use_imu_orientation', True)
        self.declare_parameter('imu_use_yaw', False)
        self.declare_parameter('imu_timeout_seconds', 0.2)
        self.declare_parameter('use_lidar_ground_height', True)
        self.declare_parameter('pointcloud_timeout_seconds', 0.4)
        self.declare_parameter('lidar_ground_radius', 1.2)
        self.declare_parameter('lidar_ground_min_points', 20)
        self.declare_parameter('lidar_ground_percentile', 20.0)

        self.world_frame = str(self.get_parameter('world_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.pose_source = str(self.get_parameter('pose_source').value)
        self.min_distance = float(self.get_parameter('min_pose_distance').value)
        self.max_poses = int(self.get_parameter('max_path_poses').value)
        self.path = Path()
        self.path.header.frame_id = self.world_frame
        self.trajectory_records: List[Dict[str, float]] = []
        self.last_pose: Optional[PoseStamped] = None
        self.publisher = self.create_publisher(
            Path, str(self.get_parameter('trajectory_topic').value), 10)
        qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.tf_buffer = Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.pose_fusion = PoseFusion(
            self.tf_buffer, self.world_frame,
            use_imu_orientation=bool(self.get_parameter('use_imu_orientation').value),
            imu_use_yaw=bool(self.get_parameter('imu_use_yaw').value),
            imu_timeout_seconds=float(self.get_parameter('imu_timeout_seconds').value),
            use_lidar_ground_height=bool(self.get_parameter('use_lidar_ground_height').value),
            pointcloud_timeout_seconds=float(
                self.get_parameter('pointcloud_timeout_seconds').value),
            lidar_ground_radius=float(self.get_parameter('lidar_ground_radius').value),
            lidar_ground_min_points=int(self.get_parameter('lidar_ground_min_points').value),
            lidar_ground_percentile=float(self.get_parameter('lidar_ground_percentile').value),
            base_frame=str(self.get_parameter('base_frame').value))
        self.imu_subscription = self.create_subscription(
            Imu, str(self.get_parameter('imu_topic').value),
            self.pose_fusion.update_imu, qos_profile_sensor_data)
        self.pointcloud_subscription = self.create_subscription(
            PointCloud2, str(self.get_parameter('pointcloud_topic').value),
            self._pointcloud_callback, qos_profile_sensor_data)
        if self.pose_source == 'odom':
            self.subscription = self.create_subscription(
                Odometry, str(self.get_parameter('odom_topic').value), self._odom_callback, qos)
        elif self.pose_source == 'tf':
            rate = float(self.get_parameter('trajectory_publish_rate').value)
            self.tf_timer = self.create_timer(1.0 / max(rate, 0.1), self._tf_callback)
        else:
            raise ValueError("pose_source must be 'odom' or 'tf'")

    def _odom_callback(self, message: Odometry) -> None:
        if message.header.frame_id and message.header.frame_id != self.world_frame:
            self.get_logger().warning(
                'Odometry frame %s differs from world_frame %s; set world_frame to the odom frame '
                'or use pose_source=tf.' % (message.header.frame_id, self.world_frame),
                throttle_duration_sec=5.0)
            return
        pose = PoseStamped()
        pose.header = message.header
        pose.header.frame_id = self.world_frame
        fused = self.pose_fusion.fuse_odometry(message)
        pose.pose.position.x = fused.x
        pose.pose.position.y = fused.y
        pose.pose.position.z = fused.z
        quaternion = quaternion_from_euler(fused.roll, fused.pitch, fused.yaw)
        pose.pose.orientation.x = float(quaternion[0])
        pose.pose.orientation.y = float(quaternion[1])
        pose.pose.orientation.z = float(quaternion[2])
        pose.pose.orientation.w = float(quaternion[3])
        self._record(pose)

    def _pointcloud_callback(self, message: PointCloud2) -> None:
        try:
            self.pose_fusion.update_pointcloud(message)
        except (TransformException, ValueError) as error:
            self.get_logger().warning('LiDAR fusion skipped: %s' % error,
                                      throttle_duration_sec=5.0)

    def _tf_callback(self) -> None:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.world_frame, self.base_frame, Time())
        except TransformException as error:
            self.get_logger().warning('TF pose unavailable: %s' % error, throttle_duration_sec=5.0)
            return
        pose = PoseStamped()
        pose.header = transform.header
        pose.pose.position.x = transform.transform.translation.x
        pose.pose.position.y = transform.transform.translation.y
        pose.pose.position.z = transform.transform.translation.z
        pose.pose.orientation = transform.transform.rotation
        self._record(pose)

    def _record(self, pose: PoseStamped) -> None:
        if self.last_pose is not None:
            dx = pose.pose.position.x - self.last_pose.pose.position.x
            dy = pose.pose.position.y - self.last_pose.pose.position.y
            dz = pose.pose.position.z - self.last_pose.pose.position.z
            if math.sqrt(dx * dx + dy * dy + dz * dz) < self.min_distance:
                return
        self.path.header.stamp = pose.header.stamp
        self.path.poses.append(pose)
        orientation = pose.pose.orientation
        roll, pitch, yaw = euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w])
        timestamp_ns = (int(pose.header.stamp.sec) * 1_000_000_000 +
                        int(pose.header.stamp.nanosec))
        self.trajectory_records.append({
            'timestamp_ns': timestamp_ns,
            'x': pose.pose.position.x,
            'y': pose.pose.position.y,
            'z': pose.pose.position.z,
            'roll': roll,
            'pitch': pitch,
            'yaw': yaw,
        })
        if len(self.path.poses) > self.max_poses:
            self.path.poses = self.path.poses[-self.max_poses:]
            self.trajectory_records = self.trajectory_records[-self.max_poses:]
        self.last_pose = pose
        self.publisher.publish(self.path)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrajectoryRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        rclpy.try_shutdown()
