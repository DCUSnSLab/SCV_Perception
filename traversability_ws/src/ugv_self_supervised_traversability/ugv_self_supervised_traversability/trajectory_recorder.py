"""Record odometry or TF poses as a bounded nav_msgs/Path."""

import math
from typing import Dict, List, Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener

from .utils.transforms import euler_from_quaternion


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
        pose.pose = message.pose.pose
        self._record(pose)

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
