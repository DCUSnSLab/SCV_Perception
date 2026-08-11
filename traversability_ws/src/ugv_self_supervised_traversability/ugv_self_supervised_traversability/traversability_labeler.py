"""Delayed self-supervision node producing positive-only traversability masks."""

from collections import deque
from pathlib import Path
import json
from typing import Deque, Dict, Optional, Tuple

import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener

from . import TRAVERSABLE, UNKNOWN
from .observation_buffer import Observation, ObservationBuffer
from .trajectory_projector import TrajectoryProjector
from .utils.camera_projection import convert_depth_to_meters, validate_intrinsics
from .utils.geometry import footprint_corners
from .utils.transforms import euler_from_quaternion, transform_matrix


def stamp_to_ns(stamp) -> int:
    """Convert builtin_interfaces/Time to integer nanoseconds."""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class TraversabilityLabeler(Node):
    """Associate future robot footprints with buffered past camera frames."""

    def __init__(self) -> None:
        super().__init__('traversability_labeler')
        topic_defaults = {
            'camera_image_topic': '/camera/color/image_raw',
            'camera_depth_topic': '/camera/depth/image_rect_raw',
            'camera_info_topic': '/camera/color/camera_info',
            'odom_topic': '/odom',
            'pointcloud_topic': '/velodyne_points',
            'positive_mask_topic': '/traversability/positive_mask',
            'labeled_image_topic': '/traversability/labeled_image',
        }
        for name, default in topic_defaults.items():
            self.declare_parameter(name, default)
        for name, default in (
            ('world_frame', 'odom'), ('base_frame', 'base_link'),
            ('camera_frame', 'camera_color_optical_frame'),
                ('dataset_root', '~/traversability_dataset')):
            self.declare_parameter(name, default)
        for name, default in (
            ('robot_width', 0.8), ('robot_length', 1.2), ('footprint_margin', 0.05),
            ('observation_buffer_seconds', 10.0), ('depth_tolerance', 0.25),
                ('sensor_sync_tolerance', 0.08), ('min_traversal_distance', 0.03)):
            self.declare_parameter(name, default)
        self.declare_parameter('observation_buffer_max_frames', 150)
        self.declare_parameter('depth_check_enabled', True)
        self.declare_parameter('save_dataset', False)
        self.declare_parameter('save_empty_labels', False)

        self.world_frame = str(self.get_parameter('world_frame').value)
        self.camera_frame = str(self.get_parameter('camera_frame').value)
        self.width = float(self.get_parameter('robot_width').value)
        self.length = float(self.get_parameter('robot_length').value)
        self.margin = float(self.get_parameter('footprint_margin').value)
        sync_tolerance = float(self.get_parameter('sensor_sync_tolerance').value)
        self.sync_tolerance_ns = int(sync_tolerance * 1e9)
        self.min_distance = float(self.get_parameter('min_traversal_distance').value)
        self.save_dataset = bool(self.get_parameter('save_dataset').value)
        self.save_empty = bool(self.get_parameter('save_empty_labels').value)
        self.dataset_root = Path(str(self.get_parameter('dataset_root').value)).expanduser()
        self.bridge = CvBridge()
        self.buffer = ObservationBuffer(
            float(self.get_parameter('observation_buffer_seconds').value),
            int(self.get_parameter('observation_buffer_max_frames').value))
        self.projector = TrajectoryProjector(
            bool(self.get_parameter('depth_check_enabled').value),
            float(self.get_parameter('depth_tolerance').value))
        self.tf_buffer = Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.camera_info: Optional[CameraInfo] = None
        self.depth_messages: Deque[Image] = deque(maxlen=30)
        self.robot_poses: Deque[Tuple[int, Dict[str, float]]] = deque(maxlen=500)
        self.last_traversal_xy: Optional[Tuple[float, float]] = None
        self.mask_pub = self.create_publisher(
            Image, str(self.get_parameter('positive_mask_topic').value), 10)
        self.image_pub = self.create_publisher(
            Image, str(self.get_parameter('labeled_image_topic').value), 10)
        self.info_sub = self.create_subscription(
            CameraInfo, str(self.get_parameter('camera_info_topic').value),
            self._camera_info_callback, qos_profile_sensor_data)
        self.depth_sub = self.create_subscription(
            Image, str(self.get_parameter('camera_depth_topic').value),
            self._depth_callback, qos_profile_sensor_data)
        self.image_sub = self.create_subscription(
            Image, str(self.get_parameter('camera_image_topic').value),
            self._image_callback, qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(
            Odometry, str(self.get_parameter('odom_topic').value),
            self._odom_callback, qos_profile_sensor_data)
        if self.save_dataset:
            for directory in ('images', 'depth', 'labels', 'metadata'):
                (self.dataset_root / directory).mkdir(parents=True, exist_ok=True)
            self.get_logger().info('Dataset output: %s' % self.dataset_root)

    def _camera_info_callback(self, message: CameraInfo) -> None:
        try:
            validate_intrinsics(np.asarray(message.k).reshape(3, 3))
        except (ValueError, TypeError) as error:
            self.get_logger().warning('Rejecting invalid CameraInfo: %s' % error,
                                      throttle_duration_sec=5.0)
            return
        if message.width == 0 or message.height == 0:
            self.get_logger().warning('Rejecting zero-sized CameraInfo', throttle_duration_sec=5.0)
            return
        self.camera_info = message

    def _depth_callback(self, message: Image) -> None:
        if message.width == 0 or message.height == 0 or not message.data:
            return
        self.depth_messages.append(message)

    def _nearest_depth(self, timestamp_ns: int, shape: Tuple[int, int]) -> Optional[np.ndarray]:
        if not self.depth_messages:
            return None
        message = min(
            self.depth_messages,
            key=lambda item: abs(stamp_to_ns(item.header.stamp) - timestamp_ns))
        if abs(stamp_to_ns(message.header.stamp) - timestamp_ns) > self.sync_tolerance_ns:
            return None
        try:
            raw = self.bridge.imgmsg_to_cv2(message, desired_encoding='passthrough')
            depth = convert_depth_to_meters(np.asarray(raw), message.encoding)
        except (CvBridgeError, ValueError) as error:
            self.get_logger().warning('Depth conversion failed: %s' % error,
                                      throttle_duration_sec=5.0)
            return None
        if depth.shape != shape:
            self.get_logger().warning('RGB/depth shape mismatch; depth check skipped',
                                      throttle_duration_sec=5.0)
            return None
        return depth

    def _nearest_robot_pose(self, timestamp_ns: int) -> Dict[str, float]:
        if not self.robot_poses:
            return {}
        return min(self.robot_poses, key=lambda item: abs(item[0] - timestamp_ns))[1].copy()

    def _image_callback(self, message: Image) -> None:
        if self.camera_info is None:
            self.get_logger().warning('Waiting for valid CameraInfo', throttle_duration_sec=5.0)
            return
        if message.width == 0 or message.height == 0 or not message.data:
            return
        if (message.width != self.camera_info.width or
                message.height != self.camera_info.height):
            self.get_logger().warning(
                'RGB dimensions do not match CameraInfo; frame rejected',
                throttle_duration_sec=5.0)
            return
        try:
            rgb = np.asarray(self.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')).copy()
        except CvBridgeError as error:
            self.get_logger().warning('RGB conversion failed: %s' % error,
                                      throttle_duration_sec=5.0)
            return
        timestamp_ns = stamp_to_ns(message.header.stamp)
        depth = self._nearest_depth(timestamp_ns, rgb.shape[:2])
        frame = self.camera_frame or message.header.frame_id
        try:
            transform = self.tf_buffer.lookup_transform(
                self.world_frame, frame, Time.from_msg(message.header.stamp),
                timeout=Duration(seconds=0.1))
        except TransformException as error:
            self.get_logger().warning('Camera TF unavailable: %s' % error,
                                      throttle_duration_sec=5.0)
            return
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        world_t_camera = transform_matrix(
            [translation.x, translation.y, translation.z],
            [rotation.x, rotation.y, rotation.z, rotation.w])
        intrinsic = np.asarray(self.camera_info.k, dtype=np.float64).reshape(3, 3)
        observation = Observation(
            timestamp_ns=timestamp_ns,
            frame_id=frame,
            rgb=rgb,
            depth_m=depth,
            intrinsic=intrinsic,
            world_t_camera=world_t_camera,
            robot_pose=self._nearest_robot_pose(timestamp_ns),
            label=np.full(rgb.shape[:2], UNKNOWN, dtype=np.uint8),
            metadata={'rgb_encoding': 'bgr8'})
        for evicted in self.buffer.append(observation):
            self._finalize(evicted)

    def _odom_callback(self, message: Odometry) -> None:
        timestamp_ns = stamp_to_ns(message.header.stamp)
        if message.header.frame_id and message.header.frame_id != self.world_frame:
            self.get_logger().warning(
                'Odometry frame differs from world_frame; delayed labeling skipped. '
                'Provide odometry in world_frame.', throttle_duration_sec=5.0)
            return
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        roll, pitch, yaw = euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w])
        pose = {'timestamp_ns': timestamp_ns, 'x': position.x, 'y': position.y,
                'z': position.z, 'roll': roll, 'pitch': pitch, 'yaw': yaw}
        self.robot_poses.append((timestamp_ns, pose))
        for expired in self.buffer.pop_expired(timestamp_ns):
            self._finalize(expired)
        if self.last_traversal_xy is not None:
            distance = ((position.x - self.last_traversal_xy[0]) ** 2 +
                        (position.y - self.last_traversal_xy[1]) ** 2) ** 0.5
            if distance < self.min_distance:
                return
        self.last_traversal_xy = (position.x, position.y)
        footprint = footprint_corners(position.x, position.y, position.z, yaw,
                                      self.length, self.width, self.margin)
        for observation in self.buffer.observations_before(timestamp_ns):
            positive = self.projector.project_footprint(footprint, observation)
            new_positive = positive & (observation.label != TRAVERSABLE)
            if np.any(new_positive):
                observation.label[new_positive] = TRAVERSABLE
                observation.source_trajectory_timestamps_ns.append(timestamp_ns)
                self._publish(observation)

    def _publish(self, observation: Observation) -> None:
        stamp = Time(nanoseconds=observation.timestamp_ns).to_msg()
        mask_message = self.bridge.cv2_to_imgmsg(observation.label, encoding='mono8')
        mask_message.header.stamp = stamp
        mask_message.header.frame_id = observation.frame_id
        image_message = self.bridge.cv2_to_imgmsg(observation.rgb, encoding='bgr8')
        image_message.header = mask_message.header
        self.image_pub.publish(image_message)
        self.mask_pub.publish(mask_message)

    def _finalize(self, observation: Observation) -> None:
        traversable_pixels = int(np.count_nonzero(observation.label == TRAVERSABLE))
        if traversable_pixels:
            self._publish(observation)
        if not self.save_dataset or (not traversable_pixels and not self.save_empty):
            return
        stem = str(observation.timestamp_ns)
        cv2.imwrite(str(self.dataset_root / 'images' / (stem + '.png')), observation.rgb)
        cv2.imwrite(str(self.dataset_root / 'labels' / (stem + '.png')), observation.label)
        if observation.depth_m is not None:
            depth_mm = np.nan_to_num(observation.depth_m * 1000.0, nan=0.0,
                                     posinf=0.0, neginf=0.0).clip(0, 65535).astype(np.uint16)
            cv2.imwrite(str(self.dataset_root / 'depth' / (stem + '.png')), depth_mm)
        metadata = {
            'timestamp_ns': observation.timestamp_ns,
            'timestamp_seconds': observation.timestamp_ns / 1e9,
            'frame_id': observation.frame_id,
            'robot_pose': observation.robot_pose,
            'camera_pose_world_matrix': observation.world_t_camera.tolist(),
            'source_trajectory_timestamps_ns': observation.source_trajectory_timestamps_ns,
            'label_statistics': {
                'traversable_pixels': traversable_pixels,
                'unknown_pixels': int(np.count_nonzero(observation.label == UNKNOWN)),
                'non_traversable_pixels': 0,
            },
            'label_encoding': {'UNKNOWN': 0, 'TRAVERSABLE': 1, 'NON_TRAVERSABLE': 2},
            'depth_unit': 'millimeter_png' if observation.depth_m is not None else None,
        }
        metadata_path = self.dataset_root / 'metadata' / (stem + '.json')
        with metadata_path.open('w', encoding='utf-8') as handle:
            json.dump(metadata, handle, indent=2)

    def destroy_node(self):
        for observation in self.buffer.flush():
            self._finalize(observation)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TraversabilityLabeler()
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
