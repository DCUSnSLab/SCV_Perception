"""ROS 2 node for SegFormer masks and semantic RGB-D point clouds."""

import json
import time

import cv2
import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from rclpy.time import Time
from sensor_msgs.msg import (CameraInfo, CompressedImage, Image, PointCloud2,
                             PointField)
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, String
from tf2_ros import Buffer, TransformException, TransformListener

from .depth_geometry import (FUSED_LABELS, GEOMETRY_LABELS, GeometryConfig,
                             classify_depth_geometry,
                             fuse_cityscapes_geometry, label_colors)
from .model import SegformerRunner
from .semantic_geometry import (CITYSCAPES_LABELS, colorize_labels,
                                depth_to_meters, resize_semantics,
                                rasterize_camera_depth,
                                scale_intrinsics, semantic_rgbd_points,
                                select_projected_semantics, transform_xyz)


def quaternion_matrix(x, y, z, w):
    """Convert a quaternion into a float32 3-by-3 rotation matrix."""
    norm = np.linalg.norm((x, y, z, w))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError('invalid transform quaternion')
    x, y, z, w = np.asarray((x, y, z, w), dtype=np.float32) / norm
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),
         2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z),
         2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w),
         1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


class TerrainSegNode(Node):
    """Infer Cityscapes labels and lift synchronized RGB-D pixels into 3D."""

    def __init__(self):
        super().__init__('terrain_seg_node')
        defaults = {
            'model_id': 'nvidia/segformer-b0-finetuned-cityscapes-1024-1024',
            'device': 'auto',
            'use_fp16': True,
            'input_mode': 'cloud',
            'image_topic': '/camera/camera/color/image_raw',
            'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
            'camera_info_topic': '/camera/camera/color/camera_info',
            'compressed_image_topic':
                '/camera/camera/color/image_raw/compressed',
            'camera_cloud_topic': '/camera/camera/depth/color/points',
            'camera_calibration_width': 896,
            'camera_calibration_height': 504,
            'camera_intrinsics': [450.6982727050781, 450.1395263671875,
                                  443.96533203125, 260.06781005859375],
            'camera_distortion': [-0.0555432066321373,
                                  0.06311847269535065,
                                  0.0002300205669598654,
                                  0.0008905537542887032,
                                  -0.019217900931835175],
            'class_mask_topic': '/terrain_seg/class_mask',
            'color_mask_topic': '/terrain_seg/color_mask',
            'overlay_topic': '/terrain_seg/overlay',
            'pointcloud_topic': '/terrain_seg/semantic_points',
            'lidar_topic': '/velodyne_points',
            'lidar_pointcloud_topic': '/terrain_seg/semantic_lidar_points',
            'geometry_pointcloud_topic': '/terrain_seg/geometry_points',
            'fused_pointcloud_topic': '/terrain_seg/fused_points',
            'labels_topic': '/terrain_seg/labels',
            'geometry_labels_topic': '/terrain_seg/geometry_labels',
            'fused_labels_topic': '/terrain_seg/fused_labels',
            'target_frame': 'velodyne',
            'static_tf_only': True,
            'sync_queue_size': 5,
            'sync_slop': 0.08,
            'cloud_sync_slop': 0.15,
            'lidar_sync_tolerance': 0.12,
            'lidar_occlusion_tolerance': 0.5,
            'tf_timeout': 0.1,
            'max_inference_rate': 5.0,
            'point_stride': 2,
            'min_depth': 0.25,
            'max_depth': 12.0,
            'depth_scale': 0.001,
            'min_confidence': 0.55,
            'included_class_ids': [0, 1],
            'overlay_alpha': 0.45,
            'enable_depth_geometry': True,
            'geometry_stride': 2,
            'geometry_resolution': 0.05,
            'geometry_x_min': -1.0,
            'geometry_x_max': 12.0,
            'geometry_y_min': -5.0,
            'geometry_y_max': 5.0,
            'geometry_z_min': -1.5,
            'geometry_z_max': 1.5,
            'expected_ground_z': -0.9,
            'expected_ground_tolerance': 0.45,
            'ground_plane_tolerance': 0.05,
            'max_ground_slope_deg': 20.0,
            'curb_min_height': 0.05,
            'curb_max_height': 0.25,
            'obstacle_min_height': 0.28,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        p = {name: self.get_parameter(name).value for name in defaults}
        self._validate(p)

        self.bridge = CvBridge()
        self.target_frame = p['target_frame']
        self.input_mode = p['input_mode']
        self.tf_timeout = p['tf_timeout']
        self.max_inference_rate = p['max_inference_rate']
        self.point_stride = p['point_stride']
        self.min_depth = p['min_depth']
        self.max_depth = p['max_depth']
        self.depth_scale = p['depth_scale']
        self.min_confidence = p['min_confidence']
        self.included_class_ids = tuple(p['included_class_ids'])
        self.overlay_alpha = p['overlay_alpha']
        self.lidar_sync_tolerance = p['lidar_sync_tolerance']
        self.lidar_occlusion_tolerance = p['lidar_occlusion_tolerance']
        self.last_inference_time = -np.inf
        self.latest_lidar = None
        self.camera_calibration_size = (
            p['camera_calibration_width'], p['camera_calibration_height'])
        self.camera_intrinsics = tuple(p['camera_intrinsics'])
        self.camera_distortion = tuple(p['camera_distortion'])
        self.enable_depth_geometry = p['enable_depth_geometry']
        self.geometry_stride = p['geometry_stride']
        self.geometry_config = GeometryConfig(
            resolution=p['geometry_resolution'],
            x_min=p['geometry_x_min'], x_max=p['geometry_x_max'],
            y_min=p['geometry_y_min'], y_max=p['geometry_y_max'],
            z_min=p['geometry_z_min'], z_max=p['geometry_z_max'],
            expected_ground_z=p['expected_ground_z'],
            expected_ground_tolerance=p['expected_ground_tolerance'],
            plane_tolerance=p['ground_plane_tolerance'],
            max_ground_slope_deg=p['max_ground_slope_deg'],
            curb_min_height=p['curb_min_height'],
            curb_max_height=p['curb_max_height'],
            obstacle_min_height=p['obstacle_min_height'])

        self.get_logger().info(
            f"Loading {p['model_id']} on device={p['device']} ...")
        self.runner = SegformerRunner(
            p['model_id'], p['device'], p['use_fp16'])
        self.get_logger().info('SegFormer model loaded')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        if p['static_tf_only']:
            # Camera/LiDAR extrinsics are fixed. Avoid TF_OLD_DATA storms when
            # a recorded bag is replayed in a loop without resetting ROS time.
            self.destroy_subscription(self.tf_listener.tf_sub)
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE)

        if self.input_mode == 'rgbd':
            self.image_sub = message_filters.Subscriber(
                self, Image, p['image_topic'], qos_profile=sensor_qos)
            self.depth_sub = message_filters.Subscriber(
                self, Image, p['depth_topic'], qos_profile=sensor_qos)
            self.info_sub = message_filters.Subscriber(
                self, CameraInfo, p['camera_info_topic'],
                qos_profile=sensor_qos)
            self.synchronizer = message_filters.ApproximateTimeSynchronizer(
                [self.image_sub, self.depth_sub, self.info_sub],
                queue_size=p['sync_queue_size'], slop=p['sync_slop'],
                allow_headerless=False)
            self.synchronizer.registerCallback(self.synced_callback)
        else:
            self.image_sub = message_filters.Subscriber(
                self, CompressedImage, p['compressed_image_topic'],
                qos_profile=sensor_qos)
            self.camera_cloud_sub = message_filters.Subscriber(
                self, PointCloud2, p['camera_cloud_topic'],
                qos_profile=sensor_qos)
            self.synchronizer = message_filters.ApproximateTimeSynchronizer(
                [self.image_sub, self.camera_cloud_sub],
                queue_size=p['sync_queue_size'], slop=p['cloud_sync_slop'],
                allow_headerless=False)
            self.synchronizer.registerCallback(self.cloud_synced_callback)
        self.lidar_sub = self.create_subscription(
            PointCloud2, p['lidar_topic'], self._receive_lidar, sensor_qos)

        self.class_mask_pub = self.create_publisher(
            Image, p['class_mask_topic'], 1)
        self.color_mask_pub = self.create_publisher(
            Image, p['color_mask_topic'], 1)
        self.overlay_pub = self.create_publisher(Image, p['overlay_topic'], 1)
        self.cloud_pub = self.create_publisher(
            PointCloud2, p['pointcloud_topic'], sensor_qos)
        self.lidar_cloud_pub = self.create_publisher(
            PointCloud2, p['lidar_pointcloud_topic'], sensor_qos)
        self.geometry_cloud_pub = self.create_publisher(
            PointCloud2, p['geometry_pointcloud_topic'], sensor_qos)
        self.fused_cloud_pub = self.create_publisher(
            PointCloud2, p['fused_pointcloud_topic'], sensor_qos)
        label_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.labels_pub = self.create_publisher(
            String, p['labels_topic'], label_qos)
        self.geometry_labels_pub = self.create_publisher(
            String, p['geometry_labels_topic'], label_qos)
        self.fused_labels_pub = self.create_publisher(
            String, p['fused_labels_topic'], label_qos)
        self.labels_pub.publish(String(data=json.dumps(
            {index: name for index, name in enumerate(CITYSCAPES_LABELS)})))
        self.geometry_labels_pub.publish(String(data=json.dumps(
            GEOMETRY_LABELS)))
        self.fused_labels_pub.publish(String(data=json.dumps(FUSED_LABELS)))

        self.get_logger().info(
            f'terrain_seg ready: input_mode={self.input_mode}; '
            f'included classes={self.included_class_ids or "all"}')

    @staticmethod
    def _validate(p):
        if p['input_mode'] not in ('rgbd', 'cloud'):
            raise ValueError('input_mode must be rgbd or cloud')
        if p['sync_queue_size'] < 1:
            raise ValueError('sync_queue_size must be positive')
        if p['sync_slop'] < 0.0 or p['tf_timeout'] < 0.0:
            raise ValueError('sync_slop and tf_timeout must be non-negative')
        if p['cloud_sync_slop'] < 0.0:
            raise ValueError('cloud_sync_slop must be non-negative')
        if p['lidar_sync_tolerance'] < 0.0:
            raise ValueError('lidar_sync_tolerance must be non-negative')
        if p['lidar_occlusion_tolerance'] < 0.0:
            raise ValueError('lidar_occlusion_tolerance must be non-negative')
        if p['max_inference_rate'] <= 0.0:
            raise ValueError('max_inference_rate must be positive')
        if p['point_stride'] < 1:
            raise ValueError('point_stride must be positive')
        if p['geometry_stride'] < 1:
            raise ValueError('geometry_stride must be positive')
        if not 0.0 < p['min_depth'] < p['max_depth']:
            raise ValueError('depth limits are invalid')
        if p['depth_scale'] <= 0.0:
            raise ValueError('depth_scale must be positive')
        if not 0.0 <= p['min_confidence'] <= 1.0:
            raise ValueError('min_confidence must be in [0, 1]')
        if not 0.0 <= p['overlay_alpha'] <= 1.0:
            raise ValueError('overlay_alpha must be in [0, 1]')
        if any(value < 0 or value >= len(CITYSCAPES_LABELS)
               for value in p['included_class_ids']):
            raise ValueError('included_class_ids contains an invalid class id')
        if len(p['camera_intrinsics']) != 4:
            raise ValueError('camera_intrinsics must contain fx, fy, cx, cy')
        if p['camera_calibration_width'] <= 0 or \
                p['camera_calibration_height'] <= 0:
            raise ValueError('camera calibration dimensions must be positive')
        if not 0.0 < p['curb_min_height'] < p['curb_max_height'] \
                < p['obstacle_min_height']:
            raise ValueError('curb/obstacle height thresholds are invalid')

    def _receive_lidar(self, message):
        self.latest_lidar = message

    def cloud_synced_callback(self, image_msg, camera_cloud_msg):
        """Process compressed RGB and the recorded unorganized D555 cloud."""
        now = time.monotonic()
        if now - self.last_inference_time < 1.0 / self.max_inference_rate:
            return
        self.last_inference_time = now
        try:
            encoded = np.frombuffer(image_msg.data, dtype=np.uint8)
            bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError('failed to decode compressed RGB image')
            labels, confidence = self.runner.predict(bgr)
            self._publish_images(labels, bgr, image_msg.header)

            camera_xyz = self._read_xyz(camera_cloud_msg)
            color_frame = image_msg.header.frame_id
            cloud_frame = camera_cloud_msg.header.frame_id
            color_from_cloud = self._lookup_matrix(
                color_frame, cloud_frame, camera_cloud_msg.header.stamp)
            xyz_color = transform_xyz(
                camera_xyz, color_from_cloud[0], color_from_cloud[1])
            intrinsics = self._configured_intrinsics(bgr.shape[:2])
            indices, rgb, point_labels, point_confidence = (
                select_projected_semantics(
                    xyz_color, labels, confidence, bgr, intrinsics,
                    min_confidence=self.min_confidence,
                    included_class_ids=self.included_class_ids,
                    distortion_coeffs=self.camera_distortion))
            output_xyz = camera_xyz
            output_frame = cloud_frame
            if self.target_frame and self.target_frame != cloud_frame:
                target_from_cloud = self._lookup_matrix(
                    self.target_frame, cloud_frame,
                    camera_cloud_msg.header.stamp)
                output_xyz = transform_xyz(
                    camera_xyz, target_from_cloud[0], target_from_cloud[1])
                output_frame = self.target_frame
            selected_xyz = output_xyz[indices]
            self.cloud_pub.publish(self._create_cloud(
                selected_xyz, rgb, point_labels, point_confidence,
                output_frame, camera_cloud_msg.header.stamp))

            if self.enable_depth_geometry:
                geometry = classify_depth_geometry(
                    output_xyz, self.geometry_config)
                self._publish_geometry_clouds(
                    output_xyz, geometry.labels, geometry.confidence,
                    indices, point_labels, point_confidence, output_frame,
                    camera_cloud_msg.header.stamp)

            depth_image = rasterize_camera_depth(
                xyz_color, intrinsics, labels.shape, self.camera_distortion)
            self._publish_semantic_lidar_values(
                labels, confidence, bgr, depth_image, image_msg.header,
                intrinsics, self.camera_distortion)
        except (ValueError, RuntimeError, TransformException, KeyError) as error:
            self.get_logger().warning(
                f'cloud-mode segmentation frame skipped: {error}',
                throttle_duration_sec=2.0)

    def _publish_geometry_clouds(
            self, xyz, geometry_labels, geometry_confidence,
            semantic_indices, semantic_labels, semantic_confidence,
            frame_id, stamp):
        selected = np.flatnonzero(geometry_labels != 0)[::self.geometry_stride]
        self.geometry_cloud_pub.publish(self._create_cloud(
            xyz[selected], label_colors(geometry_labels[selected]),
            geometry_labels[selected], geometry_confidence[selected],
            frame_id, stamp))

        selected_geometry = geometry_labels[semantic_indices]
        selected_geometry_confidence = geometry_confidence[semantic_indices]
        fused_labels, fused_confidence = fuse_cityscapes_geometry(
            semantic_labels, semantic_confidence, selected_geometry,
            selected_geometry_confidence)
        self.fused_cloud_pub.publish(self._create_cloud(
            xyz[semantic_indices], label_colors(fused_labels), fused_labels,
            fused_confidence, frame_id, stamp))

    def synced_callback(self, image_msg, depth_msg, info_msg):
        """Process a synchronized color, aligned-depth and CameraInfo triplet."""
        now = time.monotonic()
        if now - self.last_inference_time < 1.0 / self.max_inference_rate:
            return
        self.last_inference_time = now
        started = time.perf_counter()
        try:
            bgr = self.bridge.imgmsg_to_cv2(
                image_msg, desired_encoding='bgr8')
            depth_raw = self.bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding='passthrough')
            depth_m = depth_to_meters(
                depth_raw, depth_msg.encoding, self.depth_scale)
            labels, confidence = self.runner.predict(bgr)
            self._publish_images(labels, bgr, image_msg.header)

            labels_depth, confidence_depth, bgr_depth = resize_semantics(
                labels, confidence, bgr, depth_m.shape)
            intrinsics = scale_intrinsics(
                info_msg.k, (info_msg.width, info_msg.height),
                (depth_m.shape[1], depth_m.shape[0]))
            xyz, rgb, point_labels, point_confidence = semantic_rgbd_points(
                depth_m, labels_depth, confidence_depth, bgr_depth,
                intrinsics, stride=self.point_stride,
                min_depth=self.min_depth, max_depth=self.max_depth,
                min_confidence=self.min_confidence,
                included_class_ids=self.included_class_ids)

            source_frame = depth_msg.header.frame_id
            output_frame = source_frame
            if self.target_frame and self.target_frame != source_frame:
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame, source_frame,
                    Time.from_msg(depth_msg.header.stamp),
                    timeout=Duration(seconds=self.tf_timeout))
                q = transform.transform.rotation
                rotation = quaternion_matrix(q.x, q.y, q.z, q.w)
                t = transform.transform.translation
                xyz = transform_xyz(xyz, rotation, (t.x, t.y, t.z))
                output_frame = self.target_frame

            cloud = self._create_cloud(
                xyz, rgb, point_labels, point_confidence,
                output_frame, depth_msg.header.stamp)
            self.cloud_pub.publish(cloud)
            distortion = tuple(info_msg.d)
            self._publish_semantic_lidar_values(
                labels, confidence, bgr, depth_m, image_msg.header,
                scale_intrinsics(
                    info_msg.k, (info_msg.width, info_msg.height),
                    (labels.shape[1], labels.shape[0])), distortion)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.get_logger().debug(
                f'published {len(xyz)} semantic points in {elapsed_ms:.1f} ms')
        except (ValueError, RuntimeError, TransformException) as error:
            self.get_logger().warning(
                f'terrain segmentation frame skipped: {error}',
                throttle_duration_sec=2.0)

    def _publish_semantic_lidar_values(
            self, labels, confidence, bgr, depth_m, image_header,
            intrinsics, distortion):
        lidar_msg = self.latest_lidar
        if lidar_msg is None:
            return
        image_stamp = self._stamp_seconds(image_header.stamp)
        lidar_stamp = self._stamp_seconds(lidar_msg.header.stamp)
        if abs(image_stamp - lidar_stamp) > self.lidar_sync_tolerance:
            return
        try:
            lidar_xyz = self._read_xyz(lidar_msg)
            if not len(lidar_xyz):
                return

            camera_frame = image_header.frame_id
            lidar_frame = lidar_msg.header.frame_id
            camera_from_lidar = self._lookup_matrix(
                camera_frame, lidar_frame, lidar_msg.header.stamp)
            xyz_camera = transform_xyz(
                lidar_xyz, camera_from_lidar[0], camera_from_lidar[1])
            if depth_m.shape != labels.shape:
                depth_for_occlusion = cv2.resize(
                    depth_m, (labels.shape[1], labels.shape[0]),
                    interpolation=cv2.INTER_NEAREST)
            else:
                depth_for_occlusion = depth_m
            indices, rgb, point_labels, point_confidence = (
                select_projected_semantics(
                    xyz_camera, labels, confidence, bgr, intrinsics,
                    min_confidence=self.min_confidence,
                    included_class_ids=self.included_class_ids,
                    aligned_depth_m=depth_for_occlusion,
                    occlusion_tolerance=self.lidar_occlusion_tolerance,
                    distortion_coeffs=distortion))
            selected_xyz = lidar_xyz[indices]
            output_frame = lidar_frame
            if self.target_frame and self.target_frame != lidar_frame:
                target_from_lidar = self._lookup_matrix(
                    self.target_frame, lidar_frame, lidar_msg.header.stamp)
                selected_xyz = transform_xyz(
                    selected_xyz, target_from_lidar[0], target_from_lidar[1])
                output_frame = self.target_frame
            output = self._create_cloud(
                selected_xyz, rgb, point_labels, point_confidence,
                output_frame, lidar_msg.header.stamp)
            self.lidar_cloud_pub.publish(output)
        except (ValueError, TransformException, KeyError) as error:
            self.get_logger().warning(
                f'semantic LiDAR unavailable: {error}',
                throttle_duration_sec=2.0)

    @staticmethod
    def _read_xyz(message):
        records = point_cloud2.read_points(
            message, field_names=('x', 'y', 'z'), skip_nans=True)
        records = np.asarray(records).reshape(-1)
        if not len(records):
            return np.empty((0, 3), dtype=np.float32)
        return np.column_stack((
            records['x'], records['y'], records['z'])).astype(np.float32)

    def _configured_intrinsics(self, image_shape):
        height, width = image_shape
        source_width, source_height = self.camera_calibration_size
        fx, fy, cx, cy = self.camera_intrinsics
        return (fx * width / source_width, fy * height / source_height,
                cx * width / source_width, cy * height / source_height)

    def _lookup_matrix(self, target_frame, source_frame, stamp):
        transform = self.tf_buffer.lookup_transform(
            target_frame, source_frame, Time.from_msg(stamp),
            timeout=Duration(seconds=self.tf_timeout))
        q = transform.transform.rotation
        rotation = quaternion_matrix(q.x, q.y, q.z, q.w)
        t = transform.transform.translation
        return rotation, np.asarray((t.x, t.y, t.z), dtype=np.float32)

    @staticmethod
    def _stamp_seconds(stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    def _publish_images(self, labels, bgr, header):
        color_mask = colorize_labels(labels)
        overlay = cv2.addWeighted(
            bgr, 1.0 - self.overlay_alpha, color_mask,
            self.overlay_alpha, 0.0)
        for publisher, array, encoding in (
                (self.class_mask_pub, labels, 'mono8'),
                (self.color_mask_pub, color_mask, 'bgr8'),
                (self.overlay_pub, overlay, 'bgr8')):
            message = self.bridge.cv2_to_imgmsg(array, encoding=encoding)
            message.header = header
            publisher.publish(message)

    @staticmethod
    def _create_cloud(xyz, rgb, labels, confidence, frame_id, stamp):
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32,
                       count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32,
                       count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32,
                       count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32,
                       count=1),
            PointField(name='label', offset=16, datatype=PointField.UINT8,
                       count=1),
            PointField(name='confidence', offset=20,
                       datatype=PointField.FLOAT32, count=1),
        ]
        dtype = point_cloud2.dtype_from_fields(fields)
        points = np.empty(len(xyz), dtype=dtype)
        points['x'], points['y'], points['z'] = xyz.T
        points['rgb'] = rgb
        points['label'] = labels
        points['confidence'] = confidence
        header = Header(frame_id=frame_id, stamp=stamp)
        return point_cloud2.create_cloud(header, fields, points)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = TerrainSegNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as error:  # Report model download/dependency failures cleanly.
        if node is not None:
            node.get_logger().fatal(str(error))
        else:
            print(f'terrain_seg startup failed: {error}')
        raise
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
