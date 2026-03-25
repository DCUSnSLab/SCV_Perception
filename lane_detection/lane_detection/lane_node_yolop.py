#!/usr/bin/env python3
import os
import time

import cv2
import numpy as np
import rclpy
import tf2_ros
import torch
import torchvision.transforms as transforms
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from scipy.spatial import KDTree
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from tf2_ros import TransformException

from lane_detection.lib.config import cfg
from lane_detection.lib.models import get_net


class YoloPLaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node_yolop')

        default_weights = os.path.expanduser('~/epoch-155_old.pth')
        self.declare_parameter('weights_path', default_weights)
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('img_size', 640)
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('depth_min', 0.1)
        self.declare_parameter('depth_max', 10.0)
        self.declare_parameter('depth_scale', 0.01)
        self.declare_parameter('voxel_size', 0.03)
        self.declare_parameter('ground_proj', True)
        self.declare_parameter('mask_point_stride', 1)
        self.declare_parameter('enable_depth_hole_fill', True)
        self.declare_parameter('enable_sor', True)
        self.declare_parameter('sor_k', 20)
        self.declare_parameter('sor_std_mul', 1.5)
        self.declare_parameter('morph_kernel_width', 3)
        self.declare_parameter('morph_kernel_height', 5)
        self.declare_parameter('manual_roll_deg', 0.3)
        self.declare_parameter('manual_pitch_deg', 0.1)
        self.declare_parameter('manual_yaw_deg', 0.0)
        self.declare_parameter('cloud_offset_x', -0.5)
        self.declare_parameter('cloud_offset_y', 0.05)
        self.declare_parameter('lane_pixel_value', 255)

        self.weights_path = os.path.expanduser(self.get_parameter('weights_path').value)
        self.image_topic = self.get_parameter('image_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.img_size = int(self.get_parameter('img_size').value)
        self.device_name = self.get_parameter('device').value
        self.depth_min = float(self.get_parameter('depth_min').value)
        self.depth_max = float(self.get_parameter('depth_max').value)
        self.depth_scale = float(self.get_parameter('depth_scale').value)
        self.voxel_size = float(self.get_parameter('voxel_size').value)
        self.ground_proj = bool(self.get_parameter('ground_proj').value)
        self.mask_point_stride = max(1, int(self.get_parameter('mask_point_stride').value))
        self.enable_depth_hole_fill = bool(self.get_parameter('enable_depth_hole_fill').value)
        self.enable_sor = bool(self.get_parameter('enable_sor').value)
        self.sor_k = int(self.get_parameter('sor_k').value)
        self.sor_std_mul = float(self.get_parameter('sor_std_mul').value)
        kernel_w = max(1, int(self.get_parameter('morph_kernel_width').value))
        kernel_h = max(1, int(self.get_parameter('morph_kernel_height').value))
        self.manual_roll_deg = float(self.get_parameter('manual_roll_deg').value)
        self.manual_pitch_deg = float(self.get_parameter('manual_pitch_deg').value)
        self.manual_yaw_deg = float(self.get_parameter('manual_yaw_deg').value)
        self.cloud_offset_x = float(self.get_parameter('cloud_offset_x').value)
        self.cloud_offset_y = float(self.get_parameter('cloud_offset_y').value)
        self.lane_pixel_value = int(self.get_parameter('lane_pixel_value').value)

        self.device = torch.device(self.device_name if torch.cuda.is_available() else 'cpu')
        self.half = self.device.type != 'cpu'
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.bridge = CvBridge()
        self.K: np.ndarray | None = None
        self._last_color_msg_time: float | None = None
        self._last_depth_msg_time: float | None = None
        self._last_callback_time: float | None = None
        self.morph_kernel = np.ones((kernel_h, kernel_w), dtype=np.uint8)
        self.extra_rotation = self._build_rpy_rotation(
            self.manual_roll_deg, self.manual_pitch_deg, self.manual_yaw_deg
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._tf_cache = {}

        self.get_logger().info(f'YOLOP 모델 로드 중: {self.weights_path}')
        self.model = self._load_model()
        self.get_logger().info('YOLOP 모델 로드 완료')

        self.create_subscription(CameraInfo, self.camera_info_topic, self._camera_info_cb, 1)
        self.create_subscription(Image, self.image_topic, self._color_monitor_cb, 1)
        self.create_subscription(Image, self.depth_topic, self._depth_monitor_cb, 1)
        sub_color = Subscriber(self, Image, self.image_topic)
        sub_depth = Subscriber(self, Image, self.depth_topic)
        self.sync = ApproximateTimeSynchronizer([sub_color, sub_depth], queue_size=2, slop=0.05)
        self.sync.registerCallback(self._callback)

        self.pub_mask = self.create_publisher(Image, '/lane_detection/lane_mask', 1)
        self.pub_cloud = self.create_publisher(PointCloud2, '/lane_detection/lane_pointcloud', 1)
        self.get_logger().info(
            f'구독: color={self.image_topic}, depth={self.depth_topic}, camera_info={self.camera_info_topic} | '
            f'imgsz={self.img_size} device={self.device} depth_scale={self.depth_scale} '
            f'rpy_deg=({self.manual_roll_deg:.2f}, {self.manual_pitch_deg:.2f}, {self.manual_yaw_deg:.2f}) '
            f'cloud_offset=({self.cloud_offset_x:.3f}, {self.cloud_offset_y:.3f}) '
            f'stride={self.mask_point_stride} depth_hole_fill={self.enable_depth_hole_fill} '
            f'sor={self.enable_sor}'
        )

    def _load_model(self):
        model = get_net(cfg)
        ckpt = torch.load(self.weights_path, map_location='cpu')
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        elif isinstance(ckpt, dict):
            model.load_state_dict(ckpt)
        elif isinstance(ckpt, torch.nn.Module):
            model = ckpt
        else:
            raise RuntimeError('Unsupported YOLOP checkpoint format')

        model.to(self.device).eval()
        if self.half:
            model.half()

        dummy = torch.zeros((1, 3, self.img_size, self.img_size), device=self.device)
        _ = model(dummy.half() if self.half else dummy)
        return model

    def _camera_info_cb(self, msg: CameraInfo):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.get_logger().info(
                f'카메라 내부 파라미터 수신 — fx={self.K[0, 0]:.1f}, '
                f'fy={self.K[1, 1]:.1f}, cx={self.K[0, 2]:.1f}, cy={self.K[1, 2]:.1f}'
            )

    def _color_monitor_cb(self, _msg: Image) -> None:
        now = time.perf_counter()
        if self._last_color_msg_time is not None:
            dt = now - self._last_color_msg_time
            if dt > 0.0:
                self.get_logger().info(
                    f'color input rate: {1.0 / dt:.2f} Hz',
                    throttle_duration_sec=2.0
                )
        self._last_color_msg_time = now

    def _depth_monitor_cb(self, _msg: Image) -> None:
        now = time.perf_counter()
        if self._last_depth_msg_time is not None:
            dt = now - self._last_depth_msg_time
            if dt > 0.0:
                self.get_logger().info(
                    f'depth input rate: {1.0 / dt:.2f} Hz',
                    throttle_duration_sec=2.0
                )
        self._last_depth_msg_time = now

    def _callback(self, color_msg: Image, depth_msg: Image):
        now = time.perf_counter()
        if self._last_callback_time is not None:
            dt = now - self._last_callback_time
            if dt > 0.0:
                self.get_logger().info(
                    f'sync callback rate: {1.0 / dt:.2f} Hz',
                    throttle_duration_sec=2.0
                )
        self._last_callback_time = now

        if self.K is None:
            self.get_logger().warn('camera_info 미수신, 프레임 스킵', throttle_duration_sec=2.0)
            return

        img_bgr = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough').astype(np.float32)

        infer_start = time.perf_counter()
        lane_mask = self._infer_lane_mask(img_bgr)
        infer_ms = (time.perf_counter() - infer_start) * 1000.0
        self.get_logger().info(
            f'inference time: {infer_ms:.1f} ms',
            throttle_duration_sec=2.0
        )

        mask_msg = self.bridge.cv2_to_imgmsg((lane_mask * 255).astype(np.uint8), encoding='mono8')
        mask_msg.header = color_msg.header
        self.pub_mask.publish(mask_msg)

        self._publish_pointcloud(lane_mask, depth_img, color_msg.header)

    def _infer_lane_mask(self, img_bgr: np.ndarray) -> np.ndarray:
        tensor = self._preprocess(img_bgr)
        with torch.no_grad():
            _, _, ll_seg_out = self.model(tensor)

        lane_mask = torch.argmax(ll_seg_out, 1).int().squeeze().cpu().numpy().astype(np.uint8)
        lane_mask = cv2.resize(
            lane_mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        lane_mask = (lane_mask == self.lane_pixel_value).astype(np.uint8)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_OPEN, self.morph_kernel, iterations=1)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, self.morph_kernel, iterations=1)
        return lane_mask

    def _preprocess(self, img_bgr: np.ndarray) -> torch.Tensor:
        resized = cv2.resize(img_bgr, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = self.transform(resized).to(self.device)
        tensor = tensor.half() if self.half else tensor.float()
        return tensor.unsqueeze(0)

    def _publish_pointcloud(self, lane_mask: np.ndarray, depth_img: np.ndarray, header) -> None:
        frame_id = header.frame_id
        if frame_id not in self._tf_cache:
            try:
                tf_stamped = self.tf_buffer.lookup_transform('base_link', frame_id, rclpy.time.Time())
            except TransformException as e:
                self.get_logger().warn(f'TF 조회 실패: {e}', throttle_duration_sec=5.0)
                return
            q = tf_stamped.transform.rotation
            tv = tf_stamped.transform.translation
            qx, qy, qz, qw = q.x, q.y, q.z, q.w
            r_mat = np.array([
                [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
            ], dtype=np.float32)
            t_vec = np.array([tv.x, tv.y, tv.z], dtype=np.float32)
            self._tf_cache[frame_id] = (r_mat, t_vec)
            self.get_logger().info(f'TF 캐시 완료: base_link ← {frame_id}')

        r_mat, t_vec = self._tf_cache[frame_id]
        dh, dw = depth_img.shape[:2]
        if lane_mask.shape != (dh, dw):
            lane_mask = cv2.resize(lane_mask, (dw, dh), interpolation=cv2.INTER_NEAREST)

        ys, xs = np.where(lane_mask > 0)
        if len(xs) == 0:
            self.pub_cloud.publish(self._make_pointcloud2(np.empty((0, 3), dtype=np.float32), header.stamp, 'base_link'))
            return

        if self.mask_point_stride > 1:
            xs = xs[::self.mask_point_stride]
            ys = ys[::self.mask_point_stride]

        depths = depth_img[ys, xs] * self.depth_scale
        valid = (depths > self.depth_min) & (depths < self.depth_max)
        if not np.any(valid):
            self.pub_cloud.publish(self._make_pointcloud2(np.empty((0, 3), dtype=np.float32), header.stamp, 'base_link'))
            return

        if self.enable_depth_hole_fill and not np.all(valid):
            valid_coords = np.stack([xs[valid], ys[valid]], axis=1)
            invalid_coords = np.stack([xs[~valid], ys[~valid]], axis=1)
            if len(valid_coords) > 0 and len(invalid_coords) > 0:
                tree = KDTree(valid_coords)
                _, nn_idx = tree.query(invalid_coords, workers=-1)
                depths[~valid] = depths[valid][nn_idx]
                valid = np.ones(len(xs), dtype=bool)

        xs, ys, depths = xs[valid], ys[valid], depths[valid]
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        pts_cam = np.stack([
            (xs - cx) * depths / fx,
            (ys - cy) * depths / fy,
            depths,
        ], axis=-1).astype(np.float32)

        pts_cam = (self.extra_rotation @ pts_cam.T).T
        pts_cam[:, 0] *= -1
        pts_cam[:, 1] *= -1
        pts_base = (r_mat @ pts_cam.T).T + t_vec
        pts_base[:, 0] += self.cloud_offset_x
        pts_base[:, 1] += self.cloud_offset_y

        if self.ground_proj:
            pts_base[:, 2] = 0.0

        if self.enable_sor:
            pts_base = self._sor_filter(pts_base, self.sor_k, self.sor_std_mul)
        pts_f32 = self._voxel_downsample(pts_base)
        self.pub_cloud.publish(self._make_pointcloud2(pts_f32, header.stamp, 'base_link'))

    @staticmethod
    def _build_rpy_rotation(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
        roll = np.deg2rad(roll_deg)
        pitch = np.deg2rad(pitch_deg)
        yaw = np.deg2rad(yaw_deg)

        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float32)
        ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
        rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float32)
        return rz @ ry @ rx

    @staticmethod
    def _sor_filter(pts: np.ndarray, k: int, std_mul: float) -> np.ndarray:
        n = len(pts)
        if n < k + 1:
            return pts
        tree = KDTree(pts)
        dists, _ = tree.query(pts, k=k + 1, workers=-1)
        mean_dists = dists[:, 1:].mean(axis=1)
        threshold = mean_dists.mean() + std_mul * mean_dists.std()
        return pts[mean_dists <= threshold]

    def _voxel_downsample(self, pts: np.ndarray) -> np.ndarray:
        if len(pts) == 0:
            return pts
        idx = np.floor(pts / self.voxel_size).astype(np.int32)
        keys = idx[:, 0] * 1_000_003 + idx[:, 1] * 1_009 + idx[:, 2]
        _, first = np.unique(keys, return_index=True)
        return pts[first]

    @staticmethod
    def _make_pointcloud2(pts: np.ndarray, stamp, frame_id: str) -> PointCloud2:
        cloud = PointCloud2()
        cloud.header.stamp = stamp
        cloud.header.frame_id = frame_id
        cloud.height = 1
        cloud.width = len(pts)
        cloud.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud.is_bigendian = False
        cloud.point_step = 12
        cloud.row_step = 12 * len(pts)
        cloud.is_dense = True
        cloud.data = pts.astype(np.float32).tobytes()
        return cloud


def main(args=None):
    rclpy.init(args=args)
    node = YoloPLaneDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
