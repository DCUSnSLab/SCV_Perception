#!/usr/bin/env python3
import os
import time
import numpy as np
import cv2
from scipy.spatial import KDTree

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
import tf2_ros
from tf2_ros import TransformException
from ultralytics import YOLO


class Yolo26LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node_yolo26')

        default_model_path = os.path.expanduser('~/yolo26m_seg_best.pt')
        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera/color/camera_info')
        self.declare_parameter('conf', 0.0007)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('max_det', 64)
        self.declare_parameter('min_mask_area_ratio', 0.0002)
        self.declare_parameter('min_bottom_y_ratio', 0.45)
        self.declare_parameter('min_height_ratio', 0.08)
        self.declare_parameter('max_lane_instances', 8)
        self.declare_parameter('depth_min', 0.1)
        self.declare_parameter('depth_max', 10.0)
        self.declare_parameter('depth_scale', 0.01)
        self.declare_parameter('voxel_size', 0.03)
        self.declare_parameter('ground_proj', True)
        self.declare_parameter('sor_k', 20)
        self.declare_parameter('sor_std_mul', 1.5)
        self.declare_parameter('morph_kernel_width', 3)
        self.declare_parameter('morph_kernel_height', 5)
        self.declare_parameter('manual_roll_deg', 0.0)
        self.declare_parameter('manual_pitch_deg', 0.0)
        self.declare_parameter('manual_yaw_deg', 0.0)
        self.declare_parameter('cloud_offset_x', 0.0)
        self.declare_parameter('cloud_offset_y', 0.0)
        self.declare_parameter('publish_overlay', True)

        self.model_path = os.path.expanduser(self.get_parameter('model_path').value)
        self.image_topic = self.get_parameter('image_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.conf = float(self.get_parameter('conf').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.max_det = int(self.get_parameter('max_det').value)
        self.min_mask_area_ratio = float(self.get_parameter('min_mask_area_ratio').value)
        self.min_bottom_y_ratio = float(self.get_parameter('min_bottom_y_ratio').value)
        self.min_height_ratio = float(self.get_parameter('min_height_ratio').value)
        self.max_lane_instances = int(self.get_parameter('max_lane_instances').value)
        self.depth_min = float(self.get_parameter('depth_min').value)
        self.depth_max = float(self.get_parameter('depth_max').value)
        self.depth_scale = float(self.get_parameter('depth_scale').value)
        self.voxel_size = float(self.get_parameter('voxel_size').value)
        self.ground_proj = bool(self.get_parameter('ground_proj').value)
        self.sor_k = int(self.get_parameter('sor_k').value)
        self.sor_std_mul = float(self.get_parameter('sor_std_mul').value)
        kernel_w = max(1, int(self.get_parameter('morph_kernel_width').value))
        kernel_h = max(1, int(self.get_parameter('morph_kernel_height').value))
        self.manual_roll_deg = float(self.get_parameter('manual_roll_deg').value)
        self.manual_pitch_deg = float(self.get_parameter('manual_pitch_deg').value)
        self.manual_yaw_deg = float(self.get_parameter('manual_yaw_deg').value)
        self.cloud_offset_x = float(self.get_parameter('cloud_offset_x').value)
        self.cloud_offset_y = float(self.get_parameter('cloud_offset_y').value)
        self.publish_overlay = bool(self.get_parameter('publish_overlay').value)
        self.morph_kernel = np.ones((kernel_h, kernel_w), dtype=np.uint8)
        self.extra_rotation = self._build_rpy_rotation(
            self.manual_roll_deg, self.manual_pitch_deg, self.manual_yaw_deg
        )

        self.bridge = CvBridge()
        self.K: np.ndarray | None = None
        self._last_callback_time: float | None = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._tf_cache = {}

        self.get_logger().info(f'YOLO26-seg 모델 로드 중: {self.model_path}')
        self.model = YOLO(self.model_path)
        self.model.to('cuda')
        self.get_logger().info('YOLO26-seg 모델 로드 완료')

        self.create_subscription(CameraInfo, self.camera_info_topic, self._camera_info_cb, 1)
        sub_color = Subscriber(self, Image, self.image_topic)
        sub_depth = Subscriber(self, Image, self.depth_topic)
        self.sync = ApproximateTimeSynchronizer([sub_color, sub_depth], queue_size=2, slop=0.05)
        self.sync.registerCallback(self._callback)

        self.pub_mask = self.create_publisher(Image, '/lane_detection/lane_mask', 1)
        self.pub_cloud = self.create_publisher(PointCloud2, '/lane_detection/lane_pointcloud', 1)
        self.pub_overlay = self.create_publisher(Image, '/lane_detection/lane_overlay', 1)

        self.get_logger().info(
            f'구독: color={self.image_topic}, depth={self.depth_topic}, '
            f'camera_info={self.camera_info_topic} | conf={self.conf:.4f} '
            f'imgsz={self.imgsz} max_det={self.max_det} '
            f'depth_scale={self.depth_scale} '
            f'rpy_deg=({self.manual_roll_deg:.2f}, {self.manual_pitch_deg:.2f}, {self.manual_yaw_deg:.2f}) '
            f'cloud_offset=({self.cloud_offset_x:.3f}, {self.cloud_offset_y:.3f})'
        )

    def _camera_info_cb(self, msg: CameraInfo):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.get_logger().info(
                f'카메라 내부 파라미터 수신 — fx={self.K[0, 0]:.1f}, '
                f'fy={self.K[1, 1]:.1f}, cx={self.K[0, 2]:.1f}, cy={self.K[1, 2]:.1f}'
            )

    def _callback(self, color_msg: Image, depth_msg: Image):
        now = time.perf_counter()
        if self._last_callback_time is not None:
            dt = now - self._last_callback_time
            if dt > 0.0:
                self.get_logger().info(f'lane_detection rate: {1.0 / dt:.2f} Hz',
                                       throttle_duration_sec=2.0)
        self._last_callback_time = now

        if self.K is None:
            self.get_logger().warn('camera_info 미수신, 프레임 스킵', throttle_duration_sec=2.0)
            return

        img_bgr = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough').astype(np.float32)

        lane_mask, overlay = self._infer_lane_mask(img_bgr)

        mask_msg = self.bridge.cv2_to_imgmsg((lane_mask * 255).astype(np.uint8), encoding='mono8')
        mask_msg.header = color_msg.header
        self.pub_mask.publish(mask_msg)

        if self.publish_overlay:
            overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
            overlay_msg.header = color_msg.header
            self.pub_overlay.publish(overlay_msg)

        self._publish_pointcloud(lane_mask, depth_img, color_msg.header)

    def _infer_lane_mask(self, img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = img_bgr.shape[:2]
        results = self.model.predict(
            img_bgr,
            verbose=False,
            conf=self.conf,
            imgsz=self.imgsz,
            max_det=self.max_det,
            half=True,
            device='cuda',
            retina_masks=True,
        )
        result = results[0]
        overlay = img_bgr.copy()

        if result.masks is None or result.masks.data is None or result.masks.data.shape[0] == 0:
            return np.zeros((h, w), dtype=np.uint8), overlay

        boxes = result.boxes
        masks = result.masks.data
        lane_mask = np.zeros((h, w), dtype=np.uint8)
        candidates = []

        for idx in range(masks.shape[0]):
            mask = masks[idx].detach().cpu().numpy()
            if mask.ndim != 2:
                continue
            mask = (mask > 0).astype(np.uint8)
            ys, xs = np.where(mask > 0)
            if ys.size == 0:
                continue

            area_ratio = float(mask.sum()) / float(h * w)
            y_min, y_max = int(ys.min()), int(ys.max())
            x_min, x_max = int(xs.min()), int(xs.max())
            height_ratio = float(y_max - y_min + 1) / float(h)
            bottom_ratio = float(y_max) / float(h)
            width_ratio = float(x_max - x_min + 1) / float(w)
            box_conf = float(boxes.conf[idx].item()) if boxes is not None and boxes.conf is not None else 0.0

            # Lane fragments tend to be long, thin, and extend downward.
            geometry_ok = (
                area_ratio >= self.min_mask_area_ratio and
                (bottom_ratio >= self.min_bottom_y_ratio or height_ratio >= self.min_height_ratio)
            )
            score = box_conf + 0.15 * bottom_ratio + 0.10 * height_ratio - 0.05 * width_ratio
            candidates.append((score, geometry_ok, idx, mask, (x_min, y_min, x_max, y_max), box_conf))

        if not candidates:
            return lane_mask, overlay

        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = [c for c in candidates if c[1]]
        if not selected:
            selected = candidates[: min(2, len(candidates))]
        else:
            selected = selected[: self.max_lane_instances]

        for rank, (_, _, _, mask, bbox, box_conf) in enumerate(selected, start=1):
            lane_mask = np.maximum(lane_mask, mask)
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (255, 180, 0), 2)
            cv2.putText(
                overlay,
                f'lane {box_conf:.2f} #{rank}',
                (x_min, max(20, y_min - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 180, 0),
                1,
                cv2.LINE_AA,
            )

        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, self.morph_kernel)
        overlay[lane_mask > 0] = cv2.addWeighted(
            overlay[lane_mask > 0], 0.4, np.full_like(overlay[lane_mask > 0], (255, 0, 0)), 0.6, 0.0
        )
        return lane_mask, overlay

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
            self.get_logger().warn('lane mask는 있지만 유효한 픽셀이 없어 pointcloud 생성을 건너뜁니다.',
                                   throttle_duration_sec=2.0)
            self.pub_cloud.publish(self._make_pointcloud2(np.empty((0, 3), dtype=np.float32), header.stamp, 'base_link'))
            return

        depths = depth_img[ys, xs] * self.depth_scale
        self.get_logger().info(
            f'lane px={len(xs)} raw_depth[min,max]=({float(depth_img[ys, xs].min()):.3f}, '
            f'{float(depth_img[ys, xs].max()):.3f}) scaled[min,max]=({float(depths.min()):.3f}, '
            f'{float(depths.max()):.3f})',
            throttle_duration_sec=2.0,
        )
        valid = (depths > self.depth_min) & (depths < self.depth_max)
        if not np.any(valid):
            self.get_logger().warn(
                f'모든 depth가 범위를 벗어났습니다. depth_scale={self.depth_scale}, '
                f'range=({self.depth_min}, {self.depth_max})',
                throttle_duration_sec=2.0,
            )
            self.pub_cloud.publish(self._make_pointcloud2(np.empty((0, 3), dtype=np.float32), header.stamp, 'base_link'))
            return

        if not np.all(valid):
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

        pts_base = self._sor_filter(pts_base, self.sor_k, self.sor_std_mul)
        pts_f32 = self._voxel_downsample(pts_base)
        if len(pts_f32) == 0:
            self.get_logger().warn('후처리 이후 남은 lane pointcloud 포인트가 없습니다.',
                                   throttle_duration_sec=2.0)
            self.pub_cloud.publish(self._make_pointcloud2(np.empty((0, 3), dtype=np.float32), header.stamp, 'base_link'))
            return

        self.get_logger().info(
            f'lane pointcloud publish: {len(pts_f32)} pts in base_link',
            throttle_duration_sec=2.0,
        )
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
    node = Yolo26LaneDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
