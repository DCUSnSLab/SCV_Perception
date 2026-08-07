#!/usr/bin/env python3
import os
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

from lane_detection.device_utils import move_model_to_device, resolve_device


class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')

        # ── 파라미터 ──────────────────────────────────────────────────────────
        default_model_path = os.path.expanduser('~/yolo26m_seg_best.pt')
        self.declare_parameter('model_path',   default_model_path)
        self.declare_parameter(
            'image_topic', '/front_right/front_right/color/image_raw')
        self.declare_parameter(
            'depth_topic',
            '/front_right/front_right/aligned_depth_to_color/image_raw')
        self.declare_parameter('device',       'cuda:0')  # 'cpu' 로 GPU 격리 가능
        self.declare_parameter('conf',         0.00065)  # YOLO-seg confidence threshold
        self.declare_parameter('depth_min',    0.1)   # m
        self.declare_parameter('depth_max',    10.0)  # m
        self.declare_parameter('voxel_size',   0.03)  # m
        # True: 차선 포인트를 지면(z=0)으로 투영, False: 뎁스 값 그대로 사용
        self.declare_parameter('ground_proj',  True)
        # SOR 파라미터
        self.declare_parameter('sor_k',        50)    # 이웃 포인트 수
        self.declare_parameter('sor_std_mul',  1.0)   # 표준편차 배수 (낮을수록 공격적 제거)
        self.declare_parameter('morph_kernel_width', 5)
        self.declare_parameter('morph_kernel_height', 9)

        model_path  = os.path.expanduser(
            self.get_parameter('model_path').get_parameter_value().string_value)
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        requested_device = self.get_parameter('device').get_parameter_value().string_value
        self.conf       = self.get_parameter('conf').get_parameter_value().double_value
        self.depth_min  = self.get_parameter('depth_min').get_parameter_value().double_value
        self.depth_max  = self.get_parameter('depth_max').get_parameter_value().double_value
        self.voxel_size = self.get_parameter('voxel_size').get_parameter_value().double_value
        self.ground_proj  = self.get_parameter('ground_proj').get_parameter_value().bool_value
        self.sor_k        = self.get_parameter('sor_k').get_parameter_value().integer_value
        self.sor_std_mul  = self.get_parameter('sor_std_mul').get_parameter_value().double_value
        morph_kernel_width = max(
            1, self.get_parameter('morph_kernel_width').get_parameter_value().integer_value)
        morph_kernel_height = max(
            1, self.get_parameter('morph_kernel_height').get_parameter_value().integer_value)
        self.morph_kernel = np.ones((morph_kernel_height, morph_kernel_width), dtype=np.uint8)

        # ── 내부 상태 ─────────────────────────────────────────────────────────
        self.bridge = CvBridge()
        self.K: np.ndarray | None = None  # 3×3 카메라 내부 행렬

        # ── TF2 ───────────────────────────────────────────────────────────────
        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ── YOLO-seg 모델 로드 ────────────────────────────────────────────────
        self.device, self.half = resolve_device(self.get_logger(), requested_device)
        self.get_logger().info(f'모델 로드 중: {model_path}')
        self.model = YOLO(model_path)
        self.device, self.half = move_model_to_device(
            self.get_logger(), self.model, self.device, self.half)
        self.get_logger().info(f'YOLO-seg 모델 로드 완료 (device={self.device})')

        # ── 구독 ──────────────────────────────────────────────────────────────
        self.create_subscription(
            CameraInfo, '/front_right/front_right/color/camera_info',
            self._camera_info_cb, 1)

        sub_color = Subscriber(self, Image, image_topic)
        sub_depth = Subscriber(self, Image, depth_topic)
        self.sync = ApproximateTimeSynchronizer(
            [sub_color, sub_depth], queue_size=2, slop=0.05)
        self.sync.registerCallback(self._callback)

        # ── TF 캐시 (static transform 캐싱) ──────────────────────────────────
        self._tf_cache: dict = {}  # frame_id → (R, T numpy arrays)

        # ── 발행 ──────────────────────────────────────────────────────────────
        self.pub_mask    = self.create_publisher(Image,       '/lane_detection/lane_mask',        1)
        self.pub_cloud   = self.create_publisher(PointCloud2, '/lane_detection/lane_pointcloud',  1)

        self.get_logger().info(
            f'구독: color={image_topic}, depth={depth_topic} | '
            f'conf={self.conf}, depth={self.depth_min}~{self.depth_max}m, '
            f'voxel={self.voxel_size}m, ground_proj={self.ground_proj}')

    # ──────────────────────────────────────────────────────────────────────────
    # 콜백
    # ──────────────────────────────────────────────────────────────────────────

    def _camera_info_cb(self, msg: CameraInfo):
        if self.K is None:
            self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            self.get_logger().info(
                f'카메라 내부 파라미터 수신 — '
                f'fx={self.K[0,0]:.1f}, fy={self.K[1,1]:.1f}, '
                f'cx={self.K[0,2]:.1f}, cy={self.K[1,2]:.1f}')

    def _callback(self, color_msg: Image, depth_msg: Image):
        if self.K is None:
            self.get_logger().warn(
                'camera_info 미수신, 프레임 스킵', throttle_duration_sec=2.0)
            return

        img_bgr   = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        depth_img = self.bridge.imgmsg_to_cv2(
            depth_msg, desired_encoding='passthrough').astype(np.float32)

        # 1) YOLO-seg 차선 마스크 추론
        lane_score = self._infer_lane_score(img_bgr)
        lane_mask = self._finalize_lane_mask(lane_score)

        # 2) 마스크 이미지 발행
        mask_msg = self.bridge.cv2_to_imgmsg(
            (lane_mask * 255).astype(np.uint8), encoding='mono8')
        mask_msg.header = color_msg.header
        self.pub_mask.publish(mask_msg)

        # 3) 포인트클라우드 발행
        self._publish_pointcloud(lane_mask, depth_img, color_msg.header)

    # ──────────────────────────────────────────────────────────────────────────
    # YOLO-seg 추론
    # ──────────────────────────────────────────────────────────────────────────

    def _infer_lane_score(self, img_bgr: np.ndarray) -> np.ndarray:
        """YOLO-seg 추론 → 원본 해상도 점수 맵 (float32 0~1)."""
        h, w = img_bgr.shape[:2]
        results = self.model.predict(
            img_bgr, verbose=False, conf=self.conf,
            half=self.half, device=self.device)

        result = results[0]
        if result.masks is None or result.masks.data is None or result.masks.data.shape[0] == 0:
            return np.zeros((h, w), dtype=np.float32)

        # 모든 인스턴스 마스크를 합친 뒤 원본 해상도로 복원한다.
        combined = result.masks.data.max(dim=0).values.cpu().numpy()  # (mask_h, mask_w)
        resized  = cv2.resize(combined, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        return np.clip(resized, 0.0, 1.0)

    def _finalize_lane_mask(self, lane_score: np.ndarray) -> np.ndarray:
        """현재 프레임 마스크를 morphology close로 다듬는다."""
        lane_mask = (lane_score > 0.0).astype(np.uint8)
        lane_mask = cv2.morphologyEx(lane_mask, cv2.MORPH_CLOSE, self.morph_kernel)
        return lane_mask

    # ──────────────────────────────────────────────────────────────────────────
    # 포인트클라우드 생성 · 발행
    # ──────────────────────────────────────────────────────────────────────────

    def _publish_pointcloud(self, lane_mask: np.ndarray,
                            depth_img: np.ndarray, header) -> None:
        # ① TF 조회: 카메라 광학 프레임 → base_link
        #    static transform이므로 첫 프레임에만 조회 후 캐싱
        frame_id = header.frame_id
        if frame_id not in self._tf_cache:
            try:
                tf_stamped = self.tf_buffer.lookup_transform(
                    'base_link', frame_id, rclpy.time.Time())
            except TransformException as e:
                self.get_logger().warn(
                    f'TF 조회 실패: {e}', throttle_duration_sec=5.0)
                return
            q  = tf_stamped.transform.rotation
            tv = tf_stamped.transform.translation
            qx, qy, qz, qw = q.x, q.y, q.z, q.w
            R = np.array([
                [1 - 2*(qy**2 + qz**2),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
                [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),   2*(qy*qz - qx*qw)],
                [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
            ], dtype=np.float32)
            T = np.array([tv.x, tv.y, tv.z], dtype=np.float32)
            self._tf_cache[frame_id] = (R, T)
            self.get_logger().info(f'TF 캐시 완료: base_link ← {frame_id}')
        R, T = self._tf_cache[frame_id]
        # ② 마스크를 뎁스 해상도에 맞게 리사이즈
        dh, dw = depth_img.shape[:2]
        if lane_mask.shape != (dh, dw):
            lane_mask = cv2.resize(
                lane_mask, (dw, dh), interpolation=cv2.INTER_NEAREST)

        # ③ 유효 픽셀 좌표 추출
        ys, xs = np.where(lane_mask > 0)
        if len(xs) == 0:
            return

        # ④ 깊이 추출 및 범위 필터 (RealSense: mm → m)
        depths = depth_img[ys, xs] * 1e-3
        valid  = (depths > self.depth_min) & (depths < self.depth_max)
        if not np.any(valid):
            return

        # ④-1 마스크 내 depth 구멍 보간
        #     invalid 픽셀 → 마스크 내 가장 가까운 valid 픽셀의 depth 사용
        #     ground_proj=True 이므로 z는 버리고 x,y 계산에만 쓰임 → 근사값으로 충분
        if not np.all(valid):
            valid_coords = np.stack([xs[valid], ys[valid]], axis=1)
            invalid_mask = ~valid
            invalid_coords = np.stack([xs[invalid_mask], ys[invalid_mask]], axis=1)
            tree = KDTree(valid_coords)
            _, nn_idx = tree.query(invalid_coords, workers=-1)
            depths[invalid_mask] = depths[valid][nn_idx]
            valid = np.ones(len(xs), dtype=bool)  # 모두 유효

        xs, ys, depths = xs[valid], ys[valid], depths[valid]

        # ⑤ 픽셀 → 카메라 광학 좌표계 역투영 (핀홀 모델)
        #    카메라 광학 좌표계: X=오른쪽, Y=아래, Z=앞
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        pts_cam = np.stack([
            (xs - cx) * depths / fx,   # X_cam
            (ys - cy) * depths / fy,   # Y_cam
            depths,                    # Z_cam
        ], axis=-1).astype(np.float32)  # (N, 3) — float32로 통일 (코스트맵 0.1m 해상도에 충분)

        # RealSense 180° 소프트웨어 회전 보정
        # URDF TF에 roll=π가 이미 있는데 RealSense config에서도 180° 회전하면
        # X, Y가 이중으로 반전됨 → 역보정
        pts_cam[:, 0] *= -1
        pts_cam[:, 1] *= -1

        # ⑥ TF 변환: 카메라 → base_link (캐싱된 R, T 사용)
        pts_base = (R @ pts_cam.T).T + T  # (N, 3)

        # ⑦ 지면 투영 옵션
        #    차선은 지면 위에 있으므로 z=0 강제 → 뎁스 노이즈 제거 효과
        #    실제 z 값을 보려면 launch에서 ground_proj:=false 로 설정
        if self.ground_proj:
            pts_base[:, 2] = 0.0

        # ⑧ SOR — 뎁스 카메라 노이즈 제거
        pts_base = self._sor_filter(pts_base, self.sor_k, self.sor_std_mul)

        # ⑨ 복셀 다운샘플링
        pts_f32 = self._voxel_downsample(pts_base)
        if len(pts_f32) == 0:
            return

        # ⑪ PointCloud2 발행
        self.pub_cloud.publish(
            self._make_pointcloud2(pts_f32, header.stamp, 'base_link'))

    # ──────────────────────────────────────────────────────────────────────────
    # 유틸
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _sor_filter(pts: np.ndarray, k: int = 50, std_mul: float = 1.0) -> np.ndarray:
        """Statistical Outlier Removal — k-최근접 이웃 평균 거리 기반 노이즈 제거"""
        n = len(pts)
        if n < k + 1:
            return pts

        tree = KDTree(pts)
        # 자기 자신 포함 k+1개 조회 → [1:] 로 자기 제외
        dists, _ = tree.query(pts, k=k + 1, workers=-1)
        mean_dists = dists[:, 1:].mean(axis=1)   # (N,) 각 포인트의 이웃 평균 거리

        threshold = mean_dists.mean() + std_mul * mean_dists.std()
        return pts[mean_dists <= threshold]

    def _voxel_downsample(self, pts: np.ndarray) -> np.ndarray:
        """복셀 그리드 다운샘플링 — 각 복셀에서 첫 번째 포인트 선택"""
        if len(pts) == 0:
            return pts
        idx  = np.floor(pts / self.voxel_size).astype(np.int32)
        keys = idx[:, 0] * 1_000_003 + idx[:, 1] * 1_009 + idx[:, 2]
        _, first = np.unique(keys, return_index=True)
        return pts[first]

    @staticmethod
    def _make_pointcloud2(pts: np.ndarray, stamp, frame_id: str) -> PointCloud2:
        """(N, 3) float32 numpy 배열 → PointCloud2 메시지"""
        cloud               = PointCloud2()
        cloud.header.stamp  = stamp
        cloud.header.frame_id = frame_id
        cloud.height        = 1
        cloud.width         = len(pts)
        cloud.fields        = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        cloud.is_bigendian  = False
        cloud.point_step    = 12
        cloud.row_step      = 12 * len(pts)
        cloud.is_dense      = True
        cloud.data          = pts.tobytes()
        return cloud


# ─────────────────────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
