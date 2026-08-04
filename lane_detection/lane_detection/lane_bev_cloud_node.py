#!/usr/bin/env python3
"""
IPM BEV + PointCloud 노드

lane_mask (mono8) + camera_info → cv2.remap (IPM) → BEV 이미지
BEV white 픽셀 → PointCloud2 (base_link, 3cm 격자)

remap map 계산:
  BEV 픽셀 → 월드 XY → 카메라 좌표 → forward distortion 적용 → 원본 이미지 픽셀
  K/D 를 remap map 에 통합 (undistort 별도 불필요)
"""
import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from cv_bridge import CvBridge


class LaneBevCloudNode(Node):
    def __init__(self):
        super().__init__('lane_bev_cloud_node')

        # ── 파라미터 선언 ──────────────────────────────────────────────
        self.declare_parameter('mask_topic',        '/lane_detection/lane_mask')
        self.declare_parameter('camera_info_topic', '/ardu_cam_link/camera_info')
        self.declare_parameter('cloud_topic',       '/lane_detection/bev_cloud')
        self.declare_parameter('bev_image_topic',   '/lane_detection/bev_image')

        self.declare_parameter('camera_height_m',   1.2)
        self.declare_parameter('pitch_deg',         15.0)
        self.declare_parameter('roll_deg',          0.0)
        self.declare_parameter('yaw_deg',           0.0)
        self.declare_parameter('camera_x_offset_m', 0.0)
        self.declare_parameter('camera_y_offset_m', 0.0)

        self.declare_parameter('forward_m',         15.0)
        self.declare_parameter('backward_m',         2.0)
        self.declare_parameter('left_m',             4.0)
        self.declare_parameter('right_m',            4.0)
        self.declare_parameter('meters_per_pixel',   0.03)

        self.declare_parameter('output_frame',      'base_link')
        self.declare_parameter('lateral_scale',     1.0)

        # ── 파라미터 로드 ──────────────────────────────────────────────
        self.mask_topic        = self.get_parameter('mask_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.cloud_topic       = self.get_parameter('cloud_topic').value
        self.bev_image_topic   = self.get_parameter('bev_image_topic').value

        self.camera_height_m   = float(self.get_parameter('camera_height_m').value)
        self.pitch_deg         = float(self.get_parameter('pitch_deg').value)
        self.roll_deg          = float(self.get_parameter('roll_deg').value)
        self.yaw_deg           = float(self.get_parameter('yaw_deg').value)
        self.camera_x_offset_m = float(self.get_parameter('camera_x_offset_m').value)
        self.camera_y_offset_m = float(self.get_parameter('camera_y_offset_m').value)

        self.forward_m         = float(self.get_parameter('forward_m').value)
        self.backward_m        = float(self.get_parameter('backward_m').value)
        self.left_m            = float(self.get_parameter('left_m').value)
        self.right_m           = float(self.get_parameter('right_m').value)
        self.mpp               = float(self.get_parameter('meters_per_pixel').value)

        self.output_frame      = self.get_parameter('output_frame').value
        self.lateral_scale     = float(self.get_parameter('lateral_scale').value)

        # ── BEV 출력 해상도 계산 ───────────────────────────────────────
        self.bev_w = int(round((self.left_m  + self.right_m)   / self.mpp))
        self.bev_h = int(round((self.forward_m + self.backward_m) / self.mpp))
        # BEV 이미지에서 차량(카메라 정사영) 위치
        self.bev_cx = int(round(self.left_m   / self.mpp))   # 열
        self.bev_cy = int(round(self.forward_m / self.mpp))  # 행 (전방이 위)

        # ── 내부 상태 ──────────────────────────────────────────────────
        self.bridge  = CvBridge()
        self.map1    = None   # cv2.remap map (float32)
        self.map2    = None
        self._last_time: float | None = None

        # ── 카메라 외부 파라미터 회전행렬 ─────────────────────────────
        # R0: 카메라 광학 좌표계(X=right, Y=down, Z=forward) →
        #     base_link 좌표계(X=forward, Y=left, Z=up) 기저 변환
        #   cam_X(right) → -base_Y  /  cam_Y(down) → -base_Z  /  cam_Z(fwd) → +base_X
        R0 = np.array([[0,  0,  1],
                       [-1, 0,  0],
                       [0,  -1, 0]], dtype=np.float64)
        R_extra = self._build_rotation(self.roll_deg, self.pitch_deg, self.yaw_deg)
        # R_cam: camera → world (base_link)
        self.R_cam = R_extra @ R0
        # 카메라 위치 (base_link 기준)
        self.t_cam = np.array([
            self.camera_x_offset_m,
            self.camera_y_offset_m,
            self.camera_height_m,
        ], dtype=np.float64)

        # ── 구독 / 퍼블리셔 ───────────────────────────────────────────
        self.create_subscription(CameraInfo, self.camera_info_topic, self._camera_info_cb, 1)
        self.create_subscription(Image,      self.mask_topic,        self._mask_cb,        1)
        self.pub_cloud = self.create_publisher(PointCloud2, self.cloud_topic,     1)
        self.pub_bev   = self.create_publisher(Image,       self.bev_image_topic, 1)

        self.get_logger().info(
            f'BEV 크기: {self.bev_w}×{self.bev_h} px  '
            f'({self.left_m+self.right_m:.1f}m × {self.forward_m+self.backward_m:.1f}m)  '
            f'mpp={self.mpp*100:.0f}cm/px'
        )
        self.get_logger().info(f'camera_info 대기 중: {self.camera_info_topic}')

    # ── camera_info 수신 → remap map 계산 (1회) ───────────────────────
    def _camera_info_cb(self, msg: CameraInfo):
        if self.map1 is not None:
            return

        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        D = np.array(msg.d, dtype=np.float64)

        self.get_logger().info(
            f'camera_info 수신 — fx={K[0,0]:.1f} fy={K[1,1]:.1f} '
            f'cx={K[0,2]:.1f} cy={K[1,2]:.1f} | remap map 계산 중...'
        )

        t0 = time.perf_counter()
        self.map1, self.map2 = self._build_remap_maps(K, D)
        dt = (time.perf_counter() - t0) * 1000

        self.get_logger().info(f'remap map 계산 완료 ({dt:.1f} ms)')

    # ── IPM remap map 계산 ────────────────────────────────────────────
    def _build_remap_maps(
        self,
        K: np.ndarray,
        D: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        BEV 픽셀 (u_bev, v_bev) 각각에 대해
        원본 이미지에서 샘플링해야 할 픽셀 좌표 (map1, map2) 를 계산한다.

        BEV 좌표계:
          - u 축: 오른쪽이 +  (left_m 이 u=0)
          - v 축: 위쪽이 +  (forward_m 이 v=0, 차량 후방이 v=bev_h)
          - 원점: 카메라 정사영 지점 (bev_cx, bev_cy)
        """
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # BEV 픽셀 그리드
        u_bev = np.arange(self.bev_w, dtype=np.float64)
        v_bev = np.arange(self.bev_h, dtype=np.float64)
        uu, vv = np.meshgrid(u_bev, v_bev)  # (bev_h, bev_w)

        # BEV 픽셀 → 월드 좌표 (base_link, Z=0 ground plane)
        # X: 전방 (+앞), Y: 좌측 (+왼쪽), Z=0
        Xw = (self.bev_cy - vv) * self.mpp   # v=0 이 전방
        Yw = (self.bev_cx - uu) * self.mpp   # u=0 이 좌측

        # 월드 → 카메라 좌표
        # p_cam = R_cam^T @ (p_world - t_cam)
        # p_world = [Xw, Yw, 0]
        pts_world = np.stack([Xw, Yw, np.zeros_like(Xw)], axis=-1)  # (H,W,3)
        pts_shifted = pts_world - self.t_cam                          # (H,W,3)
        R_inv = self.R_cam.T
        pts_cam = (R_inv @ pts_shifted.reshape(-1, 3).T).T.reshape(self.bev_h, self.bev_w, 3)

        Xc = pts_cam[..., 0]
        Yc = pts_cam[..., 1]
        Zc = pts_cam[..., 2]

        # Zc <= 0 이면 카메라 뒤쪽 → 유효하지 않은 픽셀
        valid = Zc > 0.0

        # 정규화 좌표
        xn = np.where(valid, Xc / Zc, 0.0)
        yn = np.where(valid, Yc / Zc, 0.0)

        # forward distortion 적용 (왜곡 → 원본 이미지 픽셀)
        r2 = xn * xn + yn * yn
        if D.size >= 5:
            k1, k2, p1, p2 = D[0], D[1], D[2], D[3]
            k3 = D[4] if D.size > 4 else 0.0
        else:
            k1 = k2 = p1 = p2 = k3 = 0.0

        radial = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3
        xd = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn**2)
        yd = yn * radial + p1 * (r2 + 2.0 * yn**2) + 2.0 * p2 * xn * yn

        # 이미지 픽셀 좌표
        u_img = (fx * xd + cx).astype(np.float32)
        v_img = (fy * yd + cy).astype(np.float32)

        # 유효하지 않은 픽셀은 -1 (cv2.remap BORDER_CONSTANT → 검정)
        u_img[~valid] = -1.0
        v_img[~valid] = -1.0

        return u_img, v_img

    # ── 마스크 수신 → remap → BEV → PointCloud ────────────────────────
    def _mask_cb(self, msg: Image):
        if self.map1 is None:
            self.get_logger().warn(
                'camera_info 미수신, 마스크 스킵', throttle_duration_sec=3.0
            )
            return

        t0 = time.perf_counter()

        mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        t1 = time.perf_counter()

        # IPM remap
        bev = cv2.remap(
            mask, self.map1, self.map2,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        t2 = time.perf_counter()

        # BEV 이미지 퍼블리시 (디버그)
        bev_msg = self.bridge.cv2_to_imgmsg(bev, encoding='mono8')
        bev_msg.header = msg.header
        bev_msg.header.frame_id = self.output_frame
        self.pub_bev.publish(bev_msg)

        # BEV white 픽셀 → PointCloud2
        cloud_msg = self._bev_to_pointcloud(bev, msg.header.stamp)
        self.pub_cloud.publish(cloud_msg)
        t3 = time.perf_counter()

        now = time.perf_counter()
        if self._last_time is not None:
            dt_interval = (now - self._last_time) * 1000
            hz = 1000.0 / dt_interval if dt_interval > 0 else 0.0
        else:
            dt_interval = hz = 0.0
        self._last_time = now

        self.get_logger().info(
            f'interval={dt_interval:.1f}ms ({hz:.1f}Hz)  '
            f'decode={( t1-t0)*1000:.1f}ms  '
            f'remap={(  t2-t1)*1000:.1f}ms  '
            f'cloud={( t3-t2)*1000:.1f}ms  '
            f'total={( t3-t0)*1000:.1f}ms  '
            f'pts={int(np.count_nonzero(bev))}',
            throttle_duration_sec=2.0,
        )

    # ── BEV 이미지 → PointCloud2 (base_link, Z=0) ─────────────────────
    def _bev_to_pointcloud(self, bev: np.ndarray, stamp) -> PointCloud2:
        rows, cols = np.where(bev > 0)

        # BEV 픽셀 → 월드 XY (lateral_scale로 횡방향 확장)
        Xw = (self.bev_cy - rows.astype(np.float32)) * self.mpp
        Yw = (self.bev_cx - cols.astype(np.float32)) * self.mpp * self.lateral_scale
        Zw = np.zeros(len(rows), dtype=np.float32)

        # base_link 오프셋 적용
        Xw += self.camera_x_offset_m
        Yw += self.camera_y_offset_m

        pts = np.stack([Xw, Yw, Zw], axis=-1).astype(np.float32)

        cloud = PointCloud2()
        cloud.header.stamp    = stamp
        cloud.header.frame_id = self.output_frame
        cloud.height    = 1
        cloud.width     = len(pts)
        cloud.fields    = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        cloud.is_bigendian = False
        cloud.point_step   = 12
        cloud.row_step     = 12 * len(pts)
        cloud.is_dense     = True
        cloud.data         = pts.tobytes()
        return cloud

    # ── 회전행렬 (roll/pitch/yaw → R) ────────────────────────────────
    @staticmethod
    def _build_rotation(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
        r = np.deg2rad(roll_deg)
        p = np.deg2rad(pitch_deg)
        y = np.deg2rad(yaw_deg)
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]], dtype=np.float64)
        Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]], dtype=np.float64)
        Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]], dtype=np.float64)
        return Rz @ Ry @ Rx


def main(args=None):
    rclpy.init(args=args)
    node = LaneBevCloudNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
