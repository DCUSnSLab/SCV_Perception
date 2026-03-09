#!/usr/bin/env python3
import sys
import os

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
import tf2_ros
from tf2_ros import TransformException

import torch
import numpy as np

torch.backends.cudnn.benchmark = True  # 고정 입력 크기에서 cuDNN 최적 커널 자동 선택
import cv2

from lib.models import get_net
from lib.config import cfg

# GPU 정규화용 상수 (초기화 시 device로 이동)
_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

def preprocess(img_bgr, device, img_size=640):
    h0, w0 = img_bgr.shape[:2]
    r = img_size / max(h0, w0)
    new_h, new_w = int(h0 * r), int(w0 * r)
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dh, dw = img_size - new_h, img_size - new_w
    top, left = dh // 2, dw // 2
    padded = cv2.copyMakeBorder(resized, top, dh - top, left, dw - left,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    # CPU: numpy → uint8 tensor (zero-copy), GPU로 전송 후 정규화
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device, non_blocking=True).float().div_(255.0)
    tensor.sub_(_MEAN.to(device)).div_(_STD.to(device))
    return tensor, (h0, w0), (new_h / h0, new_w / w0), (left, top)

def postprocess_mask(out, shapes, img_size=640):
    h0, w0 = shapes[0]
    ry, rx = shapes[2]
    pad_x, pad_y = shapes[3]
    _, _, H, W = out.shape
    cropped = out[:, :, pad_y:H - pad_y, pad_x:W - pad_x]
    upsampled = torch.nn.functional.interpolate(
        cropped, scale_factor=(1 / ry, 1 / rx), mode='bilinear', align_corners=False
    )
    mask = upsampled.argmax(1).squeeze().cpu().numpy().astype(np.uint8)
    return cv2.resize(mask, (w0, h0), interpolation=cv2.INTER_NEAREST)


class YolopNode(Node):
    def __init__(self):
        super().__init__('yolop_node')

        # 파라미터
        self.declare_parameter('model_path', '')
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/camera/depth/image_rect_raw')
        self.declare_parameter('input_size', 640)

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.input_size = self.get_parameter('input_size').get_parameter_value().integer_value

        if not model_path:
            model_path = os.path.join(os.path.dirname(__file__), '..', '..', 'weights', 'yolop_aug_best_compat.pth')

        self.bridge = CvBridge()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 모델 로드
        self.get_logger().info(f'모델 로드 중: {model_path}')
        self.model = get_net(cfg)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()
        self.get_logger().info('YOLOP 모델 로드 완료')

        # 카메라 파라미터 (camera_info로 자동 업데이트)
        self.fx = self.fy = self.cx = self.cy = None
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 1)

        # 복셀 다운샘플링 크기 (m)
        self.voxel_size = 0.05

        # TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 발행
        self.pub_lane = self.create_publisher(Image, '/lane_detection/lane_mask', 1)
        self.pub_cloud = self.create_publisher(PointCloud2, '/lane_detection/lane_pointcloud', 1)

        # 컬러 + 뎁스 시간 동기화 구독
        self.sub_color = Subscriber(self, Image, image_topic)
        self.sub_depth = Subscriber(self, Image, depth_topic)
        self.sync = ApproximateTimeSynchronizer(
            [self.sub_color, self.sub_depth], queue_size=5, slop=0.05
        )
        self.sync.registerCallback(self.callback)

        self.get_logger().info(f'구독: {image_topic}, {depth_topic}')

    def camera_info_callback(self, msg):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def callback(self, color_msg, depth_msg):
        if self.fx is None:
            return  # camera_info 아직 못 받은 경우
        import time
        t0 = time.time()
        img_bgr = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
        t_color = time.time()
        depth_img = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough').astype(np.float32)
        t_depth = time.time()

        # 전처리 (GPU 정규화 포함)
        tensor, shapes_hw, ratio, pad = preprocess(img_bgr, self.device, self.input_size)
        t_pre = time.time()
        t1 = t_pre

        with torch.inference_mode():
            _, seg_out, lane_out = self.model(tensor)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        t2 = time.time()

        # 후처리 (패딩 제거 + 원본 크기 복원)
        shapes = (shapes_hw, None, ratio, pad)
        drive_mask = postprocess_mask(seg_out, shapes, self.input_size)
        lane_mask = postprocess_mask(lane_out, shapes, self.input_size)
        t3 = time.time()

        stamp = color_msg.header.stamp
        frame_id = color_msg.header.frame_id

        # 마스크 발행
        lane_msg = self.bridge.cv2_to_imgmsg((lane_mask * 255).astype(np.uint8), encoding='mono8')
        lane_msg.header.stamp = stamp
        lane_msg.header.frame_id = frame_id
        self.pub_lane.publish(lane_msg)
        t4 = time.time()

        # 차선 포인트클라우드 생성 (뎁스 + 마스크 융합)
        self._publish_pointcloud(lane_mask, drive_mask, depth_img, stamp, color_msg.header.frame_id)
        t5 = time.time()
        self.get_logger().info(
            f'cv={1000*(t_pre-t_depth):.0f}ms  infer={1000*(t2-t1):.0f}ms  '
            f'post={1000*(t3-t2):.0f}ms  pub={1000*(t4-t3):.0f}ms  cloud={1000*(t5-t4):.0f}ms  '
            f'total={1000*(t5-t0):.0f}ms',
            throttle_duration_sec=2.0)

    def _publish_pointcloud(self, lane_mask, drive_mask, depth_img, stamp, src_frame):
        # TF: src_frame → base_link 변환 조회
        try:
            tf = self.tf_buffer.lookup_transform(
                'base_link', src_frame, rclpy.time.Time())
        except TransformException as e:
            self.get_logger().warn(f'TF lookup failed: {e}', throttle_duration_sec=5.0)
            return

        # depth 해상도에 맞게 마스크 리사이즈
        dh, dw = depth_img.shape[:2]
        lane_resized = cv2.resize(lane_mask, (dw, dh), interpolation=cv2.INTER_NEAREST)
        drive_resized = cv2.resize(drive_mask, (dw, dh), interpolation=cv2.INTER_NEAREST)

        # 비차선 요소 제거: drivable area를 팽창시켜 경계 차선도 포함
        kernel = np.ones((15, 15), np.uint8)
        drive_dilated = cv2.dilate(drive_resized.astype(np.uint8), kernel)
        filtered_lane = (lane_resized > 0) & (drive_dilated > 0)

        fx, fy = self.fx, self.fy
        cx, cy = self.cx, self.cy
        depth_scale = 0.001  # mm → m

        ys, xs = np.where(filtered_lane)
        if len(xs) == 0:
            return

        depths = depth_img[ys, xs] * depth_scale
        valid = (depths > 0.1) & (depths < 10.0)
        xs, ys, depths = xs[valid], ys[valid], depths[valid]

        if len(xs) == 0:
            return

        # optical frame 기준 3D 좌표 (X=오른쪽, Y=아래, Z=앞)
        X = (xs - cx) * depths / fx
        Y = (ys - cy) * depths / fy
        Z = depths
        pts_cam = np.stack([X, Y, Z], axis=-1).astype(np.float64)  # (N, 3)

        # TF 변환 행렬 구성 (쿼터니언 → 회전행렬)
        q = tf.transform.rotation
        t = tf.transform.translation
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        R = np.array([
            [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
            [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
        ])
        T = np.array([t.x, t.y, t.z])

        # base_link 좌표로 변환
        pts_base = (R @ pts_cam.T).T + T  # (N, 3)
        pts_base[:, 1] *= -1  # 카메라 180° 뒤집힘: 드라이버가 이미지는 보정했지만 TF는 물리 방향 기준이라 Y 반전
        pts_base[:, 2] = 0.0  # 차선은 항상 지면(z=0)에 있으므로 강제 고정

        # 복셀 다운샘플링으로 노이즈 감소 및 포인트 안정화
        points = self._voxel_downsample(pts_base.astype(np.float32))

        # PointCloud2 메시지 생성
        cloud = PointCloud2()
        cloud.header.stamp = stamp
        cloud.header.frame_id = 'base_link'
        cloud.height = 1
        cloud.width = len(points)
        cloud.fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        ]
        cloud.is_bigendian = False
        cloud.point_step = 12
        cloud.row_step = 12 * len(points)
        cloud.is_dense = True
        cloud.data = points.tobytes()
        self.pub_cloud.publish(cloud)


    def _voxel_downsample(self, pts: np.ndarray) -> np.ndarray:
        """복셀 그리드 다운샘플링 - 각 복셀에서 첫 번째 포인트만 선택"""
        if len(pts) == 0:
            return pts
        voxel_idx = np.floor(pts / self.voxel_size).astype(np.int32)
        keys = voxel_idx[:, 0] * 1000003 + voxel_idx[:, 1] * 1009 + voxel_idx[:, 2]
        _, unique_idx = np.unique(keys, return_index=True)
        return pts[unique_idx]


def main(args=None):
    rclpy.init(args=args)
    node = YolopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
