#!/usr/bin/env python3
import os
import time
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

# 레인별 고정 색상 (BGR)
_LANE_COLORS = [
    (0,   255,   0),
    (255,   0,   0),
    (0,   0,   255),
    (0,   255, 255),
    (255,   0, 255),
    (255, 255,   0),
    (128, 255,   0),
    (0,  128, 255),
]


class Yolo26OverlayNode(Node):
    def __init__(self):
        super().__init__('lane_detection_overlay_node')

        default_model = os.path.expanduser('~/yolo26m_seg_best.pt')
        self.declare_parameter('model_path',           default_model)
        self.declare_parameter(
            'image_topic', '/front_right/front_right/color/image_raw')
        self.declare_parameter('output_topic',         '/lane_detection/overlay')
        self.declare_parameter('conf',                 0.0007)
        self.declare_parameter('imgsz',                640)
        self.declare_parameter('max_det',              64)
        self.declare_parameter('min_mask_area_ratio',  0.0002)
        self.declare_parameter('min_bottom_y_ratio',   0.45)
        self.declare_parameter('min_height_ratio',     0.08)
        self.declare_parameter('max_lane_instances',   8)
        self.declare_parameter('overlay_alpha',        0.45)

        self.model_path          = os.path.expanduser(self.get_parameter('model_path').value)
        self.image_topic         = self.get_parameter('image_topic').value
        self.output_topic        = self.get_parameter('output_topic').value
        self.conf                = float(self.get_parameter('conf').value)
        self.imgsz               = int(self.get_parameter('imgsz').value)
        self.max_det             = int(self.get_parameter('max_det').value)
        self.min_mask_area_ratio = float(self.get_parameter('min_mask_area_ratio').value)
        self.min_bottom_y_ratio  = float(self.get_parameter('min_bottom_y_ratio').value)
        self.min_height_ratio    = float(self.get_parameter('min_height_ratio').value)
        self.max_lane_instances  = int(self.get_parameter('max_lane_instances').value)
        self.overlay_alpha       = float(self.get_parameter('overlay_alpha').value)

        self.bridge = CvBridge()
        self._last_time: float | None = None

        self.get_logger().info(f'모델 로드 중: {self.model_path}')
        self.model = YOLO(self.model_path)
        self.model.to('cuda')
        self.get_logger().info('모델 로드 완료')

        self.create_subscription(Image, self.image_topic, self._callback, 1)
        self.pub_overlay = self.create_publisher(Image, self.output_topic, 1)
        self.pub_mask    = self.create_publisher(Image, '/lane_detection/lane_mask', 1)

        self.get_logger().info(
            f'구독: {self.image_topic} | 퍼블리시: {self.output_topic} | '
            f'conf={self.conf} imgsz={self.imgsz} alpha={self.overlay_alpha}'
        )

    def _callback(self, msg: Image):
        t0 = time.perf_counter()

        if self._last_time is not None:
            dt = t0 - self._last_time
            if dt > 0.0:
                self.get_logger().info(
                    f'interval: {dt * 1000:.1f} ms ({1.0 / dt:.1f} Hz)',
                    throttle_duration_sec=2.0
                )
        self._last_time = t0

        t1 = time.perf_counter()
        img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        t2 = time.perf_counter()

        overlay, lane_mask = self._infer_and_overlay(img_bgr)
        t3 = time.perf_counter()

        out_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
        out_msg.header = msg.header
        self.pub_overlay.publish(out_msg)

        mask_msg = self.bridge.cv2_to_imgmsg(lane_mask, encoding='mono8')
        mask_msg.header = msg.header
        self.pub_mask.publish(mask_msg)
        t4 = time.perf_counter()

        self.get_logger().info(
            f'decode={( t2-t1)*1000:.1f}ms  '
            f'infer={(  t3-t2)*1000:.1f}ms  '
            f'encode={(  t4-t3)*1000:.1f}ms  '
            f'total={(   t4-t0)*1000:.1f}ms',
            throttle_duration_sec=2.0
        )

    def _infer_and_overlay(self, img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = img_bgr.shape[:2]
        empty_mask = np.zeros((h, w), dtype=np.uint8)

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
            return overlay, empty_mask

        boxes = result.boxes
        masks = result.masks.data
        candidates = []

        for idx in range(masks.shape[0]):
            mask = masks[idx].detach().cpu().numpy()
            if mask.ndim != 2:
                continue
            mask = (mask > 0).astype(np.uint8)
            ys, xs = np.where(mask > 0)
            if ys.size == 0:
                continue

            area_ratio   = float(mask.sum()) / float(h * w)
            y_min, y_max = int(ys.min()), int(ys.max())
            x_min, x_max = int(xs.min()), int(xs.max())
            height_ratio = float(y_max - y_min + 1) / float(h)
            bottom_ratio = float(y_max) / float(h)
            width_ratio  = float(x_max - x_min + 1) / float(w)
            box_conf     = float(boxes.conf[idx].item()) if boxes is not None and boxes.conf is not None else 0.0

            geometry_ok = (
                area_ratio >= self.min_mask_area_ratio and
                (bottom_ratio >= self.min_bottom_y_ratio or height_ratio >= self.min_height_ratio)
            )
            score = box_conf + 0.15 * bottom_ratio + 0.10 * height_ratio - 0.05 * width_ratio
            candidates.append((score, geometry_ok, mask))

        if not candidates:
            return overlay, empty_mask

        candidates.sort(key=lambda c: c[0], reverse=True)
        selected = [c for c in candidates if c[1]]
        if not selected:
            selected = candidates[: min(2, len(candidates))]
        else:
            selected = selected[: self.max_lane_instances]

        lane_mask = np.zeros((h, w), dtype=np.uint8)
        color_layer = img_bgr.copy()
        for rank, (_, _, mask) in enumerate(selected):
            color = _LANE_COLORS[rank % len(_LANE_COLORS)]
            color_layer[mask > 0] = color
            lane_mask = np.maximum(lane_mask, mask * 255)

        cv2.addWeighted(color_layer, self.overlay_alpha,
                        overlay,     1.0 - self.overlay_alpha, 0, overlay)
        return overlay, lane_mask


def main(args=None):
    rclpy.init(args=args)
    node = Yolo26OverlayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
