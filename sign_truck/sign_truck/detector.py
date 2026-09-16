from __future__ import annotations

import json
import time
from pathlib import Path

from cv_bridge import CvBridge
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Int32, String
import torch
from ultralytics import YOLO

from .logic import GREEN, RED, assign_lane_states, state_at_anchor


COLORS = {
    GREEN: (60, 220, 80),
    RED: (60, 60, 235),
}


class SignTruckDetector(Node):
    def __init__(self) -> None:
        super().__init__('sign_truck_detector')

        self.model_path = Path(self.declare_parameter('model_path', 'best.pt').value).expanduser()
        self.image_topic = str(self.declare_parameter('image_topic', '/panorama/image_raw').value)
        self.annotated_topic = str(
            self.declare_parameter('annotated_topic', '/sign_truck/annotated').value
        )
        self.conf = float(self.declare_parameter('conf_threshold', 0.25).value)
        self.iou = float(self.declare_parameter('iou_threshold', 0.70).value)
        self.imgsz = int(self.declare_parameter('image_size', 1280).value)
        requested_device = str(self.declare_parameter('device', 'auto').value)
        self.device = (
            0 if requested_device == 'auto' and torch.cuda.is_available()
            else 'cpu' if requested_device == 'auto'
            else requested_device
        )
        self.roi_top = float(self.declare_parameter('roi_top_ratio', 0.40).value)
        self.roi_left = float(self.declare_parameter('roi_left_ratio', 0.30).value)
        self.roi_right = float(self.declare_parameter('roi_right_ratio', 0.70).value)
        self.left_anchor_ratio = float(
            self.declare_parameter('left_lane_anchor_ratio', 0.35).value
        )
        self.right_anchor_ratio = float(
            self.declare_parameter('right_lane_anchor_ratio', 0.65).value
        )
        self.ego_anchor_ratio = float(self.declare_parameter('ego_anchor_ratio', 0.50).value)
        self.match_max_ratio = float(self.declare_parameter('lane_match_max_ratio', 0.22).value)
        self.current_lane = str(self.declare_parameter('current_lane', 'auto').value).lower()

        if not self.model_path.is_file():
            raise FileNotFoundError(f'Model not found: {self.model_path}')
        if not (0.0 <= self.roi_left < self.roi_right <= 1.0 and 0.0 < self.roi_top <= 1.0):
            raise ValueError('ROI ratios must satisfy 0 <= left < right <= 1 and 0 < top <= 1')
        if self.current_lane not in {'auto', 'left', 'right'}:
            raise ValueError('current_lane must be auto, left, or right')

        self.bridge = CvBridge()
        self.model = YOLO(str(self.model_path))
        self.class_names = self.model.names
        wanted = {'green_sign', 'red_sign'}
        self.class_ids = [index for index, name in self.class_names.items() if name in wanted]
        found = {self.class_names[index] for index in self.class_ids}
        if found != wanted:
            raise RuntimeError(f'Model classes must include {sorted(wanted)}; found {self.class_names}')

        self.annotated_pub = self.create_publisher(Image, self.annotated_topic, 10)
        self.left_state_pub = self.create_publisher(String, '/sign_truck/left_lane_state', 10)
        self.right_state_pub = self.create_publisher(String, '/sign_truck/right_lane_state', 10)
        self.current_state_pub = self.create_publisher(String, '/sign_truck/current_lane_state', 10)
        self.debug_pub = self.create_publisher(String, '/sign_truck/debug', 10)
        self.create_subscription(Image, self.image_topic, self.image_callback, qos_profile_sensor_data)
        self.create_subscription(Int32, '/sign_truck/current_lane', self.current_lane_callback, 10)

        self.fps = 0.0
        self.get_logger().info(f'Model: {self.model_path}')
        self.get_logger().info(f'YOLO classes used: {self.class_ids} (truck excluded)')
        self.get_logger().info(
            f'Image: {self.image_topic}, ROI: top {self.roi_top:.0%}, '
            f'horizontal {self.roi_left:.0%}..{self.roi_right:.0%}'
        )

    def current_lane_callback(self, msg: Int32) -> None:
        if msg.data in (0, 1):
            self.current_lane = 'left' if msg.data == 0 else 'right'
        else:
            self.get_logger().warning('current_lane expects 0 (left) or 1 (right)')

    def image_callback(self, msg: Image) -> None:
        started = time.perf_counter()
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        height, width = frame.shape[:2]
        x0, x1 = int(width * self.roi_left), int(width * self.roi_right)
        y1 = int(height * self.roi_top)
        roi = frame[:y1, x0:x1]

        result = self.model.predict(
            roi,
            classes=self.class_ids,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )[0]

        visual_detections = []
        lane_detections: list[tuple[float, str, float]] = []
        if result.boxes is not None:
            for box, class_id, confidence in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.cls.int().cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
            ):
                bx0, by0, bx1, by1 = box
                name = self.class_names[class_id]
                lane_detections.append(((bx0 + bx1) / 2.0, name, float(confidence)))
                visual_detections.append((bx0, by0, bx1, by1, name, float(confidence)))

        roi_width = x1 - x0
        anchors = {
            'left': roi_width * self.left_anchor_ratio,
            'right': roi_width * self.right_anchor_ratio,
        }
        max_distance = roi_width * self.match_max_ratio
        lane_states = assign_lane_states(lane_detections, anchors, max_distance)
        if self.current_lane == 'auto':
            ego_x = roi_width * self.ego_anchor_ratio
            current_state = state_at_anchor(lane_detections, ego_x, max_distance)
        else:
            ego_x = anchors[self.current_lane]
            current_state = lane_states[self.current_lane]

        elapsed = time.perf_counter() - started
        instant_fps = 1.0 / max(elapsed, 1e-9)
        self.fps = instant_fps if not self.fps else 0.9 * self.fps + 0.1 * instant_fps
        self.left_state_pub.publish(String(data=lane_states['left']))
        self.right_state_pub.publish(String(data=lane_states['right']))
        self.current_state_pub.publish(String(data=current_state))
        self.debug_pub.publish(
            String(
                data=json.dumps(
                    {
                        'current_lane': self.current_lane,
                        'current_state': current_state,
                        'left_state': lane_states['left'],
                        'right_state': lane_states['right'],
                        'detections': len(lane_detections),
                        'fps': round(self.fps, 1),
                    }
                )
            )
        )

        if self.annotated_pub.get_subscription_count() > 0:
            annotated = self.draw_visualization(roi.copy(), visual_detections)
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            annotated_msg.header = msg.header
            self.annotated_pub.publish(annotated_msg)

    def draw_visualization(self, frame, detections):
        for bx0, by0, bx1, by1, name, _ in detections:
            state = GREEN if name == 'green_sign' else RED
            cv2.rectangle(
                frame, (int(bx0), int(by0)), (int(bx1), int(by1)), COLORS[state], 2
            )
        return frame

def main() -> None:
    rclpy.init()
    node = SignTruckDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
