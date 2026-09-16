from __future__ import annotations

import time
from pathlib import Path

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
import torch
from ultralytics import YOLO


UNKNOWN = 'UNKNOWN'
COLORS = {
    UNKNOWN: (150, 150, 150),
    'RED': (60, 60, 235),
    'YELLOW': (0, 215, 255),
    'GREEN': (60, 220, 80),
    'LEFT_ARROW': (220, 220, 60),
}


def state_from_class(class_name: str) -> str:
    name = class_name.strip().lower().replace('-', '_').replace(' ', '_')
    if not name.startswith('vehicular_') or 'etc' in name or 'down' in name:
        return UNKNOWN
    if 'green_arrow' in name or ('red' in name and 'green' in name):
        return 'LEFT_ARROW'
    if 'yellow' in name:
        return 'YELLOW'
    if 'green' in name:
        return 'GREEN'
    if 'red' in name:
        return 'RED'
    return UNKNOWN


class TrafficDetector(Node):
    def __init__(self) -> None:
        super().__init__('traffic_detector')

        model_path = Path(self.declare_parameter('model_path', 'best.pt').value).expanduser()
        image_topic = str(self.declare_parameter('image_topic', '/panorama/image_raw').value)
        self.conf = float(self.declare_parameter('conf_threshold', 0.20).value)
        self.imgsz = int(self.declare_parameter('image_size', 416).value)
        requested_device = str(self.declare_parameter('device', 'auto').value)
        self.device = (
            0 if requested_device == 'auto' and torch.cuda.is_available()
            else 'cpu' if requested_device == 'auto'
            else requested_device
        )

        if not model_path.is_file():
            raise FileNotFoundError(f'Model not found: {model_path}')

        self.bridge = CvBridge()
        self.model = YOLO(str(model_path))
        self.names = self.model.names
        self.class_ids = [
            class_id
            for class_id, name in self.names.items()
            if str(name).startswith('vehicular_') and 'down' not in str(name)
        ]

        input_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        output_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.state_pub = self.create_publisher(String, '/traffic/state', 10)
        self.annotated_pub = self.create_publisher(Image, '/traffic/annotated', output_qos)
        self.create_subscription(Image, image_topic, self.image_callback, input_qos)

        self.last_log = time.monotonic()
        self.frames = 0
        self.get_logger().info(
            f'Model: {model_path}, device: {self.device}, image_size: {self.imgsz}'
        )
        self.get_logger().info('ROI: top 40%, horizontal center 40%')

    def image_callback(self, msg: Image) -> None:
        started = time.perf_counter()
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        height, width = frame.shape[:2]
        roi = frame[:int(height * 0.40), int(width * 0.30):int(width * 0.70)]

        result = self.model.predict(
            roi,
            classes=self.class_ids,
            conf=self.conf,
            imgsz=self.imgsz,
            max_det=10,
            device=self.device,
            verbose=False,
        )[0]

        detections = []
        if result.boxes is not None:
            for box, class_id, confidence in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.cls.int().cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
            ):
                detections.append((*box, self.names[class_id], float(confidence)))

        resolved = [
            (confidence, state_from_class(name))
            for *_, name, confidence in detections
            if state_from_class(name) != UNKNOWN
        ]
        state = max(resolved, default=(0.0, UNKNOWN))[1]
        self.state_pub.publish(String(data=state))

        if self.annotated_pub.get_subscription_count() > 0:
            annotated = roi.copy()
            for x0, y0, x1, y1, name, _ in detections:
                cv2.rectangle(
                    annotated,
                    (int(x0), int(y0)),
                    (int(x1), int(y1)),
                    COLORS[state_from_class(name)],
                    2,
                )
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            annotated_msg.header = msg.header
            self.annotated_pub.publish(annotated_msg)

        self.frames += 1
        now = time.monotonic()
        if now - self.last_log >= 2.0:
            fps = self.frames / (now - self.last_log)
            latency_ms = (time.perf_counter() - started) * 1000.0
            self.get_logger().info(
                f'state={state} detections={len(detections)} fps={fps:.1f} latency={latency_ms:.0f}ms'
            )
            self.frames = 0
            self.last_log = now


def main() -> None:
    rclpy.init()
    node = TrafficDetector()
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
