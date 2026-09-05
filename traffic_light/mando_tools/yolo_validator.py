from __future__ import annotations

import time
from pathlib import Path
import sys
import traceback
from typing import Any

from .workspace_paths import default_image_topic
from .workspace_paths import default_model_path
from .workspace_paths import default_runtime_image_topic
from .workspace_paths import local_python_deps_path
from .workspace_paths import resolve_inference_device

# 워크스페이스 로컬 의존성을 우선 로드해 전역 ROS/시스템 환경을 건드리지 않고
# 런타임 import를 만족시킨다.
deps_path = local_python_deps_path()
if deps_path.exists():
    sys.path.insert(0, str(deps_path))

from cv_bridge import CvBridge
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from std_msgs.msg import String
from vision_msgs.msg import Detection2D
from vision_msgs.msg import Detection2DArray
from vision_msgs.msg import ObjectHypothesisWithPose


def _declare_param(node: Node, name: str, default_value: Any) -> Any:
    return node.declare_parameter(name, default_value).value


STATE_UNKNOWN = 0
STATE_RED = 1
STATE_YELLOW = 2
STATE_GREEN = 3
STATE_LEFT_ARROW = 4

STATE_LABELS = {
    STATE_UNKNOWN: 'UNKNOWN',
    STATE_RED: 'RED',
    STATE_YELLOW: 'YELLOW',
    STATE_GREEN: 'GREEN',
    STATE_LEFT_ARROW: 'LEFT ARROW',
}

STATE_COLORS = {
    STATE_UNKNOWN: (120, 120, 120),
    STATE_RED: (70, 70, 235),
    STATE_YELLOW: (0, 215, 255),
    STATE_GREEN: (70, 205, 95),
    STATE_LEFT_ARROW: (80, 220, 220),
}


class YoloValidatorNode(Node):
    def __init__(self) -> None:
        super().__init__('mando_yolo_validator')

        # launch/실행 시 튜닝 포인트를 모두 일반 ROS 파라미터로 노출한다.
        model_path = Path(_declare_param(self, 'model_path', str(default_model_path()))).expanduser()
        self.image_topic = str(
            _declare_param(self, 'image_topic', default_runtime_image_topic())
        )
        self.annotated_topic = str(
            _declare_param(self, 'annotated_topic', '/mando/yolo/annotated')
        )
        self.detections_topic = str(
            _declare_param(self, 'detections_topic', '/mando/yolo/detections')
        )
        self.state_topic = str(_declare_param(self, 'state_topic', '/tl/state'))
        self.state_label_topic = str(_declare_param(self, 'state_label_topic', '/tl/state_label'))
        self.state_reason_topic = str(_declare_param(self, 'state_reason_topic', '/tl/state_reason'))
        self.conf_threshold = float(_declare_param(self, 'conf_threshold', 0.25))
        self.iou_threshold = float(_declare_param(self, 'iou_threshold', 0.45))
        self.image_size = int(_declare_param(self, 'image_size', 640))
        self.max_detections = int(_declare_param(self, 'max_detections', 100))
        self.max_fps = float(_declare_param(self, 'max_fps', 15.0))
        requested_device = str(_declare_param(self, 'device', 'cuda:0'))
        self.device = resolve_inference_device(requested_device)
        self.publish_annotated = bool(_declare_param(self, 'publish_annotated', True))
        self.publish_detections = bool(_declare_param(self, 'publish_detections', True))
        self.publish_state = bool(_declare_param(self, 'publish_state', True))
        self.draw_labels = bool(_declare_param(self, 'draw_labels', True))
        self.draw_confidence = bool(_declare_param(self, 'draw_confidence', True))
        self.line_thickness = int(_declare_param(self, 'line_thickness', 2))
        self.font_scale = float(_declare_param(self, 'font_scale', 0.5))

        if not model_path.exists():
            raise FileNotFoundError(f'Model file not found: {model_path}')

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                f'ultralytics is not installed. Install local deps into {deps_path}.'
            ) from exc

        self.bridge = CvBridge()
        self.model = YOLO(str(model_path))
        self.class_names = self.model.names
        self.detector_classes = self._resolve_detector_classes()
        self.latest_msg: Image | None = None
        self.processing = False
        self.processed_frames = 0
        self.last_status_log = 0.0
        self._last_error_signature: tuple[str, str] | None = None
        self._last_error_log_time = 0.0

        # 가장 최근 이미지 1장만 유지해서 추론 지연이 생겨도 오래된 프레임이
        # 큐에 계속 쌓이지 않게 한다.
        self.create_subscription(
            Image,
            self.image_topic,
            self._image_callback,
            qos_profile_sensor_data,
        )

        image_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        detection_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        status_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.annotated_pub = self.create_publisher(Image, self.annotated_topic, image_qos)
        self.detections_pub = self.create_publisher(
            Detection2DArray,
            self.detections_topic,
            detection_qos,
        )
        self.state_pub = self.create_publisher(Int32, self.state_topic, status_qos)
        self.state_label_pub = self.create_publisher(String, self.state_label_topic, status_qos)
        self.state_reason_pub = self.create_publisher(String, self.state_reason_topic, status_qos)

        # 카메라 publish 주기와 모델 처리 속도를 분리하기 위해
        # 타이머 기반으로 추론 루프를 돌린다.
        timer_period = 1.0 / max(self.max_fps, 0.1)
        self.create_timer(timer_period, self._process_latest_frame)

        self.get_logger().info(f'Loaded model: {model_path}')
        self.get_logger().info(
            f'Inference device: {self.device} (requested: {requested_device})'
        )
        self.get_logger().info(f'Detector class filter: {self.detector_classes}')
        self.get_logger().info(f'Subscribing to image topic: {self.image_topic}')
        self.get_logger().info(f'Annotated image topic: {self.annotated_topic}')
        self.get_logger().info(f'Detections topic: {self.detections_topic}')
        self.get_logger().info(f'State topic: {self.state_topic}')

    def _image_callback(self, msg: Image) -> None:
        self.latest_msg = msg

    def _process_latest_frame(self) -> None:
        if self.processing or self.latest_msg is None:
            return

        # 추론이 도는 동안 subscriber가 더 최신 프레임으로 교체할 수 있게
        # 대기 중인 프레임을 바로 비운다.
        msg = self.latest_msg
        self.latest_msg = None
        self.processing = True

        started = time.perf_counter()
        stage = 'decode'
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            stage = 'predict'
            results = self.model.predict(
                source=frame,
                classes=self.detector_classes,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.image_size,
                max_det=self.max_detections,
                device=self.device,
                verbose=False,
            )[0]
            stage = 'interpret'
            inferred_state, inferred_reason = self._infer_state(results)

            # 후속 ROS 노드가 바로 쓸 수 있도록 검출 결과를 표준 메시지로 발행한다.
            if self.publish_detections:
                stage = 'publish_detections'
                self.detections_pub.publish(self._build_detection_array(results, msg))

            if self.publish_state:
                stage = 'publish_state'
                self.state_pub.publish(Int32(data=int(inferred_state)))
                self.state_label_pub.publish(String(data=STATE_LABELS[inferred_state]))
                self.state_reason_pub.publish(
                    String(data=f'{STATE_LABELS[inferred_state]} {inferred_reason}')
                )

            if self.publish_annotated:
                stage = 'annotate'
                annotated = self._draw_annotations(frame, results, inferred_state, inferred_reason)
                try:
                    stage = 'convert_annotated'
                    annotated_msg = self._numpy_to_image_msg(
                        annotated,
                        header=msg.header,
                        encoding='bgr8',
                    )
                    stage = 'publish_annotated'
                    self.annotated_pub.publish(annotated_msg)
                except Exception as exc:  # noqa: BLE001
                    self._log_processing_error(stage, exc, msg)

            self.processed_frames += 1
            elapsed = time.perf_counter() - started
            now = time.monotonic()
            if now - self.last_status_log >= 2.0:
                self.last_status_log = now
                self.get_logger().info(
                    'processed_frame=%d detections=%d state=%s latency=%.3fs'
                    % (
                        self.processed_frames,
                        len(results.boxes),
                        STATE_LABELS[inferred_state],
                        elapsed,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            self._log_processing_error(stage, exc, msg)
        finally:
            self.processing = False

    def _log_processing_error(self, stage: str, exc: Exception, msg: Image) -> None:
        signature = (stage, f'{type(exc).__name__}:{exc!r}')
        now = time.monotonic()
        should_log_trace = (
            signature != self._last_error_signature
            or (now - self._last_error_log_time) >= 5.0
        )
        self._last_error_signature = signature
        self._last_error_log_time = now

        header = msg.header
        image_info = (
            f'encoding={msg.encoding} '
            f'size={msg.width}x{msg.height} '
            f'frame_id={header.frame_id} '
            f'stamp={header.stamp.sec}.{header.stamp.nanosec:09d}'
        )
        context = (
            f'Processing failed at stage={stage} '
            f'exception_type={type(exc).__name__} '
            f'exception_repr={exc!r} '
            f'device={self.device} '
            f'image_topic={self.image_topic} '
            f'imgsz={self.image_size} '
            f'conf={self.conf_threshold:.2f} '
            f'iou={self.iou_threshold:.2f} '
            f'{image_info}'
        )
        self.get_logger().error(context)
        if should_log_trace:
            self.get_logger().error(traceback.format_exc())

    def _numpy_to_image_msg(self, image: np.ndarray, header: Any, encoding: str) -> Image:
        if not isinstance(image, np.ndarray):
            raise TypeError(f'Annotated image must be numpy.ndarray, got {type(image).__name__}')
        if image.dtype != np.uint8:
            raise TypeError(f'Annotated image dtype must be uint8, got {image.dtype}')
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f'Annotated image shape must be HxWx3, got {image.shape}')

        contiguous = np.ascontiguousarray(image)
        image_msg = Image()
        image_msg.header = header
        image_msg.height = int(contiguous.shape[0])
        image_msg.width = int(contiguous.shape[1])
        image_msg.encoding = encoding
        image_msg.is_bigendian = bool(contiguous.dtype.byteorder == '>')
        image_msg.step = int(contiguous.shape[1] * contiguous.shape[2] * contiguous.dtype.itemsize)
        image_msg.data = contiguous.tobytes()
        return image_msg

    def _build_detection_array(self, results: Any, msg: Image) -> Detection2DArray:
        detections = Detection2DArray()
        detections.header = msg.header

        for index, box in enumerate(results.boxes):
            xywh = box.xywh[0].tolist()
            class_index = int(box.cls[0]) if box.cls is not None else -1
            confidence = float(box.conf[0]) if box.conf is not None else 0.0
            class_id = self._class_name(class_index)

            detection = Detection2D()
            detection.header = msg.header
            detection.id = f'{msg.header.stamp.sec}-{msg.header.stamp.nanosec}-{index}'
            # vision_msgs는 center/size 표현을 쓰므로 YOLO의 xywh 값을 그대로 매핑한다.
            detection.bbox.center.position.x = float(xywh[0])
            detection.bbox.center.position.y = float(xywh[1])
            detection.bbox.center.theta = 0.0
            detection.bbox.size_x = float(xywh[2])
            detection.bbox.size_y = float(xywh[3])

            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = class_id
            hypothesis.hypothesis.score = confidence
            detection.results.append(hypothesis)
            detections.detections.append(detection)

        return detections

    def _draw_annotations(
        self,
        frame: Any,
        results: Any,
        inferred_state: int,
        inferred_reason: str,
    ) -> Any:
        annotated = frame.copy()
        self._draw_state_banner(annotated, inferred_state, inferred_reason)

        if len(results.boxes) == 0:
            return annotated

        for box in results.boxes:
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            class_index = int(box.cls[0]) if box.cls is not None else -1
            confidence = float(box.conf[0]) if box.conf is not None else 0.0
            color = self._class_color(class_index)

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                self.line_thickness,
                lineType=cv2.LINE_AA,
            )

            label = self._label_text(class_index, confidence)
            if not label:
                continue

            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                1,
            )
            text_top = max(y1 - text_height - baseline - 4, 0)
            text_bottom = text_top + text_height + baseline + 4
            text_right = x1 + text_width + 8

            cv2.rectangle(
                annotated,
                (x1, text_top),
                (text_right, text_bottom),
                color,
                thickness=-1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 4, text_bottom - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )

        return annotated

    def _draw_state_banner(
        self,
        annotated: Any,
        inferred_state: int,
        inferred_reason: str,
    ) -> None:
        summary = f'{STATE_LABELS[inferred_state]} | {inferred_reason}'
        color = self._state_color(inferred_state)
        text_color = (18, 18, 18) if inferred_state == STATE_YELLOW else (255, 255, 255)
        (text_width, text_height), baseline = cv2.getTextSize(
            summary,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            1,
        )
        top_left = (12, 12)
        bottom_right = (top_left[0] + text_width + 16, top_left[1] + text_height + baseline + 12)
        cv2.rectangle(annotated, top_left, bottom_right, color, thickness=-1)
        cv2.putText(
            annotated,
            summary,
            (top_left[0] + 8, bottom_right[1] - baseline - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text_color,
            1,
            lineType=cv2.LINE_AA,
        )

    def _label_text(self, class_index: int, confidence: float) -> str:
        parts: list[str] = []
        if self.draw_labels:
            parts.append(self._class_name(class_index))
        if self.draw_confidence:
            parts.append(f'{confidence:.2f}')
        return ' '.join(parts)

    def _resolve_detector_classes(self) -> list[int] | None:
        if isinstance(self.class_names, dict):
            normalized = {
                int(class_index): str(name).strip().lower()
                for class_index, name in self.class_names.items()
            }
        elif isinstance(self.class_names, (list, tuple)):
            normalized = {
                index: str(name).strip().lower()
                for index, name in enumerate(self.class_names)
            }
        else:
            normalized = {}

        detector_classes = [
            class_index
            for class_index, name in normalized.items()
            if name.startswith('vehicular_') or name == 'traffic light'
        ]
        return detector_classes or None

    def _infer_state(self, results: Any) -> tuple[int, str]:
        if results.boxes is None or len(results.boxes) == 0:
            return STATE_UNKNOWN, 'no_detection'

        best_resolved: tuple[float, int, str] | None = None
        best_unknown: tuple[float, str] | None = None

        for box in results.boxes:
            class_index = int(box.cls[0]) if box.cls is not None else -1
            confidence = float(box.conf[0]) if box.conf is not None else 0.0
            class_name = self._class_name(class_index)
            state, resolved = self._state_from_class_name(class_name)

            if best_unknown is None or confidence > best_unknown[0]:
                best_unknown = (confidence, class_name)

            if resolved and (best_resolved is None or confidence > best_resolved[0]):
                best_resolved = (confidence, state, class_name)

        if best_resolved is not None:
            confidence, state, class_name = best_resolved
            return state, f'model {class_name}:{confidence:.2f}'
        if best_unknown is not None:
            confidence, class_name = best_unknown
            return STATE_UNKNOWN, f'model_unknown {class_name}:{confidence:.2f}'
        return STATE_UNKNOWN, 'no_detection'

    def _state_from_class_name(self, class_name: str) -> tuple[int, bool]:
        normalized = class_name.strip().lower().replace('-', '_').replace(' ', '_')
        if normalized in {'traffic_light', 'trafficlight'} or 'etc' in normalized:
            return STATE_UNKNOWN, False

        has_red = 'red' in normalized
        has_yellow = 'yellow' in normalized
        has_green = 'green' in normalized
        has_left_arrow = 'left' in normalized and 'arrow' in normalized

        if has_left_arrow or (has_red and has_green):
            return STATE_LEFT_ARROW, True
        if has_yellow:
            return STATE_YELLOW, True
        if has_green:
            return STATE_GREEN, True
        if has_red:
            return STATE_RED, True
        return STATE_UNKNOWN, False

    def _class_color(self, class_index: int) -> tuple[int, int, int]:
        # 같은 클래스가 프레임마다 같은 색을 유지하도록 결정론적 색상을 만든다.
        seed = max(class_index, 0)
        return (
            (37 * seed + 70) % 256,
            (17 * seed + 160) % 256,
            (29 * seed + 220) % 256,
        )

    def _state_color(self, state: int) -> tuple[int, int, int]:
        return STATE_COLORS.get(state, STATE_COLORS[STATE_UNKNOWN])

    def _class_name(self, class_index: int) -> str:
        if isinstance(self.class_names, dict):
            return str(self.class_names.get(class_index, class_index))
        if 0 <= class_index < len(self.class_names):
            return str(self.class_names[class_index])
        return str(class_index)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = YoloValidatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
