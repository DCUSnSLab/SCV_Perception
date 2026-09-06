from __future__ import annotations

import math
import sys
import time
import traceback
from collections import Counter
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .workspace_paths import local_python_deps_path
from .workspace_paths import default_runtime_image_topic
from .workspace_paths import resolve_inference_device
from .workspace_paths import workspace_root_or_none

deps_path = local_python_deps_path()
if deps_path is not None and deps_path.exists():
    sys.path.insert(0, str(deps_path))

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from std_msgs.msg import Int32
from std_msgs.msg import String
from ultralytics import YOLO


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

COLOR_ORDER = ('red', 'yellow', 'green')
COLOR_TO_STATE = {
    'red': STATE_RED,
    'yellow': STATE_YELLOW,
    'green': STATE_GREEN,
}


def default_tl_model_path() -> str:
    root = workspace_root_or_none()
    if root is None:
        return 'best.pt'

    best = root / 'model' / 'best.pt'
    if best.exists():
        return str(best)
    candidate = root / 'yolo11s.pt'
    return str(candidate) if candidate.exists() else 'yolo11s.pt'


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class DetectionCandidate:
    box: tuple[int, int, int, int]
    conf: float
    class_id: int
    class_name: str
    model_state: int
    model_resolved: bool
    selection_score: float = 0.0


@dataclass
class ColorAnalysisResult:
    state: int
    decisive: bool
    reason: str
    valid_pixels: int
    top_score: float
    score_gap: float
    scores: dict[str, float]
    highlighted: np.ndarray


@dataclass
class DecisionResult:
    proposed_state: int
    source: str
    reason: str


@dataclass
class OverlayCandidate:
    box: tuple[int, int, int, int]
    label: str
    color: tuple[int, int, int]
    selected: bool


class TLFusionNode(Node):
    def __init__(self) -> None:
        super().__init__('tl_fusion')

        self.bridge = CvBridge()
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self.fallback_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        self.model_path = self._declare_param('model_path', default_tl_model_path())
        self.image_topic = str(self._declare_param('image_topic', default_runtime_image_topic()))
        self.state_topic = str(self._declare_param('state_topic', '/tl/state_id'))
        self.show_windows = bool(self._declare_param('show_windows', False))
        self.input_timeout_s = float(self._declare_param('input_timeout_s', 3.0))

        self.max_fps = float(self._declare_param('max_fps', 15.0))
        requested_device = str(self._declare_param('detector_device', 'cuda:0'))
        self.detector_device = resolve_inference_device(requested_device)
        self.detector_image_size = int(self._declare_param('detector_image_size', 640))
        self.detector_conf_threshold = float(self._declare_param('detector_conf_threshold', 0.10))
        self.detector_iou_threshold = float(self._declare_param('detector_iou_threshold', 0.45))
        self.detector_max_detections = int(self._declare_param('detector_max_detections', 50))

        self.detect_top_ratio = float(self._declare_param('detect_top_ratio', 0.00))
        self.detect_bottom_ratio = float(self._declare_param('detect_bottom_ratio', 1.00))
        self.detect_left_ratio = float(self._declare_param('detect_left_ratio', 0.00))
        self.detect_right_ratio = float(self._declare_param('detect_right_ratio', 1.00))

        self.preferred_top_ratio = float(self._declare_param('preferred_top_ratio', 0.00))
        self.preferred_bottom_ratio = float(self._declare_param('preferred_bottom_ratio', 0.50))
        self.preferred_left_ratio = float(self._declare_param('preferred_left_ratio', 0.25))
        self.preferred_right_ratio = float(self._declare_param('preferred_right_ratio', 0.75))

        self.min_box_side_px = int(self._declare_param('min_box_side_px', 5))
        self.min_box_area_px = int(self._declare_param('min_box_area_px', 40))
        self.edge_margin_px = int(self._declare_param('edge_margin_px', 2))

        self.model_confidence_threshold = float(self._declare_param('model_confidence_threshold', 0.60))
        self.model_min_confidence_threshold = float(
            self._declare_param('model_min_confidence_threshold', 0.30)
        )

        self.fallback_expand_ratio = float(self._declare_param('fallback_expand_ratio', 1.80))
        self.fallback_min_margin_px = int(self._declare_param('fallback_min_margin_px', 4))
        self.fallback_saturation_gain = float(self._declare_param('fallback_saturation_gain', 1.80))
        self.fallback_value_gain = float(self._declare_param('fallback_value_gain', 1.25))
        self.fallback_gamma = float(self._declare_param('fallback_gamma', 0.85))
        self.fallback_s_min = int(self._declare_param('fallback_s_min', 55))
        self.fallback_v_min = int(self._declare_param('fallback_v_min', 70))
        self.fallback_min_valid_pixels = int(self._declare_param('fallback_min_valid_pixels', 12))
        self.fallback_min_component_pixels = int(
            self._declare_param('fallback_min_component_pixels', 6)
        )
        self.fallback_score_threshold = float(self._declare_param('fallback_score_threshold', 0.50))
        self.fallback_score_gap = float(self._declare_param('fallback_score_gap', 0.14))
        self.fallback_red_green_red_min = float(self._declare_param('fallback_red_green_red_min', 0.30))
        self.fallback_red_green_green_min = float(
            self._declare_param('fallback_red_green_green_min', 0.18)
        )
        self.fallback_red_green_yellow_max = float(
            self._declare_param('fallback_red_green_yellow_max', 0.12)
        )
        self.fallback_gamma_lut = self._build_gamma_lut(self.fallback_gamma)

        self.state_window_size = int(self._declare_param('state_window_size', 5))
        self.hold_ms = int(self._declare_param('hold_ms', 250))
        self.missing_timeout_ms = int(self._declare_param('missing_timeout_ms', 400))
        self.reset_tracking_ms = int(self._declare_param('reset_tracking_ms', 1200))
        self.overlay_hold_ms = int(self._declare_param('overlay_hold_ms', 220))
        self.overlay_smoothing_alpha = float(self._declare_param('overlay_smoothing_alpha', 0.55))
        # Some recorded bags publish valid images with zero header stamps.
        self.require_image_header_stamp = bool(
            self._declare_param('require_image_header_stamp', False)
        )
        self.require_monotonic_image_stamp = bool(
            self._declare_param('require_monotonic_image_stamp', True)
        )

        model_path = Path(self.model_path).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f'Model file not found: {model_path}')

        self.get_logger().info(f'Loading YOLO model: {model_path}')
        self.model = YOLO(str(model_path))
        self.class_names = self.model.names
        self.detector_classes = self._resolve_detector_classes()
        self.get_logger().info(f'YOLO model loaded: {model_path}')
        self.get_logger().info(
            f'Inference device: {self.detector_device} (requested: {requested_device})'
        )
        self.get_logger().info(f'Detector class filter: {self.detector_classes}')
        self.get_logger().info(f'Subscribing to image topic: {self.image_topic}')
        self.get_logger().info(f'Publishing traffic-light state: {self.state_topic}')

        self.latest_msg: Image | None = None
        self.processing = False
        self.last_image_received_monotonic_ns = time.monotonic_ns()
        self.input_timeout_active = False
        self.current_state = STATE_UNKNOWN
        self.current_source = 'init'
        self.current_reason = 'init'
        self.last_state_change_ns = self._now_ns()
        self.last_seen_candidate_ns = self._now_ns()
        self.last_candidate_box: tuple[int, int, int, int] | None = None
        self.last_overlay_candidate: OverlayCandidate | None = None
        self.last_overlay_update_ns = self._now_ns()
        self.state_history: deque[int] = deque(maxlen=max(1, self.state_window_size))
        self.processed_frames = 0
        self.last_status_log = time.monotonic()
        self.last_status_frames = 0
        self.last_image_stamp_ns: int | None = None
        self._last_error_signature: tuple[str, str] | None = None
        self._last_error_log_time = 0.0
        self._last_invalid_input_reason: str | None = None
        self._last_invalid_input_log_time = 0.0

        self.create_subscription(
            Image,
            self.image_topic,
            self._image_callback,
            qos_profile_sensor_data,
        )

        image_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        status_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.debug_pub = self.create_publisher(Image, '/tl/debug_image', image_qos)
        self.state_pub = self.create_publisher(Int32, self.state_topic, status_qos)
        self.state_label_pub = self.create_publisher(String, '/tl/state_label', status_qos)
        self.state_reason_pub = self.create_publisher(String, '/tl/state_reason', status_qos)
        self.input_valid_pub = self.create_publisher(Bool, '/tl/input_valid', status_qos)

        timer_period = 1.0 / max(self.max_fps, 0.1)
        self.create_timer(timer_period, self._process_latest_frame)

    def _declare_param(self, name: str, default_value: Any) -> Any:
        return self.declare_parameter(name, default_value).value

    def _image_callback(self, msg: Image) -> None:
        self.last_image_received_monotonic_ns = time.monotonic_ns()
        self.latest_msg = msg

    def _process_latest_frame(self) -> None:
        if self.processing:
            return
        if self.latest_msg is None:
            self._publish_timeout_if_needed()
            return

        msg = self.latest_msg
        self.latest_msg = None
        self.processing = True
        started = time.perf_counter()
        stage = 'decode'

        try:
            stamp_reason = self._validate_input_timestamp(msg)
            if stamp_reason is not None:
                self._handle_invalid_input(msg, stamp_reason)
                return
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            stage = 'detect'
            detections, _ = self._detect_candidates(frame)
            stage = 'select'
            selected = self._select_candidate(detections, frame.shape)

            if selected is not None:
                self.last_seen_candidate_ns = self._now_ns()
                self.last_candidate_box = selected.box

            if selected is not None and selected.model_resolved and selected.conf >= self.model_confidence_threshold:
                stage = 'model_high_conf'
                analysis = self._empty_analysis('model_high_conf_skip')
                decision = DecisionResult(
                    proposed_state=selected.model_state,
                    source='model',
                    reason=f'{selected.class_name}:{selected.conf:.2f}',
                )
            else:
                stage = 'analyze'
                analysis = self._analyze_selected_candidate(frame, selected)
                stage = 'decide'
                decision = self._decide_state(selected, analysis)
            stage = 'stabilize'
            stable_state = self._update_stable_state(decision.proposed_state, selected is not None)
            debug_image = None
            if self._should_render_debug():
                stage = 'build_overlay'
                overlay_candidates = self._build_overlay_candidates(
                    detections,
                    selected,
                    stable_state,
                )
                stage = 'render_debug'
                debug_image = self._build_debug_image(
                    frame,
                    overlay_candidates,
                    selected,
                    analysis,
                )
            stage = 'publish_outputs'
            self._publish_outputs(msg, debug_image, stable_state, decision)

            self.processed_frames += 1
            now = time.monotonic()
            elapsed = time.perf_counter() - started
            if now - self.last_status_log >= 2.0:
                delta_t = max(now - self.last_status_log, 1e-6)
                delta_frames = self.processed_frames - self.last_status_frames
                effective_fps = delta_frames / delta_t
                self.last_status_log = now
                self.last_status_frames = self.processed_frames
                summary = (
                    f'stable={STATE_LABELS[stable_state]} '
                    f'source={decision.source} '
                    f'reason={decision.reason}'
                )
                self.get_logger().info(
                    'processed_frame=%d detections=%d latency=%.3fs fps=%.2f %s'
                    % (
                        self.processed_frames,
                        len(detections),
                        elapsed,
                        effective_fps,
                        summary,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            self._log_processing_error(stage, exc, msg)
        finally:
            self.processing = False

    def _publish_timeout_if_needed(self) -> None:
        if self.input_timeout_s <= 0.0:
            return

        elapsed_s = (
            time.monotonic_ns() - self.last_image_received_monotonic_ns
        ) / 1e9
        if elapsed_s < self.input_timeout_s:
            return

        first_timeout_publish = not self.input_timeout_active
        self.input_timeout_active = True
        self.current_state = STATE_UNKNOWN
        self.current_source = 'input_timeout'
        self.current_reason = f'no_image_for={elapsed_s:.1f}s'
        self.state_history.clear()
        self.state_history.append(STATE_UNKNOWN)
        self.last_candidate_box = None
        self.last_overlay_candidate = None

        self.input_valid_pub.publish(Bool(data=False))
        self.state_pub.publish(Int32(data=int(STATE_UNKNOWN)))
        self.state_label_pub.publish(String(data=STATE_LABELS[STATE_UNKNOWN]))
        self.state_reason_pub.publish(
            String(data=f'{STATE_LABELS[STATE_UNKNOWN]} input_timeout {self.current_reason}')
        )

        if first_timeout_publish:
            self.get_logger().error(
                f'No image received for {elapsed_s:.1f}s on {self.image_topic}; '
                f'publishing UNKNOWN on {self.state_topic}'
            )

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
            f'TL fusion failed at stage={stage} '
            f'exception_type={type(exc).__name__} '
            f'exception_repr={exc!r} '
            f'device={self.detector_device} '
            f'image_topic={self.image_topic} '
            f'imgsz={self.detector_image_size} '
            f'conf={self.detector_conf_threshold:.2f} '
            f'iou={self.detector_iou_threshold:.2f} '
            f'{image_info}'
        )
        self.get_logger().error(context)
        if should_log_trace:
            self.get_logger().error(traceback.format_exc())

    def _validate_input_timestamp(self, msg: Image) -> str | None:
        stamp_ns = self._stamp_to_ns(msg)
        if self.require_image_header_stamp and stamp_ns <= 0:
            return 'invalid_image_stamp_zero'
        if (
            self.require_monotonic_image_stamp
            and stamp_ns > 0
            and self.last_image_stamp_ns is not None
            and stamp_ns < self.last_image_stamp_ns
        ):
            return (
                f'invalid_image_stamp_backward current={stamp_ns} '
                f'previous={self.last_image_stamp_ns}'
            )
        if stamp_ns > 0:
            self.last_image_stamp_ns = stamp_ns
        return None

    def _handle_invalid_input(self, msg: Image, reason: str) -> None:
        now = time.monotonic()
        should_log = (
            reason != self._last_invalid_input_reason
            or (now - self._last_invalid_input_log_time) >= 5.0
        )
        self._last_invalid_input_reason = reason
        self._last_invalid_input_log_time = now

        self.current_state = STATE_UNKNOWN
        self.current_source = 'invalid_input'
        self.current_reason = reason
        self.state_history.clear()
        self.state_history.append(STATE_UNKNOWN)
        self.last_candidate_box = None
        self.last_overlay_candidate = None

        self.input_valid_pub.publish(Bool(data=False))
        self.state_pub.publish(Int32(data=int(STATE_UNKNOWN)))
        self.state_label_pub.publish(String(data=STATE_LABELS[STATE_UNKNOWN]))
        self.state_reason_pub.publish(String(data=f'{STATE_LABELS[STATE_UNKNOWN]} invalid_input {reason}'))

        if should_log:
            header = msg.header
            self.get_logger().error(
                'Rejecting image frame due to invalid timestamp '
                f'reason={reason} frame_id={header.frame_id} '
                f'stamp={header.stamp.sec}.{header.stamp.nanosec:09d} '
                f'image_topic={self.image_topic}'
            )

    def _detect_candidates(
        self,
        frame: np.ndarray,
    ) -> tuple[list[DetectionCandidate], tuple[int, int, int, int]]:
        x0, y0, x1, y1 = self._window_from_ratios(
            frame.shape,
            self.detect_left_ratio,
            self.detect_right_ratio,
            self.detect_top_ratio,
            self.detect_bottom_ratio,
        )
        detect_window = (x0, y0, x1, y1)
        if x1 <= x0 or y1 <= y0:
            return [], detect_window

        detect_frame = frame[y0:y1, x0:x1]
        result = self.model.predict(
            source=detect_frame,
            classes=self.detector_classes,
            conf=self.detector_conf_threshold,
            iou=self.detector_iou_threshold,
            imgsz=self.detector_image_size,
            max_det=self.detector_max_detections,
            device=self.detector_device,
            verbose=False,
        )[0]

        detections: list[DetectionCandidate] = []
        if result.boxes is None:
            return detections, detect_window

        for box in result.boxes:
            x_a, y_a, x_b, y_b = [int(v) for v in box.xyxy[0].tolist()]
            width = x_b - x_a
            height = y_b - y_a
            if width <= 0 or height <= 0:
                continue
            if min(width, height) < self.min_box_side_px:
                continue
            if width * height < self.min_box_area_px:
                continue
            if (
                x_a <= self.edge_margin_px
                or y_a <= self.edge_margin_px
                or (detect_frame.shape[1] - x_b) <= self.edge_margin_px
                or (detect_frame.shape[0] - y_b) <= self.edge_margin_px
            ):
                continue

            class_id = int(box.cls[0]) if box.cls is not None else -1
            class_name = self._class_name(class_id)
            normalized_class_name = (
                class_name.strip().lower().replace('-', '_').replace(' ', '_')
            )
            # The model also contains a downward arrow used by the terminal-branch
            # detector. It must not be interpreted as a drivable green light.
            if 'green_arrow' in normalized_class_name and 'down' in normalized_class_name:
                continue
            model_state, model_resolved = self._state_from_class_name(class_name)
            detections.append(
                DetectionCandidate(
                    box=(x_a + x0, y_a + y0, x_b + x0, y_b + y0),
                    conf=float(box.conf[0]) if box.conf is not None else 0.0,
                    class_id=class_id,
                    class_name=class_name,
                    model_state=model_state,
                    model_resolved=model_resolved,
                )
            )

        return detections, detect_window

    def _select_candidate(
        self,
        detections: list[DetectionCandidate],
        frame_shape: tuple[int, ...],
    ) -> DetectionCandidate | None:
        if not detections:
            return None

        preferred_window = self._window_from_ratios(
            frame_shape,
            self.preferred_left_ratio,
            self.preferred_right_ratio,
            self.preferred_top_ratio,
            self.preferred_bottom_ratio,
        )
        height, width = frame_shape[:2]
        frame_area = float(max(1, width * height))

        selected: DetectionCandidate | None = None
        best_score = -1.0
        for detection in detections:
            x_a, y_a, x_b, y_b = detection.box
            center_x = 0.5 * (x_a + x_b)
            center_y = 0.5 * (y_a + y_b)
            center_y_norm = center_y / float(max(1, height))
            vertical_bonus = 1.0 + 0.35 * (1.0 - center_y_norm)
            area_bonus = 1.0 + 0.20 * min(1.0, ((x_b - x_a) * (y_b - y_a)) / (frame_area * 0.02))
            preferred_bonus = (
                1.25 if self._point_in_window(center_x, center_y, preferred_window) else 1.0
            )
            tracking_bonus = 1.0 + 0.20 * self._tracking_similarity(detection.box, self.last_candidate_box, frame_shape)
            score = detection.conf * vertical_bonus * area_bonus * preferred_bonus * tracking_bonus
            detection.selection_score = score
            if score > best_score:
                best_score = score
                selected = detection

        return selected

    def _analyze_selected_candidate(
        self,
        frame: np.ndarray,
        candidate: DetectionCandidate | None,
    ) -> ColorAnalysisResult:
        if candidate is None:
            empty = np.zeros((160, 320, 3), dtype=np.uint8)
            cv2.putText(
                empty,
                'NO CANDIDATE',
                (22, 86),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            return ColorAnalysisResult(
                state=STATE_UNKNOWN,
                decisive=False,
                reason='no_candidate',
                valid_pixels=0,
                top_score=0.0,
                score_gap=0.0,
                scores={name: 0.0 for name in COLOR_ORDER},
                highlighted=empty,
            )

        crop = self._expanded_crop(frame, candidate.box)
        enhanced = self._enhance_crop(crop)
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].astype(np.float32) / 255.0
        value = hsv[:, :, 2].astype(np.float32) / 255.0
        weights = 0.25 + 0.40 * saturation + 0.35 * value

        red_mask_1 = cv2.inRange(hsv, (0, self.fallback_s_min, self.fallback_v_min), (9, 255, 255))
        red_mask_2 = cv2.inRange(hsv, (165, self.fallback_s_min, self.fallback_v_min), (179, 255, 255))
        red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
        yellow_mask = cv2.inRange(
            hsv,
            (14, self.fallback_s_min, self.fallback_v_min),
            (38, 255, 255),
        )
        green_mask = cv2.inRange(
            hsv,
            (40, self.fallback_s_min, self.fallback_v_min),
            (95, 255, 255),
        )

        red_mask = self._clean_mask(red_mask)
        yellow_mask = self._clean_mask(yellow_mask)
        green_mask = self._clean_mask(green_mask)

        masks = {
            'red': red_mask,
            'yellow': yellow_mask,
            'green': green_mask,
        }
        raw_scores = {
            name: float(weights[mask > 0].sum())
            for name, mask in masks.items()
        }
        total_score = float(sum(raw_scores.values()))
        scores = {
            name: (raw_scores[name] / total_score) if total_score > 0.0 else 0.0
            for name in COLOR_ORDER
        }
        valid_pixels = int(sum(int(np.count_nonzero(mask)) for mask in masks.values()))
        top_color = max(COLOR_ORDER, key=lambda name: scores[name])
        top_score = float(scores[top_color])
        second_score = max(
            (float(scores[name]) for name in COLOR_ORDER if name != top_color),
            default=0.0,
        )
        score_gap = top_score - second_score

        red_green_decisive = (
            valid_pixels >= self.fallback_min_valid_pixels
            and scores['red'] >= self.fallback_red_green_red_min
            and scores['green'] >= self.fallback_red_green_green_min
            and scores['yellow'] <= self.fallback_red_green_yellow_max
        )

        component_size = 0
        if (
            valid_pixels >= self.fallback_min_valid_pixels
            and top_score >= self.fallback_score_threshold
            and score_gap >= self.fallback_score_gap
        ):
            component_size = self._largest_component(masks[top_color])

        decisive = (
            valid_pixels >= self.fallback_min_valid_pixels
            and top_score >= self.fallback_score_threshold
            and score_gap >= self.fallback_score_gap
            and component_size >= self.fallback_min_component_pixels
        )

        if red_green_decisive:
            state = STATE_LEFT_ARROW
            decisive = True
            reason = 'color_left_arrow'
        elif decisive:
            state = COLOR_TO_STATE[top_color]
            reason = f'color_{top_color}'
        else:
            state = STATE_UNKNOWN
            reason = 'color_ambiguous'

        highlighted = self._highlight_masks(enhanced, masks, scores)
        return ColorAnalysisResult(
            state=state,
            decisive=decisive,
            reason=reason,
            valid_pixels=valid_pixels,
            top_score=top_score,
            score_gap=score_gap,
            scores=scores,
            highlighted=highlighted,
        )

    def _empty_analysis(self, reason: str) -> ColorAnalysisResult:
        return ColorAnalysisResult(
            state=STATE_UNKNOWN,
            decisive=False,
            reason=reason,
            valid_pixels=0,
            top_score=0.0,
            score_gap=0.0,
            scores={name: 0.0 for name in COLOR_ORDER},
            highlighted=np.zeros((1, 1, 3), dtype=np.uint8),
        )

    def _decide_state(
        self,
        candidate: DetectionCandidate | None,
        analysis: ColorAnalysisResult,
    ) -> DecisionResult:
        if candidate is None:
            return DecisionResult(
                proposed_state=STATE_UNKNOWN,
                source='none',
                reason='no_candidate',
            )

        if candidate.model_resolved and candidate.conf >= self.model_confidence_threshold:
            return DecisionResult(
                proposed_state=candidate.model_state,
                source='model',
                reason=f'{candidate.class_name}:{candidate.conf:.2f}',
            )

        if analysis.decisive and (
            not candidate.model_resolved or candidate.conf < self.model_confidence_threshold
        ):
            return DecisionResult(
                proposed_state=analysis.state,
                source='color_fallback',
                reason=f'{analysis.reason}:score={analysis.top_score:.2f}',
            )

        if candidate.model_resolved and candidate.conf >= self.model_min_confidence_threshold:
            return DecisionResult(
                proposed_state=candidate.model_state,
                source='model_low_conf',
                reason=f'{candidate.class_name}:{candidate.conf:.2f}',
            )

        if analysis.decisive:
            return DecisionResult(
                proposed_state=analysis.state,
                source='color_only',
                reason=f'{analysis.reason}:score={analysis.top_score:.2f}',
            )

        if candidate.model_resolved:
            return DecisionResult(
                proposed_state=candidate.model_state,
                source='model_weak',
                reason=f'{candidate.class_name}:{candidate.conf:.2f}',
            )

        return DecisionResult(
            proposed_state=STATE_UNKNOWN,
            source='unknown',
            reason='etc_or_ambiguous',
        )

    def _update_stable_state(self, proposed_state: int, has_candidate: bool) -> int:
        now_ns = self._now_ns()
        missing_ms = self._ns_to_ms(now_ns - self.last_seen_candidate_ns)

        if not has_candidate and missing_ms < self.missing_timeout_ms:
            proposed_state = self.current_state
        elif not has_candidate and missing_ms >= self.reset_tracking_ms:
            self.last_candidate_box = None

        self.state_history.append(proposed_state)
        majority_state = self._majority_state(self.state_history)
        can_change = self._ns_to_ms(now_ns - self.last_state_change_ns) >= self.hold_ms

        if majority_state != self.current_state and can_change:
            self.current_state = majority_state
            self.last_state_change_ns = now_ns

        return self.current_state

    def _publish_outputs(
        self,
        msg: Image,
        debug_image: np.ndarray | None,
        stable_state: int,
        decision: DecisionResult,
    ) -> None:
        self.input_timeout_active = False
        self.input_valid_pub.publish(Bool(data=True))
        if debug_image is not None:
            debug_msg = self._numpy_to_image_msg(
                debug_image,
                header=msg.header,
                encoding='bgr8',
            )
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)

        self.state_pub.publish(Int32(data=int(stable_state)))
        self.state_label_pub.publish(String(data=STATE_LABELS[stable_state]))
        self.state_reason_pub.publish(
            String(data=f'{STATE_LABELS[stable_state]} {decision.source} {decision.reason}')
        )

        if self.show_windows and debug_image is not None:
            cv2.imshow('TL Debug', debug_image)
            cv2.waitKey(1)

    def _numpy_to_image_msg(self, image: np.ndarray, header: Any, encoding: str) -> Image:
        if not isinstance(image, np.ndarray):
            raise TypeError(f'Debug image must be numpy.ndarray, got {type(image).__name__}')
        if image.dtype != np.uint8:
            raise TypeError(f'Debug image dtype must be uint8, got {image.dtype}')
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f'Debug image shape must be HxWx3, got {image.shape}')

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

    def _build_debug_image(
        self,
        frame: np.ndarray,
        overlay_candidates: list[OverlayCandidate],
        selected: DetectionCandidate | None,
        analysis: ColorAnalysisResult,
    ) -> np.ndarray:
        debug = frame.copy()
        for overlay in overlay_candidates:
            x_a, y_a, x_b, y_b = overlay.box
            thickness = 2 if overlay.selected else 1
            cv2.rectangle(debug, (x_a, y_a), (x_b, y_b), overlay.color, thickness)
            self._draw_box_label(debug, overlay.label, x_a, y_a, overlay.color)
        if selected is not None and analysis.highlighted.size > 0:
            inset = self._fit_to_canvas(analysis.highlighted, 200, 120)
            self._draw_debug_inset(debug, inset, 'Color Mask')
        return debug

    def _build_overlay_candidates(
        self,
        detections: list[DetectionCandidate],
        selected: DetectionCandidate | None,
        stable_state: int,
    ) -> list[OverlayCandidate]:
        overlays: list[OverlayCandidate] = []
        selected_box = selected.box if selected is not None else None
        selected_overlay: OverlayCandidate | None = None

        for detection in detections:
            is_selected = selected_box is not None and detection.box == selected_box
            if is_selected:
                smoothed_box = self._smooth_overlay_box(detection.box)
                selected_overlay = OverlayCandidate(
                    box=smoothed_box,
                    label=f'{detection.class_name} {detection.conf:.2f} | {STATE_LABELS[stable_state]}',
                    color=self._state_color(stable_state),
                    selected=True,
                )
            else:
                overlays.append(
                    OverlayCandidate(
                        box=detection.box,
                        label=f'{detection.class_name} {detection.conf:.2f}',
                        color=(0, 128, 255),
                        selected=False,
                    )
                )

        if selected_overlay is not None:
            self.last_overlay_candidate = selected_overlay
            self.last_overlay_update_ns = self._now_ns()
            overlays.append(selected_overlay)
            return overlays

        fallback_overlay = self._fallback_overlay_candidate()
        if fallback_overlay is not None:
            overlays.append(fallback_overlay)
        return overlays

    def _fallback_overlay_candidate(self) -> OverlayCandidate | None:
        if self.last_overlay_candidate is None:
            return None

        now_ns = self._now_ns()
        elapsed_ms = self._ns_to_ms(now_ns - self.last_overlay_update_ns)
        if elapsed_ms > self.overlay_hold_ms:
            self.last_overlay_candidate = None
            return None
        return self.last_overlay_candidate

    def _smooth_overlay_box(self, current_box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        previous = self.last_overlay_candidate
        if previous is None:
            return current_box

        alpha = clamp(self.overlay_smoothing_alpha, 0.0, 1.0)
        if alpha <= 0.0:
            return previous.box
        if alpha >= 1.0:
            return current_box

        blended = []
        for current_value, previous_value in zip(current_box, previous.box, strict=False):
            blended_value = alpha * float(current_value) + (1.0 - alpha) * float(previous_value)
            blended.append(int(round(blended_value)))
        return tuple(blended)  # type: ignore[return-value]

    def _window_from_ratios(
        self,
        frame_shape: tuple[int, ...],
        left_ratio: float,
        right_ratio: float,
        top_ratio: float,
        bottom_ratio: float,
    ) -> tuple[int, int, int, int]:
        height, width = frame_shape[:2]
        x0 = int(width * left_ratio)
        x1 = int(width * right_ratio)
        y0 = int(height * top_ratio)
        y1 = int(height * bottom_ratio)
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(width, x1)
        y1 = min(height, y1)
        return x0, y0, x1, y1

    def _class_name(self, class_id: int) -> str:
        if isinstance(self.class_names, dict):
            return str(self.class_names.get(class_id, class_id))
        if isinstance(self.class_names, (list, tuple)) and 0 <= class_id < len(self.class_names):
            return str(self.class_names[class_id])
        return str(class_id)

    def _resolve_detector_classes(self) -> list[int] | None:
        if isinstance(self.class_names, dict):
            normalized = {
                int(class_id): str(name).strip().lower()
                for class_id, name in self.class_names.items()
            }
        elif isinstance(self.class_names, (list, tuple)):
            normalized = {
                index: str(name).strip().lower()
                for index, name in enumerate(self.class_names)
            }
        else:
            normalized = {}

        detector_classes = [
            class_id
            for class_id, name in normalized.items()
            if name.startswith('vehicular_') or name == 'traffic light'
        ]
        return detector_classes or None

    def _state_from_class_name(self, class_name: str) -> tuple[int, bool]:
        normalized = class_name.strip().lower().replace('-', '_').replace(' ', '_')
        if normalized in {'traffic_light', 'trafficlight'} or 'etc' in normalized:
            return STATE_UNKNOWN, False
        if 'green_arrow' in normalized and 'down' in normalized:
            return STATE_UNKNOWN, False

        has_red = 'red' in normalized
        has_yellow = 'yellow' in normalized
        has_green = 'green' in normalized

        has_left_arrow = 'left' in normalized and 'arrow' in normalized
        has_green_arrow = 'green_arrow' in normalized and 'down' not in normalized
        if has_left_arrow or has_green_arrow or (has_red and has_green):
            return STATE_LEFT_ARROW, True
        if has_yellow:
            return STATE_YELLOW, True
        if has_green:
            return STATE_GREEN, True
        if has_red:
            return STATE_RED, True
        return STATE_UNKNOWN, False

    def _expanded_crop(
        self,
        frame: np.ndarray,
        box: tuple[int, int, int, int],
    ) -> np.ndarray:
        x_a, y_a, x_b, y_b = box
        width = x_b - x_a
        height = y_b - y_a
        expand_w = max(int(width * self.fallback_expand_ratio), width + 2 * self.fallback_min_margin_px)
        expand_h = max(int(height * self.fallback_expand_ratio), height + 2 * self.fallback_min_margin_px)
        center_x = 0.5 * (x_a + x_b)
        center_y = 0.5 * (y_a + y_b)
        x0 = int(round(center_x - 0.5 * expand_w))
        x1 = int(round(center_x + 0.5 * expand_w))
        y0 = int(round(center_y - 0.5 * expand_h))
        y1 = int(round(center_y + 0.5 * expand_h))
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(frame.shape[1], x1)
        y1 = min(frame.shape[0], y1)
        if x1 <= x0 or y1 <= y0:
            return np.zeros((64, 64, 3), dtype=np.uint8)
        return frame[y0:y1, x0:x1]

    def _enhance_crop(self, crop: np.ndarray) -> np.ndarray:
        if crop.size == 0:
            return np.zeros((64, 64, 3), dtype=np.uint8)

        working = crop
        if min(working.shape[:2]) < 64:
            scale = 64.0 / float(max(1, min(working.shape[:2])))
            new_width = max(1, int(round(working.shape[1] * scale)))
            new_height = max(1, int(round(working.shape[0] * scale)))
            working = cv2.resize(working, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_channel = self.clahe.apply(l_channel)
        enhanced = cv2.cvtColor(
            cv2.merge((l_channel, a_channel, b_channel)),
            cv2.COLOR_LAB2BGR,
        )

        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
        h_channel, s_channel, v_channel = cv2.split(hsv)
        s_channel = cv2.convertScaleAbs(s_channel, alpha=self.fallback_saturation_gain)
        v_channel = cv2.convertScaleAbs(v_channel, alpha=self.fallback_value_gain)
        hsv = cv2.merge((h_channel, s_channel, v_channel))
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        enhanced = cv2.LUT(enhanced, self.fallback_gamma_lut)

        blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        enhanced = cv2.addWeighted(enhanced, 1.35, blurred, -0.35, 0.0)
        return enhanced

    def _build_gamma_lut(self, gamma: float) -> np.ndarray:
        gamma = max(0.1, gamma)
        inv_gamma = 1.0 / gamma
        return np.array(
            [((index / 255.0) ** inv_gamma) * 255.0 for index in range(256)],
            dtype=np.uint8,
        )

    def _should_render_debug(self) -> bool:
        if self.show_windows:
            return True
        try:
            return self.debug_pub.get_subscription_count() > 0
        except AttributeError:
            return True

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.fallback_kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.fallback_kernel, iterations=1)
        return mask

    def _largest_component(self, mask: np.ndarray) -> int:
        if np.count_nonzero(mask) == 0:
            return 0
        component_count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if component_count <= 1:
            return 0
        return int(stats[1:, cv2.CC_STAT_AREA].max())

    def _highlight_masks(
        self,
        enhanced: np.ndarray,
        masks: dict[str, np.ndarray],
        scores: dict[str, float],
    ) -> np.ndarray:
        highlighted = enhanced.copy()
        dimmed = (highlighted * 0.25).astype(np.uint8)
        highlighted = cv2.addWeighted(dimmed, 1.0, highlighted, 0.6, 0.0)

        overlays = {
            'red': (0, 0, 255),
            'yellow': (0, 255, 255),
            'green': (0, 255, 0),
        }
        for name in COLOR_ORDER:
            mask = masks[name]
            if np.count_nonzero(mask) == 0:
                continue
            color = np.zeros_like(highlighted)
            color[:, :] = overlays[name]
            color_strength = 0.30 + 0.50 * float(scores[name])
            blended = cv2.addWeighted(highlighted, 1.0, color, color_strength, 0.0)
            highlighted[mask > 0] = blended[mask > 0]

        return highlighted

    def _fit_to_canvas(self, image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        if image.size == 0:
            return np.zeros((target_height, target_width, 3), dtype=np.uint8)

        source_height, source_width = image.shape[:2]
        scale = min(target_width / float(source_width), target_height / float(source_height))
        new_width = max(1, int(round(source_width * scale)))
        new_height = max(1, int(round(source_height * scale)))
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        x0 = (target_width - new_width) // 2
        y0 = (target_height - new_height) // 2
        canvas[y0:y0 + new_height, x0:x0 + new_width] = resized
        return canvas

    def _state_color(self, state: int) -> tuple[int, int, int]:
        return STATE_COLORS.get(state, STATE_COLORS[STATE_UNKNOWN])

    def _draw_box_label(
        self,
        image: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.42
        thickness = 1
        text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
        label_x = max(4, x)
        label_y = max(24, y - 4)
        y0 = label_y - text_size[1] - 6
        y1 = label_y + baseline + 2
        x1 = min(image.shape[1] - 4, label_x + text_size[0] + 10)
        cv2.rectangle(image, (label_x, y0), (x1, y1), color, -1)
        cv2.putText(
            image,
            text,
            (label_x + 5, label_y - 2),
            font,
            scale,
            (16, 16, 16),
            thickness,
            cv2.LINE_AA,
        )

    def _draw_debug_inset(
        self,
        image: np.ndarray,
        inset: np.ndarray,
        title: str,
    ) -> None:
        inset_height, inset_width = inset.shape[:2]
        title_height = 18
        margin = 12
        x0 = max(margin, image.shape[1] - inset_width - margin)
        y0 = margin + title_height
        if y0 + inset_height + margin > image.shape[0]:
            y0 = max(margin + title_height, image.shape[0] - inset_height - margin)

        panel_x0 = max(0, x0 - 4)
        panel_y0 = max(0, y0 - 22)
        panel_x1 = min(image.shape[1], x0 + inset_width + 4)
        panel_y1 = min(image.shape[0], y0 + inset_height + 4)

        cv2.rectangle(image, (panel_x0, panel_y0), (panel_x1, panel_y1), (10, 12, 16), -1)
        cv2.rectangle(image, (panel_x0, panel_y0), (panel_x1, panel_y1), (120, 120, 120), 1)
        cv2.putText(
            image,
            title,
            (panel_x0 + 8, panel_y0 + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (245, 247, 250),
            1,
            cv2.LINE_AA,
        )
        image[y0:y0 + inset_height, x0:x0 + inset_width] = inset

    def _point_in_window(
        self,
        x: float,
        y: float,
        window: tuple[int, int, int, int],
    ) -> bool:
        x0, y0, x1, y1 = window
        return x0 <= x <= x1 and y0 <= y <= y1

    def _tracking_similarity(
        self,
        current_box: tuple[int, int, int, int],
        previous_box: tuple[int, int, int, int] | None,
        frame_shape: tuple[int, ...],
    ) -> float:
        if previous_box is None:
            return 1.0

        current_center = self._box_center(current_box)
        previous_center = self._box_center(previous_box)
        diagonal = max(1.0, math.hypot(frame_shape[1], frame_shape[0]))
        center_distance = math.hypot(
            current_center[0] - previous_center[0],
            current_center[1] - previous_center[1],
        )
        center_score = clamp(1.0 - (center_distance / diagonal), 0.0, 1.0)
        overlap_score = self._iou(current_box, previous_box)
        return max(center_score, overlap_score)

    def _box_center(self, box: tuple[int, int, int, int]) -> tuple[float, float]:
        x_a, y_a, x_b, y_b = box
        return 0.5 * (x_a + x_b), 0.5 * (y_a + y_b)

    def _iou(
        self,
        box_a: tuple[int, int, int, int],
        box_b: tuple[int, int, int, int],
    ) -> float:
        ax0, ay0, ax1, ay1 = box_a
        bx0, by0, bx1, by1 = box_b
        inter_x0 = max(ax0, bx0)
        inter_y0 = max(ay0, by0)
        inter_x1 = min(ax1, bx1)
        inter_y1 = min(ay1, by1)
        inter_width = max(0, inter_x1 - inter_x0)
        inter_height = max(0, inter_y1 - inter_y0)
        inter_area = inter_width * inter_height
        area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
        area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return float(inter_area) / float(union)

    def _majority_state(self, values: deque[int]) -> int:
        if not values:
            return STATE_UNKNOWN
        counts = Counter(values)
        return max(counts.items(), key=lambda item: (item[1], item[0] != STATE_UNKNOWN))[0]

    def _now_ns(self) -> int:
        return int(self.get_clock().now().nanoseconds)

    def _stamp_to_ns(self, msg: Image) -> int:
        stamp = msg.header.stamp
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _ns_to_ms(self, value: int) -> float:
        return value / 1e6


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = TLFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.show_windows:
            try:
                cv2.destroyAllWindows()
            except Exception:  # noqa: BLE001
                pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
