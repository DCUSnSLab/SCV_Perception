from __future__ import annotations

import itertools
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from .workspace_paths import default_runtime_image_topic
from .workspace_paths import local_python_deps_path

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
from std_msgs.msg import Float32
from std_msgs.msg import Int32
from std_msgs.msg import String


GREEN_ARROW = 0
RED_X = 1
NAMES = {GREEN_ARROW: 'GREEN_ARROW', RED_X: 'RED_X'}
COLORS = {GREEN_ARROW: (0, 255, 0), RED_X: (0, 0, 255)}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class ColorCandidate:
    class_id: int
    center: tuple[float, float]
    side: float
    score: float


@dataclass(frozen=True)
class RigCandidate:
    boxes: tuple[tuple[int, int, int, int], ...]
    score: float


@dataclass(frozen=True)
class PanelColor:
    class_id: int | None
    red_ratio: float
    green_ratio: float
    vivid_green_ratio: float
    confidence: float


def build_color_masks(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 70, 65), (15, 255, 255)) | cv2.inRange(
        hsv, (165, 70, 65), (179, 255, 255)
    )
    green = cv2.inRange(hsv, (30, 70, 65), (100, 255, 255))
    vivid_green = cv2.inRange(hsv, (30, 90, 65), (100, 255, 255))
    return red, green, vivid_green


def _color_candidates(
    mask: np.ndarray,
    class_id: int,
    offset: tuple[int, int],
    min_pixels: int,
    max_side_px: int,
) -> list[ColorCandidate]:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    connected = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    connected = cv2.dilate(connected, np.ones((3, 3), np.uint8), iterations=1)
    count, _, stats, _ = cv2.connectedComponentsWithStats(connected, connectivity=8)
    candidates = []
    for component in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[component])
        side = float(max(width, height))
        if area < min_pixels or side < 7 or side > max_side_px:
            continue
        aspect = width / max(float(height), 1.0)
        if not 0.32 <= aspect <= 3.2:
            continue
        raw_pixels = cv2.countNonZero(mask[y : y + height, x : x + width])
        if raw_pixels < min_pixels:
            continue
        candidates.append(
            ColorCandidate(
                class_id=class_id,
                center=(x + offset[0] + width / 2.0, y + offset[1] + height / 2.0),
                side=side,
                score=min(1.0, raw_pixels / max(side * side * 0.20, 1.0)),
            )
        )
    return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)[:24]


def find_three_panel_rig(
    image: np.ndarray,
    roi: tuple[int, int, int, int],
    min_pixels: int = 10,
    max_side_px: int = 180,
) -> RigCandidate | None:
    left, top, right, bottom = roi
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        return None
    red, _, vivid_green = build_color_masks(crop)
    candidates = _color_candidates(red, RED_X, (left, top), min_pixels, max_side_px)
    candidates += _color_candidates(
        vivid_green,
        GREEN_ARROW,
        (left, top),
        min_pixels,
        max_side_px,
    )

    best: RigCandidate | None = None
    for group in itertools.combinations(candidates, 3):
        ordered = sorted(group, key=lambda candidate: candidate.center[0])
        classes = [candidate.class_id for candidate in ordered]
        if classes.count(GREEN_ARROW) != 1 or classes.count(RED_X) != 2:
            continue
        if classes[-1] == GREEN_ARROW:
            continue

        sides = np.array([candidate.side for candidate in ordered], dtype=np.float32)
        centers_x = np.array([candidate.center[0] for candidate in ordered], dtype=np.float32)
        centers_y = np.array([candidate.center[1] for candidate in ordered], dtype=np.float32)
        mean_side = float(sides.mean())
        gaps = np.diff(centers_x)
        if centers_x[-1] - centers_x[0] < 100.0:
            continue
        if sides.max() / max(float(sides.min()), 1.0) > 2.2:
            continue
        if centers_y.max() - centers_y.min() > max(12.0, mean_side * 0.75):
            continue
        if not all(mean_side * 1.25 <= gap <= mean_side * 5.2 for gap in gaps):
            continue
        if abs(float(gaps[0] - gaps[1])) / max(float(gaps.mean()), 1.0) > 0.28:
            continue

        alignment = 1.0 - clamp(
            float(centers_y.max() - centers_y.min()) / max(mean_side * 0.75, 1.0),
            0.0,
            1.0,
        )
        spacing = 1.0 - clamp(
            abs(float(gaps[0] - gaps[1])) / max(float(gaps.mean()), 1.0),
            0.0,
            1.0,
        )
        size_match = 1.0 - clamp(
            float(sides.max() - sides.min()) / max(mean_side, 1.0),
            0.0,
            1.0,
        )
        color_strength = float(np.mean([candidate.score for candidate in ordered]))
        score = 0.32 * alignment + 0.30 * spacing + 0.23 * size_match + 0.15 * color_strength
        if score < 0.62:
            continue

        panel_side = max(14, min(int(round(mean_side * 1.28)), max_side_px))
        boxes = tuple(
            (
                round(candidate.center[0] - panel_side / 2),
                round(candidate.center[1] - panel_side / 2),
                panel_side,
                panel_side,
            )
            for candidate in ordered
        )
        rig = RigCandidate(boxes, score)
        if best is None or rig.score > best.score:
            best = rig
    return best


def classify_panel(image: np.ndarray, box: tuple[int, int, int, int]) -> PanelColor:
    x, y, width, height = box
    image_height, image_width = image.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(image_width, x + width), min(image_height, y + height)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return PanelColor(None, 0.0, 0.0, 0.0, 0.0)

    red, green, vivid_green = build_color_masks(crop)
    red_ratio = cv2.countNonZero(red) / red.size
    green_ratio = cv2.countNonZero(green) / green.size
    vivid_green_ratio = cv2.countNonZero(vivid_green) / vivid_green.size

    class_id = None
    confidence = 0.0
    if red_ratio >= 0.040 and red_ratio >= green_ratio * 1.35:
        class_id = RED_X
        confidence = min(1.0, red_ratio / 0.20)
    elif vivid_green_ratio >= 0.030 and green_ratio >= red_ratio * 1.35:
        class_id = GREEN_ARROW
        confidence = min(1.0, vivid_green_ratio / 0.12)
    return PanelColor(class_id, red_ratio, green_ratio, vivid_green_ratio, confidence)


def rigs_are_close(first: RigCandidate, second: RigCandidate) -> bool:
    first_center = np.mean(
        [(x + width / 2, y + height / 2) for x, y, width, height in first.boxes],
        axis=0,
    )
    second_center = np.mean(
        [(x + width / 2, y + height / 2) for x, y, width, height in second.boxes],
        axis=0,
    )
    first_side = float(np.mean([box[2] for box in first.boxes]))
    second_side = float(np.mean([box[2] for box in second.boxes]))
    center_distance = float(np.linalg.norm(first_center - second_center))
    size_ratio = max(first_side, second_side) / max(min(first_side, second_side), 1.0)
    return center_distance <= max(30.0, first_side * 1.2) and size_ratio <= 1.8


class GreenDownArrowNode(Node):
    def __init__(self) -> None:
        super().__init__('green_down_arrow_detector')
        self.bridge = CvBridge()
        self.latest_msg: Image | None = None
        self.processing = False
        self.last_image_ns = time.monotonic_ns()

        self.image_topic = str(self._param('image_topic', default_runtime_image_topic()))
        self.show_windows = bool(self._param('show_windows', False))
        self.publish_debug_image = bool(self._param('publish_debug_image', True))
        self.max_fps = float(self._param('max_fps', 15.0))
        self.input_timeout_s = float(self._param('input_timeout_s', 3.0))
        self.roi_top_ratio = float(self._param('roi_top_ratio', 0.00))
        self.roi_bottom_ratio = float(self._param('roi_bottom_ratio', 0.38))
        self.roi_left_ratio = float(self._param('roi_left_ratio', 0.30))
        self.roi_right_ratio = float(self._param('roi_right_ratio', 0.75))
        self.min_pixels = int(self._param('min_pixels', 10))
        self.max_panel_side_px = int(self._param('max_panel_side_px', 180))
        self.rig_hold_s = float(self._param('rig_hold_s', 1.0))
        self.majority_window = int(self._param('majority_window', 5))
        self.required_positive_count = int(self._param('required_positive_count', 2))

        self.detected_topic = str(
            self._param('detected_topic', '/tl/green_down_arrow_detected')
        )
        self.score_topic = str(self._param('score_topic', '/tl/green_down_arrow_score'))
        self.reason_topic = str(self._param('reason_topic', '/tl/green_down_arrow_reason'))
        self.debug_image_topic = str(
            self._param('debug_image_topic', '/tl/green_down_arrow_debug')
        )
        self.red_x_topic = str(self._param('red_x_topic', '/tl/red_x_detected'))
        self.red_x_count_topic = str(self._param('red_x_count_topic', '/tl/red_x_count'))

        self.green_history: deque[int] = deque(maxlen=max(1, self.majority_window))
        self.red_history: deque[int] = deque(maxlen=max(1, self.majority_window))
        self.rig: RigCandidate | None = None
        self.pending_rig: RigCandidate | None = None
        self.pending_rig_count = 0
        self.rig_misses = 0

        status_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        image_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.green_pub = self.create_publisher(Bool, self.detected_topic, status_qos)
        self.score_pub = self.create_publisher(Float32, self.score_topic, status_qos)
        self.reason_pub = self.create_publisher(String, self.reason_topic, status_qos)
        self.red_pub = self.create_publisher(Bool, self.red_x_topic, status_qos)
        self.red_count_pub = self.create_publisher(Int32, self.red_x_count_topic, status_qos)
        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, image_qos)

        self.create_subscription(Image, self.image_topic, self._image_callback, qos_profile_sensor_data)
        self.create_timer(1.0 / max(self.max_fps, 0.1), self._process_latest_frame)
        self.create_timer(0.5, self._publish_timeout_if_needed)
        self.get_logger().info(
            f'Color-only three-panel detector subscribing to {self.image_topic}'
        )

    def _param(self, name: str, default: Any) -> Any:
        return self.declare_parameter(name, default).value

    def _image_callback(self, msg: Image) -> None:
        self.latest_msg = msg
        self.last_image_ns = time.monotonic_ns()

    def _roi(self, image: np.ndarray) -> tuple[int, int, int, int]:
        height, width = image.shape[:2]
        left = int(clamp(self.roi_left_ratio, 0.0, 1.0) * width)
        right = int(clamp(self.roi_right_ratio, 0.0, 1.0) * width)
        top = int(clamp(self.roi_top_ratio, 0.0, 1.0) * height)
        bottom = int(clamp(self.roi_bottom_ratio, 0.0, 1.0) * height)
        return left, top, max(left + 1, right), max(top + 1, bottom)

    def _process_latest_frame(self) -> None:
        if self.processing or self.latest_msg is None:
            return
        msg = self.latest_msg
        self.latest_msg = None
        self.processing = True
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            roi = self._roi(image)
            acquired = find_three_panel_rig(
                image,
                roi,
                min_pixels=self.min_pixels,
                max_side_px=self.max_panel_side_px,
            )
            if self.rig is None and acquired is not None:
                if self.pending_rig is not None and rigs_are_close(self.pending_rig, acquired):
                    self.pending_rig_count += 1
                else:
                    self.pending_rig_count = 1
                self.pending_rig = acquired
                if self.pending_rig_count >= 2:
                    self.rig = acquired
                    self.rig_misses = 0
            elif self.rig is not None and acquired is not None and rigs_are_close(self.rig, acquired):
                self.rig = acquired
                self.rig_misses = 0
                self.pending_rig = None
                self.pending_rig_count = 0
            else:
                self.rig_misses += 1
                if self.rig is None and acquired is None:
                    self.pending_rig = None
                    self.pending_rig_count = 0
                if self.rig_misses > max(2, round(self.rig_hold_s * self.max_fps)):
                    self.rig = None
                    self.pending_rig = acquired
                    self.pending_rig_count = int(acquired is not None)

            panels = (
                [classify_panel(image, box) for box in self.rig.boxes]
                if self.rig is not None
                else []
            )
            instant_green = any(panel.class_id == GREEN_ARROW for panel in panels)
            instant_red_count = min(2, sum(panel.class_id == RED_X for panel in panels))
            self.green_history.append(int(instant_green))
            self.red_history.append(int(instant_red_count > 0))
            stable_green = sum(self.green_history) >= self.required_positive_count
            stable_red = sum(self.red_history) >= self.required_positive_count
            score = max((panel.confidence for panel in panels), default=0.0)
            reason = self._reason(acquired, panels, stable_green, stable_red)
            self._publish(stable_green, stable_red, instant_red_count, score, reason)
            if self.publish_debug_image:
                debug = self._draw_debug(image, roi, panels, stable_green, stable_red, reason)
                debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
                debug_msg.header = msg.header
                self.debug_pub.publish(debug_msg)
            if self.show_windows:
                cv2.imshow(
                    'green_down_arrow_debug',
                    self._draw_debug(image, roi, panels, stable_green, stable_red, reason),
                )
                cv2.waitKey(1)
        finally:
            self.processing = False

    def _reason(
        self,
        acquired: RigCandidate | None,
        panels: list[PanelColor],
        green: bool,
        red: bool,
    ) -> str:
        source = 'color_rig' if acquired is not None else ('held_rig' if self.rig is not None else 'no_rig')
        ratios = ';'.join(
            f'r={panel.red_ratio:.3f},g={panel.green_ratio:.3f},v={panel.vivid_green_ratio:.3f}'
            for panel in panels
        )
        return (
            f'green={str(green).lower()} red_x={str(red).lower()} source={source} '
            f'rig_score={(self.rig.score if self.rig else 0.0):.2f} panels=[{ratios}]'
        )

    def _publish(
        self,
        green: bool,
        red: bool,
        red_count: int,
        score: float,
        reason: str,
    ) -> None:
        self.green_pub.publish(Bool(data=green))
        self.red_pub.publish(Bool(data=red))
        self.red_count_pub.publish(Int32(data=red_count))
        self.score_pub.publish(Float32(data=float(score)))
        self.reason_pub.publish(String(data=reason))

    def _publish_timeout_if_needed(self) -> None:
        elapsed_s = (time.monotonic_ns() - self.last_image_ns) / 1e9
        if elapsed_s <= self.input_timeout_s:
            return
        self.rig = None
        self.pending_rig = None
        self.pending_rig_count = 0
        self.green_history.clear()
        self.red_history.clear()
        self._publish(False, False, 0, 0.0, f'input_timeout elapsed={elapsed_s:.1f}s')

    def _draw_debug(
        self,
        image: np.ndarray,
        roi: tuple[int, int, int, int],
        panels: list[PanelColor],
        green: bool,
        red: bool,
        reason: str,
    ) -> np.ndarray:
        debug = image.copy()
        left, top, right, bottom = roi
        cv2.rectangle(debug, (left, top), (right, bottom), (255, 180, 0), 2)
        if self.rig is not None:
            for box, panel in zip(self.rig.boxes, panels):
                x, y, width, height = box
                color = COLORS.get(panel.class_id, (160, 160, 160))
                name = NAMES.get(panel.class_id, 'UNKNOWN')
                cv2.rectangle(debug, (x, y), (x + width, y + height), color, 2)
                cv2.putText(
                    debug,
                    name,
                    (x, max(14, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )
        cv2.putText(
            debug,
            f'ARROW={"ON" if green else "OFF"} RED_X={"ON" if red else "OFF"}',
            (20, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            debug,
            reason[:180],
            (20, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        return debug


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = GreenDownArrowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.show_windows:
            cv2.destroyAllWindows()
        node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
