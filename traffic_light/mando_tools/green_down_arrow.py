from __future__ import annotations

import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from .workspace_paths import default_runtime_image_topic
from .workspace_paths import local_python_deps_path

deps_path = local_python_deps_path()
if deps_path.exists():
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
from std_msgs.msg import String


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class ArrowCandidate:
    detected: bool
    score: float
    reason: str
    bbox: tuple[int, int, int, int] | None = None
    contour: np.ndarray | None = None
    tip_point: tuple[int, int] | None = None
    area: float = 0.0


class GreenDownArrowNode(Node):
    def __init__(self) -> None:
        super().__init__('green_down_arrow_detector')

        self.bridge = CvBridge()
        self.latest_msg: Image | None = None
        self.processing = False
        self.last_positive_ns: int | None = None

        self.image_topic = str(self._declare_param('image_topic', default_runtime_image_topic()))
        self.show_windows = bool(self._declare_param('show_windows', False))
        self.publish_debug_image = bool(self._declare_param('publish_debug_image', True))
        self.max_fps = float(self._declare_param('max_fps', 15.0))

        self.roi_top_ratio = float(self._declare_param('roi_top_ratio', 0.00))
        self.roi_bottom_ratio = float(self._declare_param('roi_bottom_ratio', 0.70))
        self.roi_left_ratio = float(self._declare_param('roi_left_ratio', 0.00))
        self.roi_right_ratio = float(self._declare_param('roi_right_ratio', 1.00))

        self.h_min = int(self._declare_param('h_min', 40))
        self.h_max = int(self._declare_param('h_max', 95))
        self.s_min = int(self._declare_param('s_min', 90))
        self.s_max = int(self._declare_param('s_max', 255))
        self.v_min = int(self._declare_param('v_min', 90))
        self.v_max = int(self._declare_param('v_max', 255))

        self.morph_open_iterations = int(self._declare_param('morph_open_iterations', 1))
        self.morph_close_iterations = int(self._declare_param('morph_close_iterations', 2))
        self.dilate_iterations = int(self._declare_param('dilate_iterations', 1))
        self.kernel_size = int(self._declare_param('kernel_size', 5))

        self.min_area_px = int(self._declare_param('min_area_px', 120))
        self.min_side_px = int(self._declare_param('min_side_px', 12))
        self.min_bbox_fill_ratio = float(self._declare_param('min_bbox_fill_ratio', 0.12))
        self.max_bbox_fill_ratio = float(self._declare_param('max_bbox_fill_ratio', 0.78))
        self.min_aspect_ratio = float(self._declare_param('min_aspect_ratio', 0.70))
        self.max_aspect_ratio = float(self._declare_param('max_aspect_ratio', 1.80))
        self.min_tip_prominence = float(self._declare_param('min_tip_prominence', 0.12))
        self.max_tip_center_offset = float(self._declare_param('max_tip_center_offset', 0.45))
        self.min_bottom_top_width_ratio = float(
            self._declare_param('min_bottom_top_width_ratio', 1.25)
        )
        self.min_template_iou = float(self._declare_param('min_template_iou', 0.28))
        self.min_shape_score = float(self._declare_param('min_shape_score', 0.30))
        self.detection_score_threshold = float(
            self._declare_param('detection_score_threshold', 0.58)
        )

        self.template_size = int(self._declare_param('template_size', 96))
        self.majority_window = int(self._declare_param('majority_window', 5))
        self.required_positive_count = int(self._declare_param('required_positive_count', 3))
        self.hold_ms = int(self._declare_param('hold_ms', 250))

        self.detected_topic = str(
            self._declare_param('detected_topic', '/tl/green_down_arrow_detected')
        )
        self.score_topic = str(self._declare_param('score_topic', '/tl/green_down_arrow_score'))
        self.reason_topic = str(self._declare_param('reason_topic', '/tl/green_down_arrow_reason'))
        self.debug_image_topic = str(
            self._declare_param('debug_image_topic', '/tl/green_down_arrow_debug')
        )

        kernel_size = max(3, self.kernel_size)
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        self.template_mask, self.template_contour = self._build_down_arrow_template(
            self.template_size
        )
        self.history: deque[int] = deque(maxlen=max(1, self.majority_window))

        self.get_logger().info(f'Subscribing to image topic: {self.image_topic}')
        self.get_logger().info(
            'Green down arrow detector uses HSV+shape rules only, without model inference.'
        )

        self.create_subscription(
            Image,
            self.image_topic,
            self._image_callback,
            qos_profile_sensor_data,
        )

        status_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        image_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        self.detected_pub = self.create_publisher(Bool, self.detected_topic, status_qos)
        self.score_pub = self.create_publisher(Float32, self.score_topic, status_qos)
        self.reason_pub = self.create_publisher(String, self.reason_topic, status_qos)
        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, image_qos)

        timer_period = 1.0 / max(self.max_fps, 0.1)
        self.create_timer(timer_period, self._process_latest_frame)

    def _declare_param(self, name: str, default_value: Any) -> Any:
        return self.declare_parameter(name, default_value).value

    def _image_callback(self, msg: Image) -> None:
        self.latest_msg = msg

    def _process_latest_frame(self) -> None:
        if self.processing or self.latest_msg is None:
            return

        msg = self.latest_msg
        self.latest_msg = None
        self.processing = True

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if frame is None or frame.size == 0:
                self._publish(False, 0.0, 'invalid_frame empty_image')
                return

            roi, roi_rect = self._extract_roi(frame)
            if roi.size == 0:
                self._publish(False, 0.0, 'invalid_frame empty_roi')
                return

            green_mask = self._build_green_mask(roi)
            candidate = self._find_best_candidate(green_mask)
            stable_detected = self._update_temporal_state(candidate.detected)
            reason = self._compose_reason(candidate, stable_detected)

            self._publish(stable_detected, candidate.score, reason)

            if self.publish_debug_image:
                debug_image = self._draw_debug(frame, roi_rect, candidate, stable_detected)
                debug_msg = self.bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
                debug_msg.header = msg.header
                self.debug_pub.publish(debug_msg)

            if self.show_windows:
                debug_image = self._draw_debug(frame, roi_rect, candidate, stable_detected)
                cv2.imshow('green_down_arrow_debug', debug_image)
                cv2.waitKey(1)
        finally:
            self.processing = False

    def _extract_roi(self, frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        height, width = frame.shape[:2]
        top = int(clamp(self.roi_top_ratio, 0.0, 1.0) * height)
        bottom = int(clamp(self.roi_bottom_ratio, 0.0, 1.0) * height)
        left = int(clamp(self.roi_left_ratio, 0.0, 1.0) * width)
        right = int(clamp(self.roi_right_ratio, 0.0, 1.0) * width)

        top = max(0, min(top, height - 1))
        bottom = max(top + 1, min(bottom, height))
        left = max(0, min(left, width - 1))
        right = max(left + 1, min(right, width))

        return frame[top:bottom, left:right].copy(), (left, top, right, bottom)

    def _build_green_mask(self, roi: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            (self.h_min, self.s_min, self.v_min),
            (self.h_max, self.s_max, self.v_max),
        )

        if self.morph_open_iterations > 0:
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                self.morph_kernel,
                iterations=self.morph_open_iterations,
            )
        if self.morph_close_iterations > 0:
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                self.morph_kernel,
                iterations=self.morph_close_iterations,
            )
        if self.dilate_iterations > 0:
            mask = cv2.dilate(mask, self.morph_kernel, iterations=self.dilate_iterations)

        return mask

    def _find_best_candidate(self, mask: np.ndarray) -> ArrowCandidate:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_candidate = ArrowCandidate(False, 0.0, 'no_green_blob')

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area_px:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w < self.min_side_px or h < self.min_side_px:
                continue

            bbox_area = float(w * h)
            fill_ratio = area / max(bbox_area, 1.0)
            if fill_ratio < self.min_bbox_fill_ratio or fill_ratio > self.max_bbox_fill_ratio:
                continue

            aspect_ratio = h / max(float(w), 1.0)
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue

            local_mask = np.zeros((h, w), dtype=np.uint8)
            shifted_contour = contour - np.array([[[x, y]]], dtype=np.int32)
            cv2.drawContours(local_mask, [shifted_contour], -1, 255, thickness=cv2.FILLED)

            template_iou, shape_score = self._measure_template_similarity(local_mask)
            if template_iou < self.min_template_iou or shape_score < self.min_shape_score:
                continue

            tip_prominence, tip_center_offset, bottom_top_width_ratio, shaft_score, tip_point = (
                self._measure_arrow_geometry(local_mask, shifted_contour)
            )

            if tip_prominence < self.min_tip_prominence:
                continue
            if tip_center_offset > self.max_tip_center_offset:
                continue
            if bottom_top_width_ratio < self.min_bottom_top_width_ratio:
                continue

            tip_score = clamp(
                (tip_prominence - self.min_tip_prominence) / 0.22,
                0.0,
                1.0,
            )
            center_score = 1.0 - clamp(
                tip_center_offset / max(self.max_tip_center_offset, 1e-6),
                0.0,
                1.0,
            )
            width_ratio_score = clamp(
                (bottom_top_width_ratio - self.min_bottom_top_width_ratio) / 1.2,
                0.0,
                1.0,
            )

            score = (
                0.30 * template_iou
                + 0.25 * shape_score
                + 0.15 * tip_score
                + 0.15 * center_score
                + 0.10 * width_ratio_score
                + 0.05 * shaft_score
            )

            reason = (
                f'score={score:.2f} iou={template_iou:.2f} shape={shape_score:.2f} '
                f'tip={tip_prominence:.2f} center={tip_center_offset:.2f} '
                f'width_ratio={bottom_top_width_ratio:.2f}'
            )

            detected = score >= self.detection_score_threshold
            candidate = ArrowCandidate(
                detected=detected,
                score=score,
                reason=reason,
                bbox=(x, y, w, h),
                contour=contour,
                tip_point=(x + tip_point[0], y + tip_point[1]),
                area=area,
            )

            if candidate.score > best_candidate.score:
                best_candidate = candidate

        if best_candidate.bbox is None:
            return best_candidate

        if not best_candidate.detected:
            best_candidate.reason = f'best_candidate_rejected {best_candidate.reason}'
        else:
            best_candidate.reason = f'detected {best_candidate.reason}'
        return best_candidate

    def _measure_template_similarity(self, mask: np.ndarray) -> tuple[float, float]:
        resized = cv2.resize(
            mask,
            (self.template_size, self.template_size),
            interpolation=cv2.INTER_NEAREST,
        )
        resized_binary = resized > 0
        template_binary = self.template_mask > 0
        intersection = np.logical_and(resized_binary, template_binary).sum()
        union = np.logical_or(resized_binary, template_binary).sum()
        template_iou = float(intersection / union) if union > 0 else 0.0

        resized_contours, _ = cv2.findContours(
            resized.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not resized_contours:
            return template_iou, 0.0

        largest = max(resized_contours, key=cv2.contourArea)
        match_distance = cv2.matchShapes(
            largest,
            self.template_contour,
            cv2.CONTOURS_MATCH_I1,
            0.0,
        )
        shape_score = 1.0 / (1.0 + 5.0 * float(match_distance))
        return template_iou, shape_score

    def _measure_arrow_geometry(
        self,
        mask: np.ndarray,
        contour: np.ndarray,
    ) -> tuple[float, float, float, float, tuple[int, int]]:
        moments = cv2.moments(contour)
        if abs(moments['m00']) < 1e-6:
            cy = mask.shape[0] / 2.0
        else:
            cy = moments['m01'] / moments['m00']

        tip_idx = contour[:, :, 1].argmax()
        tip_point = tuple(int(v) for v in contour[tip_idx][0])

        height, width = mask.shape[:2]
        tip_prominence = (tip_point[1] - cy) / max(float(height), 1.0)
        tip_center_offset = abs(tip_point[0] - (width / 2.0)) / max(width / 2.0, 1.0)

        row_widths = (mask > 0).sum(axis=1).astype(np.float32)
        top_end = max(1, int(height * 0.45))
        bottom_start = min(height - 1, int(height * 0.55))
        top_peak = float(row_widths[:top_end].max()) if top_end > 0 else 0.0
        bottom_peak = float(row_widths[bottom_start:].max()) if bottom_start < height else 0.0
        bottom_top_width_ratio = bottom_peak / max(top_peak, 1.0)

        center_band_half = max(1, int(width * 0.10))
        center_x = width // 2
        band_left = max(0, center_x - center_band_half)
        band_right = min(width, center_x + center_band_half + 1)
        shaft_region = mask[:top_end, band_left:band_right]
        shaft_score = float((shaft_region > 0).mean()) if shaft_region.size > 0 else 0.0

        return (
            float(tip_prominence),
            float(tip_center_offset),
            float(bottom_top_width_ratio),
            float(shaft_score),
            tip_point,
        )

    def _update_temporal_state(self, instant_detected: bool) -> bool:
        self.history.append(1 if instant_detected else 0)
        now_ns = self.get_clock().now().nanoseconds
        if instant_detected:
            self.last_positive_ns = now_ns

        stable_by_majority = sum(self.history) >= self.required_positive_count
        if stable_by_majority:
            return True

        if self.last_positive_ns is None:
            return False

        hold_ns = int(self.hold_ms * 1e6)
        return now_ns - self.last_positive_ns <= hold_ns

    def _compose_reason(self, candidate: ArrowCandidate, stable_detected: bool) -> str:
        instant = 'true' if candidate.detected else 'false'
        stable = 'true' if stable_detected else 'false'
        positives = sum(self.history)
        return (
            f'stable={stable} instant={instant} history={positives}/{len(self.history)} '
            f'{candidate.reason}'
        )

    def _draw_debug(
        self,
        frame: np.ndarray,
        roi_rect: tuple[int, int, int, int],
        candidate: ArrowCandidate,
        stable_detected: bool,
    ) -> np.ndarray:
        debug = frame.copy()
        left, top, right, bottom = roi_rect
        cv2.rectangle(debug, (left, top), (right, bottom), (255, 180, 0), 2)
        cv2.putText(
            debug,
            'search ROI',
            (left + 4, max(18, top - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 180, 0),
            2,
            cv2.LINE_AA,
        )

        status_text = f'DOWN_ARROW: {"ON" if stable_detected else "OFF"} score={candidate.score:.2f}'
        status_color = (0, 220, 0) if stable_detected else (0, 120, 255)
        cv2.putText(
            debug,
            status_text,
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            status_color,
            2,
            cv2.LINE_AA,
        )

        if candidate.bbox is not None:
            x, y, w, h = candidate.bbox
            global_x = left + x
            global_y = top + y
            box_color = (0, 255, 0) if candidate.detected else (0, 165, 255)
            cv2.rectangle(
                debug,
                (global_x, global_y),
                (global_x + w, global_y + h),
                box_color,
                2,
            )

            if candidate.contour is not None:
                contour = candidate.contour.copy()
                contour[:, :, 0] += left
                contour[:, :, 1] += top
                cv2.drawContours(debug, [contour], -1, box_color, 2)

            if candidate.tip_point is not None:
                tip_x = left + candidate.tip_point[0]
                tip_y = top + candidate.tip_point[1]
                cv2.circle(debug, (tip_x, tip_y), 5, (0, 0, 255), -1)

            cv2.putText(
                debug,
                candidate.reason[:120],
                (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                debug,
                candidate.reason,
                (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

        return debug

    def _publish(self, detected: bool, score: float, reason: str) -> None:
        self.detected_pub.publish(Bool(data=bool(detected)))
        self.score_pub.publish(Float32(data=float(score)))
        self.reason_pub.publish(String(data=reason))

    def _build_down_arrow_template(self, size: int) -> tuple[np.ndarray, np.ndarray]:
        mask = np.zeros((size, size), dtype=np.uint8)
        center_x = size // 2
        shaft_half = int(size * 0.12)
        shaft_top = int(size * 0.10)
        shaft_bottom = int(size * 0.56)
        head_half = int(size * 0.32)
        head_top = shaft_bottom
        head_bottom = int(size * 0.92)

        points = np.array(
            [
                [center_x - shaft_half, shaft_top],
                [center_x + shaft_half, shaft_top],
                [center_x + shaft_half, shaft_bottom],
                [center_x + head_half, head_top],
                [center_x, head_bottom],
                [center_x - head_half, head_top],
                [center_x - shaft_half, shaft_bottom],
            ],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(mask, points, 255)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        template_contour = max(contours, key=cv2.contourArea)
        return mask, template_contour


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
