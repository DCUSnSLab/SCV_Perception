#!/usr/bin/env python3
"""상단 ROI의 빨강/초록 색 영역마다 빨강=0, 초록=1로 발행한다.

처리 흐름: BGR 영상 → ROI → HSV 색 마스크 → 연결 영역 → 추적/EMA → 비트.
검출기는 영상/시간을 입력받고, ROS 노드는 구독·주기 제한·발행을 담당한다.
출력은 확정된 박스만 영상의 왼쪽부터 나열한 가변 길이 배열이다. 미확정은
0으로 대체하지 않으며, 최초 유효 검출 전이나 모든 트랙 만료 시에는 []이다.
사각형 모양은 요구하지 않는다. 가까운 동색 LED를 묶고 주변의 어두운 비율을 검사한다.
어두운 창문/옷의 유색 반사까지 완벽하게 구분하는 의미 기반 검출기는 아니다.
기존 실행 파일·클래스·토픽 이름은 호환성을 위해 유지한다.
"""

from __future__ import annotations

import sys
import time
from math import hypot
from dataclasses import dataclass
from typing import Any

from .workspace_paths import local_python_deps_path
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
from std_msgs.msg import MultiArrayDimension
from std_msgs.msg import MultiArrayLayout
from std_msgs.msg import UInt8MultiArray


@dataclass
class DetectorConfig:
    """검출/추적 설정. ROS 기본값 변경 시 노드의 파라미터 선언도 함께 맞춘다.

    권장 튜닝 순서: ROI → HSV 범위 → 색 영역 크기 필터 → 시간 안정화.
    픽셀 크기 기준은 입력 해상도에 의존하며, 모든 비율의 기준은 아래와 같다.
    """

    # 입력 영상 전체 기준 경계 비율(0~1); 중앙 절반 너비의 상단 1/2만 처리한다.
    roi_top_ratio: float = 0.0
    roi_bottom_ratio: float = 0.50
    roi_left_ratio: float = 0.25
    roi_right_ratio: float = 0.75

    # close는 같은 색의 틈을 메우고 open은 작은 색 잡음을 제거한다.
    morphology_kernel_size: int = 5
    morphology_close_iterations: int = 0
    morphology_open_iterations: int = 0
    led_group_gap_px: int = 6
    surround_margin_px: int = 4
    surround_v_max: int = 70
    surround_min_dark_ratio: float = 0.55

    # 폭/높이는 색 영역 bounding rect, 면적은 영역 내 원본 유색 픽셀 수다.
    # 최대 크기는 절대 픽셀 제한과 ROI 대비 비율 제한을 모두 만족해야 한다.
    min_box_width_px: int = 3
    min_box_height_px: int = 3
    min_box_area_px: int = 12
    max_box_width_px: int = 160
    max_box_height_px: int = 160
    max_box_width_ratio: float = 0.35
    max_box_height_ratio: float = 0.80

    # YOLO 모드에서 박스 내부 색상을 측정할 때만 사용하는 각 변의 여백 비율.
    inner_margin_ratio: float = 0.20
    # uint8 OpenCV HSV: H=0~179, S/V=0~255. 빨강은 hue 양 끝의 합집합이다.
    color_s_min: int = 80
    color_v_min: int = 45
    red_hue_high: int = 10
    red_hue_low_wrap: int = 170
    green_hue_low: int = 35
    green_hue_high: int = 95
    # 점수는 색 영역 bounding rect 대비 해당 연결 영역의 원본 색상 픽셀 비율이다.
    # 우세 점수와 점수 차이가 각각 임계값을 넘어야 갱신한다. 작은 alpha는 반응을 늦춘다.
    color_score_threshold: float = 0.04
    color_hysteresis_delta: float = 0.08
    color_ema_alpha: float = 0.45

    # 중심 매칭 거리=박스 최대 변 길이×비율(최소 12px); 크면 오연결 위험이 커진다.
    match_distance_ratio: float = 2.5
    # 박스의 마지막 관측 이후 유지 시간(초). 마지막 유효 색상의 나이가 아니다.
    hold_timeout_s: float = 0.5


@dataclass
class BoxObservation:
    """현재 프레임 후보. bbox는 전체 영상 좌표 (x0, y0, x1, y1), 끝점 제외.

    점수는 EMA 이전 값이며 track_id/stable_bit은 추적 단계에서 채운다.
    stable_bit=None은 미확정으로, 빨강(0)과 구별해야 한다.
    """

    bbox: tuple[int, int, int, int]
    red_score: float
    green_score: float
    track_id: int | None = None
    stable_bit: int | None = None


@dataclass
class BoxTrack:
    """프레임 간 상태. bbox 좌표계는 BoxObservation과 동일하다.

    last_seen_ns는 time.monotonic_ns() 계열의 박스 관측 시간이다.
    ROS header.stamp/시뮬레이션 시간과 혼용하지 않는다.
    """

    track_id: int
    bbox: tuple[int, int, int, int]
    red_ema: float
    green_ema: float
    stable_bit: int | None
    last_seen_ns: int


def _clamp_ratio(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    first_x0, first_y0, first_x1, first_y1 = first
    second_x0, second_y0, second_x1, second_y1 = second
    intersection_x0 = max(first_x0, second_x0)
    intersection_y0 = max(first_y0, second_y0)
    intersection_x1 = min(first_x1, second_x1)
    intersection_y1 = min(first_y1, second_y1)
    intersection_width = max(0, intersection_x1 - intersection_x0)
    intersection_height = max(0, intersection_y1 - intersection_y0)
    intersection_area = intersection_width * intersection_height
    first_area = max(0, first_x1 - first_x0) * max(0, first_y1 - first_y0)
    second_area = max(0, second_x1 - second_x0) * max(0, second_y1 - second_y0)
    union_area = first_area + second_area - intersection_area
    if union_area <= 0:
        return 0.0
    return intersection_area / union_area


class BlackBoxColorDetector:
    """검출/시간 안정화를 담당하는 상태 객체. 한 영상 스트림당 하나를 사용한다.

    ROS 통신 없이 process()로 합성 영상을 시험할 수 있다. 내부 트랙을 변경하므로
    병렬 호출은 지원하지 않는다. 위치 기반 추적은 교차/가림 시 ID 유지를 보장하지 않는다.
    """

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        kernel_size = max(3, int(self.config.morphology_kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.morphology_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (kernel_size, kernel_size),
        )
        radius = max(0, int(self.config.led_group_gap_px)) // 2
        self.grouping_kernel = (
            cv2.getStructuringElement(cv2.MORPH_RECT, (2 * radius + 1, 2 * radius + 1))
            if radius else None
        )
        self.tracks: list[BoxTrack] = []
        self.next_track_id = 0
        self.last_bits: list[int] = []

    def process(
        self,
        frame: np.ndarray,
        now_ns: int | None = None,
    ) -> tuple[list[int], list[BoxObservation], tuple[int, int, int, int]]:
        """uint8 BGR 영상에서 (비트 배열, 현재 관측 목록, 전체 좌표 ROI)를 반환한다.

        now_ns는 테스트용 시간 주입 인자이며 같은 단조 증가 시간축을 사용한다.
        비트에는 잠시 미검출된 트랙도 포함될 수 있어 관측 목록 길이와 다를 수 있다.
        """
        if frame is None or frame.size == 0 or frame.ndim < 2:
            self._expire_tracks(now_ns or time.monotonic_ns())
            return self._bits_after_tracking(now_ns or time.monotonic_ns()), [], (0, 0, 0, 0)

        timestamp_ns = now_ns if now_ns is not None else time.monotonic_ns()
        roi, roi_bounds = self._extract_roi(frame)
        observations = self._detect_observations(roi, roi_bounds)
        self._update_tracks(observations, timestamp_ns)
        self._expire_tracks(timestamp_ns)
        return self._bits_after_tracking(timestamp_ns), observations, roi_bounds

    def _extract_roi(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """ROI를 복사 없이 잘라낸다. 경계를 보정해 최소 1×1 픽셀을 확보한다."""
        frame_height, frame_width = frame.shape[:2]
        x0 = int(_clamp_ratio(self.config.roi_left_ratio) * frame_width)
        x1 = int(_clamp_ratio(self.config.roi_right_ratio) * frame_width)
        y0 = int(_clamp_ratio(self.config.roi_top_ratio) * frame_height)
        y1 = int(_clamp_ratio(self.config.roi_bottom_ratio) * frame_height)

        x0 = max(0, min(x0, frame_width - 1))
        x1 = max(x0 + 1, min(x1, frame_width))
        y0 = max(0, min(y0, frame_height - 1))
        y1 = max(y0 + 1, min(y1, frame_height))
        return frame[y0:y1, x0:x1], (x0, y0, x1, y1)

    def _detect_observations(
        self,
        roi: np.ndarray,
        roi_bounds: tuple[int, int, int, int],
    ) -> list[BoxObservation]:
        """빨강/초록 마스크를 각각 연결 영역으로 나눠 색 영역당 관측 하나를 만든다.

        서로 닿은 빨강과 초록은 별도로 검출한다. 가까운 동색 발광부는 묶고,
        원본 픽셀로 영역 크기/점수와 주변의 어두운 비율을 확인한다.
        """
        if roi.size == 0:
            return []
        hue, saturation, value = cv2.split(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV))
        valid = (saturation >= self.config.color_s_min) & (value >= self.config.color_v_min)
        masks = [
            valid & ((hue <= self.config.red_hue_high) | (hue >= self.config.red_hue_low_wrap)),
            valid & (hue >= self.config.green_hue_low) & (hue <= self.config.green_hue_high),
        ]
        observations: list[BoxObservation] = []
        roi_height, roi_width = roi.shape[:2]
        roi_x0, roi_y0, _, _ = roi_bounds
        for bit, mask in enumerate(masks):
            raw_mask = mask.astype(np.uint8)
            cleaned = raw_mask
            for operation, iterations in [
                (cv2.MORPH_CLOSE, self.config.morphology_close_iterations),
                (cv2.MORPH_OPEN, self.config.morphology_open_iterations),
            ]:
                if iterations > 0:
                    cleaned = cv2.morphologyEx(
                        cleaned, operation, self.morphology_kernel, iterations=int(iterations),
                    )
            if self.grouping_kernel is not None:
                grouped = cv2.dilate(cleaned, self.grouping_kernel)
            else:
                grouped = cleaned
            count, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
            for label_index in range(1, count):
                left, top, width, height, area = (int(item) for item in stats[label_index])
                if (area < self.config.min_box_area_px
                        or width < self.config.min_box_width_px
                        or height < self.config.min_box_height_px):
                    continue
                component = labels[top:top + height, left:left + width] == label_index
                source_pixels = cv2.bitwise_and(
                    raw_mask[top:top + height, left:left + width], component.astype(np.uint8),
                )
                color_pixels = cv2.countNonZero(source_pixels)
                if color_pixels < self.config.min_box_area_px or color_pixels == 0:
                    continue
                source_left, source_top, width, height = cv2.boundingRect(source_pixels)
                left += source_left
                top += source_top
                if width < self.config.min_box_width_px or height < self.config.min_box_height_px:
                    continue
                if width > self.config.max_box_width_px or height > self.config.max_box_height_px:
                    continue
                if width > roi_width * self.config.max_box_width_ratio:
                    continue
                if height > roi_height * self.config.max_box_height_ratio:
                    continue
                if (self.config.surround_min_dark_ratio > 0.0
                        and self._surrounding_dark_ratio(value, (left, top, left + width, top + height)) < self.config.surround_min_dark_ratio):
                    continue
                score = float(color_pixels / (width * height))
                observations.append(BoxObservation(
                    bbox=(left + roi_x0, top + roi_y0, left + width + roi_x0, top + height + roi_y0),
                    red_score=score if bit == 0 else 0.0,
                    green_score=score if bit == 1 else 0.0,
                ))
        observations.sort(key=lambda observation: (observation.bbox[0], observation.bbox[1]))
        return observations

    def _surrounding_dark_ratio(
        self, value: np.ndarray, bbox: tuple[int, int, int, int],
    ) -> float:
        """발광부 bbox 바깥 띠의 V 임계값 이하 비율. ROI 경계 밖은 세지 않는다."""
        left, top, right, bottom = bbox
        margin = max(1, int(self.config.surround_margin_px))
        outer_left, outer_top = max(0, left - margin), max(0, top - margin)
        outer_right = min(value.shape[1], right + margin)
        outer_bottom = min(value.shape[0], bottom + margin)
        ring_pixels = ((outer_bottom - outer_top) * (outer_right - outer_left)
                       - (bottom - top) * (right - left))
        if ring_pixels == 0:
            return 0.0
        dark = value[outer_top:outer_bottom, outer_left:outer_right] <= self.config.surround_v_max
        inner_dark = dark[top - outer_top:bottom - outer_top, left - outer_left:right - outer_left]
        return float((np.count_nonzero(dark) - np.count_nonzero(inner_dark)) / ring_pixels)

    def _update_tracks(
        self,
        observations: list[BoxObservation],
        timestamp_ns: int,
    ) -> None:
        """왼쪽 관측부터 가까운 미사용 트랙에 일대일 greedy 매칭하고 EMA를 갱신한다.

        전역 최적 할당/속도 예측은 없다. 빠른 이동이나 교차가 많으면 이 매칭을 교체한다.
        """
        unmatched_track_indices = set(range(len(self.tracks)))
        observations.sort(key=lambda observation: (observation.bbox[0], observation.bbox[1]))

        for observation in observations:
            best_index: int | None = None
            best_distance = float('inf')
            observation_center = self._center(observation.bbox)
            for track_index in unmatched_track_indices:
                track = self.tracks[track_index]
                track_center = self._center(track.bbox)
                distance = hypot(
                    observation_center[0] - track_center[0],
                    observation_center[1] - track_center[1],
                )
                max_dimension = max(
                    observation.bbox[2] - observation.bbox[0],
                    observation.bbox[3] - observation.bbox[1],
                    track.bbox[2] - track.bbox[0],
                    track.bbox[3] - track.bbox[1],
                    1,
                )
                distance_limit = max(12.0, self.config.match_distance_ratio * max_dimension)
                # 중심이 가깝거나 IoU가 충분하면 매칭 후보로 인정한다(둘 중 하나).
                if distance > distance_limit and _iou(observation.bbox, track.bbox) < 0.05:
                    continue
                if distance < best_distance:
                    best_distance = distance
                    best_index = track_index

            # 새 트랙은 현재 점수로 EMA를 초기화하므로 충분한 색상이면 즉시 확정된다.
            if best_index is None:
                track = BoxTrack(
                    track_id=self.next_track_id,
                    bbox=observation.bbox,
                    red_ema=observation.red_score,
                    green_ema=observation.green_score,
                    stable_bit=None,
                    last_seen_ns=timestamp_ns,
                )
                self.next_track_id += 1
                self.tracks.append(track)
                observation.track_id = track.track_id
                self._update_track_bit(track)
                observation.stable_bit = track.stable_bit
                continue

            unmatched_track_indices.remove(best_index)
            track = self.tracks[best_index]
            track.bbox = observation.bbox
            alpha = max(0.0, min(1.0, self.config.color_ema_alpha))
            track.red_ema = alpha * observation.red_score + (1.0 - alpha) * track.red_ema
            track.green_ema = alpha * observation.green_score + (1.0 - alpha) * track.green_ema
            # 색 영역을 다시 관측하면 수명을 연장한다. 무색 프레임은 관측을 만들지 않는다.
            track.last_seen_ns = timestamp_ns
            self._update_track_bit(track)
            observation.track_id = track.track_id
            observation.stable_bit = track.stable_bit

    def _update_track_bit(self, track: BoxTrack) -> None:
        """EMA 우세 점수와 차이 임계값을 모두 통과한 경우에만 비트를 변경한다.

        여기서 히스테리시스는 점수 차이의 dead band이며 상태별 상/하한 방식은 아니다.
        불확실하면 이전 비트(None 포함)를 유지한다. 원시 색상 무효 여부로 EMA 갱신을
        건너뛰지는 않으므로, 판정은 항상 누적된 EMA를 기준으로 한다.
        """
        strongest_score = max(track.red_ema, track.green_ema)
        if strongest_score < self.config.color_score_threshold:
            return

        score_gap = abs(track.green_ema - track.red_ema)
        if score_gap < self.config.color_hysteresis_delta:
            return

        desired_bit = 1 if track.green_ema > track.red_ema else 0
        if track.stable_bit is None or track.stable_bit != desired_bit:
            track.stable_bit = desired_bit

    def _expire_tracks(self, timestamp_ns: int) -> None:
        """미관측 시간이 제한을 초과한 트랙을 제거한다(제한과 같으면 유지)."""
        timeout_ns = max(0.0, self.config.hold_timeout_s) * 1_000_000_000
        self.tracks = [
            track for track in self.tracks
            if timestamp_ns - track.last_seen_ns <= timeout_ns
        ]

    def _bits_after_tracking(self, timestamp_ns: int) -> list[int]:
        """미만료·확정 트랙만 마지막 bbox의 x 순서로 반환한다.

        미확정 박스의 자리는 생략된다. 배열 인덱스는 영구 ID가 아니며 박스의
        추가/만료에 따라 바뀐다. last_bits는 결과 캐시이지 별도 유지 타이머가 아니다.
        """
        timeout_ns = max(0.0, self.config.hold_timeout_s) * 1_000_000_000
        active_tracks = [
            track for track in self.tracks
            if track.stable_bit is not None
            and timestamp_ns - track.last_seen_ns <= timeout_ns
        ]
        active_tracks.sort(key=lambda track: (track.bbox[0], track.bbox[1]))
        current_bits = [int(track.stable_bit) for track in active_tracks]
        self.last_bits = current_bits
        return list(self.last_bits)

    @staticmethod
    def _center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
        return (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))


class BlackBoxColorBitsNode(Node):
    """최신 영상만 처리하고 타이머마다 상태 배열을 발행하는 ROS 2 어댑터."""

    def __init__(self) -> None:
        super().__init__('black_box_color_bits')
        self.bridge = CvBridge()
        self.latest_msg: Image | None = None
        self.processing = False

        self.image_topic = str(self._declare_param('image_topic', '/panorama/image_raw'))
        self.bits_topic = str(self._declare_param('bits_topic', '/tl/box_color_bits'))
        self.debug_image_topic = str(
            self._declare_param('debug_image_topic', '/tl/box_color_bits/debug')
        )
        self.publish_debug_image = bool(self._declare_param('publish_debug_image', False))
        self.max_fps = float(self._declare_param('max_fps', 15.0))
        opencv_threads = int(self._declare_param('opencv_threads', 1))
        if opencv_threads < 1:
            raise ValueError('opencv_threads must be positive')
        cv2.setNumThreads(opencv_threads)

        # 설정은 시작 시 한 번 읽는다. 런타임 set_parameters 반영 콜백은 없다.
        config = DetectorConfig(
            roi_top_ratio=float(self._declare_param('roi_top_ratio', 0.0)),
            roi_bottom_ratio=float(self._declare_param('roi_bottom_ratio', 0.5)),
            roi_left_ratio=float(self._declare_param('roi_left_ratio', 0.25)),
            roi_right_ratio=float(self._declare_param('roi_right_ratio', 0.75)),
            morphology_kernel_size=int(self._declare_param('morphology_kernel_size', 5)),
            morphology_close_iterations=int(self._declare_param('morphology_close_iterations', 0)),
            morphology_open_iterations=int(self._declare_param('morphology_open_iterations', 0)),
            led_group_gap_px=int(self._declare_param('led_group_gap_px', 6)),
            surround_margin_px=int(self._declare_param('surround_margin_px', 4)),
            surround_v_max=int(self._declare_param('surround_v_max', 70)),
            surround_min_dark_ratio=float(self._declare_param('surround_min_dark_ratio', 0.55)),
            min_box_width_px=int(self._declare_param('min_box_width_px', 3)),
            min_box_height_px=int(self._declare_param('min_box_height_px', 3)),
            min_box_area_px=int(self._declare_param('min_box_area_px', 12)),
            max_box_width_px=int(self._declare_param('max_box_width_px', 160)),
            max_box_height_px=int(self._declare_param('max_box_height_px', 160)),
            max_box_width_ratio=float(self._declare_param('max_box_width_ratio', 0.35)),
            max_box_height_ratio=float(self._declare_param('max_box_height_ratio', 0.80)),
            inner_margin_ratio=float(self._declare_param('inner_margin_ratio', 0.20)),
            color_s_min=int(self._declare_param('color_s_min', 80)),
            color_v_min=int(self._declare_param('color_v_min', 45)),
            red_hue_high=int(self._declare_param('red_hue_high', 10)),
            red_hue_low_wrap=int(self._declare_param('red_hue_low_wrap', 170)),
            green_hue_low=int(self._declare_param('green_hue_low', 35)),
            green_hue_high=int(self._declare_param('green_hue_high', 95)),
            color_score_threshold=float(self._declare_param('color_score_threshold', 0.04)),
            color_hysteresis_delta=float(self._declare_param('color_hysteresis_delta', 0.08)),
            color_ema_alpha=float(self._declare_param('color_ema_alpha', 0.45)),
            match_distance_ratio=float(self._declare_param('match_distance_ratio', 2.5)),
            hold_timeout_s=float(self._declare_param('hold_timeout_s', 0.5)),
        )
        detector_mode = str(self._declare_param('detector_mode', 'color_regions'))
        root = workspace_root_or_none()
        default_box_model = str(root / 'model' / 'box_best.pt') if root is not None else 'box_best.pt'
        box_model_path = str(self._declare_param('box_model_path', default_box_model))
        box_device = str(self._declare_param('box_device', 'auto'))
        box_confidence = float(self._declare_param('box_confidence', 0.25))
        box_image_size = int(self._declare_param('box_image_size', 640))
        if detector_mode == 'color_regions':
            self.detector = BlackBoxColorDetector(config)
        elif detector_mode == 'yolo_boxes':
            from .yolo_box_color_bits import YoloBoxColorDetector
            self.detector = YoloBoxColorDetector(
                config, box_model_path, device=box_device,
                confidence=box_confidence, image_size=box_image_size,
            )
            self.get_logger().info(
                f'YOLO box model: {box_model_path}; device={self.detector.device}; '
                f'classes={self.detector.class_ids}; bit source=HSV'
            )
        else:
            raise ValueError('detector_mode must be color_regions or yolo_boxes')
        self.get_logger().info(f'Detector mode: {detector_mode}')

        self.get_logger().info(f'Subscribing to image topic: {self.image_topic}')
        self.get_logger().info(f'Publishing color bits on: {self.bits_topic}')

        # 영상은 sensor-data QoS로 지연을 줄이고, 비트는 reliable 전송을 사용한다.
        self.create_subscription(
            Image,
            self.image_topic,
            self._image_callback,
            qos_profile_sensor_data,
        )
        status_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.bits_pub = self.create_publisher(UInt8MultiArray, self.bits_topic, status_qos)
        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, qos_profile_sensor_data)

        timer_period = 1.0 / max(self.max_fps, 0.1)
        self.create_timer(timer_period, self._process_latest_frame)

    def _declare_param(self, name: str, default_value: Any) -> Any:
        return self.declare_parameter(name, default_value).value

    def _image_callback(self, msg: Image) -> None:
        """처리를 예약하는 대신 최신 메시지로 덮어써 입력 적체를 방지한다."""
        self.latest_msg = msg

    def _process_latest_frame(self) -> None:
        """최신 프레임을 최대 max_fps 주기로 처리한다(실제 처리 속도는 연산량에 의존).

        영상이 끊겨도 타이머가 트랙 만료/빈 배열 발행을 진행한다. 유지 시간은
        이미지 촬영 시각이나 rosbag 시간이 아닌 로컬 단조 증가 시간 기준이다.
        기본 단일 스레드 spin을 전제로 하며 processing은 스레드 동기화 락이 아니다.
        """
        if self.processing:
            return
        if self.latest_msg is None:
            timestamp_ns = time.monotonic_ns()
            self.detector._expire_tracks(timestamp_ns)
            self._publish_bits(self.detector._bits_after_tracking(timestamp_ns))
            return
        message = self.latest_msg
        self.latest_msg = None
        self.processing = True
        try:
            frame = self.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
            bits, observations, roi_bounds = self.detector.process(frame)
            self._publish_bits(bits)
            if self.publish_debug_image and self.debug_pub.get_subscription_count() > 0:
                debug_image = self._draw_debug(frame, observations, roi_bounds, bits)
                debug_message = self.bridge.cv2_to_imgmsg(debug_image, encoding='bgr8')
                debug_message.header = message.header
                self.debug_pub.publish(debug_message)
        except Exception as exc:
            self.get_logger().error(f'Failed to process image: {exc}')
        finally:
            self.processing = False

    def _publish_bits(self, bits: list[int]) -> None:
        """UInt8MultiArray로 발행한다. 빈 배열은 유효 비트 없음이며 [0]과 다르다.

        메시지에는 timestamp/track_id/신뢰도가 없으므로 수신 측은 인덱스를 ID로 쓰지 않는다.
        """
        message = UInt8MultiArray()
        message.data = [int(bit) for bit in bits]
        if bits:
            message.layout = MultiArrayLayout(
                dim=[MultiArrayDimension(label='boxes', size=len(bits), stride=len(bits))],
                data_offset=0,
            )
        self.bits_pub.publish(message)

    def _draw_debug(
        self,
        frame: np.ndarray,
        observations: list[BoxObservation],
        roi_bounds: tuple[int, int, int, int],
        bits: list[int],
    ) -> np.ndarray:
        """ROI만 복사해 현재 관측을 그린다. r/g는 원시 점수, 색은 확정 비트다.

        표시 인덱스는 관측 순서이며 미확정/유지 중 트랙 때문에 출력 인덱스와 다를 수 있다.
        트랙은 전체 영상 좌표를 유지하고 표시 좌표만 ROI 시작점만큼 이동한다.
        """
        roi_x0, roi_y0, roi_x1, roi_y1 = roi_bounds
        debug = frame[roi_y0:roi_y1, roi_x0:roi_x1].copy()
        if debug.size == 0:
            return debug
        for index, observation in enumerate(observations):
            x0, y0, x1, y1 = observation.bbox
            x0, x1 = x0 - roi_x0, x1 - roi_x0
            y0, y1 = y0 - roi_y0, y1 - roi_y0
            if observation.stable_bit == 1:
                color = (0, 255, 0)
                label = 'G:1'
            elif observation.stable_bit == 0:
                color = (0, 0, 255)
                label = 'R:0'
            else:
                color = (0, 165, 255)
                label = '?:-'
            cv2.rectangle(debug, (x0, y0), (x1 - 1, y1 - 1), color, 2)
            text = f'{index} {label} r={observation.red_score:.2f} g={observation.green_score:.2f}'
            self._put_debug_text(debug, text, (x0, y0 - 5), color)
        self._put_debug_text(debug, f'bits={bits}', (10, 25), (255, 255, 255))
        return debug

    @staticmethod
    def _put_debug_text(
        debug: np.ndarray,
        text: str,
        origin: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        """라벨을 크롭 영상 안으로 이동한다. 작은 ROI에서는 글꼴 크기도 줄인다."""
        height, width = debug.shape[:2]
        if height < 4 or width < 4:
            return
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, 1)
        scale *= min(1.0, (width - 3) / max(text_width, 1), (height - 3) / max(text_height + baseline, 1))
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, 1)
        if text_width > width - 2 or text_height + baseline > height - 2:
            return
        text_x = max(1, min(origin[0], width - text_width - 1))
        text_y = max(text_height + 1, min(origin[1], height - baseline - 1))
        cv2.putText(debug, text, (text_x, text_y), font, scale, color, 1, cv2.LINE_AA)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = BlackBoxColorBitsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
