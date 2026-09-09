#!/usr/bin/env python3
"""상단 ROI에서 검은 바탕 디스플레이를 찾아 빨강=0, 초록=1로 발행한다.

처리 흐름: BGR 영상 → ROI → 어두운 사각형 → 내부 HSV 점수 → 추적/EMA → 비트.
검출기는 영상/시간을 입력받고, ROS 노드는 구독·주기 제한·발행을 담당한다.
출력은 확정된 박스만 영상의 왼쪽부터 나열한 가변 길이 배열이다. 미확정은
0으로 대체하지 않으며, 최초 유효 검출 전이나 모든 트랙 만료 시에는 []이다.
검은 외곽선만 있는 물체가 아니라 검은 픽셀 비율이 높은 디스플레이가 대상이다.
형태/색상 기반이므로 어두운 배경 물체와의 의미적 구분을 보장하지 않는다.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any

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
from std_msgs.msg import MultiArrayDimension
from std_msgs.msg import MultiArrayLayout
from std_msgs.msg import UInt8MultiArray


@dataclass
class DetectorConfig:
    """검출/추적 설정. ROS 기본값 변경 시 노드의 파라미터 선언도 함께 맞춘다.

    권장 튜닝 순서: ROI → 검은색/크기 필터 → 내부 색상 → 시간 안정화.
    픽셀 크기 기준은 입력 해상도에 의존하며, 모든 비율의 기준은 아래와 같다.
    """

    # 입력 영상 전체를 기준으로 한 경계 비율(0~1); 기본값은 상단 35%.
    roi_top_ratio: float = 0.0
    roi_bottom_ratio: float = 0.35
    roi_left_ratio: float = 0.0
    roi_right_ratio: float = 1.0

    # OpenCV HSV의 V(0~255) 상한. close는 틈을 메우고 open은 작은 잡음을 제거한다.
    # 큰 커널/open 반복은 멀리 있는 작은 디스플레이까지 지울 수 있다.
    black_v_max: int = 45
    morphology_kernel_size: int = 5
    morphology_close_iterations: int = 0
    morphology_open_iterations: int = 1

    # 폭/높이는 bounding rect, 면적은 contourArea(px²) 기준이다.
    # 최대 크기는 절대 픽셀 제한과 ROI 대비 비율 제한을 모두 만족해야 한다.
    min_box_width_px: int = 10
    min_box_height_px: int = 6
    min_box_area_px: int = 80
    max_box_width_px: int = 160
    max_box_height_px: int = 160
    max_box_width_ratio: float = 0.35
    max_box_height_ratio: float = 0.80
    # 종횡비=폭/높이, 직사각형성=외곽 contour 면적/bounding rect 면적.
    min_aspect_ratio: float = 0.65
    max_aspect_ratio: float = 1.50
    min_rectangularity: float = 0.65
    # 원본 검은색 마스크의 점유율로, 외곽 형태만 사각형인 물체를 추가로 거른다.
    min_black_ratio: float = 0.65
    duplicate_iou_threshold: float = 0.45

    # 각 변에서 폭/높이의 해당 비율만큼 제외한다(0.20이면 중앙 약 60%씩 사용).
    inner_margin_ratio: float = 0.20
    # uint8 OpenCV HSV: H=0~179, S/V=0~255. 빨강은 hue 양 끝의 합집합이다.
    color_s_min: int = 80
    color_v_min: int = 70
    red_hue_high: int = 10
    red_hue_low_wrap: int = 170
    green_hue_low: int = 35
    green_hue_high: int = 95
    # 점수는 내부 전체 픽셀 대비 색상 픽셀 비율이다. 우세 점수와 두 점수의
    # 차이가 각각 임계값을 넘어야 갱신한다. alpha가 작을수록 반응이 느려진다.
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
        """어두운 연결 영역에 형태/검은색 점유율 필터를 적용하고 색상을 측정한다."""
        if roi.size == 0:
            return []

        # V는 채널 최댓값이므로 밝은 유색 발광부를 검은 바탕과 분리할 수 있다.
        value = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 2]
        raw_black_mask = cv2.inRange(value, 0, int(self.config.black_v_max))
        black_mask = raw_black_mask.copy()
        if self.config.morphology_close_iterations > 0:
            black_mask = cv2.morphologyEx(
                black_mask,
                cv2.MORPH_CLOSE,
                self.morphology_kernel,
                iterations=int(self.config.morphology_close_iterations),
            )
        if self.config.morphology_open_iterations > 0:
            black_mask = cv2.morphologyEx(
                black_mask,
                cv2.MORPH_OPEN,
                self.morphology_kernel,
                iterations=int(self.config.morphology_open_iterations),
            )

        # 내부 발광부가 마스크의 구멍이어도 외곽선 하나로 디스플레이를 표현한다.
        contours, _ = cv2.findContours(
            black_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        candidates: list[tuple[tuple[int, int, int, int], float]] = []
        roi_height, roi_width = roi.shape[:2]
        roi_x0, roi_y0, _, _ = roi_bounds

        for contour in contours:
            contour_area = float(cv2.contourArea(contour))
            x, y, width, height = cv2.boundingRect(contour)
            if width < self.config.min_box_width_px or height < self.config.min_box_height_px:
                continue
            if contour_area < self.config.min_box_area_px:
                continue
            if width > self.config.max_box_width_px or height > self.config.max_box_height_px:
                continue
            if width > roi_width * self.config.max_box_width_ratio:
                continue
            if height > roi_height * self.config.max_box_height_ratio:
                continue

            aspect_ratio = width / max(float(height), 1.0)
            if not self.config.min_aspect_ratio <= aspect_ratio <= self.config.max_aspect_ratio:
                continue
            rectangularity = contour_area / max(float(width * height), 1.0)
            if rectangularity < self.config.min_rectangularity:
                continue
            # morphology로 메운 픽셀을 검은색으로 세지 않도록 원본 마스크를 사용한다.
            black_ratio = np.count_nonzero(raw_black_mask[y:y + height, x:x + width]) / (width * height)
            if black_ratio < self.config.min_black_ratio:
                continue

            bbox = (x + roi_x0, y + roi_y0, x + width + roi_x0, y + height + roi_y0)
            candidates.append((bbox, contour_area))

        # 큰 후보를 우선하는 IoU 중복 억제 후 영상 x 순서로 되돌린다.
        candidates.sort(key=lambda item: item[1], reverse=True)
        selected: list[tuple[int, int, int, int]] = []
        for bbox, _ in candidates:
            if any(_iou(bbox, existing) >= self.config.duplicate_iou_threshold for existing in selected):
                continue
            selected.append(bbox)

        selected.sort(key=lambda box: (box[0], box[1]))
        observations: list[BoxObservation] = []
        for bbox in selected:
            red_score, green_score = self._color_scores(roi, bbox, roi_bounds)
            observations.append(
                BoxObservation(
                    bbox=bbox,
                    red_score=red_score,
                    green_score=green_score,
                )
            )
        return observations

    def _color_scores(
        self,
        roi: np.ndarray,
        bbox: tuple[int, int, int, int],
        roi_bounds: tuple[int, int, int, int],
    ) -> tuple[float, float]:
        """테두리를 제외한 내부에서 (빨강 비율, 초록 비율)을 반환한다.

        bbox를 ROI 로컬 좌표로 변환한 뒤 잘라낸다. 리사이즈/다운샘플링은 하지 않는다.
        검은 바탕/저채도/노랑은 색상 분자에 포함되지 않지만 전체 픽셀 분모에는 포함된다.
        """
        roi_x0, roi_y0, _, _ = roi_bounds
        x0, y0, x1, y1 = bbox
        local_x0 = x0 - roi_x0
        local_y0 = y0 - roi_y0
        local_x1 = x1 - roi_x0
        local_y1 = y1 - roi_y0
        box_width = local_x1 - local_x0
        box_height = local_y1 - local_y0
        margin_x = int(box_width * _clamp_ratio(self.config.inner_margin_ratio))
        margin_y = int(box_height * _clamp_ratio(self.config.inner_margin_ratio))
        inner_x0 = min(local_x1 - 1, local_x0 + margin_x)
        inner_y0 = min(local_y1 - 1, local_y0 + margin_y)
        inner_x1 = max(inner_x0 + 1, local_x1 - margin_x)
        inner_y1 = max(inner_y0 + 1, local_y1 - margin_y)
        inner = roi[inner_y0:inner_y1, inner_x0:inner_x1]
        if inner.size == 0:
            return 0.0, 0.0

        hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        valid_mask = (saturation >= self.config.color_s_min) & (value >= self.config.color_v_min)
        red_mask = (
            valid_mask
            & ((hue <= self.config.red_hue_high) | (hue >= self.config.red_hue_low_wrap))
        )
        green_mask = (
            valid_mask
            & (hue >= self.config.green_hue_low)
            & (hue <= self.config.green_hue_high)
        )
        # 유색 픽셀만 분모로 쓰면 작은 잡음 하나도 높은 점수가 되므로 전체 면적을 쓴다.
        pixel_count = max(1, inner.shape[0] * inner.shape[1])
        return float(np.count_nonzero(red_mask) / pixel_count), float(
            np.count_nonzero(green_mask) / pixel_count
        )

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
                distance = float(np.hypot(
                    observation_center[0] - track_center[0],
                    observation_center[1] - track_center[1],
                ))
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
            # 색상이 불명확해도 박스가 보이면 수명은 연장된다. 별도 색상 만료는 없다.
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

        # 설정은 시작 시 한 번 읽는다. 런타임 set_parameters 반영 콜백은 없다.
        config = DetectorConfig(
            roi_top_ratio=float(self._declare_param('roi_top_ratio', 0.0)),
            roi_bottom_ratio=float(self._declare_param('roi_bottom_ratio', 0.35)),
            roi_left_ratio=float(self._declare_param('roi_left_ratio', 0.0)),
            roi_right_ratio=float(self._declare_param('roi_right_ratio', 1.0)),
            black_v_max=int(self._declare_param('black_v_max', 45)),
            morphology_kernel_size=int(self._declare_param('morphology_kernel_size', 5)),
            morphology_close_iterations=int(self._declare_param('morphology_close_iterations', 0)),
            morphology_open_iterations=int(self._declare_param('morphology_open_iterations', 1)),
            min_box_width_px=int(self._declare_param('min_box_width_px', 10)),
            min_box_height_px=int(self._declare_param('min_box_height_px', 6)),
            min_box_area_px=int(self._declare_param('min_box_area_px', 80)),
            max_box_width_px=int(self._declare_param('max_box_width_px', 160)),
            max_box_height_px=int(self._declare_param('max_box_height_px', 160)),
            max_box_width_ratio=float(self._declare_param('max_box_width_ratio', 0.35)),
            max_box_height_ratio=float(self._declare_param('max_box_height_ratio', 0.80)),
            min_aspect_ratio=float(self._declare_param('min_aspect_ratio', 0.65)),
            max_aspect_ratio=float(self._declare_param('max_aspect_ratio', 1.50)),
            min_rectangularity=float(self._declare_param('min_rectangularity', 0.65)),
            duplicate_iou_threshold=float(self._declare_param('duplicate_iou_threshold', 0.45)),
            min_black_ratio=float(self._declare_param('min_black_ratio', 0.65)),
            inner_margin_ratio=float(self._declare_param('inner_margin_ratio', 0.20)),
            color_s_min=int(self._declare_param('color_s_min', 80)),
            color_v_min=int(self._declare_param('color_v_min', 70)),
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
        self.detector = BlackBoxColorDetector(config)

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
            if self.publish_debug_image:
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
        """원본 복사본에 ROI와 현재 관측을 그린다. r/g는 원시 점수, 색은 확정 비트다.

        표시 인덱스는 관측 순서이며 미확정/유지 중 트랙 때문에 출력 인덱스와 다를 수 있다.
        """
        debug = frame.copy()
        roi_x0, roi_y0, roi_x1, roi_y1 = roi_bounds
        cv2.rectangle(debug, (roi_x0, roi_y0), (roi_x1, roi_y1), (255, 0, 255), 2)
        for index, observation in enumerate(observations):
            x0, y0, x1, y1 = observation.bbox
            if observation.stable_bit == 1:
                color = (0, 255, 0)
                label = 'G:1'
            elif observation.stable_bit == 0:
                color = (0, 0, 255)
                label = 'R:0'
            else:
                color = (0, 165, 255)
                label = '?:-'
            cv2.rectangle(debug, (x0, y0), (x1, y1), color, 2)
            text = f'{index} {label} r={observation.red_score:.2f} g={observation.green_score:.2f}'
            text_y = max(18, y0 - 5)
            cv2.putText(
                debug,
                text,
                (x0, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            debug,
            f'bits={bits}',
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return debug


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
