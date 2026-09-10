"""YOLO 검출과 선택된 신호등의 색상 분석을 결합하는 ROS 2 노드.

처리 흐름:
    Image 수신 -> BGR 변환 -> YOLO 후보 검출 -> 대표 후보 선택
    -> 모델/색상 판정 결합 -> 시간축 안정화 -> 상태 및 선택적 디버그 발행.

협업 시 먼저 확인할 계약:
    * STATE_* 정수는 Behavior Planner가 소비하는 외부 인터페이스다.
    * 영상은 uint8 BGR, 박스는 원본 영상의 (x0, y0, x1, y1) 픽셀 좌표다.
      검출 ROI 내부 좌표는 _detect_candidates()에서 원본 좌표로 복원한다.
    * 판정/추적 시간은 ROS 시계, 수신 끊김과 로그 간격은 monotonic 시계를 쓴다.
      영상 header stamp는 순서 검증용이므로 이 세 시간 기준을 혼용하지 않는다.
    * 디버그 영상과 화면용 박스 보간은 판정 입력으로 되돌려 사용하지 않는다.

실행 진입점은 main(), 핵심 조정 지점은 _decide_state()와
_update_stable_state()다. 변경 시 test/test_tl_fusion_logic.py의 회귀 테스트와
동일 영상 재생으로 신호 전환, 아래 방향 화살표 제외, 입력 끊김을 확인한다.
"""

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

# 외부 라이브러리를 import하기 전에 프로젝트의 .deps를 우선 탐색한다.
# 전역 Python의 torch와 노드가 실제로 사용하는 torch가 다를 수 있다.
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

try:
    import torch
    import torch.nn.functional as torch_functional
except ImportError:  # pragma: no cover - exercised only without local PyTorch.
    torch = None
    torch_functional = None


# 외부 소비자와 공유하는 상태 ID: 순서 변경이나 재번호 부여는 호환성을 깨뜨린다.
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

# OpenCV 표시용 BGR 색상이며, 아래 HSV 기반 판정 임계값과는 별개다.
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
    """직접 실행 시 쓸 기본 가중치 경로를 찾는다. launch의 기본값 선택은 별도다.

    workspace의 model/best.pt를 우선하며, 경로 문자열을 반환하는 것 자체가
    파일 존재를 보장하지는 않는다. 실제 존재 검사는 노드 초기화에서 수행한다.
    """
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
    """필터를 통과한 후보 한 개와 모델 클래스 해석 결과.

    box는 원본 영상 기준이며 conf는 YOLO의 confidence다. model_resolved는
    클래스 이름을 상태로 해석할 수 있다는 뜻이지, 검출 정확도 보장이 아니다.
    selection_score는 위치/면적/추적 보너스를 곱한 순위 점수로 확률이 아니다.
    """

    box: tuple[int, int, int, int]
    conf: float
    class_id: int
    class_name: str
    model_state: int
    model_resolved: bool
    selection_score: float = 0.0


@dataclass
class ColorAnalysisResult:
    """보정된 ROI에서 얻은 색상 근거. 모델 confidence와 다른 척도다.

    scores는 세 색상의 가중치 합을 정규화한 비율이며, 근거가 없으면 모두 0이다.
    valid_pixels는 보정/확대 후 정제된 마스크들의 픽셀 수 합계다.
    decisive는 색상 확정 조건 통과 여부이고 highlighted는 표시 전용이다.
    highlighted=None은 시각화를 생략했다는 뜻이며 판정 실패를 의미하지 않는다.
    """

    state: int
    decisive: bool
    reason: str
    valid_pixels: int
    top_score: float
    score_gap: float
    scores: dict[str, float]
    highlighted: np.ndarray | None


@dataclass
class DecisionResult:
    """현재 프레임의 제안 상태와 근거. proposed_state는 아직 안정화 전이다."""

    proposed_state: int
    source: str
    reason: str


@dataclass
class OverlayCandidate:
    """화면에만 쓰는 박스/라벨. 보간된 box를 검출이나 색 분석에 재사용하지 않는다."""

    box: tuple[int, int, int, int]
    label: str
    color: tuple[int, int, int]
    selected: bool


class TLFusionNode(Node):
    """최신 입력 슬롯과 프레임 간 판정 이력을 소유하는 노드.

    현재 main()은 기본 단일 스레드 executor를 사용한다. processing은 중복
    처리 방지용 플래그이지 락이 아니므로, 병렬 처리 도입 시 latest_msg,
    상태 이력, 모델 호출과 callback group의 동기화를 함께 재검토해야 한다.
    파라미터는 초기화 때 멤버에 복사하며 동적 갱신 콜백은 구현하지 않는다.
    """

    def __init__(self) -> None:
        """파라미터, 재사용 연산 객체, 모델, ROS 통신과 타이머를 순서대로 준비한다."""
        super().__init__('tl_fusion')

        # 프레임마다 만들 필요가 없는 영상 변환 객체와 형태학 커널을 재사용한다.
        self.bridge = CvBridge()
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self.fallback_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # 입출력 설정. image_topic 등은 launch에서 전달한 값이 이 기본값보다 우선한다.
        # input_timeout_s는 '영상 수신 끊김' 기준이며, 신호등 미검출 시간과 다르다.
        self.model_path = self._declare_param('model_path', default_tl_model_path())
        self.image_topic = str(self._declare_param('image_topic', default_runtime_image_topic()))
        self.state_topic = str(self._declare_param('state_topic', '/tl/state_id'))
        self.show_windows = bool(self._declare_param('show_windows', False))
        self.input_timeout_s = float(self._declare_param('input_timeout_s', 3.0))

        # max_fps는 처리 타이머의 목표 주기다. 실제 처리율은 추론 시간에도 제한된다.
        # CUDA를 요청해도 사용 불가능하면 helper가 CPU로 전환하므로 시작 로그를 확인한다.
        self.max_fps = float(self._declare_param('max_fps', 15.0))
        requested_device = str(self._declare_param('detector_device', 'cuda:0'))
        self.detector_device = resolve_inference_device(requested_device)
        requested_color_device = str(
            self._declare_param('color_fallback_device', 'auto')
        )
        if requested_color_device.strip().lower() == 'auto':
            self.color_fallback_device = self.detector_device
        else:
            self.color_fallback_device = resolve_inference_device(requested_color_device)
        self.use_torch_color_fallback = bool(
            torch is not None
            and self.color_fallback_device.lower().startswith('cuda')
            and torch.cuda.is_available()
        )
        self.detector_image_size = int(self._declare_param('detector_image_size', 640))
        self.detector_conf_threshold = float(self._declare_param('detector_conf_threshold', 0.10))
        self.detector_iou_threshold = float(self._declare_param('detector_iou_threshold', 0.45))
        self.detector_max_detections = int(self._declare_param('detector_max_detections', 50))

        # detect_*: YOLO에 실제로 전달할 crop 범위. 비율은 원본 영상의 0~1 기준이다.
        self.detect_top_ratio = float(self._declare_param('detect_top_ratio', 0.00))
        self.detect_bottom_ratio = float(self._declare_param('detect_bottom_ratio', 1.0 / 3.0))
        self.detect_left_ratio = float(self._declare_param('detect_left_ratio', 0.25))
        self.detect_right_ratio = float(self._declare_param('detect_right_ratio', 0.75))

        # preferred_*: 대표 후보 선택 시 가산점만 주는 영역. 추론 범위를 줄이지 않는다.
        self.preferred_top_ratio = float(self._declare_param('preferred_top_ratio', 0.00))
        self.preferred_bottom_ratio = float(self._declare_param('preferred_bottom_ratio', 0.50))
        self.preferred_left_ratio = float(self._declare_param('preferred_left_ratio', 0.25))
        self.preferred_right_ratio = float(self._declare_param('preferred_right_ratio', 0.75))

        # 크기와 경계 필터는 리사이즈된 추론 텐서가 아닌 검출 crop의 픽셀 단위다.
        self.min_box_side_px = int(self._declare_param('min_box_side_px', 5))
        self.min_box_area_px = int(self._declare_param('min_box_area_px', 40))
        self.edge_margin_px = int(self._declare_param('edge_margin_px', 2))

        # detector_conf_threshold는 후보 수집 기준, 아래 두 값은 수집 후 판정 기준이다.
        # 상위 기준 이상이면 색 분석을 생략하고, 중간 기준은 색상이 모호할 때 사용한다.
        self.model_confidence_threshold = float(self._declare_param('model_confidence_threshold', 0.60))
        self.model_min_confidence_threshold = float(
            self._declare_param('model_min_confidence_threshold', 0.30)
        )
        # 저신뢰 모델도 색상 fallback으로 재검증하면 정확도는 높지만 CPU 비용이 크다.
        # 기본값은 기존 동작을 유지하고, 실시간 성능 모드에서만 fallback을 생략한다.
        self.enable_low_confidence_color_fallback = bool(
            self._declare_param('enable_low_confidence_color_fallback', True)
        )

        # 색상 fallback: 박스 주변 확장 -> 명암/채도 보정 -> HSV 마스크 분석.
        # S/V는 OpenCV uint8의 0~255 범위이며, 픽셀 수 기준은 보정된 ROI에 적용된다.
        self.fallback_expand_ratio = float(self._declare_param('fallback_expand_ratio', 1.80))
        self.fallback_min_margin_px = int(self._declare_param('fallback_min_margin_px', 4))
        self.fallback_max_side_px = int(self._declare_param('fallback_max_side_px', 640))
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
        # 프로젝트 규칙: 적색/녹색 동시 점등을 좌회전 상태로 해석한다.
        # 화살표 형상 검출이 아니므로 다른 신호등 배치에 적용할 때 별도 검증이 필요하다.
        self.fallback_red_green_red_min = float(self._declare_param('fallback_red_green_red_min', 0.30))
        self.fallback_red_green_green_min = float(
            self._declare_param('fallback_red_green_green_min', 0.18)
        )
        self.fallback_red_green_yellow_max = float(
            self._declare_param('fallback_red_green_yellow_max', 0.12)
        )
        self.fallback_gamma_lut = self._build_gamma_lut(self.fallback_gamma)

        # 이력 길이는 프레임 수, hold/missing/reset은 ms다. FPS 변경 시 시간 응답도 달라진다.
        # hold는 마지막 상태 전환 이후 최소 간격, missing은 짧은 미검출의 상태 유지,
        # reset은 오래 미검출됐을 때 이전 후보 위치를 버리는 기준이다.
        self.state_window_size = int(self._declare_param('state_window_size', 5))
        self.hold_ms = int(self._declare_param('hold_ms', 250))
        self.missing_timeout_ms = int(self._declare_param('missing_timeout_ms', 400))
        self.reset_tracking_ms = int(self._declare_param('reset_tracking_ms', 1200))
        # 화면의 깜빡임/박스 흔들림만 줄이는 설정이며 실제 상태 안정화와 분리한다.
        self.overlay_hold_ms = int(self._declare_param('overlay_hold_ms', 220))
        self.overlay_smoothing_alpha = float(self._declare_param('overlay_smoothing_alpha', 0.55))
        # 일부 rosbag은 정상 영상에도 0 stamp를 사용하므로 기본적으로 허용한다.
        # 양수 stamp의 역행은 기본적으로 거부하지만 같은 stamp의 재입력은 허용한다.
        self.require_image_header_stamp = bool(
            self._declare_param('require_image_header_stamp', False)
        )
        self.require_monotonic_image_stamp = bool(
            self._declare_param('require_monotonic_image_stamp', True)
        )

        # 잘못된 모델 경로는 시작 단계에서 드러낸다. 추론 루프에서 재로딩하지 않는다.
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
        self.get_logger().info(
            'Color fallback backend: '
            f"{'torch_cuda' if self.use_torch_color_fallback else 'opencv_cpu'} "
            f'(device: {self.color_fallback_device})'
        )
        self.get_logger().info(
            f'Color fallback ROI max side: {self.fallback_max_side_px}px'
        )
        self.get_logger().info(f'Detector class filter: {self.detector_classes}')
        self.get_logger().info(f'Subscribing to image topic: {self.image_topic}')
        self.get_logger().info(f'Publishing traffic-light state: {self.state_topic}')

        # 수신 콜백은 단일 슬롯을 덮어쓴다. 소비한 메시지는 타이머가 슬롯에서 제거한다.
        # DDS 내부 대기열까지 없애는 구조는 아니므로 절대적인 최신 프레임 보장은 아니다.
        self.latest_msg: Image | None = None
        self.processing = False
        self.last_image_received_monotonic_ns = time.monotonic_ns()
        self.input_timeout_active = False
        self.current_state = STATE_UNKNOWN
        self.current_source = 'init'
        self.current_reason = 'init'
        # 아래 판정/추적 시각은 ROS 시계다. monotonic 기반 수신 시각과 빼지 않는다.
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

        # 큰 디버그 영상은 깊이 1, 상태 메시지는 깊이 10의 reliable QoS를 사용한다.
        # input_valid는 입력 처리 상태이며, 신호등 검출 성공이나 GREEN 여부가 아니다.
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
        """시작 시 ROS override를 반영한 값을 읽는다. 멤버의 실시간 갱신 기능은 아니다."""
        return self.declare_parameter(name, default_value).value

    # 입력 수신, 프레임 처리와 비정상 입력 처리

    def _image_callback(self, msg: Image) -> None:
        """추론 없이 수신 시각과 최신 슬롯만 갱신해 콜백 작업을 짧게 유지한다."""
        self.last_image_received_monotonic_ns = time.monotonic_ns()
        self.latest_msg = msg

    def _process_latest_frame(self) -> None:
        """대기 중인 한 프레임을 처리하고 다음 호출을 위해 processing을 해제한다.

        selected는 현재 프레임의 대표 후보, decision은 안정화 전 판단,
        stable_state는 이력/유지 시간을 거친 외부 발행 상태다.
        stage는 장애 로그의 위치 표시이며 단계별 시간 측정값은 아니다.
        """
        if self.processing:
            return
        if self.latest_msg is None:
            self._publish_timeout_if_needed()
            return

        # 처리할 입력을 확보한 뒤 슬롯을 비운다. 새 executor 설계에서는 이 교환도 보호해야 한다.
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
            # 프레임 시작에 시각화 수요를 한 번만 확인한다. 창 표시와 ROS 발행은 별개다.
            # 색 분석 내부까지 이 값을 전달해야 보이지 않는 하이라이트 생성도 생략된다.
            publish_debug_image = self._should_publish_debug()
            render_debug = self.show_windows or publish_debug_image
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            stage = 'detect'
            detections, _ = self._detect_candidates(frame)
            stage = 'select'
            selected = self._select_candidate(detections, frame.shape)

            # 현재 후보 선택은 이전 위치를 참고한다. 선택이 끝난 뒤에 추적 위치를 갱신한다.
            if selected is not None:
                self.last_seen_candidate_ns = self._now_ns()
                self.last_candidate_box = selected.box

            # confidence가 높아도 클래스가 상태로 해석되지 않으면 색상 fallback이 필요하다.
            # 이 빠른 경로의 기준은 _decide_state()의 첫 모델 분기와 동일하게 유지한다.
            if selected is not None and selected.model_resolved and selected.conf >= self.model_confidence_threshold:
                stage = 'model_high_conf'
                analysis = self._empty_analysis('model_high_conf_skip')
                decision = DecisionResult(
                    proposed_state=selected.model_state,
                    source='model',
                    reason=f'{selected.class_name}:{selected.conf:.2f}',
                )
            elif (
                selected is not None
                and selected.model_resolved
                and selected.conf >= self.model_min_confidence_threshold
                and not self.enable_low_confidence_color_fallback
            ):
                stage = 'model_low_conf_fast'
                analysis = self._empty_analysis('model_low_conf_fallback_disabled')
                decision = self._decide_state(selected, analysis)
            else:
                stage = 'analyze'
                analysis = self._analyze_selected_candidate(
                    frame, selected, render_debug=render_debug,
                )
                stage = 'decide'
                decision = self._decide_state(selected, analysis)
            stage = 'stabilize'
            stable_state = self._update_stable_state(decision.proposed_state, selected is not None)
            debug_image = None
            if render_debug:
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
            self._publish_outputs(
                msg, debug_image, stable_state, decision,
                publish_debug_image=publish_debug_image,
            )

            # latency는 이 콜백의 처리 시간이다. 촬영/전송/DDS 대기를 포함한 지연이 아니다.
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
        # 일반 처리 예외는 로그만 남긴다. 아래 별도 입력 오류/타임아웃 경로와 달리
        # 이 경로 자체가 UNKNOWN이나 input_valid=false를 발행하지는 않는다.
        except Exception as exc:  # noqa: BLE001
            self._log_processing_error(stage, exc, msg)
        finally:
            self.processing = False

    def _publish_timeout_if_needed(self) -> None:
        """수신이 끊기면 다수결/hold를 우회해 UNKNOWN과 입력 무효를 반복 발행한다.

        input_timeout_s <= 0이면 비활성화한다. 정상 수신 때 갱신한 monotonic
        시각을 사용하므로 ROS 시간 정지와 별개지만, 같은 처리 타이머에서 호출되어
        긴 추론 중에도 정확히 제한 시각에 실행되는 독립 watchdog은 아니다.
        """
        if self.input_timeout_s <= 0.0:
            return

        elapsed_s = (
            time.monotonic_ns() - self.last_image_received_monotonic_ns
        ) / 1e9
        if elapsed_s < self.input_timeout_s:
            return

        # 상태 발행은 계속하되, 수신이 회복되기 전 같은 타임아웃의 경고는 한 번만 남긴다.
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
        """재현에 필요한 입력/장치/추론 설정을 기록한다. 상태 변경은 하지 않는다."""
        signature = (stage, f'{type(exc).__name__}:{exc!r}')
        now = time.monotonic()
        # 문맥은 매번, traceback은 서명이 바뀌거나 직전 오류에서 5초 이상 지났을 때 출력한다.
        # 아래 시각을 매번 갱신하므로 연속된 동일 오류의 '5초 주기 출력'은 아니다.
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
        """입력 시각이 허용되면 None, 거부할 경우 이유 문자열을 반환한다.

        양수 stamp에 대해서만 역행을 검사하고 마지막 시각을 갱신한다.
        같은 stamp는 허용하며, 촬영 시각이 현재보다 얼마나 오래됐는지는 검사하지 않는다.
        """
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
        """잘못된 header 입력을 처리하지 않고 상태/추적을 초기화해 입력 무효를 알린다."""
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

    # YOLO 후보 검출과 대표 후보 선택

    def _detect_candidates(
        self,
        frame: np.ndarray,
    ) -> tuple[list[DetectionCandidate], tuple[int, int, int, int]]:
        """BGR 원본에서 후보 목록과 실제 검출 창의 원본 좌표를 반환한다.

        YOLO 결과 좌표는 detect_frame 기준이다. 크기/경계 필터를 먼저 적용한 뒤
        crop 오프셋을 더하므로 이후 단계는 모두 원본 좌표를 사용한다.
        """
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

        # GPU 결과를 박스마다 읽으면 작은 전송/동기화가 반복되므로 한 번에 CPU로 옮긴다.
        # 순서와 int 절삭 방식을 유지해야 경계 필터 및 동점 후보 선택이 달라지지 않는다.
        boxes = result.boxes.cpu()
        for box in boxes:
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
            # 아래 방향 초록 화살표는 종료분기용 별도 검출 대상이다.
            # 주행 가능한 GREEN/LEFT ARROW 후보로 넣지 않도록 상태 해석 전에 제외한다.
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
        """confidence에 상단/면적/선호 영역/이전 위치 보너스를 곱해 하나를 고른다.

        각 후보의 selection_score를 갱신한다. 점수가 같으면 먼저 들어온 후보가
        유지되므로 검출 목록을 임의로 정렬하면 선택 결과도 달라질 수 있다.
        """
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

    # 선택된 후보의 색상 근거와 프레임 단위 판정

    def _analyze_selected_candidate(
        self,
        frame: np.ndarray,
        candidate: DetectionCandidate | None,
        *,
        render_debug: bool = False,
    ) -> ColorAnalysisResult:
        """대표 후보의 확장 ROI에서 적/황/녹 근거를 계산한다.

        검출 후보를 새로 찾는 함수는 아니다. 후보가 없으면 UNKNOWN 근거를 반환한다.
        render_debug는 시각화 생성만 제어하며 점수/decisive/판정 상태는 바꾸지 않는다.
        """
        if candidate is None:
            return self._empty_analysis('no_candidate')

        crop = self._expanded_crop(frame, candidate.box)
        enhanced = self._enhance_crop(crop)
        if self.use_torch_color_fallback:
            try:
                raw_scores, valid_pixels, masks = self._torch_color_measurements(enhanced)
            except Exception as exc:  # noqa: BLE001
                self.use_torch_color_fallback = False
                self.get_logger().warning(
                    f'PyTorch color fallback failed; switching to OpenCV CPU: {exc!r}'
                )

        if not self.use_torch_color_fallback:
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
            # 채도와 밝기가 높은 픽셀에 더 큰 가중치를 준다. 합산은 float32 기준이며,
            # 벡터화/합산 순서를 바꾸면 임계값 근처 결과가 달라질 수 있어 회귀 검증이 필요하다.
            saturation = hsv[:, :, 1].astype(np.float32) / 255.0
            value = hsv[:, :, 2].astype(np.float32) / 255.0
            weights = 0.25 + 0.40 * saturation + 0.35 * value

            # uint8 OpenCV HSV의 H는 0~179다. 적색은 hue 경계 양쪽에 있어 두 구간을 합친다.
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
            valid_pixels = int(sum(int(np.count_nonzero(mask)) for mask in masks.values()))

        # 분모는 ROI 면적이 아니라 세 색상의 가중치 합이다. 배경이 많아도 한 색만
        # 조금 남으면 비율이 커질 수 있으므로 아래 픽셀 수/연결요소 조건을 함께 사용한다.
        total_score = float(sum(raw_scores.values()))
        scores = {
            name: (raw_scores[name] / total_score) if total_score > 0.0 else 0.0
            for name in COLOR_ORDER
        }
        top_color = max(COLOR_ORDER, key=lambda name: scores[name])
        top_score = float(scores[top_color])
        second_score = max(
            (float(scores[name]) for name in COLOR_ORDER if name != top_color),
            default=0.0,
        )
        score_gap = top_score - second_score

        # 동시 적/녹 점등은 단일 우세 색상보다 먼저 해석하는 프로젝트별 좌회전 규칙이다.
        red_green_decisive = (
            valid_pixels >= self.fallback_min_valid_pixels
            and scores['red'] >= self.fallback_red_green_red_min
            and scores['green'] >= self.fallback_red_green_green_min
            and scores['yellow'] <= self.fallback_red_green_yellow_max
        )

        # 비싼 연결요소 분석은 나머지 확정 조건을 통과할 때만 수행한다.
        # 적/녹 조합이 이미 확정되면 이 결과를 사용하지 않으므로 계산도 생략한다.
        component_size = 0
        if (
            not red_green_decisive
            and valid_pixels >= self.fallback_min_valid_pixels
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

        # 이 이미지는 표시 전용이다. 디버그를 끄면 마스크 합성/복사 비용까지 없앤다.
        highlighted = self._highlight_masks(enhanced, masks, scores) if render_debug else None
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
        """후보 없음/색 분석 생략에 공통으로 쓰는 근거 객체. 빈 표시 영상은 할당하지 않는다."""
        return ColorAnalysisResult(
            state=STATE_UNKNOWN,
            decisive=False,
            reason=reason,
            valid_pixels=0,
            top_score=0.0,
            score_gap=0.0,
            scores={name: 0.0 for name in COLOR_ORDER},
            highlighted=None,
        )

    def _decide_state(
        self,
        candidate: DetectionCandidate | None,
        analysis: ColorAnalysisResult,
    ) -> DecisionResult:
        """모델/색상 근거의 우선순위를 적용해 안정화 전 제안 상태를 반환한다.

        고신뢰 모델 -> 확정 색상 -> 중간 신뢰 모델 -> 남은 확정 색상
        -> 약한 모델 -> UNKNOWN 순서다. 후보가 없으면 즉시 UNKNOWN을 제안한다.
        높은 모델 판정을 색상으로 덮지 않으며, 중간 기준 미만 모델도 해석 가능한
        클래스이면 model_weak로 사용한다. 기준 미만이 곧 UNKNOWN인 정책은 아니다.
        """
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

        # 일반적인 유한 confidence는 앞 분기에서 처리된다. 남은 확정 색상의 방어적 경로다.
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

    # 시간축 안정화와 외부 발행

    def _update_stable_state(self, proposed_state: int, has_candidate: bool) -> int:
        """짧은 미검출 보완 -> 최근 프레임 다수결 -> 전환 간격 제한을 적용한다.

        missing_timeout_ms가 지나도 바로 UNKNOWN으로 바꾸지는 않고 이력에 반영한다.
        hold_ms는 새 후보의 지속 시간이 아니라 마지막 상태 변경 이후 경과 시간이다.
        영상 자체의 끊김/무효 입력은 별도 경로에서 이 안정화 규칙을 우회한다.
        """
        now_ns = self._now_ns()
        missing_ms = self._ns_to_ms(now_ns - self.last_seen_candidate_ns)

        if not has_candidate and missing_ms < self.missing_timeout_ms:
            proposed_state = self.current_state
        elif not has_candidate and missing_ms >= self.reset_tracking_ms:
            self.last_candidate_box = None

        # 고정 길이 deque라 FPS를 낮추면 같은 이력 길이가 더 긴 시간 범위를 나타낸다.
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
        *,
        publish_debug_image: bool,
    ) -> None:
        """유효 입력 처리 결과를 발행하고, 필요한 경우에만 화면을 표시한다.

        input_valid=True는 이번 프레임이 처리됐다는 뜻이며 후보 없음도 포함한다.
        state_id/label은 안정화 결과, reason의 source/근거는 현재 프레임 결과라
        상태 유지 중에는 서로 다른 신호를 가리킬 수 있다. 상태 토픽은 매번 발행한다.
        """
        self.input_timeout_active = False
        self.input_valid_pub.publish(Bool(data=True))
        # 창만 켜진 경우에는 이미지 직렬화와 ROS 발행을 하지 않는다.
        if publish_debug_image and debug_image is not None:
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

        # ROS 발행과 창 표시를 모두 요청해도 같은 렌더링 결과를 재사용한다.
        if self.show_windows and debug_image is not None:
            cv2.imshow('TL Debug', debug_image)
            cv2.waitKey(1)

    def _numpy_to_image_msg(self, image: np.ndarray, header: Any, encoding: str) -> Image:
        """uint8 HxWx3 영상을 ROS Image로 직렬화한다. 색 순서 변환은 하지 않는다.

        호출자가 실제 채널 순서에 맞는 encoding을 지정해야 한다. 현재 호출은 bgr8이며
        원본 header를 보존한다. tobytes()는 복사를 수행하므로 발행할 때만 호출한다.
        """
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

    # 화면 표시용 데이터: 판정/추적 입력과 분리한다.

    def _build_debug_image(
        self,
        frame: np.ndarray,
        overlay_candidates: list[OverlayCandidate],
        selected: DetectionCandidate | None,
        analysis: ColorAnalysisResult,
    ) -> np.ndarray:
        """검출 ROI만 복사한다. 후보는 원본 좌표를 유지하고 표시 좌표만 이동한다."""
        roi_x0, roi_y0, roi_x1, roi_y1 = self._window_from_ratios(
            frame.shape, self.detect_left_ratio, self.detect_right_ratio,
            self.detect_top_ratio, self.detect_bottom_ratio,
        )
        debug = frame[roi_y0:roi_y1, roi_x0:roi_x1].copy()
        if debug.size == 0:
            return debug
        for overlay in overlay_candidates:
            x_a, y_a, x_b, y_b = overlay.box
            x_a, x_b = x_a - roi_x0, x_b - roi_x0
            y_a, y_b = y_a - roi_y0, y_b - roi_y0
            if x_b <= 0 or y_b <= 0 or x_a >= debug.shape[1] or y_a >= debug.shape[0]:
                continue
            thickness = 2 if overlay.selected else 1
            cv2.rectangle(debug, (x_a, y_a), (x_b, y_b), overlay.color, thickness)
            self._draw_box_label(debug, overlay.label, x_a, y_a, overlay.color)
        if selected is not None and analysis.highlighted is not None and analysis.highlighted.size > 0:
            inset_width = min(200, debug.shape[1] - 24)
            inset_height = min(120, debug.shape[0] - 42)
            if inset_width > 0 and inset_height > 0:
                inset = self._fit_to_canvas(analysis.highlighted, inset_width, inset_height)
                self._draw_debug_inset(debug, inset, 'Color Mask')
        return debug

    def _build_overlay_candidates(
        self,
        detections: list[DetectionCandidate],
        selected: DetectionCandidate | None,
        stable_state: int,
    ) -> list[OverlayCandidate]:
        """대표 박스에만 화면용 보간을 적용하고, 나머지 후보도 함께 표시한다.

        마지막 대표 overlay와 표시 시각을 갱신한다. 후보가 사라지면 잠깐 기존
        overlay를 재사용하지만, 이 보간/유지 결과는 실제 신호 판정에 관여하지 않는다.
        """
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
        """ROS 시간 기준 overlay_hold_ms 동안만 마지막 대표 표시를 남긴다."""
        if self.last_overlay_candidate is None:
            return None

        now_ns = self._now_ns()
        elapsed_ms = self._ns_to_ms(now_ns - self.last_overlay_update_ns)
        if elapsed_ms > self.overlay_hold_ms:
            self.last_overlay_candidate = None
            return None
        return self.last_overlay_candidate

    def _smooth_overlay_box(self, current_box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """좌표별 지수 보간. alpha=1은 현재 박스, alpha=0은 이전 표시 박스를 사용한다."""
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

    # 좌표 변환, 모델 클래스 해석과 색상 전처리

    def _window_from_ratios(
        self,
        frame_shape: tuple[int, ...],
        left_ratio: float,
        right_ratio: float,
        top_ratio: float,
        bottom_ratio: float,
    ) -> tuple[int, int, int, int]:
        """영상 비율을 픽셀 창으로 변환하고 시작 음수/끝 경계 초과를 제한한다.

        입력은 0~1 비율을 전제로 하며 좌우/상하 순서를 자동 교환하지 않는다.
        검출 호출자는 폭/높이가 양수인지 확인한 뒤 frame[y0:y1, x0:x1]로 사용한다.
        """
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
        """모델 names의 dict/list 형식을 모두 지원하며, 없는 ID는 문자열 ID로 남긴다."""
        if isinstance(self.class_names, dict):
            return str(self.class_names.get(class_id, class_id))
        if isinstance(self.class_names, (list, tuple)) and 0 <= class_id < len(self.class_names):
            return str(self.class_names[class_id])
        return str(class_id)

    def _resolve_detector_classes(self) -> list[int] | None:
        """vehicular_* 또는 traffic light 클래스만 골라 YOLO classes 인자로 전달한다.

        일치하는 이름이 없으면 []가 아니라 None을 반환해 모델의 전체 클래스를
        허용한다. 이후 _state_from_class_name()이 실제 상태 해석 가능 여부를 정한다.
        """
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
        """클래스 이름의 키워드를 (상태 ID, 해석 가능 여부)로 매핑한다.

        generic/기타 클래스는 색 분석에 맡긴다. 아래 방향 화살표 제외와 좌회전
        조합을 단색보다 먼저 검사해야 green 부분 문자열만 보고 오분류하지 않는다.
        새 가중치를 적용할 때 클래스 이름과 이 규칙의 호환성을 함께 검증한다.
        """
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
        """등화 주변을 함께 보기 위해 비율 확장과 최소 양쪽 여백 중 큰 값을 적용한다.

        유효한 범위에서는 원본의 view를 반환하므로 호출자는 직접 덮어쓰지 않는다.
        잘못된 범위는 검은 영상으로 대체해 이후 색상 근거가 없도록 처리한다.
        """
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
        roi_x0, roi_y0, roi_x1, roi_y1 = self._window_from_ratios(
            frame.shape, self.detect_left_ratio, self.detect_right_ratio,
            self.detect_top_ratio, self.detect_bottom_ratio,
        )
        x0 = max(roi_x0, x0)
        y0 = max(roi_y0, y0)
        x1 = min(roi_x1, x1)
        y1 = min(roi_y1, y1)
        if x1 <= x0 or y1 <= y0:
            return np.zeros((64, 64, 3), dtype=np.uint8)
        return frame[y0:y1, x0:x1]

    def _enhance_crop(self, crop: np.ndarray) -> np.ndarray:
        """작은 ROI 확대 -> LAB 명암 보정 -> HSV 채도/밝기 보정 -> 감마 -> 선명화.

        전처리 순서와 보간 방법도 HSV 마스크/픽셀 수에 영향을 주는 판정의 일부다.
        BGR/HSV 왕복 사이에 감마와 선명화가 있으므로 단순 중복 변환으로 지우지 않는다.
        """
        if crop.size == 0:
            return np.zeros((64, 64, 3), dtype=np.uint8)

        working = crop
        # 모델이 큰 배경 영역을 후보로 반환해도 색상 fallback은 신호등 판정에
        # 필요한 해상도만 유지한다. 이 제한이 없으면 4K ROI에 CPU CLAHE를 적용해
        # GPU HSV 경로보다 전처리 시간이 훨씬 커질 수 있다.
        max_side = self.fallback_max_side_px
        if max_side > 0 and max(working.shape[:2]) > max_side:
            scale = max_side / float(max(working.shape[:2]))
            new_width = max(1, int(round(working.shape[1] * scale)))
            new_height = max(1, int(round(working.shape[0] * scale)))
            working = cv2.resize(working, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # 짧은 변을 최소 64px로 만든다. 연결요소 크기 기준은 이렇게 확대된 영상 기준이다.
        if min(working.shape[:2]) < 64:
            scale = 64.0 / float(max(1, min(working.shape[:2])))
            new_width = max(1, int(round(working.shape[1] * scale)))
            new_height = max(1, int(round(working.shape[0] * scale)))
            working = cv2.resize(working, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # CLAHE는 LAB의 명도 L에만 적용해 국소 대비를 보정한다.
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

        # 흐린 영상을 빼는 unsharp 방식이다. 마스크 정제 전 경계/픽셀 값도 함께 달라진다.
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        enhanced = cv2.addWeighted(enhanced, 1.35, blurred, -0.35, 0.0)
        return enhanced

    def _build_gamma_lut(self, gamma: float) -> np.ndarray:
        """입력^(1/gamma)의 uint8 LUT. 이 정의에서는 gamma<1일 때 중간톤이 어두워진다."""
        gamma = max(0.1, gamma)
        inv_gamma = 1.0 / gamma
        return np.array(
            [((index / 255.0) ** inv_gamma) * 255.0 for index in range(256)],
            dtype=np.uint8,
        )

    def _should_publish_debug(self) -> bool:
        """구독자가 있을 때만 이미지 발행을 요청한다. 창 표시 여부는 호출자가 따로 판단한다."""
        try:
            return self.debug_pub.get_subscription_count() > 0
        except AttributeError:
            # 구독 수 조회 API가 없는 대체 publisher에서는 기존 표시 호환성을 보존한다.
            return True

    def _torch_color_measurements(
        self,
        enhanced: np.ndarray,
    ) -> tuple[dict[str, float], int, dict[str, np.ndarray]]:
        """PyTorch CUDA로 HSV 마스크와 색상 점수를 계산한다.

        ROS/OpenCV 영상은 CPU 메모리에 있으므로 입력 ROI만 CUDA로 옮긴다. CLAHE와
        연결요소 분석은 기존 OpenCV 구현을 유지하고, 이 함수에서는 픽셀별 HSV,
        마스크 정제, 가중 점수 합산을 GPU에서 처리한다. 마지막 마스크는 기존
        디버그/연결요소 코드와 호환되도록 uint8 NumPy로 한 번만 되돌린다.
        """
        if torch is None or torch_functional is None:
            raise RuntimeError('PyTorch is not available')

        with torch.inference_mode():
            bgr = torch.from_numpy(np.ascontiguousarray(enhanced)).to(
                device=self.color_fallback_device,
                non_blocking=True,
            )
            rgb = bgr[..., (2, 1, 0)].to(dtype=torch.float32).div_(255.0)
            red, green, blue = rgb.unbind(dim=-1)
            max_value = rgb.amax(dim=-1)
            min_value = rgb.amin(dim=-1)
            delta = max_value - min_value
            safe_delta = delta.clamp_min(1.0e-6)

            hue_red = torch.remainder((green - blue) / safe_delta, 6.0)
            hue_green = ((blue - red) / safe_delta) + 2.0
            hue_blue = ((red - green) / safe_delta) + 4.0
            hue = torch.where(
                max_value == red,
                hue_red,
                torch.where(max_value == green, hue_green, hue_blue),
            )
            hue = torch.where(delta > 0.0, torch.remainder(hue, 6.0) * 30.0, torch.zeros_like(hue))
            saturation = torch.where(
                max_value > 0.0,
                delta / max_value * 255.0,
                torch.zeros_like(max_value),
            )
            value = max_value * 255.0
            weights = 0.25 + 0.40 * (saturation / 255.0) + 0.35 * (value / 255.0)

            red_mask = (
                ((hue >= 0.0) & (hue <= 9.0))
                | ((hue >= 165.0) & (hue < 180.0))
            ) & (saturation >= self.fallback_s_min) & (value >= self.fallback_v_min)
            yellow_mask = (
                (hue >= 14.0) & (hue <= 38.0)
                & (saturation >= self.fallback_s_min)
                & (value >= self.fallback_v_min)
            )
            green_mask = (
                (hue >= 40.0) & (hue <= 95.0)
                & (saturation >= self.fallback_s_min)
                & (value >= self.fallback_v_min)
            )

            masks_gpu = {
                'red': self._torch_clean_mask(red_mask),
                'yellow': self._torch_clean_mask(yellow_mask),
                'green': self._torch_clean_mask(green_mask),
            }
            raw_values = torch.stack([
                (weights * masks_gpu[name].to(dtype=weights.dtype)).sum()
                for name in COLOR_ORDER
            ])
            valid_value = torch.stack([
                masks_gpu[name].sum(dtype=torch.int64) for name in COLOR_ORDER
            ]).sum()
            metrics = torch.cat((raw_values, valid_value.reshape(1))).cpu().tolist()
            masks = {
                name: mask.to(dtype=torch.uint8).mul(255).cpu().numpy()
                for name, mask in masks_gpu.items()
            }

        raw_scores = {
            name: float(metrics[index]) for index, name in enumerate(COLOR_ORDER)
        }
        valid_pixels = int(metrics[-1])
        return raw_scores, valid_pixels, masks

    def _torch_clean_mask(self, mask: Any) -> Any:
        """OpenCV 3x3 타원 커널과 같은 십자형 morphology를 PyTorch로 수행한다."""
        eroded = self._torch_ellipse_reduce(mask, reduce='min', border_value=1.0)
        opened = self._torch_ellipse_reduce(eroded, reduce='max', border_value=0.0)
        dilated = self._torch_ellipse_reduce(opened, reduce='max', border_value=0.0)
        return self._torch_ellipse_reduce(dilated, reduce='min', border_value=1.0)

    def _torch_ellipse_reduce(
        self,
        mask: Any,
        *,
        reduce: str,
        border_value: float,
    ) -> Any:
        """3x3 타원 커널의 중심/상하좌우 픽셀에 min/max를 적용한다."""
        values = mask.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        padded = torch_functional.pad(
            values,
            (1, 1, 1, 1),
            mode='constant',
            value=border_value,
        )
        neighbors = torch.stack(
            (
                padded[:, :, 1:-1, 1:-1],
                padded[:, :, :-2, 1:-1],
                padded[:, :, 2:, 1:-1],
                padded[:, :, 1:-1, :-2],
                padded[:, :, 1:-1, 2:],
            ),
            dim=0,
        )
        reduced = neighbors.amin(dim=0) if reduce == 'min' else neighbors.amax(dim=0)
        return reduced.squeeze(0).squeeze(0) > 0.5

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """opening으로 작은 잡음을 없앤 뒤 closing으로 작은 빈틈을 메운다."""
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.fallback_kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.fallback_kernel, iterations=1)
        return mask

    def _largest_component(self, mask: np.ndarray) -> int:
        """8방향 연결 성분 중 최대 면적(px)을 구한다. stats[0] 배경은 제외한다."""
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
        """분석 마스크와 점수를 BGR 영상에 합성하는 표시 전용 함수.

        반환 영상은 색상 값이 바뀌므로 판정에 재사용하면 안 된다. 복사/블렌딩을
        포함하기 때문에 디버그 수요가 있을 때만 호출해야 한다.
        """
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
        """종횡비를 유지해 표시용 canvas에 맞추고 남는 영역은 검은 여백으로 채운다."""
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
        """상태의 표시용 BGR 색상을 반환한다. 미등록 ID는 UNKNOWN 색상을 사용한다."""
        return STATE_COLORS.get(state, STATE_COLORS[STATE_UNKNOWN])

    def _draw_box_label(
        self,
        image: np.ndarray,
        text: str,
        x: int,
        y: int,
        color: tuple[int, int, int],
    ) -> None:
        """박스 위쪽에 배경과 텍스트를 그린다. 전달한 표시 영상에 직접 그린다."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.42
        thickness = 1
        text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
        height, width = image.shape[:2]
        if width < 20 or height < 20:
            return
        scale *= min(1.0, (width - 16) / max(text_size[0], 1), (height - 12) / max(text_size[1] + baseline, 1))
        text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
        if text_size[0] + 16 > width or text_size[1] + baseline + 12 > height:
            return
        label_x = max(4, min(x, width - text_size[0] - 14))
        label_y = max(text_size[1] + 6, min(y - 4, height - baseline - 3))
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
        """표시 영상 우상단에 색상 패널을 직접 합성한다.

        inset 자체를 축소하지 않으므로 호출자가 영상 안에 들어갈 크기로 준비해야 한다.
        """
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

    # 후보 순위 계산과 시간/좌표 단위 보조 함수

    def _point_in_window(
        self,
        x: float,
        y: float,
        window: tuple[int, int, int, int],
    ) -> bool:
        """후보 중심이 선호 영역에 속하는지 검사한다. 이 점 검사는 양 끝 경계를 포함한다."""
        x0, y0, x1, y1 = window
        return x0 <= x <= x1 and y0 <= y <= y1

    def _tracking_similarity(
        self,
        current_box: tuple[int, int, int, int],
        previous_box: tuple[int, int, int, int] | None,
        frame_shape: tuple[int, ...],
    ) -> float:
        """이전 대표 박스와의 중심 거리/IoU 중 큰 값을 순위 보너스로 사용한다.

        객체 ID를 유지하는 tracker는 아니다. 거리 점수는 원본 영상 대각선으로
        정규화하며, 이전 박스가 없으면 모든 후보에 동일한 최대 보너스를 준다.
        """
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
        """원본 픽셀 박스의 중심을 정수 절삭 없이 반환한다."""
        x_a, y_a, x_b, y_b = box
        return 0.5 * (x_a + x_b), 0.5 * (y_a + y_b)

    def _iou(
        self,
        box_a: tuple[int, int, int, int],
        box_b: tuple[int, int, int, int],
    ) -> float:
        """원본 좌표 박스의 교집합/합집합 면적 비율. 겹치지 않거나 퇴화하면 0이다."""
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
        """최빈 상태를 고르며 동률이면 UNKNOWN보다 알려진 상태를 우선한다.

        알려진 상태끼리 동률이면 이력에서 먼저 등장한 상태가 유지된다.
        상태 ID 크기나 적색 우선 규칙으로 동률을 해소하는 구현은 아니다.
        """
        if not values:
            return STATE_UNKNOWN
        counts = Counter(values)
        return max(counts.items(), key=lambda item: (item[1], item[0] != STATE_UNKNOWN))[0]

    def _now_ns(self) -> int:
        """판정/추적용 ROS 시각(ns). use_sim_time의 정지/점프 영향을 받는다."""
        return int(self.get_clock().now().nanoseconds)

    def _stamp_to_ns(self, msg: Image) -> int:
        """영상 header의 sec/nanosec를 순서 검증용 정수 ns로 합친다."""
        stamp = msg.header.stamp
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _ns_to_ms(self, value: int) -> float:
        """같은 시계에서 구한 ns 차이를 ms 파라미터와 비교할 단위로 바꾼다."""
        return value / 1e6


def main(args: list[str] | None = None) -> None:
    """mando_tl_fusion 진입점. 생성된 노드를 spin하고 종료 시 ROS/OpenCV 자원을 정리한다."""
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
