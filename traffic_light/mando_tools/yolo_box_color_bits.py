"""YOLO로 디스플레이 위치를 찾고 HSV로 내부 색을 독립 판정한다.

흐름: 부모 클래스의 ROI 추출 → YOLO 박스 → 중복 억제 → 내부 HSV 점수
→ 부모 클래스의 위치 매칭/EMA/비트 확정. ROS 발행은 BlackBoxColorBitsNode가 담당한다.
이 모드는 색 영역 모드의 LED 묶기·주변 어둠·박스 크기 필터를 사용하지 않는다.
YOLO가 놓친 디스플레이를 색 영역 검출로 보완하는 자동 fallback도 없다.
"""

from pathlib import Path

from .black_box_color_bits import BlackBoxColorDetector, BoxObservation, DetectorConfig, _iou
from .workspace_paths import resolve_inference_device

import cv2
import numpy as np


class YoloBoxColorDetector(BlackBoxColorDetector):
    """YOLO 클래스는 위치 필터에만 사용하며 클래스 이름을 비트로 변환하지 않는다.

    예를 들어 green_sign으로 검출된 박스라도 내부 HSV가 빨강이면 0을 제안한다.
    클래스 확률과 HSV 면적 비율은 별개이며, HSV 점수만 시간 안정화에 전달한다.
    부모의 트랙 상태를 변경하므로 스트림별 인스턴스를 사용하고 병렬 호출하지 않는다.
    """

    def __init__(
        self, config: DetectorConfig, model_path: str, device: str = 'auto',
        confidence: float = 0.25, image_size: int = 640, model=None,
    ) -> None:
        """가중치를 한 번 로드하고 허용 클래스를 찾는다.

        confidence는 YOLO 후보 확률 하한(0~1), image_size는 추론 imgsz 값이다.
        반환 박스는 imgsz 텐서가 아닌 전달한 ROI의 픽셀 좌표다.
        model 주입은 테스트용이며 파일 검사·가중치 로드·device 자동 해석을 건너뛴다.
        """
        super().__init__(config)
        if not 0.0 < confidence <= 1.0 or image_size <= 0:
            raise ValueError('confidence must be in (0,1] and image_size must be positive')
        self.device = resolve_inference_device(device) if model is None else device
        self.confidence = confidence
        self.image_size = image_size
        if model is None:
            # 잘못된 경로를 모델 이름으로 해석해 다운로드하지 않도록 먼저 파일을 검사한다.
            path = Path(model_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f'Box model not found: {path}')
            from ultralytics import YOLO
            model = YOLO(str(path))
        if getattr(model, 'task', 'detect') != 'detect':
            raise ValueError('Box model must be an object detection model')
        self.model = model
        names = model.names
        # 클래스 ID를 고정하지 않는다. 모델의 이름에서 찾아 truck 등 다른 클래스를 제외한다.
        entries = names.items() if isinstance(names, dict) else enumerate(names)
        self.class_ids = [int(index) for index, name in entries if name in ('green_sign', 'red_sign')]
        if not self.class_ids:
            raise ValueError('Box model must contain green_sign or red_sign classes')

    def _detect_observations(self, roi, roi_bounds):
        """uint8 BGR ROI에서 관측 목록을 만든다. roi_bounds는 전체 영상 좌표다.

        내부 계산은 ROI 로컬 (left, top, right, bottom), 슬라이싱 끝점은 제외한다.
        관측 bbox를 만들 때만 ROI 시작점을 더해 부모 추적기의 전체 좌표계로 복원한다.
        """
        if roi.size == 0:
            return []
        result = self.model.predict(
            source=roi, classes=self.class_ids, conf=self.confidence,
            imgsz=self.image_size, device=self.device, iou=0.45,
            agnostic_nms=True, max_det=50, verbose=False,
        )[0]
        if result.boxes is None:
            return []
        # 검출 결과는 [x0, y0, x1, y1, confidence, class_id]; GPU 결과를 한 번에 옮긴다.
        rows = result.boxes.data.cpu().numpy()
        selected = []
        # 높은 confidence를 우선해 동일 물체의 red/green 중복 박스도 하나만 남긴다.
        # 모델 종류별 NMS 동작 차이를 고려해 아래에서도 클래스 무관 IoU 억제를 수행한다.
        for row in sorted(rows, key=lambda item: float(item[4]), reverse=True):
            if not np.isfinite(row[:6]).all() or int(row[5]) not in self.class_ids or row[4] < self.confidence:
                continue
            left, top, right, bottom = (int(value) for value in row[:4])
            # 원본 ROI 바깥 박스는 잘라내고 빈 영역은 제거해 이후 HSV 변환을 보호한다.
            left, right = max(0, left), min(roi.shape[1], right)
            top, bottom = max(0, top), min(roi.shape[0], bottom)
            if right <= left or bottom <= top:
                continue
            bbox = (left, top, right, bottom)
            if any(_iou(bbox, existing) >= 0.45 for existing in selected):
                continue
            selected.append(bbox)
        observations = []
        offset_x, offset_y = roi_bounds[:2]
        # 좌→우, 같은 x이면 위→아래 순서. 중복 판정 기준 IoU는 현재 0.45로 고정이다.
        margin = max(0.0, min(0.49, self.config.inner_margin_ratio))
        for left, top, right, bottom in sorted(selected):
            # 검출 테두리/배경을 제외한다. 기본 20%씩 제외해 중앙 약 60%×60%를 사용한다.
            # 여백 상한을 49%로 제한해 작은 박스도 내부 픽셀이 남도록 한다.
            margin_x, margin_y = int((right-left)*margin), int((bottom-top)*margin)
            inner = roi[top+margin_y:bottom-margin_y, left+margin_x:right-margin_x]
            hue, saturation, value = cv2.split(cv2.cvtColor(inner, cv2.COLOR_BGR2HSV))
            # uint8 OpenCV HSV: H=0~179, S/V=0~255. 빨강은 hue 양 끝 구간의 합집합이다.
            valid = (saturation >= self.config.color_s_min) & (value >= self.config.color_v_min)
            red = valid & ((hue <= self.config.red_hue_high) | (hue >= self.config.red_hue_low_wrap))
            green = valid & (hue >= self.config.green_hue_low) & (hue <= self.config.green_hue_high)
            # 분모에는 검정/노랑/저채도 픽셀도 포함해 작은 색 잡음이 높은 점수가 되지 않게 한다.
            # 색상이 불확실해도 박스 관측은 남는다. 부모는 이전 비트를 유지할 수 있으며,
            # hold_timeout은 마지막 유효 색상이 아니라 마지막 박스 관측 이후에 적용된다.
            pixels = inner.shape[0] * inner.shape[1]
            observations.append(BoxObservation(
                bbox=(left+offset_x, top+offset_y, right+offset_x, bottom+offset_y),
                red_score=float(np.count_nonzero(red)/pixels),
                green_score=float(np.count_nonzero(green)/pixels),
            ))
        return observations
