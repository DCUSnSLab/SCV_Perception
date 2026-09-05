# 신호등 검출 코드 가이드

`mando_tools` 안에서 신호등을 검출하거나 최종 상태로 변환하는 핵심 코드는 아래 3개다.

- `src/mando_tools/mando_tools/yolo_validator.py`
- `src/mando_tools/mando_tools/tl_roi_hist.py`
- `src/mando_tools/mando_tools/tl_fusion.py`

이 문서는 현재 코드 기준으로 각 노드의 역할, 실행 방식, 입력/출력, 내부 판단 흐름을 빠르게 정리한다.

## 1. 공통 상태 ID

| ID | 의미 |
| --- | --- |
| `0` | `UNKNOWN` |
| `1` | `RED` |
| `2` | `YELLOW` |
| `3` | `GREEN` |
| `4` | `LEFT ARROW` |

클래스 이름에 `green_arrow`, `left+arrow`, `red+green`이 포함되면 `LEFT ARROW`로 해석한다.
종료분기용 `green_arrow(down)`은 신호등 주행 상태에서 제외한다.

## 2. 공통 실행 환경

- 기본 bag profile은 `stop_points`다.
- `stop_points`의 기본 이미지 토픽은 `/zed/zed_node/left/image_rect_color`다.
- `mando_ros2`의 기본 이미지 토픽은 `/zed_node/left/image_rect_color`다.
- 기본 모델은 `model/best.pt`를 먼저 찾고, 없으면 `yolo11s.pt`를 사용한다.

기본 환경 확인:

```bash
source /opt/ros/humble/setup.bash
source /home/ssc/SSC/src/perception/install/setup.bash
ros2 run mando_tools workspace_info
```

## 3. 코드별 역할 요약

| 파일 | 역할 | 대표 출력 |
| --- | --- | --- |
| `yolo_validator.py` | YOLO 클래스 결과만으로 상태를 빠르게 검증 | annotated image, `Detection2DArray`, `/tl/yolo_validator/state` |
| `tl_roi_hist.py` | YOLO로 ROI를 찾고 Hue histogram 기반으로 상태 판정 | `/tl/debug_image`, `/tl/zoom_image`, `/tl/hist_image`, `/tl/roi_hist/state` |
| `tl_fusion.py` | 모델 클래스와 색 분석을 결합하고 안정화까지 적용 | `/tl/debug_image`, `/tl/state_id`, `/tl/state_label`, `/tl/state_reason` |

## 4. `yolo_validator.py`

### 4.1 역할

`yolo_validator.py`는 가장 단순한 기준 노드다. 모델이 어떤 박스와 클래스 이름을 내는지 확인하는 데 초점을 둔다.

### 4.2 처리 구조

1. 가장 최근 프레임 한 장만 `latest_msg`에 유지한다.
2. 타이머 기반으로 `max_fps` 주기마다 최신 프레임만 추론한다.
3. YOLO 검출 결과에서 상태로 해석 가능한 클래스를 찾는다.
4. `Detection2DArray`, annotated image, 전용 검증 상태 토픽을 발행한다.

### 4.3 상태 판정

- `vehicular_*` 또는 `traffic light` 클래스만 검출 대상으로 사용한다.
- `traffic light`, `trafficlight`, `etc` 계열은 상태를 직접 해석하지 않는다.
- 상태로 해석 가능한 클래스가 여러 개면 confidence가 가장 높은 결과를 사용한다.
- 색 분석 fallback은 없다.

### 4.4 언제 쓰면 좋은가

- 새 모델의 라벨링 결과를 빠르게 점검할 때
- `Detection2DArray`와 시각화 이미지를 같이 확인하고 싶을 때
- 후처리 없이 모델 자체 결과를 보고 싶을 때

실행:

```bash
ros2 launch mando_tools validate_mando_bag.launch.py
```

## 5. `tl_roi_hist.py`

### 5.1 역할

`tl_roi_hist.py`는 YOLO를 위치 검출기로 사용하고, 최종 상태는 ROI 내부의 색 분포로 판정한다.

### 5.2 처리 구조

1. subscriber는 최신 프레임 한 장만 유지한다.
2. `_process_latest_frame()`이 `max_fps` 주기로 최신 프레임만 처리한다.
3. detection window 안에서 YOLO 후보를 찾는다.
4. preferred window, 박스 크기, 상단 가중치, tracking 점수로 대표 박스를 선택한다.
5. 선택한 ROI를 morphology 기반으로 보정하고 Hue histogram을 계산한다.
6. EMA, 다수결, hysteresis, hold time으로 상태를 안정화한다.

### 5.3 상태 판정

- `r_score`, `y_score`, `g_score`를 계산한다.
- `lt_rule_enable=true`이고 빨강/초록 조합이 강하면 `LEFT ARROW`를 사용한다.
- 일반 상태는 EMA가 가장 높은 색을 사용한다.
- 박스가 잠깐 사라져도 `missing_timeout_ms` 이내면 직전 상태를 유지한다.

### 5.4 디버그 출력

- `/tl/debug_image`
- `/tl/zoom_image`
- `/tl/hist_image` (`pub_hist_image=true`일 때)

`tl_roi_hist.launch.py`는 노드만 실행한다. bag 영상은 별도로 재생해야 한다.

실행:

```bash
ros2 launch mando_tools tl_roi_hist.launch.py
```

## 6. `tl_fusion.py`

### 6.1 역할

`tl_fusion.py`는 현재 코드베이스에서 가장 최종형에 가까운 노드다. 모델 클래스 결과와 ROI 색 분석 결과를 함께 보고, 어느 쪽을 더 신뢰할지 결정한 뒤 시간축 안정화를 적용한다.

### 6.2 처리 구조

1. 최신 프레임 한 장만 유지한다.
2. detection window 안에서 YOLO 후보를 검출한다.
3. preferred window, 박스 크기, 상단 가중치, tracking 유사도로 대표 후보를 선택한다.
4. confidence가 충분히 높으면 모델 상태를 직접 사용한다.
5. 그렇지 않으면 확대 crop과 HSV 기반 색 분석으로 fallback 판단을 만든다.
6. `state_window_size`, `hold_ms`, `missing_timeout_ms`, `reset_tracking_ms`로 상태를 안정화한다.

### 6.3 판단 우선순위

- 모델 confidence가 `model_confidence_threshold` 이상이면 모델 상태 우선
- 모델이 약하거나 상태를 직접 해석할 수 없으면 `color_fallback`
- 모델 confidence가 중간 수준이면 `model_low_conf`
- 모델이 약하지만 색 분석은 확실하면 `color_only`
- 둘 다 애매하면 `UNKNOWN`

### 6.4 현재 디버그 동작

- `/tl/debug_image`는 구독자가 있거나 `show_windows=true`일 때만 생성/발행한다.
- `show_windows=true`면 OpenCV 디버그 창을 띄운다.
- 별도 zoom/panel 창 이미지는 현재 publish하지 않는다.

실행:

```bash
ros2 launch mando_tools tl_fusion.launch.py
```

`tl_fusion.launch.py`는 bag를 직접 재생하지 않는다. 영상 입력은 외부에서 공급해야 한다.

## 7. 관련 launch 파일

| launch 파일 | 연결 코드 | 설명 |
| --- | --- | --- |
| `src/mando_tools/launch/play_mando_bag.launch.py` | bag only | 기본 rosbag2 재생 |
| `src/mando_tools/launch/validate_mando_bag.launch.py` | `yolo_validator.py` | validator 실행, 필요 시 bag도 같이 재생 |
| `src/mando_tools/launch/tl_roi_hist.launch.py` | `tl_roi_hist.py` | ROI + histogram 노드만 실행 |
| `src/mando_tools/launch/tl_fusion.launch.py` | `tl_fusion.py` | fusion 노드 실행, bag는 외부에서 별도 공급 |

## 8. 어떤 코드를 언제 쓰면 되는가

- 모델 클래스가 잘 학습됐는지 먼저 보고 싶다: `yolo_validator.py`
- 실제 색 분포를 기준으로 상태를 보고 싶다: `tl_roi_hist.py`
- 후속 노드가 바로 쓸 안정적인 최종 상태가 필요하다: `tl_fusion.py`
