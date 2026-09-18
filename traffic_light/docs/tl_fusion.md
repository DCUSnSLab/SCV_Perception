# tl_fusion

`tl_fusion`은 카메라 이미지에서 차량 신호등 후보를 찾고, 모델 클래스와 실제 색 분석 결과를 결합해 안정적인 최종 상태를 발행하는 ROS 2 노드다.

대상 파일:

- [노드 코드](../mando_tools/tl_fusion.py)
- [런치](../launch/tl_fusion.launch.py)
- [두 노드 실행 명령 및 옵션](RUN.md)

## 1. 출력 상태

- `0`: `UNKNOWN`
- `1`: `RED`
- `2`: `YELLOW`
- `3`: `GREEN`
- `4`: `LEFT ARROW`

## 2. 현재 실행 방식

빌드 후 직접 실행:

```bash
source /opt/ros/humble/setup.bash
source /home/ki/SSC/install/setup.bash
ros2 run mando_tools mando_tl_fusion
```

launch로 실행:

```bash
ros2 launch mando_tools tl_fusion.launch.py
```

`tl_fusion.launch.py`는 bag를 직접 재생하지 않는다. launch의 기본 입력 토픽은 `/panorama/image_raw`이고, `ros2 run`으로 직접 띄우면 `default_runtime_image_topic()`이 정하는 `/mando/input/image`를 사용한다(`MANDO_IMAGE_TOPIC`으로 덮어쓸 수 있다).

다른 워크스페이스에서 실행할 때는 로컬 의존성 탐색을 위해 `MANDO_WS`를 지정하고,
모델 위치가 기본 경로와 다르면 `model_path`를 직접 넘긴다.

```bash
export MANDO_WS=/home/ki/SSC/src/perception/traffic_light
```

## 3. 전체 처리 흐름

1. subscriber는 최신 프레임 한 장만 `latest_msg`에 보관한다.
2. 타이머가 `max_fps` 주기로 최신 프레임만 처리한다.
3. detection window 안에서 YOLO 후보를 검출한다.
4. 후보 중 대표 신호등 하나를 선택한다.
5. 선택된 후보의 confidence와 무관하게 ROI를 확대하고 색상 보정을 수행한다.
6. 모델 confidence와 색상 분석 결과의 우선순위를 적용해 상태를 결정한다.
7. 프레임 단위 상태를 최근 이력으로 안정화한다.
8. `/tl/state_id`, `/tl/detections`, `/tl/debug_image` 세 토픽만 발행한다.
9. 입력 영상이 3초 동안 오지 않으면 `UNKNOWN(0)`과 빈 `/tl/detections`를 발행한다.
10. 디코딩·추론 오류도 즉시 `UNKNOWN(0)`과 빈 `/tl/detections`로 처리한다.
11. 디버그 이미지가 필요할 때만 `/tl/debug_image`를 생성해 발행한다.

## 4. 입력과 출력

입력:

- `image_topic`

출력:

- `/tl/debug_image`
- `/tl/state_id`
- `/tl/detections`

디버그 이미지는 아래 조건에서만 생성된다.

- `show_windows=true`
- `/tl/debug_image`에 실제 구독자가 존재

컬러 마스크 inset에는 원본 YOLO 후보 박스를 상·중·하 3등분하는 흰색 기준선 2개와 모든 색상에 적용되는 상단→중단→하단 가중치가 표시된다.

즉 평상시에는 디버그 프레임 전체 복사와 `cv2_to_imgmsg()` 직렬화를 건너뛴다.

## 5. YOLO 후보 검출

### 5.1 detection window

YOLO는 전체 프레임이 아니라 `detect_left_ratio`, `detect_right_ratio`, `detect_top_ratio`, `detect_bottom_ratio`로 정의되는 영역만 본다.

기본값은 원본 영상의 x=20~80%, y=0~1/3 영역이다.

### 5.2 클래스 필터

모델 클래스 이름 중 아래 조건에 맞는 것만 검출 대상으로 사용한다.

- 이름이 `vehicular_`로 시작
- 이름이 정확히 `traffic light`

### 5.3 박스 필터링

아래 조건을 만족하면 후보에서 제거한다.

- 너비 또는 높이가 0 이하
- 짧은 변이 `min_box_side_px`보다 작음
- 면적이 `min_box_area_px`보다 작음
- detection window 경계에 너무 가까움

## 6. 대표 후보 선택

YOLO가 여러 박스를 내면 하나만 대표 후보로 선택한다. 선택 점수는 아래 요소를 함께 본다.

- detection confidence
- 프레임 상단에 가까운 정도
- 박스 면적
- preferred window 포함 여부
- 이전 프레임 후보와의 tracking 유사도

tracking 유사도는 중심점 거리와 IoU를 함께 사용한다.

## 7. 모델 클래스에서 상태 해석

클래스 이름이 아래 규칙을 만족하면 상태로 해석한다.

- `left`와 `arrow`가 같이 있으면 `LEFT ARROW`
- `green_arrow` 클래스면 `LEFT ARROW`
- `red`와 `green`이 동시에 있으면 `LEFT ARROW`
- `green_arrow(down)`은 종료분기 표식이므로 신호등 상태에서 제외
- `yellow` 포함 시 `YELLOW`
- `green` 포함 시 `GREEN`
- `red` 포함 시 `RED`

아래는 위치 정보만 있고 상태는 직접 해석하지 않는다.

- 정확히 `traffic light`
- 이름에 `etc` 포함

## 8. 색 분석 fallback

대표 후보가 선택되면 모델 confidence와 무관하게 색 분석을 수행한다.

### 8.1 ROI 확장과 보정

- 박스를 `fallback_expand_ratio`와 `fallback_min_margin_px` 기준으로 넓혀 crop하되,
  색상 마스크와 점수는 원래 YOLO 후보 박스 내부로 제한한다.
- 작은 ROI는 최소 한 변 64px 기준으로 확대한다.
- CLAHE, saturation/value gain, gamma LUT, sharpen을 적용한다.

현재 gamma LUT는 초기화 시 한 번만 만들고 재사용한다.

### 8.2 색 점수 계산

HSV 기반으로 빨강, 노랑, 초록 마스크를 만든 뒤 가중합 점수를 계산한다.

핵심 기준:

- `fallback_min_valid_pixels`
- `fallback_score_threshold`
- `fallback_green_score_threshold`
- `fallback_score_gap`
- `fallback_red_h_max`
- `fallback_red_h_wrap_min`
- `fallback_red_v_min`
- `fallback_min_component_pixels`

초록색은 녹색 신호의 색 바램·저조도 편차를 흡수하기 위해 별도 범위를 사용한다.
기본값은 HSV H `39~90`, S `50` 이상, V `68` 이상, score `0.40`이다. 중간 1/3은 V `60`까지 허용한다.
후보 박스 내부의 모든 색상 score에 상·중·하 가중치 `0.20`·`1.50`·`0.20`을 적용한다.
중간 1/3에서는 기존 색상 마스크 픽셀을 3x3 커널로 1회 확장해 작은 끊김을 보완한다.
Hue/S/V 임계값을 낮추는 방식이 아니므로 배경색을 새로 색상 픽셀로 만들지는 않는다.
빨강은 원본 crop의 hue를 사용해 보정 과정에서 생기는 주황색 이동을 차단한다. H `0~8`
또는 `170~179`, S `55` 이상, 원본 V `85` 이상이며, 보정 영상의 V도 `70` 이상이어야 한다.
따라서 꺼진 주황색 램프가 빨강으로 확정되는 것을 제한한다. score gap은 `0.10`이다.

`LEFT ARROW` 색 규칙은 아래 조건을 동시에 볼 때 사용한다.

- `scores['red'] >= fallback_red_green_red_min`
- `scores['green'] >= fallback_red_green_green_min`
- `scores['yellow'] <= fallback_red_green_yellow_max`

## 9. 최종 상태 결정

`_decide_state()`는 대략 아래 우선순위로 동작한다.

1. 모델 confidence가 `model_confidence_threshold` 이상이면 `model`
2. 색 분석이 확실하고 모델이 약하면 `color_fallback`
3. 모델 confidence가 `model_min_confidence_threshold` 이상이면 `model_low_conf`
4. 색 분석만 확실하면 `color_only`
5. 모델 해석은 가능하지만 약하면 `model_weak`
6. 둘 다 애매하면 `unknown`

`/tl/detections`는 후보마다 원본 영상 기준 bbox, class name, confidence를
`vision_msgs/Detection2DArray`로 발행한다. 후보가 없거나 입력이 무효이면 빈 배열이다.
상태 전환 및 오류 상세 원인은 ROS 로그에서 확인한다.

## 10. 상태 안정화

`_update_stable_state()`는 순간적인 흔들림을 줄이기 위해 아래 파라미터를 사용한다.

- `state_window_size`
- `hold_ms`
- `missing_timeout_ms`
- `uncertain_hold_ms`
- `reset_tracking_ms`

동작 요약:

- 후보가 잠깐 사라져도 `missing_timeout_ms` 이내면 직전 상태를 유지한다.
- 후보는 있지만 모델/색상 근거가 모호하면 `uncertain_hold_ms` 이내 직전 상태를 유지한다.
- 최근 상태 버퍼에서 다수결을 구한다.
- `hold_ms`가 지나기 전에는 쉽게 상태를 바꾸지 않는다.
- 후보가 오래 없으면 tracking 기준 박스를 초기화한다.

## 11. launch에서 바로 조절할 수 있는 주요 인자

아래는 주요 노드 파라미터다. 런치에서 노출하는 전체 인자와 기본값은
[실행 가이드](RUN.md)에 정리했다. `fallback_score_gap`은 `tl_fusion.launch.py`의
인자가 아니므로 `ros2 run --ros-args -p`로 지정한다.

- `model_path`
- `image_topic`
- `show_windows`
- `max_fps`
- `color_fallback_device`
- `fallback_max_side_px`
- `detector_conf_threshold`
- `detector_image_size`
- `model_confidence_threshold`
- `enable_low_confidence_color_fallback`
- `debug_image_max_side_px`
- `debug_publish_period_ms`
- `publish_debug_image`
- `fallback_score_threshold`
- `fallback_green_score_threshold`
- `fallback_score_gap`
- `fallback_red_h_max`
- `fallback_red_h_wrap_min`
- `fallback_red_v_min`
- `fallback_green_h_min`
- `fallback_green_h_max`
- `fallback_green_s_min`
- `fallback_green_v_min`
- `fallback_green_middle_v_min`
- `fallback_green_top_weight`
- `fallback_green_middle_weight`
- `fallback_middle_mask_dilate_iterations`
- `fallback_green_bottom_weight`
- `uncertain_hold_ms`
- `fallback_saturation_gain`
- `fallback_value_gain`
- `fallback_gamma`

현재 기본값은 YOLO 입력 크기 `1280`, 디버그 영상 최대 변 길이 `640`, 디버그 최소 출력 간격 `200ms`, 후보 confidence `0.05`, 일반 색상 score `0.45`, 초록색 score `0.40`, 색상 score gap `0.10`, 빨강 H `0~8`/`170~179`, 빨강 최소 V `85`, 모델 신뢰도 `0.75`, 공통 색상 상·중·하 가중치 `0.20`·`1.50`·`0.20`, 채도 gain `2.20`, 밝기 gain `1.35`, gamma `1.00`이다. 기존 `fallback_green_*_weight` 파라미터 이름은 호환성을 위해 유지한다.
색상 보정은 모든 선택 후보에 항상 적용되며, 모델 confidence가 `0.75` 미만이면 색상 fallback이 최종 판단에 더 적극적으로 사용된다.

예시:

```bash
ros2 launch mando_tools tl_fusion.launch.py \
  image_topic:=/zed/zed_node/left/image_rect_color \
  max_fps:=10.0 \
  detector_conf_threshold:=0.15
```

## 12. 이 노드를 쓰는 시점

- 후속 노드가 바로 사용할 최종 상태가 필요할 때
- 모델 클래스와 색 분석을 함께 활용하고 싶을 때
- `/tl/detections`와 `/tl/debug_image`로 검출 결과와 ROI를 추적하고 싶을 때
