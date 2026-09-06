# tl_fusion

`tl_fusion`은 카메라 이미지에서 차량 신호등 후보를 찾고, 모델 클래스와 실제 색 분석 결과를 결합해 안정적인 최종 상태를 발행하는 ROS 2 노드다.

대상 파일:

- `src/perception/traffic_light/mando_tools/tl_fusion.py`
- `src/perception/traffic_light/launch/tl_fusion.launch.py`

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
source /home/ssc/SSC/src/perception/install/setup.bash
ros2 run mando_tools mando_tl_fusion
```

launch로 실행:

```bash
ros2 launch mando_tools tl_fusion.launch.py
```

`tl_fusion.launch.py`는 bag를 직접 재생하지 않는다. launch의 기본 입력 토픽은 `/panorama/image_raw`이고, `ros2 run`으로 직접 띄우면 `default_runtime_image_topic()`이 정하는 `/mando/input/image`를 사용한다(`MANDO_IMAGE_TOPIC`으로 덮어쓸 수 있다).

install space에서 실행할 때는 노드가 소스 트리를 찾지 못하므로, 기본 `model_path` 해석을 위해 `MANDO_WS`를 지정하거나 `model_path`를 직접 넘긴다.

```bash
export MANDO_WS=/home/ssc/SSC/src/perception/traffic_light
```

## 3. 전체 처리 흐름

1. subscriber는 최신 프레임 한 장만 `latest_msg`에 보관한다.
2. 타이머가 `max_fps` 주기로 최신 프레임만 처리한다.
3. detection window 안에서 YOLO 후보를 검출한다.
4. 후보 중 대표 신호등 하나를 선택한다.
5. 모델 클래스가 충분히 신뢰되면 그 상태를 바로 사용한다.
6. 그렇지 않으면 ROI를 확대하고 색 분석 fallback을 수행한다.
7. 프레임 단위 상태를 최근 이력으로 안정화한다.
8. `/tl/state_id`, `/tl/state_label`, `/tl/state_reason`을 발행한다.
9. 입력 영상이 3초 동안 오지 않으면 `UNKNOWN(0)`과 `/tl/input_valid=false`를 발행한다.
9. 디버그 이미지가 필요할 때만 `/tl/debug_image`를 생성해 발행한다.

## 4. 입력과 출력

입력:

- `image_topic`

출력:

- `/tl/debug_image`
- `/tl/state_id`
- `/tl/state_label`
- `/tl/state_reason`
- `/tl/input_valid`

디버그 이미지는 아래 조건에서만 생성된다.

- `show_windows=true`
- `/tl/debug_image`에 실제 구독자가 존재

즉 평상시에는 디버그 프레임 전체 복사와 `cv2_to_imgmsg()` 직렬화를 건너뛴다.

## 5. YOLO 후보 검출

### 5.1 detection window

YOLO는 전체 프레임이 아니라 `detect_left_ratio`, `detect_right_ratio`, `detect_top_ratio`, `detect_bottom_ratio`로 정의되는 영역만 본다.

기본값은 전체 화면이다.

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

대표 후보가 선택되었지만 모델 confidence가 충분하지 않거나 클래스가 일반적이면 색 분석을 수행한다.

### 8.1 ROI 확장과 보정

- 박스를 `fallback_expand_ratio`와 `fallback_min_margin_px` 기준으로 넓혀 crop한다.
- 작은 ROI는 최소 한 변 64px 기준으로 확대한다.
- CLAHE, saturation/value gain, gamma LUT, sharpen을 적용한다.

현재 gamma LUT는 초기화 시 한 번만 만들고 재사용한다.

### 8.2 색 점수 계산

HSV 기반으로 빨강, 노랑, 초록 마스크를 만든 뒤 가중합 점수를 계산한다.

핵심 기준:

- `fallback_min_valid_pixels`
- `fallback_score_threshold`
- `fallback_score_gap`
- `fallback_min_component_pixels`

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

`/tl/state_reason`에는 위 source와 구체적 이유가 함께 들어간다.

## 10. 상태 안정화

`_update_stable_state()`는 순간적인 흔들림을 줄이기 위해 아래 파라미터를 사용한다.

- `state_window_size`
- `hold_ms`
- `missing_timeout_ms`
- `reset_tracking_ms`

동작 요약:

- 후보가 잠깐 사라져도 `missing_timeout_ms` 이내면 직전 상태를 유지한다.
- 최근 상태 버퍼에서 다수결을 구한다.
- `hold_ms`가 지나기 전에는 쉽게 상태를 바꾸지 않는다.
- 후보가 오래 없으면 tracking 기준 박스를 초기화한다.

## 11. launch에서 바로 조절할 수 있는 주요 인자

`tl_fusion.launch.py`가 기본으로 노출하는 인자는 아래와 같다.

- `model_path`
- `image_topic`
- `show_windows`
- `max_fps`
- `detector_conf_threshold`
- `detector_image_size`
- `model_confidence_threshold`
- `fallback_score_threshold`

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
- `/tl/state_reason`까지 포함해 판정 근거를 추적하고 싶을 때
