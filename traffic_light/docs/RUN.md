# tl_fusion / black_box 실행 명령어

대상은 ROS 2 Humble의 `mando_tools` 패키지다. 이 문서에서 `black_box`는
`black_box_color_bits` 노드를 뜻한다. 옵션 기본값은 현재 두 개별 런치 파일 기준이다.

| 구분 | tl_fusion | black_box |
| --- | --- | --- |
| 런치 | `tl_fusion.launch.py` | `black_box_color_bits.launch.py` |
| 실행 파일 | `mando_tl_fusion` | `mando_black_box_color_bits` |
| 노드 이름 | `/tl_fusion` | `/black_box_color_bits` |
| 모델 | `model/best.pt` | `model/box_best.pt` |
| 입력 | `/panorama/image_raw` | `/panorama/image_raw` |
| 판정 출력 | `/tl/state_id` | `/tl/box_color_bits` |
| 디버그 출력 | `/tl/debug_image` | `/tl/box_color_bits/debug` |
| 최대 처리율 기본값 | `5 FPS` | `5 FPS` |

## 환경 준비와 빌드

SSC 워크스페이스가 `~/SSC`에 있다고 가정한다. 다른 위치라면 `SSC_WS`를 변경한다.
`MANDO_WS`는 워크스페이스 루트가 아니라 `traffic_light` 패키지 경로다.
새 터미널에서도 아래 환경 설정을 적용한다.

```bash
export SSC_WS="$HOME/SSC"
export MANDO_WS="$SSC_WS/src/perception/traffic_light"
source /opt/ros/humble/setup.bash
source "$SSC_WS/install/setup.bash"
```

처음 설치하거나 파일 정리 후 다시 빌드할 때:

```bash
export SSC_WS="$HOME/SSC"
export MANDO_WS="$SSC_WS/src/perception/traffic_light"
cd "$SSC_WS"
source /opt/ros/humble/setup.bash
colcon build --packages-select mando_tools --symlink-install
source "$SSC_WS/install/setup.bash"
```

기존 symlink 설치에서 실험 런치 이동 후 `File exists` 오류가 발생하면, 아래 명령으로
끊어진 설치 링크만 제거하고 다시 빌드한다. 소스 파일과 정상 링크는 유지한다.

```bash
for launch_name in green_down_arrow tl_roi_hist validate_mando_bag; do
  launch_link="$SSC_WS/install/mando_tools/share/mando_tools/launch/$launch_name.launch.py"
  if [ -L "$launch_link" ] && [ ! -e "$launch_link" ]; then
    rm -- "$launch_link"
  fi
done
cd "$SSC_WS"
colcon build --packages-select mando_tools --symlink-install
source "$SSC_WS/install/setup.bash"
```

ROS 의존성 외에 OpenCV, NumPy, PyTorch, Ultralytics가 필요하다. 두 노드는
`$MANDO_WS/.deps`가 있으면 이 로컬 의존성을 우선 사용한다. 가중치 파일은 Git에
포함하지 않으므로 `model/best.pt`, `model/box_best.pt`를 별도로 준비한다.
black_box 모델에는 `green_sign` 또는 `red_sign` 검출 클래스가 있어야 한다.

## 기본 실행

### tl_fusion

```bash
ros2 launch mando_tools tl_fusion.launch.py \
  model_path:="$MANDO_WS/model/best.pt"
```

기본 모델 경로는 `/home/ki/SSC/src/perception/traffic_light/model/best.pt`다.
위 예시는 다른 계정이나 설치 위치에서도 명시한 패키지의 가중치를 사용한다.

### black_box

```bash
ros2 launch mando_tools black_box_color_bits.launch.py
```

### 실차 운행용: 디버그 이미지 완전 비활성화

실차 운행에서는 아래처럼 두 노드의 디버그 발행을 명시적으로 끈다. 이 설정은
디버그 토픽에 뷰어나 RViz가 연결된 상태에서도 이미지 생성과 발행을 막는다.

```bash
# tl_fusion
"$MANDO_WS/run_traffic_light.sh" publish_debug_image:=false

# black_box_color_bits
ros2 launch mando_tools black_box_color_bits.launch.py \
  publish_debug_image:=false
```

`tl_fusion.launch.py`를 직접 사용할 때는 창도 끈다.

```bash
ros2 launch mando_tools tl_fusion.launch.py \
  model_path:="$MANDO_WS/model/best.pt" \
  publish_debug_image:=false \
  show_windows:=false
```

`run_traffic_light.sh`는 `traffic_light.launch.py`를 호출하므로
`show_windows` 인자를 사용하지 않고 `publish_debug_image:=false`만 전달한다.
두 노드의 실제 판정 토픽은 디버그 설정과 관계없이 발행된다.

두 노드를 동시에 사용할 때는 각각 별도 터미널에서 실행한다. 각 노드는 하나씩만
실행하고, 입력 카메라 또는 파노라마 노드는 별도로 실행한다.
black_box의 기본 가중치는 탐색된 패키지의 `model/box_best.pt`다.

### 입력 토픽 및 처리율 변경

```bash
ros2 launch mando_tools tl_fusion.launch.py \
  model_path:="$MANDO_WS/model/best.pt" \
  image_topic:=/panorama/image_raw max_fps:=5.0
```

```bash
ros2 launch mando_tools black_box_color_bits.launch.py \
  image_topic:=/panorama/image_raw max_fps:=5.0
```

`max_fps`는 처리 주기의 상한이며 실제 FPS를 보장하지 않는다. 두 노드 모두 최신
입력을 사용하며, 추론 시간이 길거나 영상이 없으면 실제 출력 빈도는 낮아진다.

## rosbag 재생

과거 영상은 bag의 `/clock`과 두 노드의 `use_sim_time:=true`를 함께 사용한다.
bag에 `/panorama/image_raw`가 기록되어 있다고 가정한다.

터미널 1 — bag 재생:

```bash
ros2 bag play /absolute/path/to/bag --clock
```

터미널 2 — tl_fusion:

```bash
ros2 launch mando_tools tl_fusion.launch.py \
  model_path:="$MANDO_WS/model/best.pt" \
  use_sim_time:=true
```

터미널 3 — black_box:

```bash
ros2 launch mando_tools black_box_color_bits.launch.py \
  use_sim_time:=true
```

bag의 영상 토픽이 다르면 두 런치에 `image_topic:=/기록된/영상토픽`을 추가한다.
`ros2 bag info /absolute/path/to/bag`로 기록된 토픽을 확인할 수 있다.
실시간 카메라에서는 기본값인 `use_sim_time:=false`를 사용한다.
기존 `play_mando_bag.launch.py`는 `/clock`을 자동 발행하지 않으므로 위 명령과
동일하게 취급하면 안 된다. 반복 재생이나 seek 이후 시각 역행 오류가 지속되면
입력 발행자가 하나인지, 두 노드가 같은 `/clock`을 사용하는지 확인하고 노드를 재시작한다.

## 디버그 영상

### tl_fusion: 축소 및 발행 주기 제한

```bash
ros2 launch mando_tools tl_fusion.launch.py \
  model_path:="$MANDO_WS/model/best.pt" \
  publish_debug_image:=true \
  show_windows:=false \
  debug_image_max_side_px:=640 \
  debug_publish_period_ms:=200.0
```

- 기본 최대 변은 `640px`, 디버그 ROS 발행 간격은 최소 `200ms`로 최대 `5Hz`다.
- 구독자가 없고 OpenCV 창도 꺼져 있으면 디버그 이미지를 생성하지 않는다.
- `show_windows:=true`이면 처리 프레임마다 로컬 창을 갱신한다. ROS 발행 주기 제한은 유지된다.
- `debug_image_max_side_px:=0`은 검출 ROI 원래 크기, `debug_publish_period_ms:=0`은
  구독자가 있을 때 처리 프레임마다 발행하는 설정이다. 추론 입력 크기와는 별개다.

### black_box: 필요할 때만 디버그 켜기

```bash
ros2 launch mando_tools black_box_color_bits.launch.py \
  publish_debug_image:=true
```

기본값은 `false`다. `true`로 실행하고 토픽 구독자가 있어야 영상이 발행된다.
ROI 크롭에 박스만 그리며, 초록=1, 빨강=0, 주황=미확정이다.
tl_fusion의 `debug_image_max_side_px`, `debug_publish_period_ms`, `show_windows`는
black_box 런치 옵션이 아니다. black_box 디버그는 유효 입력을 처리한 주기에 맞춰 발행된다.

영상 뷰어가 설치되어 있으면 별도 터미널에서 다음과 같이 열고 토픽을 선택한다.

```bash
ros2 run rqt_image_view rqt_image_view
```

black_box 디버그의 QoS는 `BEST_EFFORT`다. RViz에서 볼 때 Reliability를
`Best Effort`로 설정한다. 두 디버그 영상 모두 원본 header를 유지하지만
ROI 크롭 또는 리사이즈 결과이므로 원본 영상 픽셀 좌표와 다르다.

## CPU 실행과 노드 전용 파라미터

tl_fusion은 런치에서 장치를 선택할 수 있다.

```bash
ros2 launch mando_tools tl_fusion.launch.py \
  model_path:="$MANDO_WS/model/best.pt" \
  detector_device:=cpu color_fallback_device:=cpu
```

black_box 런치는 `box_device=cuda:0`, `opencv_threads=1`, 자동 탐색한
`box_model_path`를 고정해서 전달한다. CPU를 강제하거나 다른 가중치를 사용하려면
노드를 직접 실행한다.

```bash
ros2 run mando_tools mando_black_box_color_bits --ros-args \
  -p image_topic:=/panorama/image_raw \
  -p box_model_path:="$MANDO_WS/model/box_best.pt" \
  -p box_device:=cpu \
  -p opencv_threads:=1 \
  -p max_fps:=5.0
```

CUDA 요청 시 CUDA를 사용할 수 없으면 두 노드 모두 CPU로 전환한다. 실제 장치는
시작 로그를 확인한다. `auto` 장치 선택은 `MANDO_DEVICE` 환경 변수의 영향도 받는다.

`fallback_score_gap`처럼 tl_fusion 개별 런치가 노출하지 않는 노드 파라미터는
다음 형식으로 지정한다. 직접 실행 시 입력 기본값은 `/mando/input/image`이므로
`image_topic`을 명시한다.

```bash
ros2 run mando_tools mando_tl_fusion --ros-args \
  -p model_path:="$MANDO_WS/model/best.pt" \
  -p image_topic:=/panorama/image_raw \
  -p fallback_score_gap:=0.10
```

`ros2 launch`에는 `이름:=값`, `ros2 run`에는 `--ros-args -p 이름:=값`을 사용한다.
두 노드의 설정은 시작 시 읽으므로, 옵션 변경은 종료 후 재실행으로 적용한다.

## tl_fusion 런치 옵션 전체

아래 표는 `tl_fusion.launch.py`의 옵션이다.

### 입력·추론·시각화

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `model_path` | `/home/ki/SSC/src/perception/traffic_light/model/best.pt` | YOLO 가중치 경로 |
| `image_topic` | `/panorama/image_raw` | 입력 영상 |
| `state_topic` | `/tl/state_id` | 최종 상태 출력 |
| `use_sim_time` | `false` | `/clock` 사용 |
| `input_timeout_s` | `3.0` | 영상 미수신 시 UNKNOWN 전환 기준(초) |
| `max_fps` | `5.0` | 처리율 상한 |
| `detector_device` | `cuda:0` | YOLO 장치: `cuda:0`, `cpu`, `auto` |
| `color_fallback_device` | `auto` | 색 분석 장치. `auto`는 선택된 YOLO 장치를 따름 |
| `detector_conf_threshold` | `0.05` | 검출 후보 confidence 하한 |
| `detector_image_size` | `640` | YOLO 추론 크기 |
| `model_confidence_threshold` | `0.75` | 모델 상태를 직접 신뢰하는 confidence 기준 |
| `show_windows` | `false` | OpenCV 창 표시 |
| `publish_debug_image` | `false` | `/tl/debug_image` 발행 및 생성 허용 |
| `debug_image_max_side_px` | `640` | 디버그 최대 변 길이(px); `0`은 ROI 원래 크기 |
| `debug_publish_period_ms` | `200.0` | 디버그 최소 발행 간격(ms); `0`은 매 처리 프레임 |

### 검출 ROI·시간 기준

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `detect_top_ratio` | `0.0` | 전체 영상 기준 검출 영역 위 경계 |
| `detect_bottom_ratio` | `0.3333333333333333` | 아래 경계 |
| `detect_left_ratio` | `0.20` | 왼쪽 경계 |
| `detect_right_ratio` | `0.80` | 오른쪽 경계 |
| `max_image_age_ms` | `500.0` | ROS 현재 시각 대비 영상 stamp의 최대 나이(ms) |
| `future_stamp_tolerance_ms` | `50.0` | 미래 stamp 허용 오차(ms) |
| `state_confirm_ms` | `200.0` | 상태 전환 확인 시간(ms) |
| `state_max_gap_ms` | `250.0` | 상태 확인 과정에서 허용하는 관측 공백(ms) |
| `uncertain_hold_ms` | `300.0` | 후보 색상이 불확실할 때 직전 확정 상태 유지(ms) |

### 색상 분석

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `enable_low_confidence_color_fallback` | `true` | 호환성 인자; `false`여도 색 분석은 수행 |
| `fallback_score_threshold` | `0.45` | 일반 색상 정규화 점수 하한 |
| `fallback_green_score_threshold` | `0.40` | 초록 정규화 점수 하한 |
| `fallback_green_h_min` | `39.0` | 초록 hue 하한 |
| `fallback_green_h_max` | `100.0` | 초록 hue 상한 |
| `fallback_green_s_min` | `50` | 초록 saturation 하한 |
| `fallback_green_v_min` | `68` | 초록 value 하한 |
| `fallback_green_top_weight` | `0.20` | 후보 박스 상단 1/3 색상 점수 가중치 |
| `fallback_green_middle_weight` | `1.30` | 중앙 1/3 가중치 |
| `fallback_green_bottom_weight` | `0.20` | 하단 1/3 가중치 |
| `fallback_saturation_gain` | `2.20` | 색 분석 전 채도 보정 배율 |
| `fallback_value_gain` | `1.35` | 밝기 보정 배율 |
| `fallback_gamma` | `1.00` | gamma 보정 값 |
| `fallback_max_side_px` | `640` | 색 분석 ROI 최대 변 길이(px) |

이름은 `fallback_green_*_weight`지만 현재는 빨강·노랑·초록 점수 모두에 적용된다.
상·중·하 기준은 확장 ROI가 아닌 원본 YOLO 후보 박스다. 가중치는 점수에만 적용하며
유효 픽셀 수와 연결요소 크기 조건을 대체하지 않는다.
HSV 기준은 OpenCV 범위(H: 0~179, S/V: 0~255)이며 색상 보정 이후에 평가한다.

## black_box 런치 옵션 전체

아래 표는 `black_box_color_bits.launch.py`의 옵션이다.

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `image_topic` | `/panorama/image_raw` | 입력 영상 |
| `bits_topic` | `/tl/box_color_bits` | 비트 배열 출력 |
| `debug_image_topic` | `/tl/box_color_bits/debug` | 디버그 영상 출력 |
| `publish_debug_image` | `false` | 디버그 사용; 구독자도 있어야 발행 |
| `use_sim_time` | `false` | `/clock` 사용 |
| `max_fps` | `5.0` | 처리율 상한 |
| `max_image_age_ms` | `500.0` | 영상 stamp의 최대 나이(ms); 양수 필요 |
| `input_timeout_s` | `0.5` | 로컬 수신 시각 기준 입력 유효시간(초) |
| `box_confidence` | `0.25` | YOLO 박스 confidence 하한 |
| `box_image_size` | `640` | YOLO 추론 크기 |
| `roi_top_ratio` | `0.0` | 전체 영상 기준 ROI 위 경계 |
| `roi_bottom_ratio` | `0.5` | 아래 경계 |
| `roi_left_ratio` | `0.25` | 왼쪽 경계 |
| `roi_right_ratio` | `0.75` | 오른쪽 경계 |
| `color_s_min` | `80` | HSV saturation 하한 |
| `color_v_min` | `45` | HSV value 하한 |
| `color_score_threshold` | `0.04` | 확정 색의 EMA 점수 하한 |
| `color_hysteresis_delta` | `0.08` | 비트 초기화/전환에 필요한 빨강·초록 점수 차이 |
| `hold_timeout_s` | `0.5` | 마지막 박스 관측 이후 트랙 유지 시간(초) |

추가로 `ros2 run --ros-args -p`에서 지정할 수 있는 주요 노드 파라미터:

| 파라미터 | 노드 기본값 | 설명 |
| --- | --- | --- |
| `box_model_path` | 탐색된 패키지의 `model/box_best.pt` | 검출 모델 경로 |
| `box_device` | `auto` | 직접 실행 시 장치; 런치는 `cuda:0`으로 지정 |
| `opencv_threads` | `1` | OpenCV 스레드 수 |
| `inner_margin_ratio` | `0.20` | 박스 각 변에서 제외하는 여백 비율 |
| `red_hue_high` | `10` | 빨강 hue 하단 구간의 상한 |
| `red_hue_low_wrap` | `170` | 빨강 hue 상단 구간의 하한 |
| `green_hue_low` | `35` | 초록 hue 하한 |
| `green_hue_high` | `95` | 초록 hue 상한 |
| `color_ema_alpha` | `0.45` | 새 관측 색상 점수의 EMA 반영 비율 |
| `match_distance_ratio` | `2.5` | 트랙 중심 거리 매칭 기준 배율 |

black_box는 YOLO 클래스 이름을 비트로 바꾸지 않는다. `green_sign`/`red_sign`
클래스로 위치를 찾고, 박스 중앙의 HSV 색상 점수로 비트를 결정한다.

## 출력 의미와 확인 명령

| 토픽 | 메시지 형식 | QoS Reliability | 의미 |
| --- | --- | --- | --- |
| `/tl/state_id` | `std_msgs/msg/Int32` | Reliable | 최종 신호등 상태 |
| `/tl/detections` | `vision_msgs/msg/Detection2DArray` | Reliable | 검출 결과 |
| `/tl/debug_image` | `sensor_msgs/msg/Image` | Reliable | tl_fusion 디버그 |
| `/tl/box_color_bits` | `std_msgs/msg/UInt8MultiArray` | Reliable | 좌→우 비트 배열 |
| `/tl/box_color_bits/debug` | `sensor_msgs/msg/Image` | Best Effort | black_box 디버그 |

`/tl/state_id`: `0=UNKNOWN`, `1=RED`, `2=YELLOW`, `3=GREEN`, `4=LEFT ARROW`.

`/tl/box_color_bits`: `0=빨강`, `1=초록`. 예를 들어 빨강·초록·빨강은
`data: [0, 1, 0]`이다. 출력은 가변 길이이며 3개를 강제로 채우지 않는다.
미확정 박스는 제외하고, 이미 확정된 트랙은 `hold_timeout_s` 이내 유지될 수 있다.
배열 인덱스는 고정 위치나 영구 ID가 아니며, 유효 트랙이 없으면 `data: []`다.

```bash
ros2 node list
ros2 topic hz /panorama/image_raw
ros2 topic echo /tl/state_id
ros2 topic echo /tl/box_color_bits
```

`hz`와 `echo`는 계속 실행되므로 각각 별도 터미널에서 확인하거나 `Ctrl+C` 후
다음 명령을 실행한다. 디버그 빈도 확인 자체도 구독자를 추가한다.

```bash
ros2 topic hz /tl/debug_image
ros2 topic hz /tl/box_color_bits/debug
ros2 topic info -v /tl/box_color_bits/debug
```

두 노드는 기본적으로 `500ms`보다 오래된 영상을 거부한다. tl_fusion은 잘못된 입력에
UNKNOWN과 빈 detections를, black_box는 빈 비트 배열을 발행한다.
입력이 완전히 끊기면 tl_fusion은 `input_timeout_s` 이후 UNKNOWN을 발행한다.
black_box는 입력 타임아웃 시 내부 트랙을 비우지만 새 영상이 없는 동안 빈 배열을
주기적으로 발행하지는 않는다. 수신 측에서 마지막 메시지가 계속 유효하다고 해석하지 않도록 한다.

## 옵션 확인과 종료

노드를 시작하지 않고 런치가 지원하는 옵션을 확인한다.

```bash
ros2 launch mando_tools tl_fusion.launch.py --show-args
ros2 launch mando_tools black_box_color_bits.launch.py --show-args
```

실행 중인 설정을 확인한다.

```bash
ros2 param get /tl_fusion detector_image_size
ros2 param get /tl_fusion debug_publish_period_ms
ros2 param get /black_box_color_bits max_fps
ros2 param get /black_box_color_bits publish_debug_image
```

각 런치 터미널에서 `Ctrl+C`로 종료한다.
기존 `run_traffic_light.sh`와 `traffic_light.launch.py`는 tl_fusion만 실행하며
black_box를 함께 시작하지 않는다. 그 런치는 개별 런치와 노출 옵션이 다르므로
이 문서의 전체 옵션을 사용하려면 `tl_fusion.launch.py`로 실행한다.
