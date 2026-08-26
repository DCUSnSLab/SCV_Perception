# SCV Perception

SSC 자율주행 스택에서 사용하는 ROS 2 기반 인지 패키지 모음입니다. 차선, 신호등, 멀티카메라 파노라마 영상을 처리합니다.

## Packages

| Package | Description |
| --- | --- |
| `lane_detection` | YOLOP 기반 차선 인식 노드 |
| `panorama_stitcher` | 전방 RealSense 카메라의 실시간 RGB-D 파노라마 및 점군 생성 |
| `traffic_light_detection` | 파노라마 영상 기반 신호등 검출 노드 |
| `traffic_light/mando_tools` | YOLO 및 ROI 색 분석을 결합한 독립형 신호등 상태 판정 도구 |

## Requirements

- Ubuntu 22.04
- ROS 2 Humble
- Python 3, OpenCV, NumPy
- CUDA 지원 GPU 권장. 신호등 도구는 CPU 실행도 지원합니다.
- `traffic_light/mando_tools` 사용 시: PyTorch 및 [Ultralytics YOLO](https://docs.ultralytics.com/)

## Quick Start: Traffic Light Tools

신호등 도구는 기록된 rosbag 또는 실시간 카메라 이미지를 공통 입력 토픽 `/mando/input/image`로 받아 처리합니다.

### 1. Prepare assets

모델 가중치와 rosbag은 용량 문제로 Git에 포함하지 않습니다. 아래 위치에 준비하거나, 실행 시 절대 경로를 인자로 전달하세요.

```text
traffic_light/
├── models/
│   └── best.pt
└── data/
    └── bags/
        └── stop_points/   # rosbag2 directory
```

```bash
python3 -m pip install ultralytics torch
```

### 2. Build

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select mando_tools
source install/setup.bash
```

### 3. Play a bag

별도 터미널에서 rosbag을 재생합니다. 기본 프로필은 `traffic_light/data/bags/stop_points`입니다.

```bash
ros2 launch mando_tools play_mando_bag.launch.py
```

다른 rosbag2 디렉터리를 사용할 경우:

```bash
ros2 launch mando_tools play_mando_bag.launch.py \
  bag_path:=/absolute/path/to/rosbag2_directory
```

### 4. Run a detector

두 번째 터미널에서 아래 중 하나를 실행합니다.

```bash
# YOLO 결과만 빠르게 검증
ros2 launch mando_tools validate_mando_bag.launch.py device:=cpu

# YOLO와 ROI 색 분석을 결합한 최종 상태 판정
ros2 launch mando_tools tl_fusion.launch.py detector_device:=cpu
```

GPU를 사용할 수 있으면 `device:=cuda:0` 또는 `detector_device:=cuda:0`으로 변경할 수 있습니다.

## Traffic Light Outputs

`tl_fusion`은 다음 상태를 발행합니다.

| Topic | Type | Description |
| --- | --- | --- |
| `/tl/state` | `std_msgs/Int32` | 상태 ID |
| `/tl/state_label` | `std_msgs/String` | 사람이 읽을 수 있는 상태 이름 |
| `/tl/state_reason` | `std_msgs/String` | 판정 근거 |
| `/tl/debug_image` | `sensor_msgs/Image` | 검출 및 분석 디버그 이미지 |

| ID | State |
| --- | --- |
| `0` | `UNKNOWN` |
| `1` | `RED` |
| `2` | `YELLOW` |
| `3` | `GREEN` |
| `4` | `LEFT ARROW` |

상태를 확인하려면 다음 명령을 사용합니다.

```bash
ros2 topic echo /tl/state_label
```

## Documentation

- [Traffic light usage guide](traffic_light/README)
- [Traffic light implementation guide](traffic_light/mando_tools/traffic_light_detection_code_guide.md)
- [Panorama stitcher guide](panorama_stitcher/README.md)

## License

Each package is distributed under the Apache-2.0 license unless stated otherwise.
