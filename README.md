# SCV Perception

ROS 2 Humble 기반의 인지 패키지 모음이다. LiDAR와 RGB-D 카메라를
이용한 객체 검출·추적, 지면 제거, 주행 가능 영역 라벨 생성을 포함한다.

## 전체 구성

```text
Velodyne PointCloud ──> pointpillars_coda_detector ──> 3D detections ──┐
        │                                                              │
        └────────────> ground_filter                                   ├─> observation_tracker
                                                                       │       └─> 3D tracks
D555 RGB ────────────> ultralytics_ros2 ─────────────> 2D detections ──┘

RGB + Depth + LiDAR + Odometry + IMU ──> ugv_self_supervised_traversability
                                             └─> traversability labels
```

Detector와 tracker는 서로 독립된 ROS 2 노드로 실행한다. 두 노드는
`tracking_msgs/DetectedObjectArray` 토픽을 통해서만 연결된다.

## 패키지 설명

### `ground_filter`

LiDAR 포인트클라우드를 극좌표 격자 기반으로 분석하여 지면과 비지면
포인트를 분리한다. 장애물 후보 추출이나 지면 제거 결과 비교에 사용한다.

- 입력: `/velodyne_points`
- 출력: `/perception/points_nonground`, `/perception/points_ground`
- 상세 설명: [ground_filter/README.md](ground_filter/README.md)

### `pointpillars_coda_detector`

CODa 데이터셋으로 학습한 PointPillars 모델을 이용해 차량과 보행자의 3D
bounding box를 검출한다. OpenPCDet와 학습 weight는 저장소 외부에 둔다.

- 입력: `/velodyne_points`
- 출력: `/detected_objects_3d`, `/detections/visual_markers`
- 기본 confidence threshold: `0.65`

### `ultralytics_ros2`

Ultralytics YOLO segmentation 모델을 ROS 2에서 실행한다. 기본 설정은
D555 압축 영상을 받아 `person`과 `car`의 2D detection을 발행한다.

- 입력: `/camera/camera/color/image_raw/compressed`
- 출력: `/yolo/detections`, `/yolo/seg_image`
- 기본 confidence threshold: `0.70`

### `observation_tracker`

3D detector 출력에 Kalman Filter와 Hungarian association을 적용해 객체
ID, 위치, 방향, 속도를 추적한다. LiDAR point support, 2D detection, depth,
odometry, IMU를 이용해 미탐지와 가림 상황에서 기존 ID를 유지한다.

- 필수 입력: `/detected_objects_3d`
- 선택 입력: LiDAR, RGB, 2D detection, aligned depth, odometry, IMU
- 출력: `/tracked_objects_3d`
- 상세 설명: [observation_tracker/README.md](observation_tracker/README.md)

### `perception_interface`

2D detection과 확장형 3D tracking 결과에 사용하는 ROS 2 message를
정의한다. 메시지 전용 패키지이므로 별도로 실행하는 노드는 없다.

- `DetectionResult`, `DetectionArray`
- `TrackedObject3D`, `TrackedObject3DArray`

### `tracking_msgs`

PointPillars detector와 Observation Tracker 사이에서 사용하는 간단한 3D
객체 message를 정의한다. 메시지 전용 패키지이므로 실행 노드는 없다.

- `DetectedObject`: ID, class, confidence, 3D pose, size, yaw
- `DetectedObjectArray`: 프레임 단위 객체 배열

### `ugv_self_supervised_traversability`

`traversability_ws` 아래에 있는 주행 가능 영역 분석 패키지다. 카메라,
depth, LiDAR, odometry, IMU와 실제 차량 통과 궤적을 이용해 traversable
영역의 positive label을 생성한다.

- 출력: trajectory, footprint, terrain 상태·marker, positive mask, debug image
- 상세 설명: [traversability_ws/README.md](traversability_ws/README.md)

## 빌드

```bash
cd ~/SCV_Perception
source /opt/ros/humble/setup.bash
rosdep install --from-paths . --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
```

작업공간 경로를 변경한 경우 기존 `build`, `install`, `log`에는 이전 절대
경로가 남을 수 있다. 이 세 디렉터리를 정리한 후 다시 빌드한다.

새 터미널을 열 때마다 다음 환경을 불러온다.

```bash
source /opt/ros/humble/setup.bash
source ~/SCV_Perception/install/setup.bash
```

## 객체 검출 및 추적 실행

### 1. 3D detector 실행

기본 경로는 `~/OpenPCDet-restored`와
`~/pcdet_ros2_ws/models/pointpillars_coda.pth`이다. 설치 위치가 다르면
launch argument로 지정한다.

```bash
ros2 launch pointpillars_coda_detector coda_pointpillar.launch.py \
  run_tracker:=false \
  score_thresh:=0.65
```

외부 파일 경로를 직접 지정하는 예시는 다음과 같다.

```bash
ros2 launch pointpillars_coda_detector coda_pointpillar.launch.py \
  openpcdet_path:=/absolute/path/to/OpenPCDet \
  cuda_ops_path:=/absolute/path/to/OpenPCDet/pcdet/ops \
  ckpt_file:=/absolute/path/to/pointpillars_coda.pth \
  run_tracker:=false
```

### 2. 2D detector 실행

Camera/depth observation을 사용하지 않는다면 이 단계는 생략할 수 있다.

```bash
ros2 launch ultralytics_ros2 seg.launch.py
```

모델이나 입력 토픽을 변경하려면 다음처럼 실행한다.

```bash
ros2 launch ultralytics_ros2 seg.launch.py \
  yolo_model:=/absolute/path/to/model.pt \
  input_topic:=/camera/camera/color/image_raw/compressed
```

### 3. Observation Tracker 실행

```bash
ros2 launch observation_tracker observation_tracker.launch.py
```

RViz 표시가 필요하면 다음 옵션을 추가한다.

```bash
ros2 launch observation_tracker observation_tracker.launch.py \
  enable_debug_visualization:=true \
  enable_fused_detection_visualization:=true \
  launch_rviz:=true
```

카메라, depth, odometry, IMU 없이 3D detection과 LiDAR만 사용할 수도 있다.

```bash
ros2 launch observation_tracker observation_tracker.launch.py \
  enable_camera_observation:=false \
  enable_depth_consistency:=false \
  enable_odometry:=false \
  enable_imu:=false
```

Ground Filter 출력을 tracker의 LiDAR evidence로 사용하려면 두 노드를
각각 실행하고 입력 토픽을 변경한다.

```bash
ros2 launch ground_filter ground_filter.launch.py use_sim_time:=false

ros2 launch observation_tracker observation_tracker.launch.py \
  lidar_points_topic:=/perception/points_nonground
```

## Traversability 실행

전체 라벨 생성 pipeline은 다음 명령으로 실행한다.

```bash
ros2 launch ugv_self_supervised_traversability \
  traversability_labeling.launch.py
```

카메라 라벨 생성 없이 LiDAR terrain 분석만 실행하려면 다음을 사용한다.

```bash
ros2 launch ugv_self_supervised_traversability \
  terrain_analysis.launch.py
```

Rosbag을 재생할 때는 필요한 노드의 `use_sim_time`을 활성화하고 별도
터미널에서 bag을 실행한다.

```bash
ros2 bag play <bag_directory> --clock
```

## 설정 파일

| 패키지 | 설정 파일 |
|---|---|
| `ground_filter` | `ground_filter/config/ground_filter.param.yaml` |
| `pointpillars_coda_detector` | `pointpillars_coda_detector/config/coda_pointpillar_vehicle_ped.yaml` |
| `observation_tracker` | `observation_tracker/config/observation_tracker.yaml` |
| `ugv_self_supervised_traversability` | `traversability_ws/src/ugv_self_supervised_traversability/config/traversability.yaml` |
