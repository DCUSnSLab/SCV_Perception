# UGV Online Self-Supervised Traversability

ROS 2 Humble 기반 실제 UGV용 **positive-only delayed self-supervision label generator**이다. 사람이 원격 조종한 UGV의 실제 통과 면적을 강한 `TRAVERSABLE` 근거로 사용해, 별도 수동 라벨링 없이 과거 카메라 프레임의 주행 가능 픽셀을 생성한다. 현재 MVP에는 딥러닝 모델이 없다.

## 핵심 원칙

UGV가 footprint 전체로 실제 통과한 지형은 해당 차량에 대해 주행 가능했다는 강한 positive evidence다. 반대로 지나가지 않은 지형에는 아무런 실패 증거가 없으므로 `NON_TRAVERSABLE`이 아니라 반드시 `UNKNOWN`으로 남긴다.

현재 encoding은 `UNKNOWN=0`, `TRAVERSABLE=1`, `NON_TRAVERSABLE=2`이지만 MVP가 실제 생성하는 값은 0과 1뿐이다. 값 2는 향후 LiDAR geometry validator가 제공할 명시적 negative evidence를 위해 예약되어 있다.

현재 카메라가 보는 전방 지형과 현재 footprint는 다르므로 같은 시각의 둘을 결합하지 않는다. 카메라 관측, depth, intrinsic, 촬영 시점 camera/robot pose를 제한된 buffer에 저장하고, 이후 UGV가 그 world 위치를 통과했을 때 future footprint를 과거 카메라 좌표로 되돌려 투영한다.

```text
RGB + Depth + CameraInfo + TF(world->camera) -> ObservationBuffer
                                                     |
Odometry/TF -> trajectory -> future footprint -------+-> projection
                                                         -> depth visibility
                                                         -> positive mask/dataset
positive mask + matching buffered RGB ------------------> debug overlay
```

## 구성과 ROS graph

- `trajectory_recorder`: odometry 또는 TF pose를 시간 순으로 기록하고 `/traversability/trajectory` (`nav_msgs/Path`) 발행
- `footprint_generator`: 차량 width/length/margin을 반영한 최신 polygon과 누적 면적 MarkerArray 발행
- `traversability_labeler`: timestamped observation buffer, future traversal association, world-to-camera projection, depth occlusion check, mask 및 dataset 생성
- `label_visualizer`: 동일 timestamp의 buffered RGB와 mask를 결합한 반투명 green overlay 발행
- `foundation_model_adapter.py`, `lidar_geometry_validator.py`: 후속 기능을 위한 의존성 없는 추상 인터페이스

주요 출력은 다음과 같다.

| Topic | Type | 용도 |
|---|---|---|
| `/traversability/trajectory` | `nav_msgs/Path` | 전체 주행 궤적 |
| `/traversability/footprint` | `geometry_msgs/PolygonStamped` | 최신 footprint |
| `/traversability/traversed_area` | `visualization_msgs/MarkerArray` | 누적 통과 면적 |
| `/traversability/positive_mask` | `sensor_msgs/Image`, `mono8` | 0/1 label mask |
| `/traversability/labeled_image` | `sensor_msgs/Image`, `bgr8` | mask와 timestamp가 맞는 과거 RGB |
| `/traversability/debug_image` | `sensor_msgs/Image`, `bgr8` | RGB + positive overlay |

## 빌드와 실행

ROS 2 Humble이 설치된 shell에서 다음을 실행한다.

```bash
cd ~/traversability_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash
ros2 launch ugv_self_supervised_traversability traversability_labeling.launch.py
```

별도 설정 파일은 `config_file` launch argument로 전달할 수 있다.

```bash
ros2 launch ugv_self_supervised_traversability traversability_labeling.launch.py \
  config_file:=/absolute/path/to/robot_traversability.yaml
```

## 파라미터

기본값은 `src/ugv_self_supervised_traversability/config/traversability.yaml`에 있다. 모든 sensor topic과 `world_frame`, `base_frame`, `camera_frame`을 변경할 수 있다. `robot_width`, `robot_length`, `footprint_margin`은 실제 차량 외곽을 기준으로 측정해야 한다. `observation_buffer_seconds`와 `observation_buffer_max_frames`가 메모리 및 delayed confirmation horizon을 제한한다.

`camera_frame`은 pinhole 식의 전제인 optical convention(+X right, +Y down, +Z forward)을 따르는 frame이어야 한다. URDF가 `camera_link`만 제공한다면 올바른 optical child frame TF를 먼저 구성한다. 외부 파라미터는 코드에 들어 있지 않으며 촬영 timestamp의 TF에서 얻는다. odometry header frame은 `world_frame`과 같아야 한다. trajectory 표시만 TF로 얻으려면 `pose_source: tf`를 쓸 수 있지만 labeler용 odometry는 현재 world frame으로 입력되어야 한다.

`depth_check_enabled: true`이면 같은 시각(기본 허용차 0.08초)의 `16UC1`(mm) 또는 `32FC1`(m) depth가 반드시 있어야 positive로 인정된다. `depth_tolerance`는 meter 단위다. RGB와 정렬되지 않은 depth를 사용하면 안 된다. 정렬 depth가 없는 카메라는 검증 목적으로만 `depth_check_enabled: false`로 설정할 수 있다.

`save_dataset: true`이면 buffer에서 확정되어 나가는 positive frame을 `dataset_root` 아래에 저장한다. 파일명은 충돌 방지를 위해 image timestamp nanoseconds를 사용한다.

```text
traversability_dataset/
├── images/<timestamp_ns>.png
├── depth/<timestamp_ns>.png       # uint16 millimeter, depth가 있을 때
├── labels/<timestamp_ns>.png      # uint8 0/1
└── metadata/<timestamp_ns>.json
```

metadata에는 image timestamp, robot pose, world camera pose matrix, positive evidence를 만든 trajectory timestamp 목록, label 통계와 encoding이 기록된다.

## RViz2 확인

Fixed Frame을 YAML의 `world_frame`과 같게 설정한 뒤 다음 display를 추가한다.

1. Path: `/traversability/trajectory`
2. Polygon: `/traversability/footprint`
3. MarkerArray: `/traversability/traversed_area`
4. Image: 원본 camera topic
5. Image: `/traversability/positive_mask`
6. Image: `/traversability/debug_image`

mask/debug image는 future traversal confirmation이 생긴 과거 timestamp 프레임이므로 live 원본 영상보다 늦게 발행되는 것이 정상이다. `rqt_image_view /traversability/debug_image`로도 확인할 수 있다.

## rosbag 검증

bag에는 RGB, RGB-aligned depth, CameraInfo, odometry, `/tf`, `/tf_static`이 포함되어야 한다. 먼저 `ros2 bag info <bag>`로 topic과 timestamp 범위를 확인하고 YAML을 bag topic/frame에 맞춘다.

```bash
# terminal 1
source ~/traversability_ws/install/setup.bash
ros2 launch ugv_self_supervised_traversability traversability_labeling.launch.py

# terminal 2 (use_sim_time을 쓰지 않는 기본 구성은 recorded header timestamp와 TF가 일치해야 함)
ros2 bag play <bag_directory>
```

재생 중 TF lookup failure가 반복되면 world/camera frame, `/tf_static` 포함 여부, bag의 timestamp를 점검한다. 빠른 재생 전에 1.0 배속으로 delayed mask와 trajectory를 확인한다.

## 테스트와 단계별 검증

```bash
cd ~/traversability_ws
/usr/bin/python3 -m pytest -q src/ugv_self_supervised_traversability/test
colcon test --packages-select ugv_self_supervised_traversability
colcon test-result --verbose
```

unit test는 footprint 크기/회전, world-camera transform, pinhole projection, image boundary/behind-camera rejection, depth occlusion, buffer eviction을 센서 없이 검증한다. 실제 차량에서는 trajectory -> footprint -> buffer/TF -> delayed projection -> depth filtering -> mask -> dataset 순서로 하나씩 확인한다.

## 현재 범위와 확장 계획

MVP는 실제 통과 footprint의 positive seed만 생성한다. SAM/Foundation Model 영역 확장, VLP-32C slope/height/roughness/obstacle 기반 negative evidence, lightweight RGB/RGB-D student inference 및 online training, Nav2 costmap 연결은 구현하지 않았다. 다음 단계에서는 placeholder adapter 뒤에 이들을 연결하되, model 추론과 geometry evidence의 provenance를 metadata에 분리하고 unknown을 근거 없이 negative로 바꾸지 않아야 한다.
