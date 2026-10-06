# UGV Online Self-Supervised Traversability

ROS 2 Humble 기반 실제 UGV용 **positive-only delayed self-supervision label generator**이다. 사람이 원격 조종한 UGV의 실제 통과 면적을 강한 `TRAVERSABLE` 근거로 사용해, 별도 수동 라벨링 없이 과거 카메라 프레임의 주행 가능 픽셀을 생성한다. 현재 MVP에는 딥러닝 모델이 없다.

2026-09 업데이트로 `odom` 단독 pose 외에 IMU와 LiDAR 포인트클라우드를 함께 사용해 footprint의 자세와 높이를 보정한다. 기본 동작은 `x/y/yaw`는 odometry, `roll/pitch`는 IMU, `z`는 로봇 주변 LiDAR ground percentile에서 추정한다.

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

- `trajectory_recorder`: odometry 또는 TF pose를 시간 순으로 기록하고 `/traversability/trajectory` (`nav_msgs/Path`) 발행. 설정 시 IMU/LiDAR 보정 pose를 반영
- `footprint_generator`: 차량 width/length/margin을 반영한 최신 polygon과 누적 면적 MarkerArray 발행. IMU roll/pitch와 LiDAR ground z를 사용해 3D footprint를 생성
- `traversability_labeler`: timestamped observation buffer, future traversal association, world-to-camera projection, depth occlusion check, mask 및 dataset 생성. future footprint는 보정된 IMU/LiDAR pose를 사용
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

IMU/LiDAR pose fusion 관련 파라미터도 추가됐다.

- `use_imu_orientation`: IMU orientation으로 roll/pitch를 대체한다. 기본값 `true`
- `imu_use_yaw`: magnetometer 또는 외부 yaw 보정이 있을 때만 `true`를 권장한다. 기본값 `false`
- `imu_timeout_seconds`: odometry와 IMU timestamp 허용 차이
- `use_lidar_ground_height`: footprint z를 LiDAR 주변 ground height로 대체
- `pointcloud_timeout_seconds`: odometry와 point cloud timestamp 허용 차이
- `lidar_ground_radius`: 로봇 중심 주변에서 ground z를 추정할 반경
- `lidar_ground_min_points`: ground z 추정 최소 포인트 수
- `lidar_ground_percentile`: 주변 포인트 z 분포에서 사용할 percentile. 기본값 `20.0`

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

bag에는 RGB, RGB-aligned depth, CameraInfo, odometry, `/tf`, `/tf_static`이 포함되어야 한다. IMU/LiDAR 보정을 쓰려면 IMU와 point cloud도 함께 포함되어야 한다. 먼저 `ros2 bag info <bag>`로 topic과 timestamp 범위를 확인하고 YAML을 bag topic/frame에 맞춘다.

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

MVP는 실제 통과 footprint의 positive seed만 생성한다. SAM/Foundation Model 영역 확장, LiDAR 판정의 카메라 negative-label 연결, lightweight RGB/RGB-D student inference 및 online training, Nav2 costmap 연결은 구현하지 않았다. 독립적인 LiDAR 3D 지형 판정은 아래 2026-09 추가 기능을 참고한다. 다음 단계에서는 placeholder adapter 뒤에 이들을 연결하되, model 추론과 geometry evidence의 provenance를 metadata에 분리하고 unknown을 근거 없이 negative로 바꾸지 않아야 한다.

## 2026-09: IMU / LiDAR 방향별 3D 지형 판정

`terrain_analyzer`를 추가했다. 카메라 라벨링과 독립적으로 LiDAR의 **현재 스캔**을 heading-aligned 높이 격자로 만들고, 차량 전방/후방 각각의 경사·횡경사·단차·낙차·거칠기를 평가한다. 새 지형 판정은 ROS status/3D marker로 출력하며, 카메라의 `NON_TRAVERSABLE` 라벨이나 Nav2 costmap에는 아직 연결하지 않았다.

### 실행

```bash
cd /home/scv/SCV/src/perception/traversability_ws
colcon build --symlink-install
source install/setup.bash
# 카메라 없이 지형 분석만 실행
ros2 launch ugv_self_supervised_traversability terrain_analysis.launch.py use_sim_time:=true
# 별도 터미널, 같은 ROS 환경
ros2 bag play rosbag2_2026_08_13-13_55_28 --clock
ros2 topic echo /traversability/terrain_status
```

실차 입력은 `use_sim_time:=false`로 실행한다. 기존 `traversability_labeling.launch.py`에도 새 노드를 포함했다. 기본 처리율은 2 Hz이고 scan 사이에는 누적하지 않는다. `/traversability/terrain_markers`를 RViz MarkerArray로 추가하고 Fixed Frame을 `odom`으로 지정한다. `forward`, `reverse` namespace를 선택해 볼 수 있으며 두 층은 표시상 z를 6 cm 분리했다. 초록은 `GEOMETRY_PASS`, 빨강은 `GEOMETRY_BLOCKED`, 회색은 `UNKNOWN`이다. 관측 자체가 없는 셀은 3D marker를 만들지 않는다.

`/traversability/terrain_status` (`std_msgs/String`, JSON)는 다음을 포함한다.

- `motion`: `ASCENDING / DESCENDING / LEVEL / STATIONARY / UNKNOWN`, 근거 source와 추정 경사.
- `roll_deg`, `pitch_deg`, `orientation_source`, `imu_fresh`: 차량 자세와 실제 IMU 사용 여부.
- `forward`, `reverse`: 차량 폭+margin 전체와 bumper부터 lookahead까지의 판정, 관측/분류 비율, 최대 경사·단차·낙차, 제한 초과 또는 미확인 이유별 셀 수.
- `limits`: 사용한 허용치와 `limits_calibrated`. 개별 셀 통과가 전체 corridor 통과를 뜻하지 않는다. corridor에 제한 초과 셀이 하나라도 있으면 BLOCKED, 전 셀이 통과해야 PASS, 나머지는 UNKNOWN이다.

### 구현 방법과 가정

- 기본 격자는 0.4 m, 차량 주변 ±6 m이고 셀당 최소 3점, 3×3 이웃 중 최소 6개의 관측 셀로 평면을 피팅한다. 각 셀 높이는 20번째 백분위수이다. 피팅한 진행축/횡축 gradient로 경사를 구하고 실제 포인트의 평면 잔차 95번째 백분위수로 거칠기/장애물 근거를 구한다. 이는 단순 하부 표면 추정이며 완전한 지면 분리 알고리즘은 아니다.
- 앞/뒤 인접 관측 셀의 높이 차이로 step/drop을 검사한다. 양쪽 경사가 일치할 때만 연속 경사분을 빼며, 불연속이나 복잡한 표면은 보수적으로 차단될 수 있다. 공간 해상도보다 작은 장애물이나 드문 포인트는 놓칠 수 있다.
- 무반사/가림/희박한 포인트는 UNKNOWN이다. 보이지 않는 곳을 낭떠러지로 단정하지 않는다. 여러 높이층, 물성·마찰·침하·미끄러짐, 바퀴 접촉/차체 간섭은 모델링하지 않는다.
- IMU quaternion의 유효성과 orientation covariance를 검사하고 촬영 시각의 `base_link <- imu_frame` 장착 TF를 보정한다. IMU는 중력 기준 orientation을 제공해야 한다. yaw는 odometry를 유지한다. 장착 TF 누락/오래된 IMU에는 IMU 보정을 적용하지 않는다.
- 상승/하강은 신선한 IMU의 차량 지지 평면과 약 1초 동안의 실제 odometry 이동 방향으로 추정한다. 후진 시 부호를 반전한다. 차체 기울기가 지면을 따른다는 가정이 있으며 실제 고도 변화의 직접 측정은 아니다. `odometry_z_valid: true`이면 IMU가 없을 때 검증된 3D odometry의 높이 변화로 대체한다. 기본 false에서는 IMU가 없을 때 motion은 UNKNOWN이다.
- 기본 `use_imu_cloud_leveling: true`에서는 신선한 IMU가 있으면 `base_link <- LiDAR` TF로 점군을 차량 좌표로 변환한 뒤, IMU roll/pitch + odometry yaw/translation으로 **스캔별 중력 정렬**한다. 따라서 odometry TF가 평면이어도 국소 경사를 계산할 수 있다. IMU가 없으면 recorded world TF를 사용하며 `cloud_orientation_source`에 대체 경로를 명시한다. 이 경로는 중력 정렬된 TF가 있어야 정확하다.
- 새 노드는 3D localization을 구현하지 않는다. odometry z가 0이면 지도 높이는 각 스캔의 차량 기준 높이이며 주행 전체의 절대 고도 누적이 아니다. IMU로 보정한 점군은 평면 TF만 사용한 원본 RViz point cloud와 위치가 달라질 수 있다.
- 센서 timestamp를 사용해 odometry와 0.25초 이내로 매칭하고, TF 또는 pose가 없으면 UNKNOWN을 출력한다. cloud가 끊기면 기존 marker를 삭제하고 UNKNOWN으로 바꾼다. 역방향 시간 점프 시 pose 버퍼를 초기화한다.

초기 한계값(등판 15°, 하강 12°, 횡경사 10°, 단차 0.15 m, 낙차 0.12 m, 거칠기 0.08 m)은 **검증용 예시**다. 실차 성능을 의미하지 않으며 `limits_calibrated: false`가 기본이다. 실제 측정한 값으로 설정해야 한다. `GEOMETRY_PASS`도 물리적인 주행 성공 보증이 아니다.

### 실제 bag 오프라인 검증과 3D 화면

```bash
# ROS Python + numpy + plotly 환경. DDS 재생 없이 recorded TF/point cloud를 직접 읽는다.
/usr/bin/python3 scripts/bag_terrain_review.py \
  rosbag2_2026_08_13-13_55_28/rosbag2_2026_08_13-13_55_28_0.db3 \
  --interval 10 --output test_results/terrain_3d
```

`terrain_replay_3d.html`은 외부 CDN 없이 브라우저에서 열 수 있으며 회전/확대/시간 슬라이더/재생을 지원한다. `report.json`은 각 스캔의 판정과 이유, 처리시간을 기록한다. 전체 2,515개 LiDAR 스캔 중 10초 간격으로 26개를 분석했다. 이 bag은 IMU가 없고 **모든 odometry z가 0**이어서 IMU 기반 상승/하강 실측 검증을 대신하지 못한다. 새 geometry 코드를 실행한 결과이며 이전에 녹화된 footprint를 그대로 표시한 것과 다르다.

단위 테스트는 기존 기하/투영/버퍼 외에 평지, 방향별 등판·하강, yaw 회전, 횡경사, 단차/낙차, 거칠기, 관측 누락, IMU 장착 보정/유효성/시간 만료, 후진 시 상승·하강 반전을 포함한다. `scripts/smoke_terrain_ros.py`는 별도 `ROS_DOMAIN_ID`에서 합성 센서를 실제 DDS로 발행하여 14° 경사 판정과 IMU 기반 상승 상태, 센서 중단 시 UNKNOWN을 검사한다.

### RViz2에서 bag과 새 지형 판정을 함께 확인

```bash
bash scripts/view_bag_rviz.sh
# 선택: 재생 속도 / 다른 bag / 별도 ROS domain
REVIEW_RATE=1.0 REVIEW_ROS_DOMAIN_ID=170 bash scripts/view_bag_rviz.sh /path/to/bag
```

이 스크립트는 ROS Humble 환경에서 소스의 terrain node와 RViz2, bag player를 실행한다. 기본값은 localhost ROS domain 170, 0.5배속 반복 재생, simulated time이며 RViz 창을 닫으면 이 스크립트가 시작한 player와 analyzer도 종료한다. 로그는 시작 시 출력되는 `/tmp/traversability-rviz-*`에 남는다. RViz 설정은 `config/terrain_review.rviz`이다. `LIVE terrain ...` display의 Namespaces에서 `forward`와 `reverse`를 전환할 수 있다. bag에 IMU가 없어 `Motion: UNKNOWN` 표시는 정상이다.
