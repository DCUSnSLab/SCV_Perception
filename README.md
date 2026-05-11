# pcdet_ros2_ws

ROS 2 Humble 기반 LiDAR perception workspace입니다.  
현재 워크스페이스는 다음 파이프라인을 기준으로 구성되어 있습니다.

```text
LiDAR -> ground filter -> 3D detector -> 3D tracker -> behavior predictor
```

## 구성

주요 디렉터리:

- `src/`: ROS 2 패키지 소스
- `third_party/OpenPCDet/`: OpenPCDet 코드
- `models/`: detector에서 사용하는 모델 파일
- `scripts/`: 보조 실행 스크립트

주요 패키지:

- `pcd_ground_filter`
  - 바닥 제거
- `pv_rcnn_kitti_detector`
  - KITTI용 PV-RCNN detector
- `pointpillars_coda_detector`
  - CODa용 PointPillars detector
- `pcdet_tracker`
  - tracking
- `behavior_predictor`
  - track history 기반 기본 행동예측
- `tracking_msgs`
  - detector / tracker 메시지 정의

## 모델 경로

모델 파일은 아래 경로에 두고 사용합니다.

- CODa detector model:
  - `~/pcdet_ros2_ws/models/pointpillars_coda.pth`
- KITTI detector model:
  - `~/pcdet_ros2_ws/models/pv-rcnn_kitti.pth`

현재 설정 파일 경로:

- CODa config:
  - `~/pcdet_ros2_ws/src/pointpillars_coda_detector/config/coda_pointpillar_vehicle_ped.yaml`
- KITTI config:
  - `~/pcdet_ros2_ws/src/pv_rcnn_kitti_detector/config/pv_rcnn_my_ver.yaml`

## 모델 다운로드

모델 파일은 직접 받아서 아래 경로에 배치하면 됩니다.

```text
~/pcdet_ros2_ws/models/pointpillars_coda.pth
~/pcdet_ros2_ws/models/pv-rcnn_kitti.pth
```

다운로드 방법과 배포 링크는 나중에 여기에 추가하면 됩니다.

## 빌드

```bash
source /opt/ros/humble/setup.bash
cd ~/pcdet_ros2_ws
colcon build
source install/setup.bash
```

새 터미널마다:

```bash
source /opt/ros/humble/setup.bash
cd ~/pcdet_ros2_ws
source install/setup.bash
```

## 실행

### 1. CODa detector + tracker + behavior predictor

```bash
ros2 launch pointpillars_coda_detector coda_pointpillar.launch.py
```

### 2. KITTI detector pipeline

```bash
ros2 launch pv_rcnn_kitti_detector pcdet_pipeline.launch.py
```

### 3. 개별 실행 예시

Ground filter:

```bash
ros2 run pcd_ground_filter ground_removal_node
```

KITTI detector:

```bash
ros2 run pv_rcnn_kitti_detector kitti_detector_node
```

CODa detector:

```bash
ros2 run pointpillars_coda_detector coda_pointpillar_node
```

Tracker:

```bash
ros2 run pcdet_tracker jay_tracker
```

Behavior predictor:

```bash
ros2 run behavior_predictor behavior_predictor_node -- \
  --tracked_topic /tracked_objects_3d \
  --prediction_marker_topic /behavior/prediction_markers
```

## 주요 토픽

- `/no_ground_points`
- `/detected_objects_3d`
- `/tracked_objects_3d`
- `/pcdet/coda_tracks`
- `/behavior/prediction_markers`

## 자주 쓰는 확인 명령

```bash
ros2 topic list
ros2 node list
ros2 topic hz /detected_objects_3d
ros2 topic hz /tracked_objects_3d
ros2 topic echo /behavior/prediction_markers
```

## 비고

- `build/`, `install/`, `log/`, `results/`는 실행 생성물입니다.
- `third_party/OpenPCDet`는 detector 실행에 필요합니다.
- 추가 설치 방법, 데이터셋 경로, 모델 다운로드 링크, 성능 결과는 이후 여기에 보강하면 됩니다.
