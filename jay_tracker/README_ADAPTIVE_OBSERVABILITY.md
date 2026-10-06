# Jay_Tracker Object-wise Sensor Observability Prototype

이 패키지는 기존 Jay_Tracker를 유지하면서 객체별 Camera–LiDAR 관측 상태를 계산하고, 그 상태에 따라 association과 track lifecycle을 선택적으로 조절한다. 모든 신규 정책은 YAML flag로 끌 수 있다.

## 기존 tracking pipeline

```text
tracking_msgs/DetectedObjectArray
  -> JayTrackerNode.detections_callback
  -> KittiHzTracker.update
       -> TimeAwareTrack.predict                 (Prediction)
       -> KittiHzTracker._cost                   (Cost + gating)
       -> scipy.optimize.linear_sum_assignment   (Hungarian matching)
       -> TimeAwareTrack.update                  (Matched KF update)
       -> unmatched detection -> TimeAwareTrack  (New track)
       -> unmatched track aging                  (Lifecycle/delete)
  -> tracked DetectedObjectArray + MarkerArray
```

핵심 파일은 `pcdet_tracker/pcdet_tracker/jay_tracker.py`이다.

- ROS node: `JayTrackerNode`
- tracker: `KittiHzTracker`
- track/Kalman filter: `TimeAwareTrack`
- detection representation: `[x, y, z, length, width, height, yaw, class_id, score]`
- KF state: `[x, y, vx, vy]`, measurement: `[x, y]`
- baseline cost: center distance, BEV IoU, size, anchor, yaw, class, motion direction
- matching: `linear_sum_assignment`
- confirmation: hit count/track age 조건
- deletion: `time_since_update_sec > max_age_sec`
- ego-motion: wheel odometry 선속도와 IMU yaw rate 사용
- 기존 tracker 자체에는 camera 또는 TF 처리가 없었다.

## 추가 구조와 integration point

```text
Raw LiDAR + camera detections + camera info + optional aligned depth
                       |
                       v
             ObservationAnalyzer
                       |
              ObservationState
                       |
       +---------------+----------------+
       |                                |
AdaptiveTrackPolicy._cost()    unmatched/new/delete policy
       |                                |
       +---------- KittiHzTracker ------+
```

Prediction 직후 동일 timestamp 근처의 센서 snapshot을 한 번 고정하고, detection 및 predicted track box의 관측 상태를 계산한다. 기존 cost와 lifecycle은 삭제하지 않고 optional adjustment만 적용한다.

신규 파일:

- `observation_state.py`: `ObservationState`, `ObservationClass`
- `observation_analyzer.py`: point ROI, projection, camera IoU, depth consistency
- `adaptive_track_policy.py`: state별 association/lifecycle 정책
- `adaptive_config.py`: YAML loader
- `observation_logging.py`: 논문 분석용 CSV
- `config/adaptive_tracking.yaml`: topic, threshold, ablation flag
- `launch/jay_tracker.launch.py`: tracker 실행

## 관측 상태

```text
MULTIMODAL_STRONG : Camera support + LiDAR strong
LIDAR_DOMINANT    : Camera weak/unavailable/FOV 밖 + LiDAR strong
CAMERA_DOMINANT   : Camera support + LiDAR weak/unavailable
WEAK_OBSERVATION  : 두 센서의 유효 support가 약함
```

`available`, `visible`, `supported`는 별도 값이다. Camera message/TF가 없거나 FOV 밖인 상태는 camera의 부정 증거로 사용하지 않는다. Depth가 없거나 유효 pixel이 부족한 경우도 0점이 아니라 weighting에서 제외한다.

LiDAR support는 확장된 oriented 3D box 내부 점을 NumPy로 일괄 필터링하여 `point_count`, `point_density`, `range`를 구한다. Camera support는 3D box 8개 corner를 영상에 투영하고 compatible 2D detection과 최대 IoU를 구한다. Depth score는 ROI median과 LiDAR box center의 camera-frame depth 차이에 `exp(-error/sigma)`를 적용한다.

## Adaptive policy

- `MULTIMODAL_STRONG`: 기존 3D cost에 multimodal/camera match bonus
- `LIDAR_DOMINANT`: camera penalty 없이 기존 geometry와 motion 사용
- `CAMERA_DOMINANT`: gate 완화, 최근 camera support가 있으면 deletion grace 증가
- `WEAK_OBSERVATION`: gate를 엄격하게 하고, 양 센서가 실제 관측 가능했는데 모두 약한 신규 track은 억제
- optional adaptive KF: observation score가 높으면 measurement covariance `R`을 낮추고, 낮으면 높임
- 최근 observation history와 confidence EMA를 track마다 유지

## Topics

기본값은 `config/adaptive_tracking.yaml`에서 변경한다.

| 용도 | Topic | Type |
|---|---|---|
| 3D detection | `/detected_objects_3d` | `tracking_msgs/DetectedObjectArray` |
| Raw LiDAR | `/velodyne_points` | `sensor_msgs/PointCloud2` |
| Camera detection | `/yolo/detections` | `perception_interface/DetectionArray` |
| CameraInfo | `/camera/camera/color/camera_info` | `sensor_msgs/CameraInfo` |
| Aligned depth | `/camera/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` |
| Track output | `/tracked_objects_3d` | `tracking_msgs/DetectedObjectArray` |
| RViz box/text | `/pcdet/jay_tracks` | `visualization_msgs/MarkerArray` |

TF는 3D detection frame에서 `camera_color_optical_frame`으로 조회한다. 정적 extrinsic은 성공 후 cache한다.

## Build and run

```bash
cd /home/scv/SCV
source /opt/ros/humble/setup.bash
colcon build --packages-select tracking_msgs pcdet_tracker pointpillars_coda_detector --symlink-install
source install/setup.bash

ros2 launch pointpillars_coda_detector coda_pointpillar.launch.py
```

PointPillars 기본 입력은 ground removal 결과인 `/no_ground_points`이다. Ground filter를 쓰지 않을 때는 `input_topic:=/velodyne_points`로 직접 바꿀 수 있지만, 학습 전처리와 같은 조건인지 확인해야 한다. 이 workspace에 없는 optional `behavior_predictor`는 기본 OFF이다.

Tracker만 실행:

```bash
ros2 launch pcdet_tracker jay_tracker.launch.py
```

완전한 기존 baseline 경로:

```bash
ros2 launch pcdet_tracker jay_tracker.launch.py adaptive_config:=''
```

다른 topic은 launch 또는 YAML에서 명시한다. PointPillars launch의 `openpcdet_path`, `ckpt_file`도 override할 수 있다.

## Ablation

`config/adaptive_tracking.yaml`의 flag를 사용한다.

| 실험 | 설정 |
|---|---|
| A0 | `enable_observation_analysis: false` 또는 empty config |
| A1 | LiDAR support만 활성화 |
| A2 | Camera support 추가 |
| A3 | `enable_adaptive_association: true` |
| A4 | `enable_adaptive_lifecycle: true` |
| A5 | `enable_adaptive_kf: true` |

현재 안전 기본값은 observation, LiDAR, camera, adaptive lifecycle ON이고 adaptive association, depth, adaptive KF는 OFF이다.

## Debugging and logging

RViz text marker에는 track ID, observation state, L/C/D score가 표시된다. CSV에는 timestamp, track/class/position, detector confidence, LiDAR point/score, camera availability/visibility/support/IoU, depth, observation state/score, association cost, match 여부, age/miss와 단계별 시간이 기록된다.

기본 CSV 위치는 `results/observability`이다. Tracker performance CSV에는 `observation_analysis_ms`, `association_ms`, `tracker_update_ms`, `total_ms`가 포함된다.

## 검증 상태

- unit/integration test 13개 통과
- 실제 rosbag의 `/velodyne_points`와 timestamp를 공유한 3D detection smoke test 통과
- 실제 calibration/TF와 합성 2D detection을 이용한 projection/matching smoke test 통과
- 해당 smoke test 82행 중 `MULTIMODAL_STRONG` 68행, TF 준비 전 초기 14행은 `LIDAR_DOMINANT`
- 관측 분석 중앙 처리시간 약 1.38 ms, tracker 전체의 일반적인 처리시간 약 4–5 ms

여기서 3D/2D detection은 integration 검증을 위해 합성했으며 PointPillars와 실제 camera detector의 정확도 평가는 아직 수행하지 않았다. Camera debug image overlay는 아직 없고 RViz text/box 및 CSV를 우선 구현했다.

PointPillars runtime은 CUDA 12 overlay 방식으로 복구했다. 완전한 OpenPCDet Python 소스는 `/home/scv/SCV_Perception/third_party/OpenPCDet`를 사용하고, workspace에 있던 CUDA 12 extension 6개를 `openpcdet_runtime.py`가 먼저 로드한다. Eager detector registry가 불필요한 CUDA 11 `bev_pool`을 읽지 않도록 BEVFusion만 격리했다. 현재 PyTorch와 실행 ABI가 맞지 않는 CUDA NMS는 OpenCV C++ rotated-box NMS로 대체했으며, PointPillars의 GPU convolution과 forward는 그대로 유지한다.

실제 rosbag의 `/velodyne_points`로 520프레임 연속 inference를 검증했다. 약 9.98 Hz 입력을 유지했고 전체 처리시간은 일반적으로 78–83 ms, GPU inference는 약 56–57 ms였다. 객체가 생긴 프레임에서 1–2개 detection 및 marker 발행까지 확인했고 CUDA/NMS crash 없이 종료했다. 검증은 detector에 raw point를 직접 연결했으므로 실제 운용 시에는 기본 입력인 `/no_ground_points`를 ground removal node에서 발행하는 구성을 사용한다.

## 단계별 구현 상태

1. 코드 분석 및 integration point: 완료
2. LiDAR object support: 완료
3. 3D box camera projection: 완료
4. Camera detection matching: 완료
5. Observation state classification: 완료
6. Adaptive lifecycle: 완료, 기본 ON
7. Adaptive association: 구현 완료, 기본 OFF
8. Depth consistency: 구현 완료, 기본 OFF
9. Adaptive KF: 구현 완료, 기본 OFF

실차 적용 순서는 A0 재현 후 A1/A2/A4를 먼저 평가하고, association과 KF는 CSV 및 ID-switch 분석으로 threshold를 조정한 뒤 활성화하는 것이 안전하다.
