# pcdet_ros2_ws

ROS 2 워크스페이스로, LiDAR 바닥 제거, 3D 탐지, 3D 트래킹 파이프라인을 실행하기 위한 구성입니다.

## 구성

워크스페이스 주요 디렉터리:

- `src/`: ROS 2 패키지 소스
- `third_party/OpenPCDet/`: detector와 `jay_tracker`가 사용하는 OpenPCDet 코드와 모델 파일
- `build/`, `install/`, `log/`: colcon 빌드 결과

주요 패키지:

- `pcd_ground_filter`
  - 입력 포인트클라우드에서 바닥을 제거합니다.
  - 주요 실행 파일: `ground_removal_node`
- `pcdet_detector`
  - OpenPCDet 기반 3D 탐지 노드와 launch 파일을 제공합니다.
  - 주요 실행 파일: `kitti_detector_node`
  - launch 파일: `pcdet_pipeline.launch.py`
- `pcdet_tracker`
  - 여러 트래커 구현을 포함합니다.
  - 주요 실행 파일:
    - `tracker_node`: 기본 tracker
    - `ab3dmot_node`: AB3DMOT tracker
    - `jay_tracker`: Jay tracker
- `tracking_msgs`
  - detector/tracker 통신에 사용하는 커스텀 메시지 패키지입니다.

## 패키지 역할

### 1. `pcd_ground_filter`

역할:

- 원본 LiDAR 포인트클라우드에서 바닥 점을 제거
- 후단 detector가 사용할 `/no_ground_points` 생성

일반적인 흐름:

- 입력: LiDAR 원본 PointCloud2
- 출력: `/no_ground_points`

### 2. `pcdet_detector`

역할:

- OpenPCDet 모델을 로드해 3D bounding box를 예측
- `jay_tracker`에서 분리된 KITTI 기반 detector 설정과 추론 로직이 포함됨
- `basic`, `ab3dmot` 모드에서 tracker로 넘길 detection 생성

일반적인 흐름:

- 입력: `/no_ground_points`
- 출력:
  - `/detected_objects_3d`
  - RViz용 marker

### 3. `pcdet_tracker`

역할:

- detector 결과를 이용해 객체를 프레임 간 연결
- tracker 종류에 따라 동작 방식이 다름

포함된 tracker:

- `tracker_node`
  - 기존 ROS 메시지 기반 기본 tracker
- `ab3dmot_node`
  - AB3DMOT 기반 tracker
- `jay_tracker`
  - `pcdet_detector`가 publish한 detection을 받아 tracking만 수행
  - 선택적으로 detection/track CSV 저장 가능

### 4. `tracking_msgs`

역할:

- detector와 tracker 사이에 쓰이는 메시지 정의
- `DetectedObject.msg`
- `DetectedObjectArray.msg`

## 빌드 방법

```bash
source /opt/ros/humble/setup.bash
cd ~/pcdet_ros2_ws
colcon build
source install/setup.bash
```

빌드 후 새 터미널을 열 때마다 아래를 다시 실행해야 합니다.

```bash
source /opt/ros/humble/setup.bash
cd ~/pcdet_ros2_ws
source install/setup.bash
```

## 권장 실행 방법: launch

가장 간단한 실행 방법은 launch입니다.

```bash
source /opt/ros/humble/setup.bash
cd ~/pcdet_ros2_ws
source install/setup.bash
ros2 launch pcdet_detector pcdet_pipeline.launch.py
```

기본값:

- `tracker_type:=kitti`
- 실행 노드:
  - `ground_removal_node`
  - `kitti_detector_node`
  - `jay_tracker`

### launch 인자

```bash
ros2 launch pcdet_detector pcdet_pipeline.launch.py --show-args
```

현재 지원 인자:

- `tracker_type`
  - `kitti`: `jay_tracker` 사용
  - `basic`: `kitti_detector_node + tracker_node`
  - `ab3dmot`: `kitti_detector_node + ab3dmot_node`
- `enable_csv_logging`
  - `jay_tracker`에서 CSV 저장 사용 여부
- `sequence_id`
  - CSV 파일 이름에 사용할 시퀀스 ID
- `detection_output_dir`
  - detection CSV 저장 경로
- `track_output_dir`
  - tracking CSV 저장 경로

### 예시

Jay tracker 실행:

```bash
ros2 launch pcdet_detector pcdet_pipeline.launch.py
```

Jay tracker + CSV 저장:

```bash
ros2 launch pcdet_detector pcdet_pipeline.launch.py \
  enable_csv_logging:=true \
  sequence_id:=seq01 \
  detection_output_dir:=results/detections \
  track_output_dir:=results/tracks
```

기본 tracker 사용:

```bash
ros2 launch pcdet_detector pcdet_pipeline.launch.py tracker_type:=basic
```

AB3DMOT 사용:

```bash
ros2 launch pcdet_detector pcdet_pipeline.launch.py tracker_type:=ab3dmot
```

## 성능 측정

최적화 전에 현재 기준선을 먼저 측정할 수 있도록 각 노드에 성능 요약 로그가 들어 있습니다.

- `ground_removal_node`
  - `read`, `voxel`, `filter`, `publish`
- `kitti_detector_node`
  - `read`, `batch`, `infer`, `post`, `pub_msg`, `pub_marker`
- `jay_tracker`
  - `parse`, `csv_det`, `ego`, `track`, `csv_track`, `history`, `markers`

기본값:

- 1초마다 성능 요약 출력
- 초기 5프레임은 warm-up으로 제외
- 프레임별 성능 CSV 저장:
  - `results/perf/ground_filter_perf.csv`
  - `results/perf/detector_perf.csv`
  - `results/perf/tracker_perf.csv`

실행:

```bash
source /opt/ros/humble/setup.bash
cd ~/pcdet_ros2_ws
source install/setup.bash
ros2 launch pcdet_detector pcdet_pipeline.launch.py
```

로그 예시:

```text
[perf:ground_filter] frames=8 wall_fps=7.95 msg_fps=8.58 avg_total=24.1ms max_total=31.8ms avg_in=18432 avg_out=6210 read=6.1ms voxel=0.0ms filter=13.7ms publish=4.3ms
[perf:detector] frames=8 wall_fps=7.82 msg_fps=8.58 avg_total=92.4ms max_total=110.6ms avg_in=6210 avg_out=5.6 read=4.8ms batch=11.3ms infer=63.5ms post=5.9ms pub_msg=0.4ms pub_marker=6.5ms
[perf:tracker] frames=8 wall_fps=7.80 msg_fps=8.58 avg_total=7.6ms max_total=12.4ms avg_in=5.6 avg_out=4.9 parse=1.2ms csv_det=0.0ms ego=0.1ms track=2.5ms csv_track=0.0ms history=0.1ms markers=3.7ms
```

CSV 예시 헤더:

```text
frame_index,msg_stamp,total_ms,input_count,output_count,...
```

해석:

- `avg_total`
  - 노드 1프레임 평균 처리 시간
- `max_total`
  - 해당 측정 구간에서 가장 느린 프레임 시간
- `wall_fps`
  - 실제 처리된 콜백 기준 FPS
- `msg_fps`
  - 입력 메시지 stamp 기준 FPS
- `avg_in`, `avg_out`
  - 프레임당 평균 입력/출력 개수
  - ground filter: point 수
  - detector: 입력 point 수, 출력 detection 수
  - tracker: 입력 detection 수, 출력 track 수

추가 확인 명령:

```bash
ros2 topic hz /velodyne_points
ros2 topic hz /no_ground_points
ros2 topic hz /detected_objects_3d
```

CSV 저장이 tracker 속도에 미치는 영향도 바로 비교할 수 있습니다.

CSV 끔:

```bash
ros2 launch pcdet_detector pcdet_pipeline.launch.py enable_csv_logging:=false
```

CSV 켬:

```bash
ros2 launch pcdet_detector pcdet_pipeline.launch.py enable_csv_logging:=true
```

## 개별 실행 방법: ros2 run

### 1. Jay tracker 단독 실행

`jay_tracker`는 detector 결과를 구독하는 tracking 전용 노드입니다.

```bash
source /opt/ros/humble/setup.bash
cd ~/pcdet_ros2_ws
source install/setup.bash
ros2 run pcd_ground_filter ground_removal_node
```

다른 터미널:

```bash
source /opt/ros/humble/setup.bash
cd ~/pcdet_ros2_ws
source install/setup.bash
ros2 run pcdet_tracker jay_tracker
```

CSV 저장까지 켜려면:

```bash
ros2 run pcdet_tracker jay_tracker -- \
  --enable_csv_logging true \
  --sequence_id seq01 \
  --detection_output_dir results/detections \
  --track_output_dir results/tracks
```

### 2. detector + basic tracker 분리 실행

터미널 1:

```bash
source /opt/ros/humble/setup.bash
cd ~/pcdet_ros2_ws
source install/setup.bash
ros2 run pcd_ground_filter ground_removal_node
```

터미널 2:

```bash
source /opt/ros/humble/setup.bash
cd ~/pcdet_ros2_ws
source install/setup.bash
ros2 run pcdet_detector kitti_detector_node
```

터미널 3:

```bash
source /opt/ros/humble/setup.bash
cd ~/pcdet_ros2_ws
source install/setup.bash
ros2 run pcdet_tracker tracker_node
```

### 3. detector + AB3DMOT 분리 실행

터미널 1:

```bash
ros2 run pcd_ground_filter ground_removal_node
```

터미널 2:

```bash
ros2 run pcdet_detector kitti_detector_node
```

터미널 3:

```bash
ros2 run pcdet_tracker ab3dmot_node
```

## 토픽 흐름

### Jay tracker 모드

```text
LiDAR 원본 -> ground_removal_node -> /no_ground_points
/no_ground_points -> kitti_detector_node -> /detected_objects_3d
/detected_objects_3d -> jay_tracker -> RViz marker + optional CSV
```

### Basic / AB3DMOT 모드

```text
LiDAR 원본 -> ground_removal_node -> /no_ground_points
/no_ground_points -> kitti_detector_node -> /detected_objects_3d
/detected_objects_3d -> tracker_node 또는 ab3dmot_node
```

## CSV 저장

`jay_tracker`에서만 지원합니다.

`enable_csv_logging:=true` 또는 `--enable_csv_logging true`로 활성화할 수 있습니다.

생성 파일:

- detection CSV: `results/detections/<sequence_id>_detections.csv`
- track CSV: `results/tracks/<sequence_id>_tracks.csv`

기본 경로는 현재 실행 디렉터리 기준입니다.

## 자주 확인할 명령

토픽 목록:

```bash
ros2 topic list
```

Detector 출력 확인:

```bash
ros2 topic echo /detected_objects_3d
```

노드 목록:

```bash
ros2 node list
```

패키지 실행 파일 확인:

```bash
ros2 pkg executables pcdet_tracker
```

## 참고

- `jay_tracker`는 기존 `kitti_based_tracker.py`의 트래킹 로직을 유지하되, detector는 분리한 버전입니다.
- 기존 이름 호환을 위해 `kitti_based_tracker` 실행 이름도 남겨두었습니다.
- OpenPCDet 관련 자산은 `third_party/OpenPCDet` 안에 복사되어 있으므로, 원래 `~/OpenPCDet`를 직접 참조하지 않도록 정리되어 있습니다.
