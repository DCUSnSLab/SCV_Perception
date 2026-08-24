# panorama_stitcher

두 D435if의 컬러·정렬 뎁스를 하나의 전방 RGB-D 시야와 점군으로 만든다.
런타임은 Python ROS 2 노드와 PyTorch CUDA만 사용한다.

## 실행

실차 카메라, TF, 파노라마, 전·후방 지면 분할, 상태 감시를 함께 실행한다.

```bash
cd /home/ssc/SSC
source install/setup.bash
ros2 launch bring_up realsense_panorama.launch.py
```

카메라가 이미 실행 중이면 파노라마만 실행한다.

```bash
ros2 launch panorama_stitcher rgbd_panorama.launch.py
```

## 출력

| 토픽 | 형식 | 용도 |
|---|---|---|
| `/panorama/image_raw` | `bgr8` | 전방 컬러 와이드 뷰 |
| `/panorama/range` | `32FC1` | rig 기준 거리 영상 |
| `/panorama/validity` | `mono8` | 유효 뎁스 마스크 |
| `/panorama/points` | `PointCloud2` | 지면·장애물 입력 점군 |

대용량 출력은 구독자가 있을 때만 생성·발행한다. 각 출력은 독립적인
latest-only 발행 스레드를 사용해 DDS 적체가 다른 출력을 막지 않게 한다.

## 구조

```text
panorama_stitcher/
├── config/                 # 현재 런타임·캘리브레이션 값
├── launch/                 # 파노라마와 캘리브레이션 실행
├── panorama_stitcher_py/   # 지면 분할 라이브러리
├── scripts/                # ROS 노드와 캘리브레이션 도구
└── test/                   # PyTorch 투영·지면 분할 검사
```

주요 파일:

- `scripts/rgbd_panorama_torch_node.py`: RGB-D 동기화, PyTorch 3D 재투영,
  z-buffer, 컬러 합성, 점군 발행
- `panorama_stitcher_py/ground_segmentation_node.py`: RANSAC 지면/장애물 분리
- `config/rgbd_panorama.yaml`: 런타임 파라미터
- `config/rig_extrinsics.yaml`: TF와 투영이 공유하는 외부 파라미터
- `config/rig_calibration_report.yaml`: 캘리브레이션 검증 기록

런타임 값은 코드에 기본값을 두지 않는다. 파노라마는
`config/rgbd_panorama.yaml`, 전방 지면 분할은
`config/panorama_ground_segmentation.yaml`, 후방은
`config/rear_ground_segmentation.yaml`만 수정한다. 다른 파노라마 설정을
시험할 때는 원본을 건드리지 않고 별도 YAML을 만들어 다음처럼 지정한다.

```bash
ros2 launch bring_up realsense_panorama.launch.py \
  panorama_config_file:=/absolute/path/to/profile.yaml
```

## 캘리브레이션

두 카메라에 ChArUco 보드가 동시에 보이는 상태에서 실행한다.

```bash
ros2 launch panorama_stitcher rig_calibration.launch.py
```

현재 변환 규약은 다음과 같다.

```text
X_right_topic_optical = R_right_from_left * X_left_topic_optical + t
X_velodyne = R_rig_to_lidar * X_front_camera_rig + t_rig_in_lidar
```

카메라가 움직이면 `rig_extrinsics.yaml`과 `rig_calibration_report.yaml`을
함께 갱신하고, 실내 보드뿐 아니라 야외 원거리 수평선/차선도 확인한다.

## 검사

```bash
cd /home/ssc/SSC
colcon build --packages-select panorama_stitcher --symlink-install
colcon test --packages-select panorama_stitcher
colcon test-result --verbose
```
