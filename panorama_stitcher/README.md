# SSC Python/PyTorch RGB-D Panorama

전방 `front_left`, `front_right` D435if의 RGB와 aligned depth를 동기화해
원통형 파노라마, 거리 영상, 3차원 점 구름을 생성합니다. 파노라마와 지면
분리는 Python으로만 실행하며, 영상·깊이 투영과 필터링은 PyTorch CUDA가
담당합니다.

실행 환경에는 CUDA 지원 PyTorch가 필요합니다. 빌드 전에 다음 검사에서
`True`가 출력되어야 합니다.

```bash
python3 -c 'import torch; print(torch.cuda.is_available())'
```

## 실행

카메라, 파노라마, 전방·후방 지면 분리, TF와 상태 감시기를 함께 실행합니다.

```bash
source /opt/ros/humble/setup.bash
source /home/ssc/SSC/install/setup.bash
ros2 launch bring_up realsense_panorama.launch.py
```

카메라가 이미 실행 중일 때 파노라마만 실행하려면 다음을 사용합니다.

```bash
ros2 launch panorama_stitcher rgbd_panorama.launch.py
```

두 실행 경로 모두 `rgbd_panorama_torch_node`를 시작합니다. C++ 백엔드 선택
옵션은 없습니다.

## 현재 처리 경로

1. 좌·우 RGB, aligned depth, CameraInfo의 최신 동기 묶음을 선택합니다.
2. 캘리브레이션과 실시간 CameraInfo로 두 카메라를 공통 원통 좌표에
   투영합니다.
3. PyTorch CUDA spatial/temporal 필터와 z-buffer로 깊이를 합칩니다.
4. 동일 프레임에서 `/panorama/image_raw`, `/panorama/range`,
   `/panorama/points`를 생성합니다.
5. Python 지면 분리 노드가 RANSAC, 지역 지면 확장, 장애물 후처리를 수행해
   지면과 장애물 점 구름을 발행합니다.

후방 카메라는 파노라마에 합치지 않습니다. RealSense가 만든 후방 점 구름을
별도 Python 지면 분리 노드가 직접 처리합니다.

## 주요 토픽

- `/panorama/image_raw`: 전방 RGB 파노라마
- `/panorama/range`: 파노라마 좌표의 거리 영상
- `/panorama/points`: 전방 파노라마 3차원 점 구름
- `/panorama/ground_points`: 전방 지면점
- `/panorama/obstacle_points`: 전방 장애물점
- `/rear/ground_points`: 후방 지면점
- `/rear/obstacle_points`: 후방 장애물점

## 주요 설정

- `config/rgbd_panorama.yaml`: 카메라 보정, 동기화, 투영, CUDA 필터와 출력
- `config/panorama_ground_segmentation.yaml`: 전방 지면·장애물 판정
- `config/rear_ground_segmentation.yaml`: 후방 지면·장애물 판정
- `config/rig_extrinsics.yaml`: 실제 실행 TF의 단일 기준값

## 빌드와 시험

```bash
cd /home/ssc/SSC
source /opt/ros/humble/setup.bash
colcon build --packages-select panorama_stitcher bring_up --symlink-install
source install/setup.bash
colcon test --packages-select panorama_stitcher bring_up
colcon test-result --verbose
```

Python 파노라마 회귀 시험은 동기화, 원통 투영, 깊이 z-buffer, 필터, 토픽
메시지 생성을 검사합니다. 지면 분리 시험은 RANSAC, 경사 전환, 지역 확장,
수평 노면 억제와 시간 지속성 필터를 검사합니다.
