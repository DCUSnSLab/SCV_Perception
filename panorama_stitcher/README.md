# SSC Real-time Panorama Stitcher

`front`와 `camera` D435i 컬러 영상을 실시간 합성해
`/panorama/image_raw`로 발행합니다.

주행용 전방 RGB-D 융합은 `front_rgbd_fusion.launch.py`를 사용합니다.
확정된 카메라 역할은 다음과 같습니다.

- 전방 왼쪽: `/front/front` (serial `239122073045`)
- 전방 오른쪽: `/camera/camera` (serial `239122071306`)
- 후방 단일: `/cam3/cam3` (serial `239722073611`, 이 패키지 입력 아님)

```bash
ros2 launch panorama_stitcher front_rgbd_fusion.launch.py
```

발행 토픽:

- `/parking/front/color_mosaic`: 원본 픽셀 보존 컬러 모자이크 (`bgr8`)
- `/parking/front/depth_mosaic`: 컬러와 동일한 배치의 각 원본 카메라
  optical-z 깊이 (`32FC1`, m, 무효 픽셀은 NaN)
- `/parking/front/points`: 두 카메라 점군을 rig optical frame에서 결합한
  `PointCloud2` (`x`, `y`, `z`, `rgb`)

`front_rgbd_rig_optical_frame`은 x 오른쪽, y 아래, z 전방인 가상 광학
좌표계입니다. 실제 제어에 연결할 때는 이 프레임에서 `base_link`까지의
외부 파라미터를 URDF/TF에 추가해야 합니다. 컬러와 뎁스 영상은 확인 및
영상 인식용이고, 거리 기반 주행 판단에는 `/parking/front/points`를
사용합니다.

2026-07-29 다중 보드 metric bundle 결과는 왼쪽 yaw `-32.00°`,
오른쪽 yaw `+33.34°`, 오른쪽 잔여 pitch `+0.26°`, 컬러 광학 중심
간격 `0.12149 m`입니다. 자로 잰 약 `0.11 m`는 결과를 고정하지 않고
검증에만 사용했습니다.

주행용 RGB-D 출력은 중앙 물체 정보를 압축하거나 깊이 재투영으로 컬러를
덮어쓰지 않습니다. Board 5로 측정한 고정 배치 `x=1900 px`,
`y=17 px`와 20 px 겹침을 컬러와 뎁스에 똑같이 적용합니다.
`panorama.launch.py`는 이전 비교용 고정 합성이고, 원통 투영 버전은
비교 실험용 `rgbd_panorama.launch.py`로 분리되어 있습니다.

두 영상을 중앙 정면 가상 카메라로 펴는 BEV 유사 원근 보정 버전은
`rectified_panorama.launch.py`이며 별도 토픽
`/panorama/rectified/image_raw`를 사용합니다.

## Build

```bash
cd /home/ssc/SSC
source /opt/ros/humble/setup.bash
colcon build --packages-select panorama_stitcher
source install/setup.bash
```

`/usr/local/cuda/bin/nvcc`가 있으면 `rgbd_panorama_stitcher_node`는
RTX 4060(`sm_89`)용 CUDA backend를 자동으로 함께 빌드합니다.
`rgbd_panorama.yaml`의 `use_cuda: true`가 기본값이며, CUDA를 사용할 수
없으면 노드는 경고를 남기고 기존 CPU 경로로 전환합니다. 실행 중
진단 로그의 `backend=CUDA`, `gpu=... ms`로 실제 사용 여부를 확인할 수
있습니다.

## Metric RGB-D wide view

기존 `/panorama/image_raw`는 그대로 유지됩니다. 거리 하나에 맞춘
homography 대신 양쪽 `aligned_depth_to_color`의 모든 유효 픽셀을 3D로
역투영하고, 캘리브레이션된 rig 중심 가상 카메라로 다시 투영하는 주행용
출력은 별도 노드로 실행합니다.

카메라가 이미 실행 중이면:

```bash
ros2 launch panorama_stitcher rgbd_metric_panorama.launch.py
```

카메라도 처음부터 켜야 하면 터미널 1에서:

```bash
ros2 launch bring_up realsense_multi.launch.py
```

발행 토픽:

- `/panorama/metric/image_raw`: `bgr8`, 3118×972 RGB-D 재투영 컬러
- `/panorama/metric/range`: `32FC1`, rig 중심에서의 수평 거리(m),
  무효 픽셀은 `0`
- `/panorama/metric/validity`: `mono8`, 실제 깊이 재투영 픽셀은 `255`,
  깊이가 없어 회전 전용 컬러로 대체된 픽셀은 `0`

2026-07-29 실내 라이브 측정에서는 RTX 4060 CUDA backend로 약
`29–30 FPS`, GPU `13–14 ms`, 전체 처리 `28–30 ms`, 유효 깊이 약
`94.7–94.9%`였습니다. CPU full-resolution median/EMA는 20 FPS 부근의
병목과 동적 물체 잔상을 만들어 이 경로에서는 사용하지 않습니다.

렌더러는 가까운 깊이를 선택하는 z-buffer와 깊이 불연속 인지 splatting을
사용합니다. 평탄한 면은 한 픽셀 확장해 forward-warp 구멍을 막고, 물체
경계는 확장하지 않아 배경이 전경으로 번지는 것을 줄입니다. 두 카메라가
여전히 약 10–22 ms 어긋나므로 빠른 물체의 완전한 시간 정합은 외부
hardware sync 없이는 보장할 수 없습니다.

## Live cameras

```bash
ros2 launch panorama_stitcher panorama.launch.py
```

```bash
ros2 run image_view image_view \
  --ros-args -r image:=/panorama/image_raw
```

## 2026-07-28 bag test

실시간 카메라와 bag 토픽이 충돌하지 않도록 bag 토픽을 remap합니다.

```bash
ros2 bag play /home/ssc/20260728/sensor_20260728_180417 \
  --remap \
  /front/front/color/image_raw:=/bag/front/color/image_raw \
  /camera/camera/color/image_raw:=/bag/camera/color/image_raw
```

```bash
ros2 launch panorama_stitcher panorama.launch.py \
  left_topic:=/bag/front/color/image_raw \
  right_topic:=/bag/camera/color/image_raw
```

## Geometry

실험용 RGB-D 버전은 다음과 같이 실행합니다.

```bash
ros2 launch panorama_stitcher rgbd_panorama.launch.py
```

정면 직사각형 보정 버전:

```bash
ros2 launch panorama_stitcher rectified_panorama.launch.py
```

- `front`: panorama 중심 기준 yaw `-32.00°`
- `camera`: panorama 중심 기준 yaw `+33.34°`, pitch `+0.26°`
- relative rotation: `65.379°`
- RGB optical-center baseline: `0.12149 m`
- projection: cylindrical, default scale `1.0`
- depth range: `0.20–15.0 m`

두 컬러 영상은 RealSense rotation filter로 180° 회전되어 있지만
`CameraInfo`는 회전되지 않습니다. 노드가 주점을
`(width-1-cx, height-1-cy)`로 변환해서 사용합니다.

주행용 metric 파노라마는 다음처럼 실행합니다.

```bash
ros2 launch panorama_stitcher rgbd_metric_panorama.launch.py
```

- `/panorama/metric/image_raw`: 실제 양안 중첩각 안에서만 RGB-D 3D 재투영
- `/panorama/metric/range`: 전 시야 rig 중심 거리 영상(`32FC1`, m)
- `/panorama/metric/validity`: 전 시야 유효 깊이 마스크(`mono8`)

중첩 구간의 픽셀 범위는 외부 캘리브레이션과 출력 투영에서 자동
계산됩니다. `depth_color_overlap_only: true`는 색상 렌더링만 제한하며,
`full_depth_reprojection: true`인 거리/유효성 출력은 제한하지 않습니다.

## Fixed calibration

기본 `panorama_stitcher_node`의 값은 초기 bag 검토 때 추정한 컬러 전용
고정 이동값입니다.

- right x offset: 1834 px
- right y offset: 9 px
- overlap/blend width: 86 px
- approximate sync tolerance: 25 ms

정밀한 최종 결과를 위해서는 두 카메라에 동시에 보이는 보드로 외부
파라미터를 한 번 더 보정해야 합니다.

현재 `/panorama/image_raw`는 카메라 마운트 조절용 풀해상도 화면입니다.
상단 먼 보드 대응점 기준 `1869/15 px`, 20 px blend를 사용합니다.
가까운 Board 5의 약 26 px 시차는 정상이며 이 화면에서 맞추지 않습니다.
주행용 `/parking/front/color_mosaic`는 별도의 `1900/17 px` 픽셀 보존
배치를 유지합니다.

주행용 `front_rgbd_fusion`은 Board 5 평면을 양쪽 영상에서 외삽해 얻은
별도 고정값을 사용합니다.

- right x offset: 1900 px
- right y offset: 17 px
- overlap: 20 px
- output: 3820 x 1063

## D435if rig extrinsic calibration

단일 Board 5의 서로 다른 절반만 이용한 해는 평면 퇴화로 잘못된
`1.923 m` baseline을 내어 거부됐습니다. 최종값은 서로 다른 거리와
자세의 Board 2와 Board 5를 함께 최적화하며, baseline과 오른쪽 카메라
yaw/pitch/roll을 강제하지 않습니다.

```bash
ros2 launch panorama_stitcher rig_calibration.launch.py
```

실시간 단일 보드 점검은 위 launch를 사용합니다. 최종 다중 보드 해는
`capture.py`로 양쪽의 같은 프레임 범위를 저장한 뒤 다음처럼 재현합니다.

```bash
ros2 run panorama_stitcher multiboard_bundle_calibrator \
  --frame-start 68 --frame-end 77 \
  --output /tmp/dual_front_extrinsics.yaml
```

다음 증거를 함께 사용합니다.

- 두 컬러 영상의 ArUco 마커 코너
- `aligned_depth_to_color`에서 얻은 같은 마커 내부의 깊이
- Board 2와 Board 5의 서로 다른 metric 좌표계
- `/home/ssc/lidar_cam_calib/board_params.yaml`의 실측 보드 치수
- D435if 설치 방향 `64°`와 컬러 렌즈 중심 간격 약 `0.11 m`

`0.11 m`는 초기값과 검증에만 쓰고 결과에 강제하지 않습니다. ChArUco
내부 코너를 주 해로 사용하고 ArUco 마커 외곽 코너를 독립 검증으로
사용하며, 전체/짝수/홀수 프레임 해의 반복성도 확인합니다.

항상 생성되는 감사 보고서:

```text
config/rig_calibration_report.yaml
```

모든 게이트를 통과한 경우에만 생성되는 적용 가능 외부 파라미터:

```text
config/rig_extrinsics.yaml
```

변환 규약은 다음과 같습니다.

```text
p_camera_color_optical = R * p_front_color_optical + t
```

출력에는 원본 `CameraInfo` 광학 좌표계와, 현재 180° 회전된 영상에 맞는
광학 좌표계 변환을 둘 다 기록합니다. RealSense 한 대 내부의 50 mm
스테레오 baseline은 두 카메라 사이 baseline으로 사용하지 않습니다.
