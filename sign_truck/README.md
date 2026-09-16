# sign_truck

현재 YOLO26m 모델로 신호차량의 `green_sign`, `red_sign`만 검출하는 임시 ROS2 패키지다. `truck` 클래스는 YOLO 추론 단계에서 제외한다.

## 동작

- 입력: `/panorama/image_raw`
- ROI: 영상 상단 40%, 화면 중앙 기준 좌우 20%씩인 가운데 40% 영역
- 시각화: RViz용 `/sign_truck/annotated`에 글씨 없이 초록/빨강 검출 박스만 표시
- 출력 상태:
  - `/sign_truck/left_lane_state`
  - `/sign_truck/right_lane_state`
  - `/sign_truck/current_lane_state`
  - `/sign_truck/debug`

색상 검출만으로는 차량이 어느 차선에 있는지 알 수 없다. `current_lane:=auto`는 임시로 화면 중앙에 가장 가까운 신호를 현재 차선 신호로 본다. 실제 차량에서는 planner/localization이 현재 차선을 정하고, `/sign_truck/current_lane`에 `0`(왼쪽) 또는 `1`(오른쪽)을 보내면 해당 차선 상태를 선택한다. 고정된 차선 기준선은 카메라 설치 각도와 주행 경로에 맞춰 보정해야 한다.

## 빌드 및 실행

```bash
cd /home/ssc/SSC
source /opt/ros/humble/setup.bash
colcon build --packages-select sign_truck --symlink-install
source install/setup.bash
ros2 launch sign_truck sign_truck.launch.py
```

다른 터미널에서 bag을 실행한다.

```bash
source /opt/ros/humble/setup.bash
ros2 bag play /home/ssc/SSC/rosbags/2026-09-06/2026_mando_test_signal6_20260906_135904 \
  --topics /panorama/image_raw
```

별도 팝업은 띄우지 않는다. RViz에서 `Image` 디스플레이를 추가하고 다음 토픽을 선택한다.

```text
/sign_truck/annotated
```

현재 차선을 명시하려면 launch의 `current_lane:=left|right`를 사용하거나 실행 중 토픽으로 갱신한다.

```bash
ros2 topic pub --once /sign_truck/current_lane std_msgs/msg/Int32 "{data: 1}"
```
