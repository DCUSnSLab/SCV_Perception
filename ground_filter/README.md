# ground_filter

3D LiDAR 점군에서 지면을 제거해 장애물 후보만 남기는 ROS 2 패키지

autoware_universe의 `autoware_ground_segmentation`(scan_ground_filter) 판정 로직 참고

## 원리

로봇을 중심으로 한 극좌표 격자(반경 `grid_size_m` x 방위 `radial_divider_angle_deg`)에 점을 담고, 부챗살을 따라 가까운 셀부터 바깥으로 진행하며 직전 지면 셀 기준으로 판정

## 빌드

```bash
colcon build --symlink-install --packages-select ground_filter
```

## 실행

```bash
ros2 launch ground_filter ground_filter.launch.py
```

| 토픽 | 타입 | 내용 |
|---|---|---|
| `/velodyne_points` (sub) | `sensor_msgs/PointCloud2` | 입력. 임의 프레임 |
| `/perception/points_nonground` (pub) | `sensor_msgs/PointCloud2` | 장애물 후보 |
| `/perception/points_ground` (pub) | `sensor_msgs/PointCloud2` | 지면. 디버그용 |

입력은 `base_frame`으로 변환된 뒤 처리된다. 알고리즘이 차량 중심·z축 상방 프레임을 전제하므로 TF가 필요하다.

## 주요 파라미터

| 이름 | 기본값 | 설명 |
|---|---|---|
| `grid_size_m` | 0.05 | 셀 반경 간격. 분해하려는 단차보다 작아야 한다 |
| `non_ground_height_threshold` | 0.05 | 지면선 위 이만큼 솟으면 장애물 [m] |
| `grid_radial_limit_m` | 15.0 | 처리 반경. 곧 장애물 탐지 거리 |
| `gnd_grid_continual_thresh` | 40 | 엄격 판정을 유지할 셀 간격. LiDAR 링 간격보다 커야 한다 |
| `global_slope_max_angle_deg` | 15.0 | 원점 기준 허용 경사 |
| `local_slope_max_angle_deg` | 13.0 | 셀 사이 허용 경사 |

## 한계

16채널 LiDAR에서 15 cm 연석은 5 m 밖부터 빔 간격보다 작아진다. 연석은 모서리를 따라 점점이 찍히고 연속된 경계로는 나오지 않는다. 연속 경계가 필요하면 시간 누적 elevation map에서 단차를 찾는 별도 모듈이 필요하다.
