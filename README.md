# SCV Perception 시스템을 위한 ROS 패키지
---

## 카메라 센서

- **traffic_light_detection**
  - 신호등 탐지 패키지

- **lane_detection**
  - YOLOPv2 모델과 스테레오 카메라를 사용하여 차선 탐지 및 좌표 추정

- **bev_converter**
  - 모노카메라를 사용하여 BEV(조감도) 기반의 3D 좌표 추정

- **ultralytics_ros**
  - YOLO 모델을 ROS 환경에서 사용하는 객체 탐지 패키지

- **depth_anything**
  - depth_anything_v2 모델 기반 모노카메라 센서를 사용한 실시간 Depth 추정 패키지

---

## LiDAR 센서

- **urban_road_filter**
  - PointCloud 데이터를 이용하여 도로 노면과 비도로 영역을 구분하는 필터 패키지

- **lidar_obstacle_detector**
  - LiDAR 데이터를 활용한 장애물 탐지 패키지

- **lidar_camera_fusion**
  - LiDAR와 카메라 데이터를 융합하는 센서 퓨전 패키지

---

## 객체 트래킹

- **object_depth_tracker**
  - 객체 탐지 결과와 Depth 정보를 결합해 객체의 3D 위치를 추적하는 패키지

---

## 위치 추정

- **slam_localization**
  - SLAM 기반의 위치 추정 패키지

---  
