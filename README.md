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

```text
~/pcdet_ros2_ws/models/pointpillars_coda.pth
~/pcdet_ros2_ws/models/pv-rcnn_kitti.pth
```

*모델 다운로드 경로*


