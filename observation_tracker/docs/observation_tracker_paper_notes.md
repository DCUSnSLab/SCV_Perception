# Observation-Adaptive 3D Multi-Object Tracker 구현 정리

## 1. 문서 목적

- ROS 2 Humble 기반 `observation_tracker`의 현재 구현 내용을 논문 작성용 개조식으로 정리
- 기존 LiDAR 기반 Tracking-by-Detection 구조에서 재사용한 요소와 새로 구현한 요소를 구분
- 제안 방법, 실험 변수, ablation 항목, 현재 한계를 명확하게 기록
- 본 문서의 수치 threshold는 초기 prototype 설정값이며, Ground Truth 기반 최종 최적값이 아님

## 2. 연구 목표

- 기존 3D Detector confidence와 실제 센서 관측 가능성을 분리하여 분석
- 객체별 LiDAR, Camera, Depth 관측 상태를 계산
- 관측 상태에 따라 Data Association 단계를 다르게 적용
- 3D Detector miss가 발생해도 실제 센서 근거가 존재하면 기존 Track ID를 유지
- 센서 근거가 부족한 3D detection은 즉시 Confirm하지 않고 suppression
- 단순 `miss_count` 기반 삭제 대신 관측 이력과 Track confidence를 함께 사용하는 lifecycle 구현
- 학습 기반 fusion network를 추가하지 않고 실제 센서 환경에서 실행 가능한 경량 구조 유지

## 3. 시스템 구성

### 3.1 사용 센서 및 플랫폼

- LiDAR: Velodyne VLP-32C
- RGB-D Camera: Intel RealSense D555i
- Middleware: ROS 2 Humble
- 3D Detector: CODa PointPillars
- 2D Detector: Ultralytics YOLO11 segmentation model의 bounding box 출력
- Base Tracker: 기존 Jay Tracker
- Calibration: 기존 TF 및 CameraInfo 사용
- Ego-motion: 기존 wheel odometry 및 IMU interface 재사용

### 3.2 주요 입력 토픽

| 구분 | 토픽 | 메시지 형식 | 용도 |
|---|---|---|---|
| Raw LiDAR | `/velodyne_points` | `sensor_msgs/PointCloud2` | 3D detection 및 객체별 LiDAR evidence 분석 |
| 3D detection | `/detected_objects_3d` | `tracking_msgs/DetectedObjectArray` | 3D bounding box와 confidence 입력 |
| RGB | `/camera/camera/color/image_raw` | `sensor_msgs/Image` | Camera availability 판단 |
| 2D detection | `/yolo/detections` | `perception_interface/DetectionArray` | Camera support 및 class consistency 판단 |
| Camera intrinsic | `/camera/camera/color/camera_info` | `sensor_msgs/CameraInfo` | 3D box projection |
| Aligned depth | `/camera/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | Depth consistency 및 2D detection의 3D 위치 추정 |
| Odometry | `/odometry/wheel` | `nav_msgs/Odometry` | Ego-motion compensation |
| IMU | `/vectornav/imu` | `sensor_msgs/Imu` | Ego-motion 정보 보조 |

### 3.3 주요 출력 토픽

| 구분 | 토픽 | 메시지 형식 | 비고 |
|---|---|---|---|
| Track 결과 | `/tracked_objects_3d` | `tracking_msgs/DetectedObjectArray` | 기존 downstream interface 유지 |
| Observation debug | `/observation_tracker/debug_markers` | `visualization_msgs/MarkerArray` | 상태, lifecycle, confidence 표시 |
| 통합 detection | `/observation_tracker/fused_detection_markers` | `visualization_msgs/MarkerArray` | 3D detection과 2D/depth evidence 통합 표시 |
| 기존 Track marker | `/pcdet/observation_tracks` | `visualization_msgs/MarkerArray` | Jay Tracker 기반 Track 결과 |

## 4. 기존 코드 재사용 범위

- `pcdet_tracker.jay_tracker.TimeAwareTrack`
  - Kalman Filter state prediction 및 update
  - Track ID 생성 및 관리
  - 객체 속도, 정적/동적 상태, yaw smoothing
- 기존 Jay Tracker ROS interface
  - `DetectedObjectArray` parsing 및 publication
  - PointCloud2 parsing
  - Camera, Depth, Odometry, IMU sensor cache
  - TF lookup 및 coordinate transformation
  - 기존 RViz marker publication
- 기존 ego-motion compensation
  - body-frame motion compensation
  - wheel odometry 및 yaw-rate 반영
- 기존 association 기반
  - SciPy `linear_sum_assignment`를 이용한 Hungarian matching
- 기존 메시지 형식 유지
  - 새로운 custom tracking message를 추가하지 않음
  - Observation state와 Track status는 marker 및 CSV에 기록

## 5. Detector 관련 수정

### 5.1 3D Detector

- PointPillars model과 학습 구조는 변경하지 않음
- `coda_pointpillar.launch.py`를 인자 없이 실행할 수 있도록 기본값 수정
  - 기본 입력: `/velodyne_points`
  - 기본 detection 출력: `/detected_objects_3d`
  - 기본 marker 출력: `/detections/visual_markers`
  - 기본 score threshold: `0.65`
  - 내장 Jay Tracker: 기본 비활성화
  - optional behavior predictor: 기본 비활성화
- Observation Tracker가 별도 Track ID를 관리하므로 PointPillars launch 내부 Tracker의 중복 실행 방지

### 5.2 2D Detector

- D555 compressed color topic 직접 입력 지원
  - 기본 입력: `/camera/camera/color/image_raw/compressed`
  - `sensor_msgs/CompressedImage`와 일반 `sensor_msgs/Image`를 입력 topic에 따라 자동 선택
- 기본 탐지 class 제한
  - `person`: COCO class `0`
  - `car`: COCO class `2`
- 기본 detection confidence threshold: `0.25`
- YOLO 내부 2D tracking: 기본 비활성화
  - 3D Tracker와 별도의 2D Track ID가 혼재하는 문제 방지
- source image와 segmentation mask publication: 기본 비활성화
  - 불필요한 message 크기와 runtime overhead 감소
- 2D detection 출력: `/yolo/detections`

## 6. 제안 Tracking Pipeline

```text
3D Detection + Existing Track Prediction
→ Object-wise Observation Region
→ LiDAR Observation Analysis
→ Camera Projection and 2D Matching
→ Depth Consistency Analysis
→ Observation State Decision
→ Uncertainty-Adaptive Unified Association
→ Adaptive Track Lifecycle
→ Track Output, RViz Marker, CSV Logging
```

## 7. Object-wise Observation Region

- 3D detection이 존재하는 경우
  - 현재 3D detection bounding box를 observation region으로 사용
- 3D detection이 없는 기존 Track의 경우
  - Kalman Filter predicted bounding box를 observation region으로 사용
- Calibration 및 prediction error를 고려하여 box margin 적용
  - XY margin: `0.15 m`
  - Z margin: `0.10 m`
- 최근 튜닝에서 margin을 축소
  - 배경 point 또는 인접 객체 point가 predicted box의 false support로 계산되는 현상 완화

## 8. LiDAR Observation Analysis

### 8.1 기본 처리

- Oriented 3D bounding box 내부의 raw LiDAR point 추출
- box yaw를 기준으로 point를 local coordinate로 회전한 후 box 내부 여부 판정
- 초기 구현은 `/velodyne_points`를 기본 사용
- ground removal point cloud 사용 가능
  - `lidar_points_topic`을 `/no_ground_points`로 변경 가능
  - 입력 point cloud 변경 시 point-count threshold 재설정 필요

### 8.2 사용 evidence

- Raw point count
- Point density
- Point centroid와 box center 사이의 거리
- Object distance
- Class × Distance 기반 minimum point threshold

### 8.3 LiDAR 상태

- `STRONG`
  - Class × Distance minimum point 조건 만족
  - Centroid error threshold 만족
- `WEAK`
  - Strong 조건은 만족하지 않지만 최소 2개 이상의 point 존재
- `NO_EVIDENCE`
  - 유효 point가 부족하거나 LiDAR 관측 범위 밖

### 8.4 현재 point threshold

| Class | Near ≤ 15 m | Middle ≤ 30 m | Far > 30 m |
|---|---:|---:|---:|
| Vehicle | 12 | 8 | 4 |
| Pedestrian | 6 | 4 | 2 |
| Cyclist | 8 | 5 | 3 |

- Maximum analysis range: `50 m`
- Centroid distance threshold: `1.25 m`
- 위 값은 validation dataset 분석 전 초기값

## 9. Camera Observation Analysis

### 9.1 3D-to-2D projection

- TF extrinsic과 CameraInfo intrinsic 사용
- 3D detection 또는 predicted Track의 eight corners를 image plane으로 projection
- image plane 내부의 유효 projected ROI 생성
- Camera FOV 밖인 경우 `camera_available = false`
- Camera FOV 밖이라는 이유로 negative evidence를 부여하지 않음

### 9.2 2D detection matching

- Class compatibility 확인
  - Vehicle ↔ car, truck, bus
  - Pedestrian ↔ person
  - Cyclist ↔ bicycle, motorcycle
- Projected ROI와 2D detection 간 다음 항목 계산
  - 2D IoU
  - Center distance
  - 2D detection confidence
- Candidate score
  - `0.65 × IoU + 0.20 × Center score + 0.15 × Detection confidence`
- 현재 gate
  - Camera IoU threshold: `0.20`
  - Center distance threshold: `100 pixel`

### 9.3 Camera 상태

- `AVAILABLE_SUPPORTED`
  - Camera FOV 내부
  - class, IoU, center-distance 조건 만족
- `AVAILABLE_UNSUPPORTED`
  - Camera FOV 내부이나 matching 조건 미충족
- `UNAVAILABLE`
  - TF/intrinsic/image 미수신, Camera FOV 외부 또는 유효 ROI 생성 실패

## 10. Depth Consistency Analysis

### 10.1 처리 절차

- Projected ROI의 중심 영역 사용
  - 전체 ROI 중 `0.60` scale 영역 사용
- Invalid depth 제거
  - minimum depth: `0.20 m`
  - maximum depth: `30.0 m`
- `16UC1` depth를 meter 단위로 변환
  - scale: `0.001`
- Predicted box의 camera-frame front depth와 ROI depth 비교
- Expected depth 주변의 support pixel 수와 support ratio 계산
- 대표 depth는 support pixel median 사용

### 10.2 Depth 상태

- `CONSISTENT`
  - 유효 support pixel 수와 support ratio가 threshold 이상
- `OCCLUDED`
  - Camera depth가 predicted object보다 충분히 가까움
- `INCONSISTENT`
  - 유효 depth는 존재하지만 expected depth와 불일치
- `UNAVAILABLE`
  - Depth image, projection 또는 valid pixel 부족

### 10.3 현재 threshold

- Depth consistency threshold: `1.5 m`
- Occlusion depth threshold: `1.0 m`
- Minimum valid/support depth pixels: `20`
- Minimum depth support ratio: `0.15`

## 11. Camera Detection의 3D 위치 추정

- 2D bounding box 중심 pixel과 aligned depth median을 이용하여 camera-frame 3D point 계산
- Camera intrinsic을 이용한 back-projection
  - `X = (u - cx) × Z / fx`
  - `Y = (v - cy) × Z / fy`
  - `Z = median depth`
- TF inverse transform을 이용하여 LiDAR/source frame으로 변환
- RViz에서 `CAMERA_DEPTH_CANDIDATE` box로 표시
- Camera-only candidate에는 신규 3D Track ID를 부여하지 않음
- 기존 Track과 Camera/Depth evidence가 일치하는 경우에만 recovery에 사용

## 12. Observation State Decision

| Observation state | 조건 | 의미 및 처리 방향 |
|---|---|---|
| `MULTIMODAL_STRONG` | LiDAR Strong + Camera Supported + Depth Consistent 또는 Unavailable | LiDAR와 Camera가 동시에 객체를 안정적으로 지지 |
| `LIDAR_DOMINANT` | LiDAR Strong, Camera support 부족 또는 Camera unavailable | Camera FOV 밖 객체를 camera failure로 처리하지 않음 |
| `CAMERA_DOMINANT` | LiDAR Weak/No evidence + Camera Supported + Depth Consistent | 기존 3D Track recovery에 사용 |
| `WEAK_OBSERVATION` | LiDAR와 Camera 모두 충분한 support 없음 | 즉시 삭제하지 않고 temporal lifecycle에서 판단 |

## 13. Uncertainty-Adaptive Unified Association

### 13.1 공통 원칙

- 활성, LOST, OCCLUDED Track과 모든 3D detection을 하나의 Hungarian assignment에 포함
- Class mismatch는 rejection
- `H P H^T + R` 기반 Mahalanobis distance를 유일한 위치 hard gate로 사용
- 3D IoU, Camera 2D IoU, Depth consistency는 soft cost로 사용
- 센서가 unavailable이면 해당 센서 cost를 적용하지 않음

### 13.2 Sensor-supported Unmatched Track Recovery

- 대상: 통합 assignment에서 match되지 않은 기존 Track
- 조건
  - 예측 box 내부 LiDAR evidence가 `STRONG`, 또는
  - class-compatible 2D detection support와 `CONSISTENT`/`UNAVAILABLE` depth
  - 다른 객체에 의한 명시적 occlusion은 occlusion policy가 우선
- 처리
  - 기존 Track ID 유지
  - Track status를 `RECOVERING`으로 변경
  - detector miss streak를 0으로 reset
  - association stage를 `SUPPORTED_PREDICTION`으로 기록
  - Camera-depth 3D center 방향으로 predicted position 일부 보정
  - Position gain: `0.40`
  - 1 frame maximum correction: `0.50 m`
- Camera-only detection으로 신규 3D Track을 생성하지 않음

### 13.3 Weak Observation Handling

- `CONFIRMED`, `RECOVERING`, `LOST`, `OCCLUDED` Track을 하나의 Hungarian assignment에 포함
- 모든 class가 동일한 uncertainty policy 사용
- `WEAK_OBSERVATION` match는 ID와 위치 연속성만 유지하며 정상 KF hit 또는 confirmation 근거로 계산하지 않음
- unmatched Track은 가림 여부와 센서 관측 가능성에 따라 서로 다른 confidence decay 적용
- 보행자 전용 recovery stage와 고정 반경 기반 신규 ID 차단 제거

## 14. Association Cost 및 Gate

### 14.1 기본 cost

```text
Cost =
  motion_weight × normalized squared Mahalanobis distance
  + iou_3d_weight × (1 - 3D IoU)
  + optional camera/depth cost
```

위치 innovation은 `S = H P H^T + R`로 정규화한다. Class mismatch만 즉시
거절하며 3D IoU는 hard gate가 아닌 soft cost이다. 따라서 miss가 누적되어
예측 공분산이 증가하면 재연결 가능 영역도 자동으로 증가한다.

### 14.2 현재 설정

| Parameter | 현재 값 | 목적 |
|---|---:|---|
| `iou_3d_weight` | 1.25 | box overlap 중요도 증가 |
| `motion_weight` | 0.75 | 공분산 정규화 innovation 반영 |
| `camera_iou_weight` | 0.50 | 2D overlap 반영 |
| `depth_weight` | 0.25 | Depth consistency 반영 |
| `mahalanobis_gate` | 9.21 | 2D 위치 측정의 99% chi-square gate |
| `maximum_association_cost` | 3.5 | 낮은 품질의 전체 match rejection |

### 14.3 정지/등속 motion mixture

- 동일 Track 안에서 stationary prediction과 constant-velocity prediction을 계산
- 직전 model 확률에 transition prior를 적용
- 새 위치 측정에 대한 두 prediction likelihood로 model 확률 갱신
- 두 prediction의 weighted mean과 between-model covariance를 KF prediction에 반영
- 차량과 보행자 모두 같은 motion mixture 사용

## 15. BEV IoU 수정

- 기존 Jay Tracker의 polygon clipping 내부 판정 방향 오류 확인
- 동일한 oriented bounding box끼리도 BEV IoU가 `0.0`으로 계산되는 문제 발생
- 원인
  - Box corner는 clockwise order
  - `inside_edge`는 counter-clockwise polygon 기준 부호 사용
- 수정
  - 내부 판정 부호를 `>= 0`에서 `<= 0`으로 변경
- 검증
  - 동일 box의 BEV IoU가 `1.0`으로 계산됨
- 영향
  - 통합 association의 3D IoU soft cost 정상화

## 16. Track Lifecycle Management

### 16.1 Track 상태

- `TENTATIVE`
- `CONFIRMED`
- `RECOVERING`
- `OCCLUDED`
- `SUPPRESSED`
- `LOST`
- `TERMINATED`

### 16.2 Track Initialization

- `MULTIMODAL_STRONG`
  - 신규 Track 생성 허용
- `LIDAR_DOMINANT`
  - 신규 Track 생성 허용
- `CAMERA_DOMINANT`
  - 신규 Track 생성 금지
  - 기존 Track recovery에만 사용
- `WEAK_OBSERVATION`
  - LiDAR와 Camera가 모두 관측 가능한데 support가 부족하면 신규 생성 억제
  - Detector false positive candidate로 판단하여 `SUPPRESSED`

### 16.3 Track Confirmation

- 일반 confirmation: 최소 hit `3`
- Strong observation 반복 시 confirmation: 최소 hit `2`
- WEAK observation 반복은 confirmation 근거로 사용하지 않음

### 16.4 Track Maintenance 및 Recovery

- Detection match 성공
  - KF update
  - hit count 증가
  - miss count 초기화
  - observation history 갱신
- Detection match 실패 + sensor support 존재
  - 즉시 삭제하지 않음
  - `RECOVERING`으로 유지
- Detection match 실패 + Camera/Depth recovery 성공
  - 기존 ID 유지
  - recovery count 증가
- Detection match 실패 + sensor support 없음
  - miss count 증가
  - Track confidence 감소
- Detection match 실패 + 가까운 보행자가 projected ROI를 가림
  - Depth `OCCLUDED`와 track 간 거리 순서를 함께 확인
  - `OCCLUDED` 상태로 전환하고 기존 ID를 제한 시간 동안 유지
  - 재등장 detection을 신규 ID 생성 전에 기존 Track에 우선 association

### 16.5 Track Termination

- 현재 설정
  - Maximum miss count: `6`
  - Recovery maximum age: `3 frame`
  - Maximum consecutive weak observation: `4 frame`
  - Confidence decay: `0.15/frame`
  - Minimum track confidence: `0.10`
- 단일 miss 조건만 사용하지 않음
- Miss count, weak history, sensor support, confidence를 함께 사용

## 17. Temporal Track Memory

- Track별 저장 정보
  - Track ID
  - Class ID
  - Age
  - Hit count
  - Miss count
  - Current observation state
  - Observation state history
  - LiDAR support history
  - Camera support history
  - Depth history
  - Track confidence
  - Recovery count
  - Current Track status
- History size: `7 frame`
- Temporal smoothing
  - Consecutive state count와 sliding-window majority 사용
  - State switch minimum consecutive count: `2`
  - 단일 frame noise로 인한 즉각적인 정책 전환 완화

## 18. RViz Visualization 정책

### 18.1 Observation debug marker

- Track ID
- Observation state
- Track status
- Track confidence
- LiDAR state 및 point count
- Camera support 여부
- Depth state 및 support ratio
- Observation state별 color 적용

### 18.2 Fused detection marker

- 하나의 MarkerArray에서 3D detection과 Camera matching 결과 확인
- Color
  - Green: 3D projected ROI와 2D detection match
  - Orange: Camera available, 2D match 없음
  - Blue: Camera evidence unavailable
  - Magenta sphere: 2D detection + valid depth 기반 3D candidate
- 객체 식별 표기
  - Tracker association 성공: `ID n`
  - Association되지 않은 raw detection: 임시 `DET n` 번호를 표시하지 않음
- 현재 최종 표시 정책
  - 현재 3D detection과 연결된 bounding box만 fused topic에 표시
  - Detection match가 사라지면 해당 bounding box와 text marker를 즉시 `DELETE`
  - 내부 Track ID와 lifecycle은 recovery를 위해 유지 가능
  - KF predicted bounding box는 fused topic에 표시하지 않음

## 19. CSV Logging 및 Runtime 계측

### 19.1 Track 단위 기록

- Timestamp 및 frame ID
- Track ID 및 class
- 3D/2D detection availability와 confidence
- Object distance
- LiDAR point count, density, centroid error, state
- Static/dynamic state
- Camera availability, support, IoU, center distance, class match
- Depth state, error, valid/support pixel count, support ratio
- Observation state
- Track status
- Hit count 및 miss count
- Track confidence 및 recovery count
- Matched detection ID
- Association stage 및 association cost

### 19.2 Runtime 기록

- Observation analysis time
- Association time
- Tracker update time
- Total processing time
- 향후 algorithm component별 overhead 비교에 활용 가능

## 20. Ablation Study 구성

### 20.1 Enable parameter

- `enable_lidar_observation`
- `enable_camera_observation`
- `enable_depth_consistency`
- `enable_observation_state`
- `enable_adaptive_lifecycle`
- `enable_temporal_smoothing`
- `enable_odometry`
- `enable_imu`

### 20.2 권장 ablation 조합

| 실험 | LiDAR observation | Camera | Depth | Unified association | Adaptive lifecycle | 목적 |
|---|---:|---:|---:|---:|---:|---|
| A0 Baseline | Off | Off | Off | Off | Off | 기존 LiDAR Tracking-by-Detection 기준 성능 |
| A1 LiDAR evidence | On | Off | Off | On | On | Raw point evidence 효과 |
| A2 LiDAR + Camera | On | On | Off | On | On | 2D support 효과 |
| A3 LiDAR + Camera + Depth | On | On | On | On | On | Depth consistency 효과 |
| A4 Full - temporal smoothing | On | On | On | On | On, smoothing Off | Temporal smoothing 효과 |
| A5 Full - weak match | On | On | On | On | On | Weak match의 ID 유지 효과 |
| A6 Full | On | On | On | On | On | 전체 제안 방법 |

## 21. 논문 평가 지표 후보

### 21.1 Tracking accuracy

- HOTA
- MOTA
- IDF1
- ID Switch count
- Fragmentation count
- Mostly Tracked / Mostly Lost

### 21.2 Observation-adaptive 정책 평가

- 3D Detector miss 구간에서 Track 유지율
- Camera-assisted recovery 성공률
- Weak Continuity 성공률
- Weak detection에 대한 false ID assignment 비율
- Suppressed candidate 중 실제 false positive 비율
- Observation state별 association precision
- Observation state별 Track survival time

### 21.3 Runtime 평가

- LiDAR observation analysis latency
- Camera projection/matching latency
- Depth consistency latency
- Association latency
- Total frame latency
- Sensor 기능별 추가 overhead

## 22. 현재 구현 검증 상태

- `observation_tracker` package build 성공
- `pcdet_tracker`와 함께 build 성공
- Unit test: `30 passed`
- 검증 항목
  - Class × Distance LiDAR threshold
  - LiDAR centroid consistency
  - Camera projection 및 2D matching
  - Depth consistency와 occlusion 분류
  - Observation state 결정
  - Camera-depth 3D candidate marker
  - 공분산 증가에 따른 association gate 확장
  - Camera-assisted unmatched Track recovery와 ID 유지
  - Camera-only 신규 Track 생성 억제
  - Unified weak match를 통한 기존 ID 연결
  - Weak match를 KF hit로 계산하지 않는 동작
  - 정지/CV model 확률 전환
  - Fused marker의 stable Track ID 표시
  - Detection miss 시 fused bounding box 즉시 삭제
  - 동일 box BEV IoU `1.0`

## 23. 현재 한계

- Class × Distance LiDAR point threshold는 GT 기반 validation 미완료
- Raw `/velodyne_points` 사용 시 ground point가 point count를 증가시킬 수 있음
- Camera-only detection으로 신규 3D Track을 생성하지 않음
- Appearance Re-ID를 사용하지 않음
- Track termination 이후 장시간 지난 객체의 ID 재식별 기능은 미구현
- Dormant Track memory 기반 ID 복원은 설계 아이디어 단계이며 현재 코드에는 포함하지 않음
- Camera-depth 3D candidate는 진단용 point/sphere이며 완전한 3D bounding box 추정기가 아님
- Depth hole, reflective surface, outdoor illumination에 따른 D555 depth degradation 추가 검증 필요
- Sensor time synchronization tolerance가 실제 timestamp jitter에 적합한지 validation 필요
- 정량적 MOT 성능과 실제 VLP-32C/D555 sequence 기반 비교 실험 필요

## 24. 논문의 핵심 기여점 문장 후보

- 객체별 실제 센서 관측 상태를 명시적으로 모델링하고, 이를 association과 Track lifecycle에 동시에 반영하는 경량 3D Multi-Object Tracking 구조를 제안하였다.
- Detector confidence와 raw sensor evidence를 분리하여, 3D detection miss와 sensor observation failure를 동일한 사건으로 취급하지 않도록 설계하였다.
- LiDAR point support, image-plane detection support, aligned-depth consistency를 결합하여 `MULTIMODAL_STRONG`, `LIDAR_DOMINANT`, `CAMERA_DOMINANT`, `WEAK_OBSERVATION`의 네 가지 관측 상태를 정의하였다.
- Kalman prediction uncertainty 기반 통합 association과 adaptive lifecycle을 적용하여 고정 거리 gate에 대한 의존성을 제거하고, Camera/Depth evidence를 이용한 기존 3D Track recovery를 수행하였다.
- Weak detection은 신규 ID 생성 근거로 사용하지 않으면서도 동일한 공분산 기반 association으로 기존 ID continuity를 보존하였다.
- 기존 Kalman Filter, Hungarian matching, Track ID 관리 및 ROS 2 interface를 유지하여 새로운 학습 모델 없이 실제 센서 시스템에 통합 가능한 구조를 구현하였다.

## 25. 논문 작성 시 구분해야 할 표현

- 사용 가능
  - “구현하였다”
  - “ROS 2 환경에서 동작을 확인하였다”
  - “단위 테스트와 RViz visualization을 통해 기능을 검증하였다”
  - “Ablation이 가능하도록 각 기능을 parameter화하였다”
- 정량 실험 전 사용 주의
  - “정확도를 향상시켰다”
  - “ID switch를 감소시켰다”
  - “기존 방법보다 우수하다”
  - “실시간성을 보장한다”
- 위 성능 주장은 GT 기반 비교 실험과 runtime 통계 분석 후 작성 필요

## 26. 구현 파일 대응 관계

| 기능 | 파일 |
|---|---|
| ROS node, parameter, sensor cache 연동 | `observation_tracker/observation_tracker_node.py` |
| Observation region | `observation_tracker/observation_region.py` |
| LiDAR evidence | `observation_tracker/lidar_observation.py` |
| Camera projection 및 2D matching | `observation_tracker/camera_observation.py` |
| Depth consistency | `observation_tracker/depth_consistency.py` |
| Evidence 통합 및 state 결정 | `observation_tracker/observation_analyzer.py` |
| Enum 및 `ObservationResult` | `observation_tracker/observation_state.py` |
| 통합 uncertainty association 및 Static/CV mixture | `observation_tracker/cascaded_association.py` |
| Adaptive lifecycle | `observation_tracker/track_lifecycle.py` |
| Temporal memory | `observation_tracker/track_memory.py` |
| RViz marker | `observation_tracker/visualization.py` |
| 실험 CSV | `observation_tracker/csv_logging.py` |
| 전체 threshold 및 enable flag | `config/observation_tracker.yaml` |
| 기본 실행 | `launch/observation_tracker.launch.py` |
| BEV IoU 방향 수정 | `jay_tracker/pcdet_tracker/pcdet_tracker/jay_tracker.py` |
| 3D Detector 기본 launch | `jay_tracker/pointpillars_coda_detector/launch/coda_pointpillar.launch.py` |
| 2D Detector 입력 및 class 설정 | `ultralytics_ros2/ultralytics_ros2/seg_run.py` |
| 2D Detector 기본 launch | `ultralytics_ros2/launch/seg.launch.py` |
