# yolo_validator

대상 코드:

- `src/mando_tools/mando_tools/yolo_validator.py`
- `src/mando_tools/launch/validate_mando_bag.launch.py`

## 1. 역할

`yolo_validator`는 YOLO 추론 결과만으로 신호등 상태를 빠르게 검증하는 기준 노드다.

## 2. 입력/출력

입력:

- `image_topic` (기본값: `/panorama/image_raw`)

출력:

- `/mando/yolo/annotated`
- `/mando/yolo/detections`
- `/tl/yolo_validator/state`
- `/tl/yolo_validator/state_label`
- `/tl/yolo_validator/state_reason`

## 3. 현재 처리 구조

1. subscriber는 최신 프레임 한 장만 유지한다.
2. `_process_latest_frame()`이 `max_fps` 주기로 최신 프레임만 추론한다.
3. YOLO 결과를 `_infer_state()`로 상태 ID로 변환한다.
4. 필요하면 annotated image, `Detection2DArray`, 전용 검증 상태 토픽을 발행한다.

## 4. 상태 판정 요약

- 클래스 필터는 `vehicular_*`와 `traffic light`만 사용한다.
- `traffic light`, `trafficlight`, `etc` 계열은 상태를 직접 해석하지 않는다.
- `green_arrow`, `left+arrow`, `red+green`이면 `LEFT ARROW`
- `green_arrow(down)`은 `UNKNOWN`
- 여러 검출 중 상태로 해석 가능한 클래스가 있으면 가장 높은 confidence를 사용한다.

## 5. 특징

- 색 분석 fallback이 없다.
- 구조가 가장 단순해서 모델 검증용으로 적합하다.
- `Detection2DArray`를 발행하므로 후속 노드 연결 전 점검용으로 좋다.

실행:

```bash
ros2 launch mando_tools validate_mando_bag.launch.py
```
