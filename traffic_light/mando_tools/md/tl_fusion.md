# tl_fusion

대상 코드:

- `src/mando_tools/mando_tools/tl_fusion.py`
- `src/mando_tools/launch/tl_fusion.launch.py`

상세 문서:

- `src/mando_tools/mando_tools/tl_fusion.md`

## 1. 역할

`tl_fusion`은 YOLO 클래스 결과와 ROI 색 분석 결과를 함께 보고 최종 신호 상태를 안정적으로 발행하는 노드다.

## 2. 입력/출력

입력:

- `image_topic`

출력:

- `/tl/debug_image`
- `/tl/state_id`
- `/tl/state_label`
- `/tl/state_reason`
- `/tl/input_valid`

## 3. 현재 처리 흐름

1. subscriber는 최신 프레임 한 장만 유지한다.
2. 타이머 기반으로 `max_fps` 주기마다 최신 프레임만 처리한다.
3. `_detect_candidates()`로 신호등 후보를 검출한다.
4. `_select_candidate()`로 대표 후보 하나를 고른다.
5. 모델 confidence가 높으면 모델 상태를 우선 사용한다.
6. 그렇지 않으면 `_analyze_selected_candidate()`로 색 분석 fallback을 수행한다.
7. `_decide_state()`와 `_update_stable_state()`로 최종 상태를 정한다.
8. 결과를 publish하고, 필요할 때만 디버그 이미지를 생성한다.
9. 입력 영상이 3초 동안 오지 않으면 `UNKNOWN(0)`과 `input_valid=false`를 발행한다.

## 4. 현재 기준으로 중요한 점

- 클래스 필터는 `vehicular_*`와 `traffic light`만 사용한다.
- `traffic light`와 `etc` 계열은 상태를 직접 해석하지 않는다.
- `LEFT ARROW`는 `green_arrow`, `left+arrow`, `red+green` 조합으로 해석한다.
- 종료분기용 `green_arrow(down)`은 주행 신호 상태에서 제외한다.
- `/tl/debug_image`는 구독자가 있거나 `show_windows=true`일 때만 만든다.

## 5. launch에서 자주 조절하는 인자

- `image_topic`
- `max_fps`
- `detector_conf_threshold`
- `detector_image_size`
- `model_confidence_threshold`
- `fallback_score_threshold`

실행:

```bash
ros2 launch mando_tools tl_fusion.launch.py
```
