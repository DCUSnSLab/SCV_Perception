# tl_roi_hist

대상 코드:

- `src/mando_tools/mando_tools/tl_roi_hist.py`
- `src/mando_tools/launch/tl_roi_hist.launch.py`

## 1. 역할

`tl_roi_hist`는 YOLO로 신호등 ROI를 찾고, ROI 내부의 Hue histogram으로 실제 상태를 판정하는 노드다.

## 2. 입력/출력

입력:

- `image_topic`

출력:

- `/tl/debug_image`
- `/tl/zoom_image`
- `/tl/roi_hist/state`
- `/tl/hist_image` (`pub_hist_image=true`일 때)

## 3. 현재 처리 구조

1. subscriber는 최신 프레임 한 장만 유지한다.
2. `_process_latest_frame()`이 `max_fps` 주기로 최신 프레임만 처리한다.
3. `_detect_boxes()`로 detection window 안에서 YOLO 후보를 찾는다.
4. `_select_box()`로 대표 박스를 고른다.
5. ROI를 morphology 기반으로 보정한다.
6. `hue_hist_and_scores()`로 `r_score`, `y_score`, `g_score`를 계산한다.
7. EMA, 다수결, hysteresis, hold time으로 최종 상태를 안정화한다.

## 4. 현재 상태 판정 요약

- 빨강/노랑/초록 중 EMA가 가장 높은 색을 기본 상태로 사용한다.
- `lt_rule_enable=true`이고 빨강/초록 조합 조건이 맞으면 `LEFT ARROW`를 사용한다.
- 박스가 잠깐 사라져도 `missing_timeout_ms` 이내면 직전 상태를 잠시 유지한다.

## 5. launch 특징

- `tl_roi_hist.launch.py`는 노드만 실행하며 bag은 별도로 재생한다.
- `start_offset`으로 bag 시작 지점을 조절할 수 있다.
- `max_fps`와 `pub_hist_image`를 launch 인자로 바로 바꿀 수 있다.

실행:

```bash
ros2 launch mando_tools tl_roi_hist.launch.py
```
