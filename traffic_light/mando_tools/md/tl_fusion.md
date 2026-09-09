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
- 디버그 영상과 색상 하이라이트는 구독자가 있거나 `show_windows=true`일 때만 만든다.
- `/tl/debug_image` 메시지 변환과 발행은 구독자가 있을 때만 수행한다. 창만 켜면 화면에만 표시하며, 창과 구독자를 함께 사용할 때는 같은 영상을 재사용한다.
- 후보가 없거나 고신뢰 모델 판정으로 색 분석을 생략하면 색상 inset용 빈 이미지를 만들지 않는다.
- 검출 박스는 한 번에 CPU로 가져온 뒤 기존 순서와 필터 조건대로 처리한다.
- 빨강+초록 조합이 확정되면 판정에 사용되지 않는 연결요소 분석을 생략한다. 모델, 판정 임계값, FPS, 상태 안정화와 타임아웃 규칙은 유지한다.

## 5. launch에서 자주 조절하는 인자

- `image_topic`
- `max_fps`
- `color_fallback_device`
- `fallback_max_side_px`
- `detector_conf_threshold`
- `detector_image_size`
- `model_confidence_threshold`
- `enable_low_confidence_color_fallback`
- `fallback_score_threshold`

실행:

```bash
ros2 launch mando_tools tl_fusion.launch.py
```

## 6. GPU 런타임

- 패키지 로컬 `.deps`에 `torch==2.10.0+cu128`, `torchvision==0.25.0+cu128`과 필요한 CUDA 12.8 런타임 라이브러리를 설치했다. 버전 조합은 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/previous-versions/)를 따른다.
- NumPy는 ROS 영상 변환 환경과의 호환성을 위해 기존 `1.26.4`를 유지한다. 모델, FPS와 판정 임계값은 변경하지 않는다.
- 노드는 로컬 `.deps`를 우선 사용하므로, 다른 Python 환경에서 확인한 PyTorch 버전과 다를 수 있다.
- `detector_device=cuda:0`이어도 CUDA를 사용할 수 없으면 기존 장치 선택 로직이 CPU로 전환한다. 아래 명령으로 노드가 사용하는 패키지의 CUDA 지원 여부를 확인한다.

```bash
source /opt/ros/humble/setup.bash
source /home/ki/SSC/install/setup.bash
export MANDO_WS=/home/ki/SSC/src/perception/traffic_light
PYTHONPATH="$MANDO_WS/.deps${PYTHONPATH:+:$PYTHONPATH}" python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
ros2 launch mando_tools tl_fusion.launch.py \\
  detector_device:=cuda:0 \\
  color_fallback_device:=cuda:0
```

저신뢰 모델 프레임에서 CPU 색상 fallback을 생략하는 성능 프로파일은 다음과 같이 실행한다. 기본값은 `true`이며, 이 옵션은 정확도 확인 후 사용한다.

```bash
/home/ki/SSC/src/perception/traffic_light/run_traffic_light.sh enable_low_confidence_color_fallback:=false
```

의존성 교체 후에는 실행 중이던 노드를 재시작해야 새 PyTorch가 로드된다.
