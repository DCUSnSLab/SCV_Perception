# Traffic light — `tl_fusion` / `black_box_color_bits`

ROS 2 패키지 이름은 `mando_tools`다. 실제 사용하는 두 노드는 다음과 같다.

| 노드 | 역할 | 주요 출력 | 기본 모델 |
| --- | --- | --- | --- |
| `tl_fusion` | YOLO + HSV 색상 분석으로 차량 신호등 상태 판정 | `/tl/state_id` | `model/best.pt` |
| `black_box_color_bits` | YOLO 디스플레이 박스 내부의 빨강/초록 비트 판정 | `/tl/box_color_bits` | `model/box_best.pt` |

**[실행 명령어와 전체 런치 옵션 → docs/RUN.md](docs/RUN.md)**

두 노드는 기본적으로 `/panorama/image_raw`를 구독하며 각각 최대 `5 FPS`로 처리한다.
모델 가중치, bag, `.deps`, 빌드 결과는 Git에 포함하지 않는다.

## 빠른 실행

워크스페이스가 `~/SSC`에 설치되어 있다고 가정한다. 각 터미널에서 환경을 불러온다.

```bash
source /opt/ros/humble/setup.bash
source "$HOME/SSC/install/setup.bash"
export MANDO_WS="$HOME/SSC/src/perception/traffic_light"
```

터미널 1 — 신호등 상태:

```bash
ros2 launch mando_tools tl_fusion.launch.py \
  model_path:="$MANDO_WS/model/best.pt"
```

터미널 2 — 디스플레이 비트:

```bash
ros2 launch mando_tools black_box_color_bits.launch.py
```

실차 운행 시에는 [디버그 이미지 완전 비활성화 명령](docs/RUN.md#실차-운행용-디버그-이미지-완전-비활성화)을 사용한다.

rosbag 재생은 [시뮬레이션 시간 설정](docs/RUN.md#rosbag-재생)을 함께 적용한다.

## 파일 구성

```text
traffic_light/
├── README.md
├── docs/
│   ├── RUN.md                      # 두 노드 실행 명령 및 옵션
│   ├── tl_fusion.md                # 신호등 판정 구조
│   ├── realtime.md                 # 시간 검증 및 상태 안정화
│   ├── experimental/              # 보조 실험 노드 문서
│   └── archive/                   # 이전 문서 보관본
├── mando_tools/
│   ├── tl_fusion.py                # 사용: 신호등 상태 노드
│   ├── black_box_color_bits.py     # 사용: 비트 노드 + 추적/EMA
│   ├── yolo_box_color_bits.py      # 사용: black_box의 YOLO 검출 구현
│   ├── workspace_paths.py         # 공통 경로 및 장치 탐색
│   ├── bag_cli.py                 # bag 재생 도구
│   ├── panorama_resize.py         # bag 영상 리사이즈 도구
│   ├── workspace_info.py          # 환경 확인 도구
│   └── experimental/             # green_down_arrow, tl_roi_hist, yolo_validator
├── launch/
│   ├── tl_fusion.launch.py
│   ├── black_box_color_bits.launch.py
│   ├── traffic_light.launch.py    # 기존 스크립트용 tl_fusion 런치
│   ├── play_mando_bag.launch.py    # bag 재생 보조 런치
│   └── experimental/             # 실험 노드 런치
├── test/                         # 두 노드와 공통 도구 회귀 테스트
├── model/                        # best.pt, box_best.pt를 별도로 준비
├── data/bags/                    # 입력 bag을 별도로 준비
├── run_traffic_light.sh          # 기존 tl_fusion 실행 스크립트
└── setup.py                      # ROS 실행 진입점과 런치 설치 목록
```

`yolo_box_color_bits.py`는 `black_box_color_bits.py`가 직접 불러오는 실제 실행 코드다.
파일명만 보고 제거하면 black_box 노드가 시작되지 않는다.

실험용 코드는 `mando_tools/experimental/`로, 해당 런치는 `launch/experimental/`로
분리했다. 패키지를 다시 빌드하면 기존 `ros2 run` 실행 이름과
`ros2 launch mando_tools green_down_arrow.launch.py`, `tl_roi_hist.launch.py`,
`validate_mando_bag.launch.py` 명령은 유지된다. 두 운영 노드는 이 실험 모듈을 불러오지 않는다.

## 문서

- [두 노드 실행 명령·옵션·토픽 확인](docs/RUN.md)
- [tl_fusion 판정 구조](docs/tl_fusion.md)
- [tl_fusion 시간 검증과 안정화](docs/realtime.md)
- [ROI 히스토그램 실험](docs/experimental/tl_roi_hist.md)
- [YOLO 모델 검증 실험](docs/experimental/yolo_validator.md)
- [이전 README](docs/archive/previous_readme.md)
- [이전 신호등 노드 비교](docs/archive/traffic_light_detection_code_guide.md)
- [이전 tl_fusion 요약본](docs/archive/tl_fusion_summary.md)
