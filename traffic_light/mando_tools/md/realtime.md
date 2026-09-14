# tl_fusion 실시간 입력 및 상태 안정화

수신과 추론을 별도 MutuallyExclusiveCallbackGroup 및 2스레드 executor로 실행한다.
수신 QoS는 Best Effort, Keep Last 1이다. 수신 콜백은 락으로 보호한 최신 슬롯만
덮어쓰며 추론 타이머는 슬롯을 가져온 후 락을 해제한다. 모델·판정 상태는 추론
그룹에서만 변경한다. Python GIL 또는 운영체제 스케줄링까지 실시간으로 보장하지는 않는다.

| 파라미터 | 기본값 | 의미 |
|---|---:|---|
| max_image_age_ms | 250 | 촬영 header stamp 이후 허용 경과 시간; 0은 나이 검사 비활성화 |
| future_stamp_tolerance_ms | 50 | 미래 타임스탬프 허용 오차 |
| require_image_header_stamp | true | 0/음수 타임스탬프 거부 |
| state_confirm_ms | 200 | 동일 제안 상태가 지속 관측되어야 하는 시간 |
| state_max_gap_ms | 250 | 이보다 긴 관측 공백은 확인 시간을 다시 시작 |
| hold_ms | 250 | 직전 상태 전환 이후 최소 유지 시간, 기존 동작 유지 |

촬영 타임스탬프를 ROS 시계와 비교해 추론 전, 판정 안정화 전, 발행 직전에 검사한다.
나이 초과·미래 시각·잘못된 타임스탬프는 UNKNOWN 및 빈 detections 배열을 발행하고
미완료 상태 확인을 초기화한다. 나이 검사가 켜져 있으면 require_image_header_stamp=false라도
0 stamp를 거부한다. 과거 stamp 역행 검사도 유지한다.

state_window_size에 의한 프레임 다수결은 사용하지 않는다. 상태가 바뀌거나 관측 공백이
state_max_gap_ms를 넘으면 확인 시간을 다시 시작한다. 실제 전환은 확인 시간과 hold_ms를
모두 만족하는 다음 관측 프레임에서 발생하므로 프레임 간격만큼 양자화된다.
짧은 미검출 유지(missing_timeout_ms)와 추적 초기화(reset_tracking_ms)는 유지한다.
ROS 시각이 역행하면 상태 확인을 다시 시작하고 기존 확정 상태를 UNKNOWN으로 초기화한다.

```bash
ros2 launch mando_tools tl_fusion.launch.py \
  max_image_age_ms:=250.0 future_stamp_tolerance_ms:=50.0 \
  state_confirm_ms:=200.0 state_max_gap_ms:=250.0
```

촬영 장치와 노드는 동기화된 시간 기준을 사용해야 한다. 과거 bag은 ROS clock을 재생하고
노드에서 use_sim_time=true를 설정해야 한다. 새 나이 검사 때문에 과거 stamp를 현재 wall
clock과 그대로 비교하면 입력이 거부된다. 검증용 재생만 나이 검사를 명시적으로 끌 수 있다.
카메라 타임스탬프가 센서 내부 부팅 기준이면 먼저 ROS 시간 기준으로 변환해야 한다.

위 기본값은 초기 조정값이며 차량 제동·계획 요구사항을 검증한 값은 아니다. 기존
input_timeout_s=3.0 및 독립 watchdog 부재는 이번 변경 범위에 포함되지 않는다.
max_fps=15와 크롭 ROI(x=25~75%, y=0~1/3)는 유지한다.
