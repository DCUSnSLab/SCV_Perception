# SCV Perception

현재 사용하는 기능을 ROS 2 패키지 단위로 분리한다.

| 패키지 | 역할 | 기본 실행 |
|---|---|---|
| `panorama_stitcher` | 전방 RGB-D 파노라마, 점군, 지면/장애물 분리 | `ros2 launch bring_up realsense_panorama.launch.py` |

각 패키지의 실행법과 토픽은 패키지 안의 `README.md`를 기준으로 한다.
빌드 산출물은 워크스페이스 최상위 `build/`, `install/`, `log/`에만 둔다.
