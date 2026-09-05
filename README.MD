# SCV Perception

SSC 자율주행 스택에서 사용하는 ROS 2 기반 인지 패키지 모음입니다. 멀티카메라 파노라마 영상·점군과 신호등 인식을 처리합니다.

## Packages

| Package | Description |
| --- | --- |
| `panorama_stitcher` | 전방 RealSense 카메라의 실시간 RGB-D 파노라마 및 점군 생성 |
| `traffic_light/mando_tools` | 파노라마 영상 기반 신호등 상태 판정 도구 |

## Requirements

- Ubuntu 22.04
- ROS 2 Humble
- Python 3, OpenCV, NumPy
- CUDA 지원 GPU 권장

## Documentation

- [Panorama stitcher guide](panorama_stitcher/README.md)
- [Traffic light guide](traffic_light/README)

## License

Each package is distributed under the Apache-2.0 license unless stated otherwise.
