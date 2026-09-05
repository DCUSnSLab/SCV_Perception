# SCV Perception

SSC 자율주행 스택에서 사용하는 ROS 2 기반 인지 패키지 모음입니다. 멀티카메라 파노라마 영상과 점군을 처리합니다.

## Packages

| Package | Description |
| --- | --- |
| `panorama_stitcher` | 전방 RealSense 카메라의 실시간 RGB-D 파노라마 및 점군 생성 |

## Requirements

- Ubuntu 22.04
- ROS 2 Humble
- Python 3, OpenCV, NumPy
- CUDA 지원 GPU 권장

## Documentation

- [Panorama stitcher guide](panorama_stitcher/README.md)

## License

Each package is distributed under the Apache-2.0 license unless stated otherwise.
