# 신호등 코드 문서 모음

이 폴더는 `mando_tools` 안의 신호등 관련 노드 문서를 요약본 형태로 모아둔 곳이다.

문서 목록:

- [yolo_validator.md](./yolo_validator.md)
- [tl_roi_hist.md](./tl_roi_hist.md)
- [tl_fusion.md](./tl_fusion.md)

상세 비교 문서:

- `src/mando_tools/mando_tools/traffic_light_detection_code_guide.md`

상세 단일 문서:

- `src/mando_tools/mando_tools/tl_fusion.md`

공통 상태 ID:

| ID | 의미 |
| --- | --- |
| `0` | `UNKNOWN` |
| `1` | `RED` |
| `2` | `YELLOW` |
| `3` | `GREEN` |
| `4` | `LEFT ARROW` |

현재 코드 기준으로:

- `yolo_validator.py`는 모델 클래스 검증용
- `tl_roi_hist.py`는 ROI 색 분포 기반 상태 판정용
- `tl_fusion.py`는 최종 상태 산출용
