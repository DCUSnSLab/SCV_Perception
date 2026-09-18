# tl_fusion2

`tl_fusion2`는 기존 `tl_fusion`과 별도 실행 파일·노드 이름으로 제공되는 traffic-light fusion 노드다.
기존 `tl_fusion` 소스와 합치지 않고 `tl_fusion2` 브랜치에서 관리한다.

## 실행

```bash
source /opt/ros/humble/setup.bash
source /home/ki/SSC/install/setup.bash

ros2 launch mando_tools tl_fusion2.launch.py \
  use_sim_time:=true \
  publish_debug_image:=true \
  detector_image_size:=960
```

실행 파일은 `mando_tl_fusion2`, 노드 이름은 `/tl_fusion2`다.
입력은 `/panorama/image_raw`이며 출력 토픽은 기존 계약과 동일하게
`/tl/state_id`, `/tl/detections`, `/tl/debug_image`를 사용한다.
