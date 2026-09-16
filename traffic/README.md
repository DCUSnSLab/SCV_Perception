# traffic

파노라마의 상단 40%, 중앙 40%만 `best.pt`에 넣는 빠른 차량 신호등 탐지 노드다.
입력 큐는 한 장만 유지해 느린 프레임이 쌓이지 않는다.

```bash
cd /home/ssc/SSC
source /opt/ros/humble/setup.bash
colcon build --packages-select traffic --symlink-install
source install/setup.bash
ros2 launch traffic traffic.launch.py
```

- 입력: `/panorama/image_raw`
- 상태: `/traffic/state`
- RViz 이미지: `/traffic/annotated`

RViz의 `Image` 디스플레이에서 `/traffic/annotated`를 선택하면 ROI와 색상별 박스만 보인다.
