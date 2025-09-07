# MemorySORT ROS 2 Node (memsort_ros2)

본 노드는 YOLO 기반 객체 탐지 및 추적(MemorySORT)을 ROS 2 환경에서 수행하는 패키지입니다.  
YOLOv11 모델을 기반으로 객체를 탐지하고, 추적 ID를 유지하며, 이미지 마스크와 함께 결과를 시각화 및 저장할 수 있습니다.

---

## 🛠 실행 명령어

```bash
ros2 run memsort_ros2 memory_sort_node \
  --ros-args \
  -p image_topic:="/zed/zed_node/left/image_rect_color" \
  -p show_window:=true \
  -p weights:="yolo11s-seg.pt" \
  -p det:="v11seg" \
  -p classes:="bicycle" \
  -p overlay_mask:=true \
  -p output_masks:=true
