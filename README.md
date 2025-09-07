##execute command
ros2 run memsort_ros2 memory_sort_node   --ros-args   -p image_topic:="/zed/zed_node/left/image_rect_color"   -p show_window:=true   -p save_path:="/home/jay/output1.mp4"   -p weights:="yolo11s-seg.pt"   -p det:="v11seg"   -p classes:="bicycle"   -p overlay_mask:=true   -p output_masks:=true
