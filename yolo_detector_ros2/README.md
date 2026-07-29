# yolo_detector_ros2

ROS2 YOLOv8 detector for vehicles and pedestrians.

Input:

- `sensor_msgs/msg/CompressedImage` by default
- raw `sensor_msgs/msg/Image` is still supported with `use_compressed_image:=false`

Outputs:

- `perception_interface/msg/DetectionArray`
- annotated `sensor_msgs/msg/Image`

Default filtered classes:

- `person`
- `car`
- `bus`
- `truck`
- `motorcycle`

Build:

```bash
cd ~/SCV
colcon build --packages-select perception_interface yolo_detector_ros2
source install/setup.bash
```

Run:

```bash
ros2 launch yolo_detector_ros2 yolo_detector.launch.py \
  image_topic:=/realsense_1/color/image_raw/compressed
```

Run both side cameras:

```bash
ros2 launch yolo_detector_ros2 yolo_detector_dual.launch.py
```

Default side mapping in this workspace:

- `realsense_1/d435i_right` -> right
- `realsense_2/d435i_left` -> left

Depth input remains the raw aligned depth topic:

- `/realsense_1/d435i_right/aligned_depth_to_color/image_raw`
- `/realsense_2/d435i_left/aligned_depth_to_color/image_raw`

Requirements in the runtime environment:

- `torch`
- `ultralytics`
- `opencv-python`
