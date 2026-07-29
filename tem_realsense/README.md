# tem_realsense

ROS2 launch package for two Intel RealSense D435i devices in the SCV workspace.

Default side mapping in this workspace:

- `realsense_1/d435i_right` -> right
- `realsense_2/d435i_left` -> left

Detected serial numbers on this machine:

- `233522076130`
- `327122078834`

Launch:

```bash
colcon build --packages-select tem_realsense
source install/setup.bash
ros2 launch tem_realsense tem_realsense.launch.py
```

Override serial numbers if camera assignments change:

```bash
ros2 launch tem_realsense tem_realsense.launch.py \
  serial_no_1:="'233522076130'" \
  serial_no_2:="'327122078834'"
```
