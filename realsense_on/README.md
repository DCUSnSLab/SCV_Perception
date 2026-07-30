# realsense_on

ROS2 launch package for the Intel RealSense front, left, and right cameras in the SCV workspace.

Default mapping in this workspace:

- `realsense_on/d555_front` -> front
- `realsense_1/d435i_right` -> right
- `realsense_2/d435i_left` -> left

Default serial numbers:

- front D555: `419222301550`
- right D435i: `233522076130`
- left D435i: `327122078834`

Default USB port IDs:

- front D555: `2-8`
- right D435i: `4-2.1`
- left D435i: `4-2.2`

Launch:

```bash
colcon build --packages-select realsense_on
source install/setup.bash
ros2 launch realsense_on realsense_on.launch.py
```

Launch all three cameras:

```bash
colcon build --packages-select realsense_on
source install/setup.bash
ros2 launch realsense_on realsense_on_all.launch.py
```

Override the serial number if the front camera changes:

```bash
ros2 launch realsense_on realsense_on.launch.py \
  serial_no:="'419222301550'" \
  usb_port_id:="'2-8'"
```

Override any serial when launching all cameras:

```bash
ros2 launch realsense_on realsense_on_all.launch.py \
  front_serial_no:="'419222301550'" \
  right_serial_no:="'233522076130'" \
  left_serial_no:="'327122078834'" \
  front_usb_port_id:="'2-8'" \
  right_usb_port_id:="'4-2.1'" \
  left_usb_port_id:="'4-2.2'"
```

The package now locks each camera by both `serial_no` and `usb_port_id`.
If you unplug and reconnect to the same physical USB port, it will keep the original mapping.
If you move a camera to a different port, launch will fail instead of silently swapping left/right/front.
