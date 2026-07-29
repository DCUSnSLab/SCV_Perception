# crosswalk_safety_ros2

ROS2 safety node that consumes left/right YOLO detections with depth-based distance estimates.

Outputs:

- `/crosswalk/safety_state` (`std_msgs/String`): `SAFE`, `CAUTION`, `UNSAFE`
- `/crosswalk/unsafe` (`std_msgs/Bool`)
- `/crosswalk/min_ttc` (`std_msgs/Float32`)

Build:

```bash
cd ~/SCV
colcon build --packages-select perception_interface yolo_detector_ros2 crosswalk_safety_ros2
source install/setup.bash
```

Run full perception chain:

```bash
ros2 launch crosswalk_safety_ros2 crosswalk_perception.launch.py
```

This launch file starts:

- both RealSense cameras from `tem_realsense`
- both YOLO detector nodes
- the crosswalk safety node
