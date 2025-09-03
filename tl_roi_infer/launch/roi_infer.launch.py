from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='tl_roi_infer',
            executable='roi_infer',
            name='roi_infer',
            output='screen',
            parameters=[{
                'image_topic': '/zed_node/left/image_rect_color',
                'debug_topic': '/tl/debug_image',
                'state_id_topic': '/tl/state_id',
            }],
        )
    ])


# source ~/rosenv/bin/activate
# source /opt/ros/humble/setup.bash
# source ~/tl_ws/install/setup.bash

# ros2 run tl_roi_infer roi_infer --ros-args \
#  -p image_topic:=/zed_node/left/image_rect_color \
#  -p debug_topic:=/tl/debug_image \
#  -p state_id_topic:=/tl/state_id