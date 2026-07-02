from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('tl_roi_hist')
    default_params = os.path.join(pkg_share, 'config', 'tl_crop_only.param.yaml')

    model_arg = DeclareLaunchArgument('model_path', default_value='yolo11s.pt')
    topic_arg = DeclareLaunchArgument('image_topic', default_value='/zed/zed_node/left/image_rect_color')
    params_arg = DeclareLaunchArgument('params_file', default_value=default_params)

    node = Node(
        package='tl_roi_hist',
        executable='tl_crop_only',
        name='tl_crop_only',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {'model_path': LaunchConfiguration('model_path')},
            {'image_topic': LaunchConfiguration('image_topic')},
        ],
    )

    return LaunchDescription([
        model_arg,
        topic_arg,
        params_arg,
        node
    ])
