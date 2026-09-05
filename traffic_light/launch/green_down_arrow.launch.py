from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from mando_tools.workspace_paths import default_runtime_image_topic


def generate_launch_description() -> LaunchDescription:
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value=default_runtime_image_topic(),
        description='Input image topic.',
    )
    show_windows_arg = DeclareLaunchArgument(
        'show_windows',
        default_value='false',
        description='Show OpenCV debug windows.',
    )
    max_fps_arg = DeclareLaunchArgument(
        'max_fps',
        default_value='15.0',
        description='Maximum processed frames per second.',
    )
    publish_debug_arg = DeclareLaunchArgument(
        'publish_debug_image',
        default_value='true',
        description='Publish annotated debug image.',
    )

    detector_node = Node(
        package='mando_tools',
        executable='mando_green_down_arrow',
        name='green_down_arrow_detector',
        output='screen',
        parameters=[
            {
                'image_topic': LaunchConfiguration('image_topic'),
                'show_windows': ParameterValue(
                    LaunchConfiguration('show_windows'),
                    value_type=bool,
                ),
                'max_fps': ParameterValue(
                    LaunchConfiguration('max_fps'),
                    value_type=float,
                ),
                'publish_debug_image': ParameterValue(
                    LaunchConfiguration('publish_debug_image'),
                    value_type=bool,
                ),
            }
        ],
    )

    return LaunchDescription(
        [
            image_topic_arg,
            show_windows_arg,
            max_fps_arg,
            publish_debug_arg,
            detector_node,
        ]
    )
