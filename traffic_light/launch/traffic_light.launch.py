from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    image_topic = LaunchConfiguration('image_topic')

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'image_topic',
                default_value='/panorama/image_raw',
                description='Shared panorama image input.',
            ),
            DeclareLaunchArgument(
                'input_timeout_s',
                default_value='3.0',
                description='Publish UNKNOWN after this input timeout.',
            ),
            DeclareLaunchArgument(
                'detector_device',
                default_value='cuda:0',
                description='YOLO inference device.',
            ),
            Node(
                package='mando_tools',
                executable='mando_tl_fusion',
                name='tl_fusion',
                output='screen',
                parameters=[
                    {
                        'image_topic': image_topic,
                        'state_topic': '/tl/state_id',
                        'input_timeout_s': ParameterValue(
                            LaunchConfiguration('input_timeout_s'),
                            value_type=float,
                        ),
                        'detector_device': LaunchConfiguration('detector_device'),
                    }
                ],
            ),
        ]
    )
