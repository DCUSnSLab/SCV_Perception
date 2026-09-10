from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    image_topic = LaunchConfiguration('image_topic')
    roi_defaults = {
        'detect_top_ratio': 0.0, 'detect_bottom_ratio': 1.0 / 3.0,
        'detect_left_ratio': 0.25, 'detect_right_ratio': 0.75,
    }

    return LaunchDescription(
        [
            *[DeclareLaunchArgument(name, default_value=str(value)) for name, value in roi_defaults.items()],
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
            DeclareLaunchArgument(
                'color_fallback_device',
                default_value='auto',
                description='PyTorch device for color fallback. auto follows detector_device.',
            ),
            DeclareLaunchArgument(
                'enable_low_confidence_color_fallback',
                default_value='true',
                description=(
                    'Recheck low-confidence resolved detections with color analysis. '
                    'Disable for the real-time performance profile.'
                ),
            ),
            DeclareLaunchArgument(
                'fallback_max_side_px',
                default_value='640',
                description='Maximum fallback ROI side before color preprocessing resize.',
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
                        **{name: ParameterValue(LaunchConfiguration(name), value_type=float) for name in roi_defaults},
                        'input_timeout_s': ParameterValue(
                            LaunchConfiguration('input_timeout_s'),
                            value_type=float,
                        ),
                        'detector_device': LaunchConfiguration('detector_device'),
                        'color_fallback_device': LaunchConfiguration('color_fallback_device'),
                        'enable_low_confidence_color_fallback': ParameterValue(
                            LaunchConfiguration('enable_low_confidence_color_fallback'),
                            value_type=bool,
                        ),
                        'fallback_max_side_px': ParameterValue(
                            LaunchConfiguration('fallback_max_side_px'),
                            value_type=int,
                        ),
                    }
                ],
            ),
        ]
    )
