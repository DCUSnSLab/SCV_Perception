from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from pathlib import Path

from mando_tools.workspace_paths import default_runtime_image_topic


def _default_tl_model() -> str:
    launch_file = Path(__file__).resolve()
    for root in [launch_file.parent, *launch_file.parents]:
        candidate = root / 'models' / 'best.pt'
        if candidate.exists():
            return str(candidate)
        if root.name == 'traffic_light':
            return str(candidate)
    return 'best.pt'

def generate_launch_description() -> LaunchDescription:
    model_arg = DeclareLaunchArgument(
        'model_path',
        default_value=_default_tl_model(),
        description='YOLO detector model for traffic-light ROI extraction.',
    )
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
    fps_arg = DeclareLaunchArgument(
        'max_fps',
        default_value='15.0',
        description='Maximum processed frames per second.',
    )
    detector_device_arg = DeclareLaunchArgument(
        'detector_device',
        default_value='cuda:0',
        description='Inference device for YOLO. Examples: cuda:0, cpu, auto.',
    )
    pub_hist_arg = DeclareLaunchArgument(
        'pub_hist_image',
        default_value='true',
        description='Publish hue histogram image on /tl/hist_image.',
    )

    tl_roi_hist = Node(
        package='mando_tools',
        executable='mando_tl_roi_hist',
        name='tl_roi_hist',
        output='screen',
        parameters=[
            {
                'model_path': LaunchConfiguration('model_path'),
                'image_topic': LaunchConfiguration('image_topic'),
                'show_windows': ParameterValue(
                    LaunchConfiguration('show_windows'),
                    value_type=bool,
                ),
                'max_fps': ParameterValue(
                    LaunchConfiguration('max_fps'),
                    value_type=float,
                ),
                'detector_device': LaunchConfiguration('detector_device'),
                'pub_hist_image': ParameterValue(
                    LaunchConfiguration('pub_hist_image'),
                    value_type=bool,
                ),
            }
        ],
    )

    return LaunchDescription(
        [
            model_arg,
            image_topic_arg,
            show_windows_arg,
            fps_arg,
            detector_device_arg,
            pub_hist_arg,
            tl_roi_hist,
        ]
    )
