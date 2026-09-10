from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from pathlib import Path
import os

def _default_tl_model() -> str:
    # Installed launch files live outside the source tree, so the walk-up
    # search below finds nothing; MANDO_WS pins the package directory.
    search_roots: list[Path] = []
    for env_name in ('MANDO_WS', 'MANDO_WORKSPACE'):
        env_root = os.environ.get(env_name)
        if env_root:
            search_roots.append(Path(env_root).expanduser())

    launch_file = Path(__file__).resolve()
    search_roots.extend([launch_file.parent, *launch_file.parents])

    for root in search_roots:
        candidate = root / 'model' / 'best.pt'
        if candidate.exists():
            return str(candidate)
        if root.name == 'traffic_light':
            return str(candidate)
    return 'best.pt'

def generate_launch_description() -> LaunchDescription:
    roi_defaults = {
        'detect_top_ratio': 0.0, 'detect_bottom_ratio': 1.0 / 3.0,
        'detect_left_ratio': 0.25, 'detect_right_ratio': 0.75,
    }
    roi_args = [
        DeclareLaunchArgument(name, default_value=str(value), description='Detection ROI boundary in full-image coordinates (0-1).')
        for name, value in roi_defaults.items()
    ]
    model_arg = DeclareLaunchArgument(
        'model_path',
        default_value=_default_tl_model(),
        description='YOLO detector model for traffic-light fusion.',
    )
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/panorama/image_raw',
        description='Input image topic.',
    )
    state_topic_arg = DeclareLaunchArgument(
        'state_topic',
        default_value='/tl/state_id',
        description='Final traffic-light state topic consumed by Behavior Planner.',
    )
    input_timeout_arg = DeclareLaunchArgument(
        'input_timeout_s',
        default_value='3.0',
        description='Publish UNKNOWN when no image arrives for this many seconds.',
    )
    show_windows_arg = DeclareLaunchArgument(
        'show_windows',
        default_value='false',
        description='Show OpenCV debug windows.',
    )
    fps_arg = DeclareLaunchArgument(
        'max_fps',
        default_value='15.0',
        description='Maximum fusion frames processed per second.',
    )
    detector_device_arg = DeclareLaunchArgument(
        'detector_device',
        default_value='cuda:0',
        description='Inference device for YOLO. Examples: cuda:0, cpu, auto.',
    )
    color_fallback_device_arg = DeclareLaunchArgument(
        'color_fallback_device',
        default_value='auto',
        description='PyTorch device for color fallback. auto follows detector_device.',
    )
    detector_conf_arg = DeclareLaunchArgument(
        'detector_conf_threshold',
        default_value='0.10',
        description='Minimum YOLO confidence for signal candidates.',
    )
    detector_size_arg = DeclareLaunchArgument(
        'detector_image_size',
        default_value='640',
        description='Inference image size for YOLO.',
    )
    model_conf_arg = DeclareLaunchArgument(
        'model_confidence_threshold',
        default_value='0.60',
        description='Confidence above which the model state is trusted directly.',
    )
    low_conf_fallback_arg = DeclareLaunchArgument(
        'enable_low_confidence_color_fallback',
        default_value='true',
        description=(
            'Recheck low-confidence resolved detections with color analysis. '
            'Disable for the real-time performance profile.'
        ),
    )
    fallback_score_arg = DeclareLaunchArgument(
        'fallback_score_threshold',
        default_value='0.50',
        description='Normalized color score required for the fallback state.',
    )
    fallback_max_side_arg = DeclareLaunchArgument(
        'fallback_max_side_px',
        default_value='640',
        description='Maximum fallback ROI side before color preprocessing resize.',
    )
    tl_fusion = Node(
        package='mando_tools',
        executable='mando_tl_fusion',
        name='tl_fusion',
        output='screen',
        parameters=[
            {
                'model_path': LaunchConfiguration('model_path'),
                'image_topic': LaunchConfiguration('image_topic'),
                'state_topic': LaunchConfiguration('state_topic'),
                'input_timeout_s': ParameterValue(
                    LaunchConfiguration('input_timeout_s'),
                    value_type=float,
                ),
                'show_windows': ParameterValue(
                    LaunchConfiguration('show_windows'),
                    value_type=bool,
                ),
                'max_fps': ParameterValue(
                    LaunchConfiguration('max_fps'),
                    value_type=float,
                ),
                'detector_device': LaunchConfiguration('detector_device'),
                **{name: ParameterValue(LaunchConfiguration(name), value_type=float) for name in roi_defaults},
                'color_fallback_device': LaunchConfiguration('color_fallback_device'),
                'detector_conf_threshold': ParameterValue(
                    LaunchConfiguration('detector_conf_threshold'),
                    value_type=float,
                ),
                'detector_image_size': ParameterValue(
                    LaunchConfiguration('detector_image_size'),
                    value_type=int,
                ),
                'model_confidence_threshold': ParameterValue(
                    LaunchConfiguration('model_confidence_threshold'),
                    value_type=float,
                ),
                'enable_low_confidence_color_fallback': ParameterValue(
                    LaunchConfiguration('enable_low_confidence_color_fallback'),
                    value_type=bool,
                ),
                'fallback_score_threshold': ParameterValue(
                    LaunchConfiguration('fallback_score_threshold'),
                    value_type=float,
                ),
                'fallback_max_side_px': ParameterValue(
                    LaunchConfiguration('fallback_max_side_px'),
                    value_type=int,
                ),
            }
        ],
    )

    return LaunchDescription(
        [
            model_arg,
            *roi_args,
            image_topic_arg,
            state_topic_arg,
            input_timeout_arg,
            show_windows_arg,
            fps_arg,
            detector_device_arg,
            color_fallback_device_arg,
            detector_conf_arg,
            detector_size_arg,
            model_conf_arg,
            low_conf_fallback_arg,
            fallback_score_arg,
            fallback_max_side_arg,
            tl_fusion,
        ]
    )
