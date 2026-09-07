from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from mando_tools.workspace_paths import default_model_path


def _default_tl_model() -> str:
    """Resolve the model from either a standalone or SSC-nested checkout."""
    try:
        return str(default_model_path())
    except RuntimeError:
        # Preserve a useful node-level FileNotFoundError when the source tree
        # really is unavailable instead of failing while parsing the launch.
        return 'best.pt'


def generate_launch_description() -> LaunchDescription:
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
    node_type_gate_arg = DeclareLaunchArgument(
        'node_type_gate_enabled',
        default_value='true',
        description=(
            'Publish continuously only while current_goal_node_type is the '
            'traffic-light type; disable for standalone detector tests.'
        ),
    )
    waypoint_topic_arg = DeclareLaunchArgument(
        'waypoint_topic',
        default_value='/multiple_waypoints',
        description='Waypoint topic that carries current_goal_node_type.',
    )
    traffic_light_node_type_arg = DeclareLaunchArgument(
        'traffic_light_node_type',
        default_value='10',
        description='Map node type that enables traffic-light state output.',
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
    fallback_score_arg = DeclareLaunchArgument(
        'fallback_score_threshold',
        default_value='0.50',
        description='Normalized color score required for the fallback state.',
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
                'node_type_gate_enabled': ParameterValue(
                    LaunchConfiguration('node_type_gate_enabled'),
                    value_type=bool,
                ),
                'waypoint_topic': LaunchConfiguration('waypoint_topic'),
                'traffic_light_node_type': ParameterValue(
                    LaunchConfiguration('traffic_light_node_type'),
                    value_type=int,
                ),
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
                'fallback_score_threshold': ParameterValue(
                    LaunchConfiguration('fallback_score_threshold'),
                    value_type=float,
                ),
            }
        ],
    )

    return LaunchDescription(
        [
            model_arg,
            image_topic_arg,
            state_topic_arg,
            node_type_gate_arg,
            waypoint_topic_arg,
            traffic_light_node_type_arg,
            input_timeout_arg,
            show_windows_arg,
            fps_arg,
            detector_device_arg,
            detector_conf_arg,
            detector_size_arg,
            model_conf_arg,
            fallback_score_arg,
            tl_fusion,
        ]
    )
