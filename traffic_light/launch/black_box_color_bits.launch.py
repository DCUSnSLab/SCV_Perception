from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from mando_tools.workspace_paths import workspace_root_or_none


def generate_launch_description() -> LaunchDescription:
    root = workspace_root_or_none()
    model_default = str(root / 'model' / 'box_best.pt') if root is not None else 'box_best.pt'
    model_args = [
        DeclareLaunchArgument('max_image_age_ms', default_value='500.0'),
        DeclareLaunchArgument('input_timeout_s', default_value='0.5'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('box_confidence', default_value='0.25'),
        DeclareLaunchArgument('box_image_size', default_value='640'),
    ]
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/panorama/image_raw',
        description='Input panorama image topic.',
    )
    bits_topic_arg = DeclareLaunchArgument(
        'bits_topic',
        default_value='/tl/box_color_bits',
        description='Output topic for left-to-right red/green bits.',
    )
    debug_topic_arg = DeclareLaunchArgument(
        'debug_image_topic',
        default_value='/tl/box_color_bits/debug',
        description='Annotated debug image topic.',
    )
    publish_debug_arg = DeclareLaunchArgument(
        'publish_debug_image',
        default_value='false',
        description='Publish the annotated debug image.',
    )
    fps_arg = DeclareLaunchArgument(
        'max_fps',
        default_value='15.0',
        description='Maximum number of processed frames per second.',
    )
    roi_bottom_arg = DeclareLaunchArgument(
        'roi_bottom_ratio',
        default_value='0.5',
        description='Bottom of the upper ROI as a frame-height ratio.',
    )
    roi_top_arg = DeclareLaunchArgument('roi_top_ratio', default_value='0.0')
    roi_left_arg = DeclareLaunchArgument('roi_left_ratio', default_value='0.25')
    roi_right_arg = DeclareLaunchArgument('roi_right_ratio', default_value='0.75')
    color_s_min_arg = DeclareLaunchArgument(
        'color_s_min',
        default_value='80',
        description='Minimum HSV saturation for a color pixel.',
    )
    color_v_min_arg = DeclareLaunchArgument(
        'color_v_min',
        default_value='45',
        description='Minimum HSV value for a color pixel.',
    )
    score_threshold_arg = DeclareLaunchArgument(
        'color_score_threshold',
        default_value='0.04',
        description='Minimum EMA color score for a valid color.',
    )
    hysteresis_arg = DeclareLaunchArgument(
        'color_hysteresis_delta',
        default_value='0.08',
        description='Minimum red/green score gap for switching a bit.',
    )
    hold_timeout_arg = DeclareLaunchArgument(
        'hold_timeout_s',
        default_value='0.5',
        description='How long a missing box keeps its last bit.',
    )

    detector = Node(
        package='mando_tools',
        executable='mando_black_box_color_bits',
        name='black_box_color_bits',
        output='screen',
        parameters=[
            {
                'image_topic': LaunchConfiguration('image_topic'),
                'max_image_age_ms': ParameterValue(LaunchConfiguration('max_image_age_ms'), value_type=float),
                'input_timeout_s': ParameterValue(LaunchConfiguration('input_timeout_s'), value_type=float),
                'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
                'box_model_path': model_default,
                'box_device': 'cuda:0',
                'box_confidence': ParameterValue(LaunchConfiguration('box_confidence'), value_type=float),
                'box_image_size': ParameterValue(LaunchConfiguration('box_image_size'), value_type=int),
                'opencv_threads': ParameterValue(1, value_type=int),
                'bits_topic': LaunchConfiguration('bits_topic'),
                'debug_image_topic': LaunchConfiguration('debug_image_topic'),
                'publish_debug_image': ParameterValue(
                    LaunchConfiguration('publish_debug_image'),
                    value_type=bool,
                ),
                'max_fps': ParameterValue(LaunchConfiguration('max_fps'), value_type=float),
                'roi_bottom_ratio': ParameterValue(
                    LaunchConfiguration('roi_bottom_ratio'),
                    value_type=float,
                ),
                'roi_top_ratio': ParameterValue(LaunchConfiguration('roi_top_ratio'), value_type=float),
                'roi_left_ratio': ParameterValue(LaunchConfiguration('roi_left_ratio'), value_type=float),
                'roi_right_ratio': ParameterValue(LaunchConfiguration('roi_right_ratio'), value_type=float),
                'color_s_min': ParameterValue(
                    LaunchConfiguration('color_s_min'),
                    value_type=int,
                ),
                'color_v_min': ParameterValue(
                    LaunchConfiguration('color_v_min'),
                    value_type=int,
                ),
                'color_score_threshold': ParameterValue(
                    LaunchConfiguration('color_score_threshold'),
                    value_type=float,
                ),
                'color_hysteresis_delta': ParameterValue(
                    LaunchConfiguration('color_hysteresis_delta'),
                    value_type=float,
                ),
                'hold_timeout_s': ParameterValue(
                    LaunchConfiguration('hold_timeout_s'),
                    value_type=float,
                ),
            }
        ],
    )

    return LaunchDescription(
        [
            image_topic_arg,
            *model_args,
            bits_topic_arg,
            debug_topic_arg,
            publish_debug_arg,
            fps_arg,
            roi_bottom_arg,
            roi_top_arg,
            roi_left_arg,
            roi_right_arg,
            color_s_min_arg,
            color_v_min_arg,
            score_threshold_arg,
            hysteresis_arg,
            hold_timeout_arg,
            detector,
        ]
    )
