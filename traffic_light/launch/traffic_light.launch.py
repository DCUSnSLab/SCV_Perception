from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    image_topic = LaunchConfiguration('image_topic')
    parameter_defaults = {
        'detect_top_ratio': 0.0, 'detect_bottom_ratio': 1.0 / 3.0,
        'detect_left_ratio': 0.20, 'detect_right_ratio': 0.80,
        'max_image_age_ms': 500.0,
        'future_stamp_tolerance_ms': 50.0,
        'state_confirm_ms': 200.0,
        'state_max_gap_ms': 250.0,
        'uncertain_hold_ms': 300.0,
    }

    return LaunchDescription(
        [
            *[DeclareLaunchArgument(name, default_value=str(value)) for name, value in parameter_defaults.items()],
            DeclareLaunchArgument('use_sim_time', default_value='false'),
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
                'publish_debug_image',
                default_value='false',
                description='Publish the debug image topic.',
            ),
            DeclareLaunchArgument(
                'debug_image_max_side_px',
                default_value='640',
                description='Maximum side length of the debug image; 0 keeps the source size.',
            ),
            DeclareLaunchArgument(
                'debug_publish_period_ms',
                default_value='200.0',
                description='Minimum interval between debug renders; 0 renders every frame.',
            ),
            DeclareLaunchArgument(
                'detector_device',
                default_value='cuda:0',
                description='YOLO inference device.',
            ),
            DeclareLaunchArgument(
                'detector_conf_threshold',
                default_value='0.05',
                description='Minimum YOLO confidence for small signal candidates.',
            ),
            DeclareLaunchArgument(
                'detector_image_size',
                default_value='640',
                description='YOLO inference size for small-object recall.',
            ),
            DeclareLaunchArgument(
                'model_path',
                default_value='/home/ki/SSC/src/perception/traffic_light/model/best.pt',
                description='Fixed YOLO traffic-light model.',
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
                    'Legacy compatibility parameter. Color analysis is always enabled.'
                ),
            ),
            DeclareLaunchArgument(
                'model_confidence_threshold',
                default_value='0.75',
                description='Confidence above which the model state is trusted directly.',
            ),
            DeclareLaunchArgument(
                'fallback_score_threshold',
                default_value='0.45',
                description='Normalized color score required for the fallback state.',
            ),
            DeclareLaunchArgument(
                'fallback_green_h_min',
                default_value='39.0',
                description='Lower HSV hue bound for green fallback pixels.',
            ),
            DeclareLaunchArgument(
                'fallback_green_h_max',
                default_value='100.0',
                description='Upper HSV hue bound for green fallback pixels.',
            ),
            DeclareLaunchArgument(
                'fallback_green_s_min',
                default_value='50',
                description='Minimum HSV saturation for green fallback pixels.',
            ),
            DeclareLaunchArgument(
                'fallback_green_v_min',
                default_value='68',
                description='Minimum HSV value for green fallback pixels.',
            ),
            DeclareLaunchArgument(
                'fallback_green_score_threshold',
                default_value='0.40',
                description='Normalized score required for a green fallback state.',
            ),
            DeclareLaunchArgument(
                'fallback_green_top_weight',
                default_value='0.20',
                description='Common color score weight at the top of the candidate box.',
            ),
            DeclareLaunchArgument(
                'fallback_green_middle_weight',
                default_value='1.30',
                description='Common color score weight in the middle of the candidate box.',
            ),
            DeclareLaunchArgument(
                'fallback_green_bottom_weight',
                default_value='0.20',
                description='Common color score weight at the bottom of the candidate box.',
            ),
            DeclareLaunchArgument(
                'fallback_score_gap',
                default_value='0.10',
                description='Minimum score gap between the top two colors.',
            ),
            DeclareLaunchArgument(
                'fallback_saturation_gain',
                default_value='2.20',
                description='Saturation gain applied before color fallback analysis.',
            ),
            DeclareLaunchArgument(
                'fallback_value_gain',
                default_value='1.35',
                description='Brightness gain applied before color fallback analysis.',
            ),
            DeclareLaunchArgument(
                'fallback_gamma',
                default_value='1.00',
                description='Gamma applied before color fallback analysis.',
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
                        'model_path': LaunchConfiguration('model_path'),
                        **{name: ParameterValue(LaunchConfiguration(name), value_type=float) for name in parameter_defaults},
                        'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
                        'input_timeout_s': ParameterValue(
                            LaunchConfiguration('input_timeout_s'),
                            value_type=float,
                        ),
                        'publish_debug_image': ParameterValue(
                            LaunchConfiguration('publish_debug_image'),
                            value_type=bool,
                        ),
                        'debug_image_max_side_px': ParameterValue(
                            LaunchConfiguration('debug_image_max_side_px'),
                            value_type=int,
                        ),
                        'debug_publish_period_ms': ParameterValue(
                            LaunchConfiguration('debug_publish_period_ms'),
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
                        'color_fallback_device': LaunchConfiguration('color_fallback_device'),
                        'enable_low_confidence_color_fallback': ParameterValue(
                            LaunchConfiguration('enable_low_confidence_color_fallback'),
                            value_type=bool,
                        ),
                        'model_confidence_threshold': ParameterValue(
                            LaunchConfiguration('model_confidence_threshold'),
                            value_type=float,
                        ),
                        'fallback_score_threshold': ParameterValue(
                            LaunchConfiguration('fallback_score_threshold'),
                            value_type=float,
                        ),
                        'fallback_green_h_min': ParameterValue(
                            LaunchConfiguration('fallback_green_h_min'), value_type=float,
                        ),
                        'fallback_green_h_max': ParameterValue(
                            LaunchConfiguration('fallback_green_h_max'), value_type=float,
                        ),
                        'fallback_green_s_min': ParameterValue(
                            LaunchConfiguration('fallback_green_s_min'), value_type=int,
                        ),
                        'fallback_green_v_min': ParameterValue(
                            LaunchConfiguration('fallback_green_v_min'), value_type=int,
                        ),
                        'fallback_green_score_threshold': ParameterValue(
                            LaunchConfiguration('fallback_green_score_threshold'), value_type=float,
                        ),
                        'fallback_green_top_weight': ParameterValue(
                            LaunchConfiguration('fallback_green_top_weight'), value_type=float,
                        ),
                        'fallback_green_middle_weight': ParameterValue(
                            LaunchConfiguration('fallback_green_middle_weight'), value_type=float,
                        ),
                        'fallback_green_bottom_weight': ParameterValue(
                            LaunchConfiguration('fallback_green_bottom_weight'), value_type=float,
                        ),
                        'fallback_score_gap': ParameterValue(
                            LaunchConfiguration('fallback_score_gap'),
                            value_type=float,
                        ),
                        'fallback_saturation_gain': ParameterValue(
                            LaunchConfiguration('fallback_saturation_gain'),
                            value_type=float,
                        ),
                        'fallback_value_gain': ParameterValue(
                            LaunchConfiguration('fallback_value_gain'),
                            value_type=float,
                        ),
                        'fallback_gamma': ParameterValue(
                            LaunchConfiguration('fallback_gamma'),
                            value_type=float,
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
