from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
DEFAULT_TL_MODEL = '/home/ki/SSC/src/perception/traffic_light/model/best.pt'

def generate_launch_description() -> LaunchDescription:
    parameter_defaults = {
        'detect_top_ratio': 0.0, 'detect_bottom_ratio': 1.0 / 3.0,
        'detect_left_ratio': 0.20, 'detect_right_ratio': 0.80,
        'max_image_age_ms': 500.0,
        'future_stamp_tolerance_ms': 50.0,
        'state_confirm_ms': 200.0,
        'state_max_gap_ms': 250.0,
        'uncertain_hold_ms': 300.0,
    }
    parameter_args = [
        DeclareLaunchArgument(name, default_value=str(value), description='Detection ROI ratio or timing threshold in milliseconds, as indicated by the parameter name.')
        for name, value in parameter_defaults.items()
    ]
    sim_time_arg = DeclareLaunchArgument('use_sim_time', default_value='false', description='Use the ROS clock published during bag playback.')
    model_arg = DeclareLaunchArgument(
        'model_path',
        default_value=DEFAULT_TL_MODEL,
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
    publish_debug_image_arg = DeclareLaunchArgument(
        'publish_debug_image',
        default_value='false',
        description='Publish the debug image topic.',
    )
    debug_image_max_side_arg = DeclareLaunchArgument(
        'debug_image_max_side_px',
        default_value='640',
        description='Maximum side length of the debug image; 0 keeps the source size.',
    )
    debug_publish_period_arg = DeclareLaunchArgument(
        'debug_publish_period_ms',
        default_value='200.0',
        description='Minimum interval between debug renders; 0 renders every frame.',
    )
    fps_arg = DeclareLaunchArgument(
        'max_fps',
        default_value='5.0',
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
        default_value='0.05',
        description='Minimum YOLO confidence for signal candidates, tuned for small lights.',
    )
    detector_size_arg = DeclareLaunchArgument(
        'detector_image_size',
        default_value='1280',
        description='Inference image size for YOLO small-object recall.',
    )
    model_conf_arg = DeclareLaunchArgument(
        'model_confidence_threshold',
        default_value='0.75',
        description='Confidence above which the model state is trusted directly.',
    )
    low_conf_fallback_arg = DeclareLaunchArgument(
        'enable_low_confidence_color_fallback',
        default_value='true',
        description=(
            'Legacy compatibility parameter. Color analysis is always enabled.'
        ),
    )
    fallback_score_arg = DeclareLaunchArgument(
        'fallback_score_threshold',
        default_value='0.45',
        description='Normalized color score required for the fallback state.',
    )
    fallback_green_h_min_arg = DeclareLaunchArgument(
        'fallback_green_h_min', default_value='39.0',
        description='Lower HSV hue bound for green fallback pixels.',
    )
    fallback_green_h_max_arg = DeclareLaunchArgument(
        'fallback_green_h_max', default_value='90.0',
        description='Upper HSV hue bound for green fallback pixels.',
    )
    fallback_green_s_min_arg = DeclareLaunchArgument(
        'fallback_green_s_min', default_value='50',
        description='Minimum HSV saturation for green fallback pixels.',
    )
    fallback_green_v_min_arg = DeclareLaunchArgument(
        'fallback_green_v_min', default_value='68',
        description='Minimum HSV value for green fallback pixels.',
    )
    fallback_green_middle_v_min_arg = DeclareLaunchArgument(
        'fallback_green_middle_v_min', default_value='60',
        description='Minimum HSV value for green pixels in the middle band.',
    )
    fallback_green_score_arg = DeclareLaunchArgument(
        'fallback_green_score_threshold', default_value='0.40',
        description='Normalized score required for a green fallback state.',
    )
    fallback_green_top_weight_arg = DeclareLaunchArgument(
        'fallback_green_top_weight', default_value='0.20',
        description='Common color score weight at the top of the candidate box.',
    )
    fallback_green_middle_weight_arg = DeclareLaunchArgument(
        'fallback_green_middle_weight', default_value='1.50',
        description='Common color score weight in the middle of the candidate box.',
    )
    fallback_middle_mask_dilate_arg = DeclareLaunchArgument(
        'fallback_middle_mask_dilate_iterations', default_value='1',
        description='Dilation iterations for existing color pixels in the middle band.',
    )
    fallback_green_bottom_weight_arg = DeclareLaunchArgument(
        'fallback_green_bottom_weight', default_value='0.20',
        description='Common color score weight at the bottom of the candidate box.',
    )
    fallback_saturation_arg = DeclareLaunchArgument(
        'fallback_saturation_gain',
        default_value='2.20',
        description='Saturation gain applied before color fallback analysis.',
    )
    fallback_value_arg = DeclareLaunchArgument(
        'fallback_value_gain',
        default_value='1.35',
        description='Brightness gain applied before color fallback analysis.',
    )
    fallback_gamma_arg = DeclareLaunchArgument(
        'fallback_gamma',
        default_value='1.00',
        description='Gamma applied before color fallback analysis.',
    )
    fallback_red_h_max_arg = DeclareLaunchArgument(
        'fallback_red_h_max',
        default_value='8.0',
        description='Upper OpenCV HSV hue bound for red near hue zero.',
    )
    fallback_red_h_wrap_min_arg = DeclareLaunchArgument(
        'fallback_red_h_wrap_min',
        default_value='170.0',
        description='Lower OpenCV HSV hue bound for red near hue 180.',
    )
    fallback_red_v_min_arg = DeclareLaunchArgument(
        'fallback_red_v_min',
        default_value='85',
        description='Minimum red brightness to reject dark orange lamps.',
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
                'max_fps': ParameterValue(
                    LaunchConfiguration('max_fps'),
                    value_type=float,
                ),
                'detector_device': LaunchConfiguration('detector_device'),
                **{name: ParameterValue(LaunchConfiguration(name), value_type=float) for name in parameter_defaults},
                'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
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
                'fallback_green_middle_v_min': ParameterValue(
                    LaunchConfiguration('fallback_green_middle_v_min'), value_type=int,
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
                'fallback_middle_mask_dilate_iterations': ParameterValue(
                    LaunchConfiguration('fallback_middle_mask_dilate_iterations'),
                    value_type=int,
                ),
                'fallback_green_bottom_weight': ParameterValue(
                    LaunchConfiguration('fallback_green_bottom_weight'), value_type=float,
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
                'fallback_red_h_max': ParameterValue(
                    LaunchConfiguration('fallback_red_h_max'),
                    value_type=float,
                ),
                'fallback_red_h_wrap_min': ParameterValue(
                    LaunchConfiguration('fallback_red_h_wrap_min'),
                    value_type=float,
                ),
                'fallback_red_v_min': ParameterValue(
                    LaunchConfiguration('fallback_red_v_min'),
                    value_type=int,
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
            *parameter_args,
            sim_time_arg,
            image_topic_arg,
            state_topic_arg,
            input_timeout_arg,
            show_windows_arg,
            publish_debug_image_arg,
            debug_image_max_side_arg,
            debug_publish_period_arg,
            fps_arg,
            detector_device_arg,
            color_fallback_device_arg,
            detector_conf_arg,
            detector_size_arg,
            model_conf_arg,
            low_conf_fallback_arg,
            fallback_score_arg,
            fallback_green_h_min_arg,
            fallback_green_h_max_arg,
            fallback_green_s_min_arg,
            fallback_green_v_min_arg,
            fallback_green_middle_v_min_arg,
            fallback_green_score_arg,
            fallback_green_top_weight_arg,
            fallback_green_middle_weight_arg,
            fallback_middle_mask_dilate_arg,
            fallback_green_bottom_weight_arg,
            fallback_saturation_arg,
            fallback_value_arg,
            fallback_gamma_arg,
            fallback_red_h_max_arg,
            fallback_red_h_wrap_min_arg,
            fallback_red_v_min_arg,
            fallback_max_side_arg,
            tl_fusion,
        ]
    )
