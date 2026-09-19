from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from pathlib import Path
import os


def _default_tl_model() -> str:
    search_roots = [
        Path(value).expanduser()
        for value in (
            os.environ.get('MANDO_WS'),
            os.environ.get('MANDO_WORKSPACE'),
        )
        if value
    ]
    search_roots.append(Path.home() / 'SSC' / 'src' / 'perception' / 'traffic_light')
    launch_file = Path(__file__).resolve()
    search_roots.extend([launch_file.parent, *launch_file.parents])
    for root in search_roots:
        candidate = root / 'model' / 'best.pt'
        if candidate.exists() or root.name == 'traffic_light':
            return str(candidate)
    return 'best.pt'

def generate_launch_description() -> LaunchDescription:
    parameter_defaults = {
        'detect_top_ratio': 0.0, 'detect_bottom_ratio': 1.0 / 3.0,
        'detect_left_ratio': 0.375, 'detect_right_ratio': 0.625,
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
    publish_debug_arg = DeclareLaunchArgument(
        'publish_debug_image',
        default_value='false',
        description='Publish /tl/debug_image when a subscriber is connected.',
    )
    debug_image_max_side_arg = DeclareLaunchArgument(
        'debug_image_max_side_px',
        default_value='640',
        description='Maximum side length of the debug image; 0 keeps the source size.',
    )
    show_color_mask_inset_arg = DeclareLaunchArgument(
        'show_color_mask_inset',
        default_value='true',
        description='Draw the Color Mask inset on the debug image. Display only; set false to hide it.',
    )
    debug_publish_period_arg = DeclareLaunchArgument(
        'debug_publish_period_ms',
        default_value='200.0',
        description='Minimum interval between debug renders; 0 renders every frame.',
    )
    fps_arg = DeclareLaunchArgument(
        'max_fps',
        default_value='30.0',
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
        default_value='480',
        description='Inference image size for the cropped ROI.',
    )
    detector_retry_gamma_arg = DeclareLaunchArgument(
        'detector_retry_gamma',
        default_value='1.20',
        description=(
            'Gamma used for one retry when raw YOLO confidence is below the '
            'model confidence threshold. '
            'Set 1.0 to disable.'
        ),
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
    model_only_arg = DeclareLaunchArgument(
        'model_only',
        default_value='false',
        description='Use YOLO classes only; skip HSV color analysis entirely.',
    )
    fallback_score_arg = DeclareLaunchArgument(
        'fallback_score_threshold',
        default_value='0.45',
        description='Normalized color score required for the fallback state.',
    )
    fallback_min_valid_pixels_arg = DeclareLaunchArgument(
        'fallback_min_valid_pixels',
        default_value='7',
        description='Minimum number of color-mask pixels required for a fallback state.',
    )
    fallback_min_component_pixels_arg = DeclareLaunchArgument(
        'fallback_min_component_pixels',
        default_value='5',
        description='Minimum connected-component size for a fallback state.',
    )
    fallback_v_min_arg = DeclareLaunchArgument(
        'fallback_v_min',
        default_value='67',
        description='Minimum HSV value for red/yellow fallback pixels.',
    )
    fallback_green_min_valid_pixels_arg = DeclareLaunchArgument(
        'fallback_green_min_valid_pixels',
        default_value='14',
        description='Minimum green-mask pixels required for a green fallback state.',
    )
    fallback_green_min_component_pixels_arg = DeclareLaunchArgument(
        'fallback_green_min_component_pixels',
        default_value='9',
        description='Minimum green connected-component size for a green fallback state.',
    )
    fallback_green_h_min_arg = DeclareLaunchArgument(
        'fallback_green_h_min', default_value='39.0',
        description='Lower HSV hue bound for green fallback pixels.',
    )
    fallback_green_h_max_arg = DeclareLaunchArgument(
        'fallback_green_h_max', default_value='100.0',
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
    fallback_green_score_arg = DeclareLaunchArgument(
        'fallback_green_score_threshold', default_value='0.40',
        description='Normalized score required for a green fallback state.',
    )
    fallback_green_top_weight_arg = DeclareLaunchArgument(
        'fallback_green_top_weight', default_value='0.20',
        description='Common color score weight at the top of the candidate box.',
    )
    fallback_green_middle_weight_arg = DeclareLaunchArgument(
        'fallback_green_middle_weight', default_value='1.30',
        description='Common color score weight in the middle of the candidate box.',
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
                'show_color_mask_inset': ParameterValue(
                    LaunchConfiguration('show_color_mask_inset'),
                    value_type=bool,
                ),
                'model_only': ParameterValue(
                    LaunchConfiguration('model_only'),
                    value_type=bool,
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
                'detector_retry_gamma': ParameterValue(
                    LaunchConfiguration('detector_retry_gamma'),
                    value_type=float,
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
                'fallback_min_valid_pixels': ParameterValue(
                    LaunchConfiguration('fallback_min_valid_pixels'),
                    value_type=int,
                ),
                'fallback_min_component_pixels': ParameterValue(
                    LaunchConfiguration('fallback_min_component_pixels'),
                    value_type=int,
                ),
                'fallback_v_min': ParameterValue(
                    LaunchConfiguration('fallback_v_min'),
                    value_type=int,
                ),
                'fallback_green_min_valid_pixels': ParameterValue(
                    LaunchConfiguration('fallback_green_min_valid_pixels'),
                    value_type=int,
                ),
                'fallback_green_min_component_pixels': ParameterValue(
                    LaunchConfiguration('fallback_green_min_component_pixels'),
                    value_type=int,
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
            publish_debug_arg,
            debug_image_max_side_arg,
            show_color_mask_inset_arg,
            model_only_arg,
            debug_publish_period_arg,
            fps_arg,
            detector_device_arg,
            color_fallback_device_arg,
            detector_conf_arg,
            detector_size_arg,
            detector_retry_gamma_arg,
            model_conf_arg,
            low_conf_fallback_arg,
            fallback_score_arg,
            fallback_min_valid_pixels_arg,
            fallback_min_component_pixels_arg,
            fallback_v_min_arg,
            fallback_green_min_valid_pixels_arg,
            fallback_green_min_component_pixels_arg,
            fallback_green_h_min_arg,
            fallback_green_h_max_arg,
            fallback_green_s_min_arg,
            fallback_green_v_min_arg,
            fallback_green_score_arg,
            fallback_green_top_weight_arg,
            fallback_green_middle_weight_arg,
            fallback_green_bottom_weight_arg,
            fallback_saturation_arg,
            fallback_value_arg,
            fallback_gamma_arg,
            fallback_max_side_arg,
            tl_fusion,
        ]
    )
