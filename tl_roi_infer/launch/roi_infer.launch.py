#!/usr/bin/env python3
# launch/roi_infer.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg = get_package_share_directory('tl_roi_infer')
    params_file = os.path.join(pkg, 'params', 'default.yaml')  # 없어도 됨
    default_model = os.path.join(pkg, 'models', 'best.pt')

    # ---- LaunchConfiguration 핸들 ----
    model_path     = LaunchConfiguration('model_path')
    image_topic    = LaunchConfiguration('image_topic')
    debug_topic    = LaunchConfiguration('debug_topic')
    state_id_topic = LaunchConfiguration('state_id_topic')

    # 시간 필터/히스테리시스
    lp_alpha_rise  = LaunchConfiguration('lp_alpha_rise')
    lp_alpha_decay = LaunchConfiguration('lp_alpha_decay')
    lp_tau_hi      = LaunchConfiguration('lp_tau_hi')
    lp_tau_lo      = LaunchConfiguration('lp_tau_lo')
    lp_hold_ms     = LaunchConfiguration('lp_hold_ms')
    lp_margin      = LaunchConfiguration('lp_margin')
    lp_unknown_to  = LaunchConfiguration('lp_unknown_timeout_ms')
    lp_acquire_ms  = LaunchConfiguration('lp_acquire_ms')

    # 이미지 LPF
    img_lpf_enable = LaunchConfiguration('img_lpf_enable')
    img_lpf_type   = LaunchConfiguration('img_lpf_type')
    img_lpf_ksize  = LaunchConfiguration('img_lpf_ksize')
    img_lpf_sigma  = LaunchConfiguration('img_lpf_sigma')
    img_lpf_roi    = LaunchConfiguration('img_lpf_roi_only')
    img_bilat_sc   = LaunchConfiguration('img_lpf_bilateral_sigma_color')
    img_bilat_ss   = LaunchConfiguration('img_lpf_bilateral_sigma_space')

    # 상태 퍼블리시 정책
    state_pub_on_change = LaunchConfiguration('state_pub_on_change')
    state_pub_rate_hz   = LaunchConfiguration('state_pub_rate_hz')

    # 존재하는 YAML만 포함 + dict에서 타입 강제
    param_list = ([params_file] if os.path.exists(params_file) else []) + [{
        # strings
        'model_path':     model_path,
        'image_topic':    image_topic,
        'debug_topic':    debug_topic,
        'state_id_topic': state_id_topic,
        'img_lpf_type':   img_lpf_type,

        # bools
        'img_lpf_enable':      ParameterValue(img_lpf_enable, value_type=bool),
        'img_lpf_roi_only':    ParameterValue(img_lpf_roi, value_type=bool),
        'state_pub_on_change': ParameterValue(state_pub_on_change, value_type=bool),

        # ints
        'img_lpf_ksize':        ParameterValue(img_lpf_ksize, value_type=int),
        'lp_hold_ms':           ParameterValue(lp_hold_ms, value_type=int),
        'lp_unknown_timeout_ms':ParameterValue(lp_unknown_to, value_type=int),
        'lp_acquire_ms':        ParameterValue(lp_acquire_ms, value_type=int),

        # floats
        'img_lpf_sigma':                    ParameterValue(img_lpf_sigma, value_type=float),
        'img_lpf_bilateral_sigma_color':    ParameterValue(img_bilat_sc, value_type=float),
        'img_lpf_bilateral_sigma_space':    ParameterValue(img_bilat_ss, value_type=float),
        'lp_alpha_rise':                    ParameterValue(lp_alpha_rise, value_type=float),
        'lp_alpha_decay':                   ParameterValue(lp_alpha_decay, value_type=float),
        'lp_tau_hi':                        ParameterValue(lp_tau_hi, value_type=float),
        'lp_tau_lo':                        ParameterValue(lp_tau_lo, value_type=float),
        'lp_margin':                        ParameterValue(lp_margin, value_type=float),
        'state_pub_rate_hz':                ParameterValue(state_pub_rate_hz, value_type=float),
    }]

    return LaunchDescription([
        # 기본값(옵션 없이 실행 시 적용됨)
        DeclareLaunchArgument('model_path', default_value=default_model),
        DeclareLaunchArgument('image_topic', default_value='/zed_node/left/image_rect_color'),
        DeclareLaunchArgument('debug_topic', default_value='/tl/debug_image'),
        DeclareLaunchArgument('state_id_topic', default_value='/tl/state_id'),

        # 시간 필터 기본값
        DeclareLaunchArgument('lp_alpha_rise',  default_value='0.30'),
        DeclareLaunchArgument('lp_alpha_decay', default_value='0.06'),
        DeclareLaunchArgument('lp_tau_hi',      default_value='0.60'),
        DeclareLaunchArgument('lp_tau_lo',      default_value='0.45'),
        DeclareLaunchArgument('lp_hold_ms',     default_value='400'),
        DeclareLaunchArgument('lp_margin',      default_value='0.10'),
        DeclareLaunchArgument('lp_unknown_timeout_ms', default_value='1500'),
        DeclareLaunchArgument('lp_acquire_ms',  default_value='400'),

        # LPF 기본값 (요청하신 세트)
        DeclareLaunchArgument('img_lpf_enable', default_value='True'),
        DeclareLaunchArgument('img_lpf_type',   default_value='bilateral'),
        DeclareLaunchArgument('img_lpf_ksize',  default_value='5'),
        DeclareLaunchArgument('img_lpf_sigma',  default_value='0.8'),
        DeclareLaunchArgument('img_lpf_roi_only', default_value='True'),
        DeclareLaunchArgument('img_lpf_bilateral_sigma_color', default_value='20.0'),
        DeclareLaunchArgument('img_lpf_bilateral_sigma_space', default_value='2.0'),

        # 상태 퍼블리시 정책 기본값 (CLI 없이도 아래가 적용됨)
        DeclareLaunchArgument('state_pub_on_change', default_value='False'),
        DeclareLaunchArgument('state_pub_rate_hz',   default_value='20.0'),

        Node(
            package='tl_roi_infer',
            executable='roi_infer',
            name='roi_infer',
            output='screen',
            parameters=param_list,
        ),
    ])
