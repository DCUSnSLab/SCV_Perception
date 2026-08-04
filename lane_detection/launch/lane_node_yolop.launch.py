from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


def generate_launch_description():
    default_weights = os.path.expanduser('~/epoch-155_old.pth')

    return LaunchDescription([
        DeclareLaunchArgument('weights_path', default_value=default_weights),
        DeclareLaunchArgument(
            'image_topic',
            default_value='/front_right/front_right/color/image_raw'),
        DeclareLaunchArgument(
            'depth_topic',
            default_value='/front_right/front_right/aligned_depth_to_color/image_raw'),
        DeclareLaunchArgument(
            'camera_info_topic',
            default_value='/front_right/front_right/color/camera_info'),
        DeclareLaunchArgument('img_size', default_value='640'),
        DeclareLaunchArgument('device', default_value='cuda:0'),
        DeclareLaunchArgument('depth_scale', default_value='0.001'),
        DeclareLaunchArgument('mask_point_stride', default_value='1'),
        DeclareLaunchArgument('enable_depth_hole_fill', default_value='true'),
        DeclareLaunchArgument('enable_sor', default_value='true'),
        DeclareLaunchArgument('manual_roll_deg', default_value='0.2'),
        DeclareLaunchArgument('manual_pitch_deg', default_value='0.0'),
        DeclareLaunchArgument('manual_yaw_deg', default_value='0.0'),
        DeclareLaunchArgument('cloud_offset_x', default_value='0.0'),
        DeclareLaunchArgument('cloud_offset_y', default_value='-0.1'),
        Node(
            package='lane_detection',
            executable='lane_node_yolop',
            name='lane_detection_node_yolop',
            output='screen',
            parameters=[{
                'weights_path': LaunchConfiguration('weights_path'),
                'image_topic': LaunchConfiguration('image_topic'),
                'depth_topic': LaunchConfiguration('depth_topic'),
                'camera_info_topic': LaunchConfiguration('camera_info_topic'),
                'img_size': LaunchConfiguration('img_size'),
                'device': LaunchConfiguration('device'),
                'depth_scale': LaunchConfiguration('depth_scale'),
                'mask_point_stride': LaunchConfiguration('mask_point_stride'),
                'enable_depth_hole_fill': LaunchConfiguration('enable_depth_hole_fill'),
                'enable_sor': LaunchConfiguration('enable_sor'),
                'depth_min': 0.1,
                'depth_max': 10.0,
                'voxel_size': 0.03,
                'ground_proj': True,
                'sor_k': 20,
                'sor_std_mul': 1.5,
                'morph_kernel_width': 3,
                'morph_kernel_height': 5,
                'manual_roll_deg': LaunchConfiguration('manual_roll_deg'),
                'manual_pitch_deg': LaunchConfiguration('manual_pitch_deg'),
                'manual_yaw_deg': LaunchConfiguration('manual_yaw_deg'),
                'cloud_offset_x': LaunchConfiguration('cloud_offset_x'),
                'cloud_offset_y': LaunchConfiguration('cloud_offset_y'),
                'lane_pixel_value': 1,
            }],
        ),
    ])
