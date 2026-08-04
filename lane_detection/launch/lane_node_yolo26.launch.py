from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


def generate_launch_description():
    default_model_path = os.path.expanduser('~/yolo26m_seg_best.pt')

    return LaunchDescription([
        DeclareLaunchArgument('model_path', default_value=default_model_path),
        DeclareLaunchArgument(
            'image_topic',
            default_value='/front_right/front_right/color/image_raw'),
        DeclareLaunchArgument(
            'depth_topic',
            default_value='/front_right/front_right/aligned_depth_to_color/image_raw'),
        DeclareLaunchArgument(
            'camera_info_topic',
            default_value='/front_right/front_right/color/camera_info'),
        DeclareLaunchArgument('conf', default_value='0.0007'),
        DeclareLaunchArgument('imgsz', default_value='640'),
        DeclareLaunchArgument('max_det', default_value='64'),
        DeclareLaunchArgument('min_mask_area_ratio', default_value='0.0002'),
        DeclareLaunchArgument('min_bottom_y_ratio', default_value='0.45'),
        DeclareLaunchArgument('min_height_ratio', default_value='0.08'),
        DeclareLaunchArgument('max_lane_instances', default_value='8'),
        DeclareLaunchArgument('bottom_region_ratio', default_value='0.2'),
        DeclareLaunchArgument('lane_outward_offset', default_value='0.5'),
        DeclareLaunchArgument('depth_scale', default_value='0.001'),
        DeclareLaunchArgument('manual_roll_deg', default_value='0.2'),
        DeclareLaunchArgument('manual_pitch_deg', default_value='0.0'),
        DeclareLaunchArgument('manual_yaw_deg', default_value='0.0'),
        DeclareLaunchArgument('cloud_offset_x', default_value='0.0'),
        DeclareLaunchArgument('cloud_offset_y', default_value='-0.1'),
        Node(
            package='lane_detection',
            executable='lane_node_yolo26',
            name='lane_detection_node_yolo26',
            output='screen',
            parameters=[{
                'model_path': LaunchConfiguration('model_path'),
                'image_topic': LaunchConfiguration('image_topic'),
                'depth_topic': LaunchConfiguration('depth_topic'),
                'camera_info_topic': LaunchConfiguration('camera_info_topic'),
                'conf': LaunchConfiguration('conf'),
                'imgsz': LaunchConfiguration('imgsz'),
                'max_det': LaunchConfiguration('max_det'),
                'min_mask_area_ratio': LaunchConfiguration('min_mask_area_ratio'),
                'min_bottom_y_ratio': LaunchConfiguration('min_bottom_y_ratio'),
                'min_height_ratio': LaunchConfiguration('min_height_ratio'),
                'max_lane_instances': LaunchConfiguration('max_lane_instances'),
                'bottom_region_ratio': LaunchConfiguration('bottom_region_ratio'),
                'lane_outward_offset': LaunchConfiguration('lane_outward_offset'),
                'depth_scale': LaunchConfiguration('depth_scale'),
                'depth_min': 0.1,
                'depth_max': 10.0,
                'voxel_size': 0.03,
                'ground_proj': True,
                'sor_k': 20,
                'sor_std_mul': 1.5,
                'morph_kernel_width': 5,
                'morph_kernel_height': 9,
                'manual_roll_deg': LaunchConfiguration('manual_roll_deg'),
                'manual_pitch_deg': LaunchConfiguration('manual_pitch_deg'),
                'manual_yaw_deg': LaunchConfiguration('manual_yaw_deg'),
                'cloud_offset_x': LaunchConfiguration('cloud_offset_x'),
                'cloud_offset_y': LaunchConfiguration('cloud_offset_y'),
            }],
        ),
    ])
