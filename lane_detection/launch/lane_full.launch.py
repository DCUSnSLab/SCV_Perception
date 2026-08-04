from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    default_model_path = os.path.expanduser('~/yolo26m_seg_best.pt')
    config = PathJoinSubstitution([
        FindPackageShare('lane_detection'), 'config', 'bev_cloud.yaml'
    ])

    return LaunchDescription([
        # ── YOLO overlay + mask 노드 파라미터 ──────────────────────────
        DeclareLaunchArgument('model_path',          default_value=default_model_path),
        DeclareLaunchArgument('image_topic',         default_value='/ardu_cam_link/image_raw'),
        DeclareLaunchArgument('conf',                default_value='0.0007'),
        DeclareLaunchArgument('imgsz',               default_value='640'),
        DeclareLaunchArgument('max_det',             default_value='64'),
        DeclareLaunchArgument('min_mask_area_ratio', default_value='0.0002'),
        DeclareLaunchArgument('min_bottom_y_ratio',  default_value='0.45'),
        DeclareLaunchArgument('min_height_ratio',    default_value='0.08'),
        DeclareLaunchArgument('max_lane_instances',  default_value='8'),
        DeclareLaunchArgument('overlay_alpha',       default_value='0.45'),

        # ── Node 1: YOLO overlay + lane_mask 퍼블리셔 ─────────────────
        Node(
            package='lane_detection',
            executable='lane_node_yolo26_overlay',
            name='lane_detection_overlay_node',
            output='screen',
            parameters=[{
                'model_path':          LaunchConfiguration('model_path'),
                'image_topic':         LaunchConfiguration('image_topic'),
                'output_topic':        '/lane_detection/overlay',
                'conf':                LaunchConfiguration('conf'),
                'imgsz':               LaunchConfiguration('imgsz'),
                'max_det':             LaunchConfiguration('max_det'),
                'min_mask_area_ratio': LaunchConfiguration('min_mask_area_ratio'),
                'min_bottom_y_ratio':  LaunchConfiguration('min_bottom_y_ratio'),
                'min_height_ratio':    LaunchConfiguration('min_height_ratio'),
                'max_lane_instances':  LaunchConfiguration('max_lane_instances'),
                'overlay_alpha':       LaunchConfiguration('overlay_alpha'),
            }],
        ),

        # ── Node 2: IPM BEV → PointCloud ──────────────────────────────
        Node(
            package='lane_detection',
            executable='lane_bev_cloud_node',
            name='lane_bev_cloud_node',
            output='screen',
            parameters=[config],
        ),
    ])
