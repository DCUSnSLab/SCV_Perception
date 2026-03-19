from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1. 바닥 제거 노드
        Node(
            package='pcd_ground_filter',
            executable='ground_removal_node',
            name='ground_removal_node',
            output='screen',
            parameters=[{
                'input_topic': '/velodyne_points',
                'output_topic': '/no_ground_points',
                'num_angular_bins': 720,
                'max_slope_deg': 10.0,
                'max_height_jump': 0.2,
                'min_range': 1.0,
                'max_range': 80.0,
                'ground_level_z': -1.5,
                'min_obj_height': 0.10,
            }]
        ),

        # 2. 군집화 노드
        Node(
            package='pcd_cluster',
            executable='cluster_node',
            name='cluster_node',
            output='screen',
            parameters=[{
                'input_topic': '/no_ground_points',
                'output_topic': '/clusters',
                'cluster_tolerance': 0.5,
                'min_cluster_size': 10,
                'max_cluster_size': 20000
            }]
        ),

        # 3. 추적 노드
        Node(
            package='pcd_tracker',
            executable='tracker_node',
            name='tracker_node',
            output='screen',
            parameters=[{
                'input_topic': '/clusters',
                'dist_thresh': 1.0
            }]
        ),
    ])
