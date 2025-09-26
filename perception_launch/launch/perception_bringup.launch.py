from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node

def generate_launch_description():

    parking_detector_node = Node(
        package='parking_detector',
        executable='main.py'
    )
    
    # uBlox GPS Launch
    ob_depth_treacker_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('object_depth_tracker'),
                'launch',
                'object_tracker.launch.py'
            ])
        ])
    )
    
    tl_roi_hist_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('tl_roi_hist'),
                'launch',
                'roi_hist.launch.py'
            ])
        ])
    )

    return LaunchDescription([
        parking_detector_node,
        ob_depth_treacker_launch,
        tl_roi_hist_launch
    ])