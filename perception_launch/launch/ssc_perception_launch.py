# multi_start.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os

from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    return LaunchDescription([
        # parking_detector 실행
        Node(
            package='parking_detector',
            executable='main.py',
            name='parking_detector',
            output='screen'
        ),

        # ssc_avoid_obstacles 런치파일 include
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('ssc_avoid_obstacles'),
                    'launch',
                    'send_stop_flag.launch.py'
                )
            )
        ),

        # tl_roi_hist 런치파일 include
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory('tl_roi_hist'),
                    'launch',
                    'roi_hist.launch.py'
                )
            )
        ),
    ])
