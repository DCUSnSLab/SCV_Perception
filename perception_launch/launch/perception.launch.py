"""SCV perception bring-up: the SCV_Perception nodes the navigation stack needs.

  traversability_grid : /perception/traversability/grid -> local_costmap
                        (traversability_costmap). Always on.
  ground_filter       : /perception/points_nonground, only needed by the old
                        accumulating local_costmap (costmap.launch.py). ground_filter:=true

Run next to command_center:
  ros2 launch perception_launch perception.launch.py use_sim_time:=false
  ros2 launch command_center_launch scv_system.launch.py use_sim_time:=false
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _inc(pkg, rel, args, condition=None):
    src = PythonLaunchDescriptionSource(
        os.path.join(get_package_share_directory(pkg), 'launch', rel))
    return IncludeLaunchDescription(src, launch_arguments=args.items(), condition=condition)


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false',
                              description='true for bag playback / simulation'),
        DeclareLaunchArgument('ground_filter', default_value='false', choices=['true', 'false'],
                              description='also run ground_filter (old local_costmap only)'),

        _inc('traversability_grid', 'traversability_grid.launch.py',
             {'use_sim_time': use_sim_time}),
        _inc('ground_filter', 'ground_filter.launch.py',
             {'use_sim_time': use_sim_time},
             condition=IfCondition(LaunchConfiguration('ground_filter'))),
    ])
