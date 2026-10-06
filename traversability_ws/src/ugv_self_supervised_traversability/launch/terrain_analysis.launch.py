"""Camera-independent terrain analysis; use_sim_time:=true for bag playback."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config=LaunchConfiguration('config_file')
    return LaunchDescription([
        DeclareLaunchArgument('config_file',default_value=PathJoinSubstitution([
            FindPackageShare('ugv_self_supervised_traversability'),'config','traversability.yaml'])),
        DeclareLaunchArgument('use_sim_time',default_value='false'),
        Node(package='ugv_self_supervised_traversability',executable='terrain_analyzer',
             name='terrain_analyzer',output='screen',parameters=[config,{
                 'use_sim_time':ParameterValue(LaunchConfiguration('use_sim_time'),value_type=bool)}])])
