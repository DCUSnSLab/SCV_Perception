from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare('realsense_on'), 'config', 'realsense_555.yaml']
    )
    rs_launch = PathJoinSubstitution(
        [FindPackageShare('realsense2_camera'), 'launch', 'rs_launch.py']
    )

    serial_arg = DeclareLaunchArgument(
        'serial_no',
        default_value="'419222301550'",
        description='RealSense serial number for the standalone camera',
    )
    usb_port_arg = DeclareLaunchArgument(
        'usb_port_id',
        default_value="'2-8'",
        description='USB port ID for the standalone camera',
    )
    namespace_arg = DeclareLaunchArgument(
        'camera_namespace',
        default_value='realsense_on',
        description='ROS namespace for the standalone camera',
    )
    name_arg = DeclareLaunchArgument(
        'camera_name',
        default_value='d555_front',
        description='Node name for the standalone camera',
    )
    config_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config,
        description='realsense2_camera YAML config file',
    )

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rs_launch),
        launch_arguments={
            'serial_no': LaunchConfiguration('serial_no'),
            'usb_port_id': LaunchConfiguration('usb_port_id'),
            'camera_namespace': LaunchConfiguration('camera_namespace'),
            'camera_name': LaunchConfiguration('camera_name'),
            'config_file': LaunchConfiguration('config_file'),
            'pointcloud.enable': 'true',
            'align_depth.enable': 'true',
            'enable_gyro': 'true',
            'enable_accel': 'true',
            'unite_imu_method': '2',
        }.items(),
    )

    return LaunchDescription([
        serial_arg,
        usb_port_arg,
        namespace_arg,
        name_arg,
        config_arg,
        camera,
    ])
