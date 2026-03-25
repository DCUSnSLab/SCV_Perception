from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'bag_dir',
            description='ROS2 bag 디렉토리 경로 (필수)'
        ),
        DeclareLaunchArgument(
            'interval',
            default_value='1.0',
            description='이미지 추출 간격 (초, 기본값: 1.0)'
        ),
        DeclareLaunchArgument(
            'output',
            default_value='/home/ssc/extracted_images',
            description='저장 디렉토리 (기본값: /home/ssc/extracted_images)'
        ),
        DeclareLaunchArgument(
            'topic',
            default_value='/camera/camera/color/image_raw',
            description='카메라 토픽 이름'
        ),
        DeclareLaunchArgument(
            'quality',
            default_value='95',
            description='JPEG 품질 0~100 (기본값: 95)'
        ),

        ExecuteProcess(
            cmd=[
                'ros2', 'run', 'lane_detection', 'bag_to_images',
                LaunchConfiguration('bag_dir'),
                '--interval', LaunchConfiguration('interval'),
                '--output',   LaunchConfiguration('output'),
                '--topic',    LaunchConfiguration('topic'),
                '--quality',  LaunchConfiguration('quality'),
            ],
            output='screen'
        ),
    ])
