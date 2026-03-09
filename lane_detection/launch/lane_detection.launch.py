from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import SetEnvironmentVariable
import os

YOLOP_PATH = os.path.expanduser('~/SSC_jeon/src/perception/YOLOP')


def generate_launch_description():
    weights_path = os.path.join(
        get_package_share_directory('lane_detection'),
        'weights',
        'yolop_aug_best_compat.pth'
        # 'epoch-155.pth'
    )

    return LaunchDescription([
        SetEnvironmentVariable('PYTHONPATH', YOLOP_PATH + ':' + os.environ.get('PYTHONPATH', '')),
        Node(
            package='lane_detection',
            executable='yolop_node',
            name='yolop_node',
            output='screen',
            parameters=[{
                'model_path': weights_path,
                'image_topic': '/camera/camera/color/image_raw',
                'depth_topic': '/camera/camera/aligned_depth_to_color/image_raw',
                'input_size': 320,
                'conf_thresh': 0.3,
            }]
        )
    ])
