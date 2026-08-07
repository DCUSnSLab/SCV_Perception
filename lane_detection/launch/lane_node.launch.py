from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os


def generate_launch_description():
    default_model_path = os.path.expanduser('~/yolo26m_seg_best.pt')
    model_path = LaunchConfiguration('model_path')
    image_topic = LaunchConfiguration('image_topic')
    depth_topic = LaunchConfiguration('depth_topic')
    conf = LaunchConfiguration('conf')

    return LaunchDescription([
        DeclareLaunchArgument(
            'model_path',
            default_value=default_model_path,
            description='YOLO-seg 모델 경로'
        ),
        DeclareLaunchArgument(
            'image_topic',
            default_value='/front_right/front_right/color/image_raw',
            description='컬러 이미지 토픽'
        ),
        DeclareLaunchArgument(
            'depth_topic',
            default_value='/front_right/front_right/aligned_depth_to_color/image_raw',
            description='뎁스 이미지 토픽'
        ),
        DeclareLaunchArgument(
            'device',
            default_value='cuda:0',
            description='추론 디바이스. GPU 격리 테스트 시 cpu 로 설정'
        ),
        DeclareLaunchArgument(
            'conf',
            default_value='0.00065',
            description='YOLO confidence 임계값'
        ),
        Node(
            package='lane_detection',
            executable='lane_node',
            name='lane_detection_node',
            output='screen',
            parameters=[{
                'model_path':  model_path,
                'image_topic': image_topic,
                'depth_topic': depth_topic,
                'device':      LaunchConfiguration('device'),
                'conf':        conf,
                'depth_min':   0.1,
                'depth_max':   10.0,
                'voxel_size':  0.03,
                'ground_proj': True,
                'sor_k':          20,
                'sor_std_mul':    1.5,
                'morph_kernel_width': 5,
                'morph_kernel_height': 9,
            }]
        ),
    ])
