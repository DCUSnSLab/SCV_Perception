from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import ExecuteProcess
from launch.actions import OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from mando_tools.bag_cli import validate_bag_path

from mando_tools.workspace_paths import default_bag_path
from mando_tools.workspace_paths import default_image_topic
from mando_tools.workspace_paths import default_runtime_image_topic
from mando_tools.workspace_paths import resolve_bag_path


def _launch_setup(context, *args, **kwargs):
    bag_selection = LaunchConfiguration('bag_path').perform(context)
    image_source_topic = LaunchConfiguration('image_source_topic').perform(context).strip()
    image_topic = LaunchConfiguration('image_topic').perform(context).strip()

    resolved_bag = resolve_bag_path(bag_selection)
    validate_bag_path(resolved_bag)
    bag_path = str(resolved_bag)
    if not image_source_topic:
        image_source_topic = default_image_topic(bag_selection)

    cmd = ['ros2', 'bag', 'play', bag_path]
    resize_images = LaunchConfiguration('resize_images').perform(context).lower() == 'true'
    if resize_images:
        if not image_source_topic or not image_topic:
            raise ValueError('Resize playback requires source and output image topics')
        width = int(LaunchConfiguration('output_width').perform(context))
        height = int(LaunchConfiguration('output_height').perform(context))
        if width <= 0 or height <= 0:
            raise ValueError('Output width and height must be positive')
        recorded_topic = image_topic.rstrip('/') + '_recorded'
        cmd.extend([
            '--topics', image_source_topic,
            '--remap', f'{image_source_topic}:={recorded_topic}',
            '--read-ahead-queue-size', '10',
            '--delay', '2',
        ])
        relay = Node(
            package='mando_tools',
            executable='mando_panorama_resize',
            output='screen',
            parameters=[{
                'input_topic': recorded_topic,
                'output_topic': image_topic,
                'output_width': width,
                'output_height': height,
            }],
        )
        return [relay, ExecuteProcess(cmd=cmd, output='screen')]
    if image_source_topic and image_topic and image_source_topic != image_topic:
        cmd.extend(['--remap', f'{image_source_topic}:={image_topic}'])

    return [ExecuteProcess(cmd=cmd, output='screen')]


def generate_launch_description() -> LaunchDescription:
    bag_arg = DeclareLaunchArgument(
        'bag_path',
        default_value=str(default_bag_path()),
        description='Path or known profile name of the rosbag2 directory to play.',
    )
    image_source_arg = DeclareLaunchArgument(
        'image_source_topic',
        default_value='',
        description='Original image topic recorded in the bag. Auto-detected for known bags.',
    )
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value=default_runtime_image_topic(),
        description='Unified image topic published during ros2 bag play.',
    )
    player = OpaqueFunction(function=_launch_setup)

    resize_arg = DeclareLaunchArgument(
        'resize_images', default_value='false', choices=['true', 'false'],
        description='Replay only the selected image topic through a resize relay.',
    )
    width_arg = DeclareLaunchArgument('output_width', default_value='1254')
    height_arg = DeclareLaunchArgument('output_height', default_value='370')
    return LaunchDescription([
        bag_arg, image_source_arg, image_topic_arg,
        resize_arg, width_arg, height_arg, player,
    ])
