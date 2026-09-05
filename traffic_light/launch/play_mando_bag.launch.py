from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import ExecuteProcess
from launch.actions import OpaqueFunction
from launch.substitutions import LaunchConfiguration

from mando_tools.workspace_paths import default_bag_path
from mando_tools.workspace_paths import default_image_topic
from mando_tools.workspace_paths import default_runtime_image_topic
from mando_tools.workspace_paths import resolve_bag_path


def _launch_setup(context, *args, **kwargs):
    bag_selection = LaunchConfiguration('bag_path').perform(context)
    image_source_topic = LaunchConfiguration('image_source_topic').perform(context).strip()
    image_topic = LaunchConfiguration('image_topic').perform(context).strip()

    bag_path = str(resolve_bag_path(bag_selection))
    if not image_source_topic:
        image_source_topic = default_image_topic(bag_selection)

    cmd = ['ros2', 'bag', 'play', bag_path]
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

    return LaunchDescription([bag_arg, image_source_arg, image_topic_arg, player])
