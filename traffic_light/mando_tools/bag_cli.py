from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

from .workspace_paths import default_bag_path
from .workspace_paths import default_image_topic
from .workspace_paths import default_runtime_image_topic
from .workspace_paths import resolve_bag_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='mando_bag',
        description='Inspect or play the default ROS 2 bag dataset.',
    )
    parser.add_argument(
        'command',
        choices=['info', 'play'],
        help='Run ros2 bag info or ros2 bag play.',
    )
    parser.add_argument(
        'bag_path',
        nargs='?',
        default=str(default_bag_path()),
        help='Path to a rosbag2 directory. Defaults to the bag resolved by MANDO_BAG.',
    )
    parser.add_argument(
        '--image-topic',
        default=default_runtime_image_topic(),
        help='Unified image topic published during ros2 bag play.',
    )
    parser.add_argument(
        '--image-source-topic',
        default='',
        help='Original image topic recorded in the bag. Auto-detected for known bag profiles.',
    )
    parser.add_argument(
        '--no-image-remap',
        action='store_true',
        help='Do not remap the bag image topic during play.',
    )
    return parser.parse_args()


def validate_bag_path(bag_path: Path) -> None:
    if not bag_path.exists():
        raise SystemExit(f'Bag path does not exist: {bag_path}')
    if not bag_path.is_dir():
        raise SystemExit(f'Bag path is not a directory: {bag_path}')
    metadata = bag_path / 'metadata.yaml'
    if not metadata.exists():
        raise SystemExit(f'Bag metadata is missing: {metadata}')


def main() -> None:
    args = parse_args()
    ros2 = shutil.which('ros2')
    if ros2 is None:
        raise SystemExit('ros2 command not found. Source your ROS 2 environment first.')

    bag_path = resolve_bag_path(args.bag_path)
    validate_bag_path(bag_path)

    cmd = [ros2, 'bag', args.command, str(bag_path)]
    if args.command == 'play' and not args.no_image_remap:
        source_topic = args.image_source_topic.strip() or default_image_topic(args.bag_path)
        target_topic = args.image_topic.strip()
        if source_topic and target_topic and source_topic != target_topic:
            cmd.extend(['--remap', f'{source_topic}:={target_topic}'])
    raise SystemExit(subprocess.run(cmd, check=False).returncode)


if __name__ == '__main__':
    main()
