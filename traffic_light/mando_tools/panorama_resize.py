"""Resize replayed panorama images without changing the recorded bag or header."""

from __future__ import annotations

import sys

from .workspace_paths import local_python_deps_path

deps_path = local_python_deps_path()
if deps_path is not None and deps_path.exists():
    sys.path.insert(0, str(deps_path))

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Image


def resize_message(message: Image, width: int, height: int, bridge: CvBridge) -> Image:
    """Keep encoding, timestamp and frame ID; change only the image dimensions.

    This is a pixel resize, not a calibrated reprojection. CameraInfo is not
    forwarded because its intrinsics would need a corresponding transformation.
    """
    if width <= 0 or height <= 0:
        raise ValueError('Output width and height must be positive')
    if message.width <= 0 or message.height <= 0:
        raise ValueError('Input image is empty')
    if message.width == width and message.height == height:
        return message
    frame = bridge.imgmsg_to_cv2(message, desired_encoding='passthrough')
    interpolation = (
        cv2.INTER_AREA if width <= message.width and height <= message.height
        else cv2.INTER_LINEAR
    )
    resized = cv2.resize(frame, (width, height), interpolation=interpolation)
    output = bridge.cv2_to_imgmsg(resized, encoding=message.encoding)
    output.header = message.header
    return output


class PanoramaResizeNode(Node):
    """Relay a separate recorded-image topic to the normal panorama topic."""

    def __init__(self) -> None:
        super().__init__('panorama_resize')
        input_topic = str(self.declare_parameter('input_topic', '/panorama/image_raw_recorded').value)
        output_topic = str(self.declare_parameter('output_topic', '/panorama/image_raw').value)
        self.width = int(self.declare_parameter('output_width', 1254).value)
        self.height = int(self.declare_parameter('output_height', 370).value)
        if self.width <= 0 or self.height <= 0:
            raise ValueError('Output width and height must be positive')
        if self.resolve_topic_name(input_topic) == self.resolve_topic_name(output_topic):
            raise ValueError('Input and output topics must differ to avoid a feedback loop')
        self.bridge = CvBridge()
        self.publisher = self.create_publisher(
            Image, output_topic, QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
        )
        self.subscription = self.create_subscription(
            Image, input_topic, self._image_callback, qos_profile_sensor_data,
        )
        self.get_logger().info(
            f'{input_topic} -> {output_topic}: {self.width}x{self.height}; header preserved'
        )

    def _image_callback(self, message: Image) -> None:
        try:
            self.publisher.publish(resize_message(message, self.width, self.height, self.bridge))
        except (ValueError, cv2.error, RuntimeError, TypeError) as error:
            self.get_logger().error(f'Failed to resize panorama: {error}', throttle_duration_sec=5.0)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = PanoramaResizeNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
