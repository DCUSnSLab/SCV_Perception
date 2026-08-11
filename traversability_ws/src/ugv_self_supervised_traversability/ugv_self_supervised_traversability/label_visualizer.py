"""Overlay delayed positive labels on their matching buffered RGB frames."""

from collections import OrderedDict

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from . import TRAVERSABLE
from .traversability_labeler import stamp_to_ns


class LabelVisualizer(Node):
    """Publish a green translucent traversability overlay."""

    def __init__(self) -> None:
        super().__init__('label_visualizer')
        self.declare_parameter('labeled_image_topic', '/traversability/labeled_image')
        self.declare_parameter('positive_mask_topic', '/traversability/positive_mask')
        self.declare_parameter('debug_image_topic', '/traversability/debug_image')
        self.declare_parameter('debug_visualization', True)
        self.declare_parameter('overlay_alpha', 0.45)
        self.enabled = bool(self.get_parameter('debug_visualization').value)
        self.alpha = float(self.get_parameter('overlay_alpha').value)
        self.bridge = CvBridge()
        self.images = OrderedDict()
        self.masks = OrderedDict()
        self.publisher = self.create_publisher(
            Image, str(self.get_parameter('debug_image_topic').value), 10)
        self.image_sub = self.create_subscription(
            Image, str(self.get_parameter('labeled_image_topic').value),
            self._image_callback, qos_profile_sensor_data)
        self.mask_sub = self.create_subscription(
            Image, str(self.get_parameter('positive_mask_topic').value),
            self._mask_callback, qos_profile_sensor_data)

    def _image_callback(self, message: Image) -> None:
        if not self.enabled:
            return
        self.images[stamp_to_ns(message.header.stamp)] = message
        self._trim(self.images)
        self._try_publish(stamp_to_ns(message.header.stamp))

    def _mask_callback(self, message: Image) -> None:
        if not self.enabled:
            return
        self.masks[stamp_to_ns(message.header.stamp)] = message
        self._trim(self.masks)
        self._try_publish(stamp_to_ns(message.header.stamp))

    @staticmethod
    def _trim(cache: OrderedDict) -> None:
        while len(cache) > 30:
            cache.popitem(last=False)

    def _try_publish(self, timestamp_ns: int) -> None:
        if timestamp_ns not in self.images or timestamp_ns not in self.masks:
            return
        image_message = self.images.pop(timestamp_ns)
        mask_message = self.masks.pop(timestamp_ns)
        try:
            image = np.asarray(self.bridge.imgmsg_to_cv2(image_message, 'bgr8')).copy()
            mask = np.asarray(self.bridge.imgmsg_to_cv2(mask_message, 'mono8'))
        except CvBridgeError as error:
            self.get_logger().warning('Overlay conversion failed: %s' % error)
            return
        if image.shape[:2] != mask.shape:
            self.get_logger().warning('Overlay image/mask size mismatch')
            return
        color = np.zeros_like(image)
        color[:, :, 1] = 255
        selected = mask == TRAVERSABLE
        image[selected] = cv2.addWeighted(image, 1.0 - self.alpha, color,
                                          self.alpha, 0.0)[selected]
        output = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        output.header = image_message.header
        self.publisher.publish(output)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LabelVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        rclpy.try_shutdown()
