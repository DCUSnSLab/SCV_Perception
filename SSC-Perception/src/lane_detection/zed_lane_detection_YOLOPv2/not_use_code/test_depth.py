#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

class DepthImageRepublisher:
    def __init__(self, input_topic, output_topic):
        self.bridge = CvBridge()

        self.image_subscriber = rospy.Subscriber(input_topic, Image, self.image_callback)

        self.image_publisher = rospy.Publisher(output_topic, Image, queue_size=10)


    def image_callback(self, msg):
        try:
            # 원본 Depth Image 그대로 퍼블리시
            self.image_publisher.publish(msg)

        except CvBridgeError as e:
            rospy.logerr(f"Error republishing depth image: {e}")

if __name__ == '__main__':
    rospy.init_node('depth_image_republisher_node', anonymous=True)
    input_topic = '/zed2i/zed_node/depth/depth_registered'
    output_topic = '/republished_depth_image'

    republisher = DepthImageRepublisher(input_topic, output_topic)

    rospy.spin()
