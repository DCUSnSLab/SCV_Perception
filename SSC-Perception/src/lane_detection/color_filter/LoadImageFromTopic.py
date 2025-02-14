import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class LoadImagesFromTopic:
    def __init__(self, topic_name):
        self.topic_name = topic_name
        self.bridge = CvBridge()
        self.image = None
        rospy.init_node('image_listener', anonymous=True)
        self.subscriber = rospy.Subscriber(self.topic_name, Image, self.callback)
        self.rate = rospy.Rate(10)  # 10 Hz

    def callback(self, msg):
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except CvBridgeError as e:
            print(f"Error converting ROS Image message to OpenCV image: {e}")

    def get_image(self):
        while self.image is None:
            pass
        return self.image

if __name__ == "__main__":
    topic_name = '/zed_node/rgb/image_rect_color'

    image_loader = LoadImagesFromTopic(topic_name)
    img = image_loader.get_image()
    print(img.shape)

    cv2.destroyAllWindows()
