import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class LoadImagesFromTopic:
    def __init__(self):
        self.topic_name = '/zed_node/left/image_rect_color'
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
            rospy.loginfo("Waiting for image...")
            self.rate.sleep()
        return self.image

if __name__ == "__main__":
    image_loader = LoadImagesFromTopic()

    try:
        while not rospy.is_shutdown():
            img = image_loader.get_image()
            cv2.imshow('Original Image', img)
            if cv2.waitKey(1) & 0xFF == ord('x'):
                break

    except rospy.ROSInterruptException:
        print("ROS node interrupted.")

    cv2.destroyAllWindows()
