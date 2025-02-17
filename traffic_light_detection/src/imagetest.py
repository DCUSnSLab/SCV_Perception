#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

def image_callback(msg):
    # CvBridge를 사용하여 ROS Image 메시지를 OpenCV 이미지로 변환
    bridge = CvBridge()
    image = bridge.imgmsg_to_cv2(msg, "bgr8")
    
    # 이미지 크기 출력
    height, width, channels = image.shape
    print(f"Image size: {width}x{height}, Channels: {channels}")

def main():
    rospy.init_node('image_size_checker_zed', anonymous=True)
    rospy.Subscriber("/zed_node/rgb/image_rect_color", Image, image_callback)
    rospy.spin()

if __name__ == '__main__':
    main()

