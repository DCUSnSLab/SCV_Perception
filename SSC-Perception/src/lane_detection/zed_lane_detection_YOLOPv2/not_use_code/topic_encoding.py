import rospy
from sensor_msgs.msg import Image

def callback(data):
    print(f"Encoding: {data.encoding}")

rospy.init_node('encoding_checker', anonymous=True)
rospy.Subscriber("/masked_depth_output", Image, callback)
rospy.spin()