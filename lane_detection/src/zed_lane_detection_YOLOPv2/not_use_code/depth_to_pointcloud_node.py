import rospy
import numpy as np
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge, CvBridgeError

class DepthToPointCloud:
    def __init__(self):
        rospy.init_node('depth_to_pointcloud_node', anonymous=True)
        self.bridge = CvBridge()
        self.camera_info = None
        
        # 카메라 정보 구독
        self.camera_info_sub = rospy.Subscriber('/zed2i/zed_node/depth/camera_info', CameraInfo, self.camera_info_callback)
        self.depth_sub = rospy.Subscriber('/masked_depth_output', Image, self.depth_callback)
        self.pc_pub = rospy.Publisher('/masked_depth_pointcloud', PointCloud2, queue_size=10)

    def camera_info_callback(self, camera_info_msg):
        self.camera_info = camera_info_msg

    def depth_callback(self, depth_msg):
        if self.camera_info is None:
            rospy.logwarn("Waiting for camera info...")
            return
        
        try:
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
            point_cloud_msg = self.depth_to_pointcloud(depth_image)
            self.pc_pub.publish(point_cloud_msg)
        except CvBridgeError as e:
            rospy.logerr(f"Failed to convert depth image to OpenCV format: {e}")

    def depth_to_pointcloud(self, depth_image):
        height, width = depth_image.shape
        fx = self.camera_info.K[0]  # focal length x
        fy = self.camera_info.K[4]  # focal length y
        cx = self.camera_info.K[2]  # optical center x
        cy = self.camera_info.K[5]  # optical center y

        points = []
        for v in range(height):
            for u in range(width):
                z = depth_image[v, u]
                if np.isfinite(z) and z > 0:
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    points.append([x, y, z])

        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1)
        ]

        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.camera_info.header.frame_id  # 카메라 프레임 ID를 camera_info에서 가져옴

        point_cloud = pc2.create_cloud(header, fields, points)
        return point_cloud

if __name__ == '__main__':
    DepthToPointCloud()
    rospy.spin()
