#!/usr/bin/env python3
"""
Fuse depth and lane mask, **and relay CameraInfo** so downstream nodes
(e.g., depth→cloud) can subscribe to `/lane_depth_fusion/camera_info`.
Publishes:
  • ~lane_depth   (Image, 32FC1)
  • ~overlay_rgb  (Image, bgr8)
  • ~camera_info  (CameraInfo)  ← NEW
"""
import rospy, cv2, numpy as np
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import message_filters

class LaneDepthFusion:
    def __init__(self):
        rospy.init_node('lane_depth_fusion', anonymous=True)

        # ───── Parameters ─────
        depth_topic  = rospy.get_param('~depth_topic',  '/zed/depth/image')
        mask_topic   = rospy.get_param('~mask_topic',   '/lane_line_node/lane_line')
        info_topic   = rospy.get_param('~camera_info_topic', '/zed/depth/camera_info')
        overlay      = rospy.get_param('~publish_overlay', True)
        self.scale_depth = rospy.get_param('~depth_scale', 1.0)

        # ───── Subscribers ─────
        self.bridge = CvBridge()
        depth_sub = message_filters.Subscriber(depth_topic, Image)
        mask_sub  = message_filters.Subscriber(mask_topic,  Image)
        self.sync = message_filters.ApproximateTimeSynchronizer([depth_sub, mask_sub], queue_size=5, slop=0.05)
        self.sync.registerCallback(self.callback)

        # camera info 저장
        self.last_info = None
        rospy.Subscriber(info_topic, CameraInfo, self.info_cb, queue_size=1)

        # ───── Publishers ─────
        self.depth_pub   = rospy.Publisher('~lane_depth', Image, queue_size=1)
        self.overlay_pub = rospy.Publisher('~overlay_rgb', Image, queue_size=1) if overlay else None
        self.info_pub    = rospy.Publisher('~camera_info', CameraInfo, queue_size=1)

        rospy.loginfo('LaneDepthFusion ready. depth="%s" mask="%s" info="%s"', depth_topic, mask_topic, info_topic)

    # ---------------------------------------------------------------
    def info_cb(self, msg: CameraInfo):
        """Store the latest CameraInfo so we can relay it with matched stamp."""
        self.last_info = msg

    # ---------------------------------------------------------------
    def callback(self, depth_msg: Image, mask_msg: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough').astype(np.float32)
            mask  = self.bridge.imgmsg_to_cv2(mask_msg,  desired_encoding='mono8')
        except CvBridgeError as e:
            rospy.logerr_throttle(5, 'CvBridge error: %s', e)
            return

        mask_bin = (mask > 0).astype(np.float32)
        fused    = depth * mask_bin * self.scale_depth

        # ─── Publish fused depth ───
        try:
            fused_msg = self.bridge.cv2_to_imgmsg(fused, encoding='32FC1')
            fused_msg.header = depth_msg.header  # copy stamp + frame
            self.depth_pub.publish(fused_msg)
        except CvBridgeError as e:
            rospy.logerr_throttle(5, 'CvBridge error publish: %s', e)

        # ─── Relay CameraInfo with synced stamp ───
        if self.last_info is not None:
            info_out = CameraInfo()
            info_out.header = fused_msg.header  # same stamp/frame
            info_out.height = self.last_info.height
            info_out.width  = self.last_info.width
            info_out.K      = list(self.last_info.K)
            info_out.D      = list(self.last_info.D)
            info_out.R      = list(self.last_info.R)
            info_out.P      = list(self.last_info.P)
            info_out.distortion_model = self.last_info.distortion_model
            self.info_pub.publish(info_out)

        # ─── Optional overlay ───
        if self.overlay_pub and self.overlay_pub.get_num_connections():
            overlay = cv2.cvtColor((mask_bin * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            overlay[:, :, 1] = 255  # green mask
            depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)
            show = cv2.addWeighted(depth_vis, 1.0, overlay, 0.6, 0)
            try:
                show_msg = self.bridge.cv2_to_imgmsg(show, encoding='bgr8')
                show_msg.header = fused_msg.header
                self.overlay_pub.publish(show_msg)
            except CvBridgeError:
                pass

    # ---------------------------------------------------------------
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        LaneDepthFusion().run()
    except rospy.ROSInterruptException:
        pass
