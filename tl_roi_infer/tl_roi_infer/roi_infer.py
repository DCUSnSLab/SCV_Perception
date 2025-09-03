#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import rclpy
from rclpy.node import Node
try:
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
except ImportError:
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy as ReliabilityPolicy, QoSHistoryPolicy as HistoryPolicy

from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Int32
from cv_bridge import CvBridge
from ultralytics import YOLO
import torch, cv2

try:
    from ament_index_python.packages import get_package_share_directory
except Exception:
    get_package_share_directory = None


class RoiInfer(Node):
    def __init__(self):
        super().__init__('roi_infer')

        # -------- default model path (패키지 내 -> 없으면 로컬) --------
        default_model_path = 'model/best.pt'
        try:
            if get_package_share_directory is not None:
                share_dir = get_package_share_directory('tl_roi_infer')
                cand = os.path.join(share_dir, 'models', 'best.pt')
                if os.path.exists(cand):
                    default_model_path = cand
        except Exception:
            pass

        # -------- params --------
        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('debug_topic', '/roi/debug_image')
        self.declare_parameter('state_id_topic', '/tl/state_id')

        self.declare_parameter('imgsz', 960)
        self.declare_parameter('conf', 0.25)
        self.declare_parameter('iou', 0.50)
        self.declare_parameter('agnostic_nms', True)
        self.declare_parameter('half', False)
        self.declare_parameter('roi_top_ratio', 0.00)
        self.declare_parameter('roi_bottom_ratio', 0.35)
        self.declare_parameter('roi_left_ratio', 0.25)
        self.declare_parameter('roi_right_ratio', 0.75)
        default_device = 0 if torch.cuda.is_available() else 'cpu'
        self.declare_parameter('device', default_device)

        g = lambda k: self.get_parameter(k).value
        self.model_path = str(g('model_path'))
        self.image_topic = str(g('image_topic'))
        self.debug_topic = str(g('debug_topic'))
        self.state_id_topic = str(g('state_id_topic'))
        self.imgsz = int(g('imgsz'))
        self.conf = float(g('conf')); self.iou = float(g('iou'))
        self.agnostic_nms = bool(g('agnostic_nms')); self.half = bool(g('half'))

        # ROI clamp
        self.r_top    = max(0.0, min(float(g('roi_top_ratio')),    1.0))
        self.r_bottom = max(0.0, min(float(g('roi_bottom_ratio')), 1.0))
        self.r_left   = max(0.0, min(float(g('roi_left_ratio')),   1.0))
        self.r_right  = max(0.0, min(float(g('roi_right_ratio')),  1.0))

        dev = g('device')
        self.device = dev if isinstance(dev, str) else (dev if (dev >= 0 and torch.cuda.is_available()) else 'cpu')

        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=5)

        # maybe... choose topic type
        types = dict(self.get_topic_names_and_types()).get(self.image_topic, [])
        if 'sensor_msgs/msg/CompressedImage' in types:
            MsgT = CompressedImage
            self.decode = lambda m: self.bridge.compressed_imgmsg_to_cv2(m)
            self.get_logger().info(f'{self.image_topic} -> CompressedImage')
        else:
            MsgT = Image
            self.decode = lambda m: self.bridge.imgmsg_to_cv2(m, desired_encoding='bgr8')
            self.get_logger().info(f'{self.image_topic} -> Image (RAW assumed)')

        self.sub = self.create_subscription(MsgT, self.image_topic, self.cb, qos)
        self.pub_dbg = self.create_publisher(Image, self.debug_topic, 10)
        self.pub_state_id = self.create_publisher(Int32, self.state_id_topic, 10)

        self.label2id = {'red': 1, 'yellow': 2, 'green': 3, 'left': 4}

        devname = f'cuda:{self.device}' if isinstance(self.device, int) else self.device
        self.get_logger().info(f'Loaded: {self.model_path} | device={devname} imgsz={self.imgsz} conf={self.conf} iou={self.iou} half={self.half}')
        self.get_logger().info(f'ROI ratios t/b/l/r = {self.r_top}/{self.r_bottom}/{self.r_left}/{self.r_right}')
        self.get_logger().info(f'Publishing debug to: {self.debug_topic}')
        self.get_logger().info(f'Publishing state_id to: {self.state_id_topic}')

    def cb(self, msg):
        frame = self.decode(msg)
        if frame is None:
            return
        H, W = frame.shape[:2]

        # ROI
        x1 = int(W * self.r_left);  x2 = int(W * self.r_right)
        y1 = int(H * self.r_top);   y2 = int(H * self.r_bottom)
        x1 = max(0, min(x1, W-1));  x2 = max(x1+1, min(x2, W))
        y1 = max(0, min(y1, H-1));  y2 = max(y1+1, min(y2, H))
        roi = frame[y1:y2, x1:x2]

        # infer
        r = self.model(
            roi,
            imgsz=self.imgsz, conf=self.conf, iou=self.iou,
            agnostic_nms=self.agnostic_nms, device=self.device,
            half=(self.half and self.device != 'cpu'),
            verbose=False
        )[0]

        # overlay
        disp = frame.copy()
        disp[y1:y2, x1:x2] = r.plot()
        cv2.rectangle(disp, (x1, y1), (x2-1, y2-1), (0,255,0), 2)

        # log + 상태 산출 (탑-1)
        label, conf = "none", 0.0
        if len(r.boxes):
            names = r.names
            best = max(r.boxes, key=lambda b: float(b.conf))
            label = names[int(best.cls)]
            conf = float(best.conf)
            items = [f"{names[int(b.cls)]}:{float(b.conf):.2f}" for b in r.boxes]
            self.get_logger().info("Det: " + ", ".join(items))

        # 결과 퍼블리시 (Int32)
        m_id = Int32();  m_id.data = self.label2id.get(label, 0)  # 0=unknown
        self.pub_state_id.publish(m_id)

        # publish debug image
        out = self.bridge.cv2_to_imgmsg(disp, encoding='bgr8')
        out.header = msg.header
        self.pub_dbg.publish(out)


def main():
    rclpy.init()
    node = RoiInfer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
