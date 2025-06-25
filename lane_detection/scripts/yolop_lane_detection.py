#!/usr/bin/env python3
"""
ROS node for real‑time lane‑line segmentation using a YOLOP‑derived network
that **only** returns `ll_seg_out` (lane‑line mask).
All detection (objects) and drivable‑area logic has been removed.
"""
import os;
import rospy
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from cv_bridge import CvBridge, CvBridgeError
import time
# ROS msgs
from sensor_msgs.msg import Image, CompressedImage

# YOLOP imports – make sure your fork that returns only ll_seg_out is on PYTHONPATH
from lib.config import cfg, update_config
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
import sys, torch, os, platform
print("cuda avail?  :", torch.cuda.is_available())

def overlay_lane_mask(img: np.ndarray, mask: np.ndarray, color=(0, 255, 0), alpha: float = 0.4):
    """Overlay binary lane mask on BGR image."""
    overlay = img.copy()
    overlay[mask == 1] = color
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)


class LaneLineNode:
    def __init__(self):
        rospy.init_node("lane_line_node", anonymous=True)

        # ==== Parameters ====
        self.weights_path = rospy.get_param("~weights_path", "weights/LaneLine.pth")
        self.device_name = rospy.get_param("~device", "cuda:0")
        self.img_size = int(rospy.get_param("~img_size", 640))
        self.conf_debug = rospy.get_param("~debug", False)
        self.camera_topic = rospy.get_param("~camera_topic", "/camera/image_raw")
        self.use_compressed = rospy.get_param("~use_compressed", False)

        # transform
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([transforms.ToTensor(), self.normalize])

        self.bridge = CvBridge()
        self.logger, _, _ = create_logger(cfg, cfg.LOG_DIR, "lane_line_ros")
        self.device = select_device(self.logger, self.device_name)
        self.half = self.device.type != "cpu"

        self._load_model()
        self._init_ros_io()
        rospy.loginfo("Lane‑line node initialised and ready.")

    # ------------------------------------------------------------------
    def _load_model(self):
        # 1) 골격
        self.model = get_net(cfg)

        # 2) 파일 읽기 (일단 CPU)
        ckpt = torch.load(self.weights_path, map_location='cpu')

        # 3) 포맷별 분기
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:      # 기존 기대 포맷
            self.model.load_state_dict(ckpt['state_dict'])

        elif isinstance(ckpt, dict):                              # state_dict 단독 저장
            self.model.load_state_dict(ckpt)

        elif isinstance(ckpt, torch.nn.Module):                   # 모델 전체 저장
            self.model = ckpt

        else:
            raise RuntimeError("Unsupported checkpoint format")

        # 4) 디바이스 이동 & FP16
        self.model.to(self.device).eval()
        if self.half:
            self.model.half()

        # 5) 워밍업
        dummy = torch.zeros((1, 3, self.img_size, self.img_size), device=self.device)
        _ = self.model(dummy.half() if self.half else dummy)

        self.logger.info("Weights loaded from %s", self.weights_path)

    # ------------------------------------------------------------------
    def _init_ros_io(self):
        # subscriber
        cb = self.compressed_cb if self.use_compressed else self.image_cb
        msg_type = CompressedImage if self.use_compressed else Image
        self.sub = rospy.Subscriber(self.camera_topic, msg_type, cb, queue_size=1)

        # publishers – lane mask and overlay
        self.lane_mask_pub = rospy.Publisher("~lane_line", Image, queue_size=1)
        self.overlay_pub = rospy.Publisher("~overlay_image", Image, queue_size=1)

    # ------------------------------------------------------------------
    def compressed_cb(self, msg: CompressedImage):
        try:
            cv_img = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
            self._process(cv_img, msg.header)
        except CvBridgeError as e:
            rospy.logerr("CvBridge error: %s", e)

    def image_cb(self, msg: Image):
        t0 = time.perf_counter() 
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self._process(cv_img, msg.header)
        except CvBridgeError as e:
            rospy.logerr("CvBridge error: %s", e)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0   # ⬅ 경과 시간(ms)
        rospy.loginfo_throttle(1.0, f"[ultralytics] 1 frame = {elapsed_ms:.1f} ms")

    # ------------------------------------------------------------------
    def _preprocess(self, img: np.ndarray):
        im = cv2.resize(img, (self.img_size, self.img_size))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        tensor = self.transform(im).to(self.device)
        tensor = tensor.half() if self.half else tensor.float()
        tensor = tensor.unsqueeze(0)
        return tensor

    # ------------------------------------------------------------------
    def _process(self, frame: np.ndarray, header):
        try:
            tensor = self._preprocess(frame)
            with torch.no_grad():
                t1 = time_synchronized()
                a,b,ll_seg_out = self.model(tensor)          # forward
                t2 = time_synchronized()

            # post‑process mask back to original resolution
            mask = torch.argmax(ll_seg_out, 1).int().squeeze().cpu().numpy()
            mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

            # publish lane mask
            mask_msg = self.bridge.cv2_to_imgmsg((mask * 255).astype(np.uint8), "mono8")
            mask_msg.header = header
            self.lane_mask_pub.publish(mask_msg)

            # overlay for visual debugging
            overlay = overlay_lane_mask(frame, mask)
            overlay_msg = self.bridge.cv2_to_imgmsg(overlay, "bgr8")
            overlay_msg.header = header
            self.overlay_pub.publish(overlay_msg)

            if self.conf_debug:
                rospy.loginfo_once("First inference done in %.1f ms", (t2 - t1) * 1000)
        except Exception as e:
            rospy.logerr("Processing error: %s", e)

    # ------------------------------------------------------------------
    def run(self):
        rospy.loginfo("Lane‑line segmentation node running …")
        rospy.spin()


if __name__ == "__main__":
    try:
        node = LaneLineNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
