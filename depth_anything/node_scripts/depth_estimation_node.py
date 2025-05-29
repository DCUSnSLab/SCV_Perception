#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import time
from utils import preprocess, INPUT_WIDTH, INPUT_HEIGHT, postprocess_depth  # 기존 유틸 그대로 사용


class DepthAnythingNode:
    def __init__(self):
        # ───────────────────────── ROS 파라미터 ─────────────────────────
        self.bridge       = CvBridge()
        self.engine_path  = rospy.get_param("~model_path", None)
        self.depth_scale  = rospy.get_param("~depth_scale", 1.0)

        if self.engine_path is None:
            rospy.logfatal("~model_path 파라미터가 없습니다!")
            raise RuntimeError("model_path missing")

        # ────────────────────── TensorRT 엔진 로딩 ──────────────────────
        logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine  = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream  = cuda.Stream()

        # ---------- I/O 텐서 이름·shape·버퍼 할당 ----------
        self.input_name  = None
        self.output_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
            else:
                self.output_name = name

        if self.input_name is None or self.output_name is None:
            raise RuntimeError("엔진의 입·출력 텐서를 찾지 못했습니다")

        self.input_shape  = tuple(self.engine.get_tensor_shape(self.input_name))   # e.g. (1,3,384,512)
        self.output_shape = tuple(self.engine.get_tensor_shape(self.output_name))  # e.g. (1,1,384,512)

        # Host(Page-locked)/Device 버퍼
        self.h_input  = cuda.pagelocked_empty(trt.volume(self.input_shape), 
                                              dtype=trt.nptype(self.engine.get_tensor_dtype(self.input_name)))
        self.h_output = cuda.pagelocked_empty(trt.volume(self.output_shape),
                                              dtype=trt.nptype(self.engine.get_tensor_dtype(self.output_name)))
        self.d_input  = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)

        # ────────────────────────── ROS I/O ────────────────────────────
        self.image_msg = None
        self.sub_image = rospy.Subscriber("~input_image",  Image, self.image_cb,  queue_size=1, buff_size=2**24)
        self.pub_depth = rospy.Publisher ("~depth_registered/image_rect", Image, queue_size=1)

    # ────────────────────── ROS 콜백 & 추론 루프 ───────────────────────
    def image_cb(self, msg):
        self.image_msg = msg

    def infer_once(self):
        if self.image_msg is None:
            return
        # 1) ROS → numpy (RGB)
        t0 = time.perf_counter() 
        img = self.bridge.imgmsg_to_cv2(self.image_msg, desired_encoding="rgb8")
        orig_h, orig_w = img.shape[:2]
        # 2) 전처리 & 입력 버퍼 복사# 8ms
        img_processed = preprocess(img).ravel() 
        np.copyto(self.h_input, img_processed)
        # 3) Host → Device
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        # 4) 텐서 주소 등록
        self.context.set_tensor_address(self.input_name,  int(self.d_input))
        self.context.set_tensor_address(self.output_name, int(self.d_output))
        # 5) 비동기 추론
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        # 6) Device → Host
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        # 7) 후처리 & 퍼블리시
        depth = self.h_output.reshape(self.output_shape).squeeze() * self.depth_scale
        depth = cv2.resize(depth, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        depth = postprocess_depth(depth)
        depth_msg = self.bridge.cv2_to_imgmsg(depth, encoding="32FC1")
        depth_msg.header.stamp  = self.image_msg.header.stamp
        depth_msg.header.frame_id = self.image_msg.header.frame_id
        self.pub_depth.publish(depth_msg)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0   # ⬅ 경과 시간(ms)
        rospy.loginfo_throttle(1.0, f"[DepthAnything] 1 frame = {elapsed_ms:.1f} ms")

        self.image_msg = None

    def spin(self):
        rate = rospy.Rate(30)      # 30 Hz 추론 시도
        while not rospy.is_shutdown():
            self.infer_once()
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("depth_anything_node")
    DepthAnythingNode().spin()
