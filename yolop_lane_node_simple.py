#!/usr/bin/env python3
"""
YOLOP Lane Detection ROS2 Node (Simplified)
이미지만 받아서 차선 검출 후 세그멘테이션 마스크 퍼블리시
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from pathlib import Path
import sys


# Normalization
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225]
)

transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])


class YOLOPLaneDetectionNode(Node):
    def __init__(self):
        super().__init__('yolop_lane_detection_node')
        
        # Parameters
        self.declare_parameter('weights', 'W_07-07_e130.pth')
        self.declare_parameter('yolop_path', '/home/jjs/lane_ws/YOLOP')
        self.declare_parameter('image_topic', '/ardu_cam_link/image_raw')
        self.declare_parameter('output_seg_topic', '/lane_node/segmentation')
        self.declare_parameter('output_overlay_topic', '/lane_node/overlay')
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('img_size', 640)
        
        # Get parameters
        weights = self.get_parameter('weights').value
        yolop_path = self.get_parameter('yolop_path').value
        image_topic = self.get_parameter('image_topic').value
        output_seg_topic = self.get_parameter('output_seg_topic').value
        output_overlay_topic = self.get_parameter('output_overlay_topic').value
        device = self.get_parameter('device').value
        self.img_size = self.get_parameter('img_size').value
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Load YOLOP model
        self.get_logger().info(f'Loading YOLOP model from {weights}...')
        self.device = torch.device(device)
        
        # YOLOP path 추가
        yolop_path = Path(yolop_path)
        if str(yolop_path) not in sys.path:
            sys.path.insert(0, str(yolop_path))
        
        from lib.models import get_net
        from lib.config import cfg
        from lib.utils import show_seg_result
        
        self.show_seg_result = show_seg_result
        
        self.model = get_net(cfg)
        checkpoint = torch.load(weights, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.get_logger().info(f'Model loaded successfully on {self.device}')
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )
        
        # Publishers
        self.seg_pub = self.create_publisher(Image, output_seg_topic, 10)
        self.overlay_pub = self.create_publisher(Image, output_overlay_topic, 10)
        
        self.get_logger().info(f'Subscribed to: {image_topic}')
        self.get_logger().info(f'Publishing to:')
        self.get_logger().info(f'  - Segmentation: {output_seg_topic}')
        self.get_logger().info(f'  - Overlay: {output_overlay_topic}')
        self.get_logger().info('YOLOP Lane Detection Node Ready!')
        
    def preprocess(self, image):
        """이미지 전처리"""
        # 원본 크기 저장
        h0, w0 = image.shape[:2]
        
        # 리사이즈 (aspect ratio 유지)
        r = self.img_size / max(h0, w0)
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            image_resized = cv2.resize(image, (int(w0 * r), int(h0 * r)), interpolation=interp)
        else:
            image_resized = image
            
        # 패딩 추가 (정사각형으로)
        h, w = image_resized.shape[:2]
        dh = self.img_size - h
        dw = self.img_size - w
        
        # 중앙 정렬 패딩
        top = dh // 2
        bottom = dh - top
        left = dw // 2
        right = dw - left
        
        image_padded = cv2.copyMakeBorder(
            image_resized, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # RGB로 변환
        image_rgb = cv2.cvtColor(image_padded, cv2.COLOR_BGR2RGB)
        
        # Tensor 변환
        img_tensor = transform(image_rgb).unsqueeze(0).to(self.device)
        
        # Shape 정보
        shapes = ((h0, w0), ((h / h0, w / w0), (left, top)))
        
        return img_tensor, image, shapes
    
    def image_callback(self, msg):
        """이미지 토픽 콜백 - 받자마자 바로 처리"""
        try:
            # ROS Image → OpenCV
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # 전처리
            img_tensor, original_img, shapes = self.preprocess(image)
            
            # 추론
            with torch.no_grad():
                det_out, da_seg_out, ll_seg_out = self.model(img_tensor)
            
            # Lane segmentation 후처리
            _, _, height, width = img_tensor.shape
            pad_w, pad_h = shapes[1][1]
            pad_w, pad_h = int(pad_w), int(pad_h)
            ratio = shapes[1][0][1]
            
            # Lane line segmentation
            ll_predict = ll_seg_out[:, :, pad_h:(height-pad_h), pad_w:(width-pad_w)]
            ll_seg_mask = torch.nn.functional.interpolate(
                ll_predict, 
                scale_factor=(1/ratio), 
                mode='bilinear'
            )
            _, ll_seg_mask = torch.max(ll_seg_mask, 1)
            ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
            
            # Drivable area segmentation
            da_predict = da_seg_out[:, :, pad_h:(height-pad_h), pad_w:(width-pad_w)]
            da_seg_mask = torch.nn.functional.interpolate(
                da_predict, 
                scale_factor=(1/ratio), 
                mode='bilinear'
            )
            _, da_seg_mask = torch.max(da_seg_mask, 1)
            da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
            
            # 원본 크기로 복원
            h0, w0 = shapes[0]
            ll_seg_mask_resized = cv2.resize(
                ll_seg_mask.astype(np.uint8), 
                (w0, h0), 
                interpolation=cv2.INTER_NEAREST
            )
            
            da_seg_mask_resized = cv2.resize(
                da_seg_mask.astype(np.uint8),
                (w0, h0),
                interpolation=cv2.INTER_NEAREST
            )
            
            # 이진화 (흰색 차선, 검은 배경)
            seg_binary = (ll_seg_mask_resized > 0).astype(np.uint8) * 255
            
            # 오버레이 이미지 생성
            overlay = self.show_seg_result(
                original_img.copy(), 
                (da_seg_mask_resized, ll_seg_mask_resized), 
                _, _, 
                is_demo=True
            )
            
            # Timestamp
            timestamp = self.get_clock().now().to_msg()
            
            # 퍼블리시 - Segmentation (흰색 차선, 검은 배경)
            seg_msg = self.bridge.cv2_to_imgmsg(seg_binary, encoding='mono8')
            seg_msg.header.stamp = timestamp
            seg_msg.header.frame_id = msg.header.frame_id
            self.seg_pub.publish(seg_msg)
            
            # 퍼블리시 - Overlay
            overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
            overlay_msg.header.stamp = timestamp
            overlay_msg.header.frame_id = msg.header.frame_id
            self.overlay_pub.publish(overlay_msg)
            
        except CvBridgeError as e:
            self.get_logger().error(f'CV Bridge Error: {e}')
        except Exception as e:
            self.get_logger().error(f'Processing Error: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = YOLOPLaneDetectionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()