#!/usr/bin/env python3
"""
YOLOP Lane Detection with Brightness/Reflection Handling
빛 반사 및 과다 노출 대응 기능 추가
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, PointField
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped
from cv_bridge import CvBridge, CvBridgeError
from tf2_ros import StaticTransformBroadcaster
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from pathlib import Path
import sys
import struct

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
        self.declare_parameter('weights', 'epoch-150.pth')
        self.declare_parameter('yolop_path', '/home/jjs/lane_ws/YOLOP')
        self.declare_parameter('image_topic', '/ardu_cam_link/image_raw')
        self.declare_parameter('camera_info_topic', '/ardu_cam_link/camera_info')
        self.declare_parameter('output_seg_topic', '/lane_node/segmentation')
        self.declare_parameter('output_overlay_topic', '/lane_node/overlay')
        self.declare_parameter('output_cloud_topic', '/lane_node/pointcloud')
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('img_size', 640)
        
        # ROI parameters
        self.declare_parameter('use_roi', False)
        self.declare_parameter('roi_x_start', 0.1)
        self.declare_parameter('roi_y_start', 0.0)
        self.declare_parameter('roi_x_end', 0.9)
        self.declare_parameter('roi_y_end', 1.0)
        
        # Noise removal parameters
        self.declare_parameter('use_noise_removal', True)
        self.declare_parameter('use_gaussian_blur', True)
        self.declare_parameter('gaussian_kernel', 5)
        self.declare_parameter('morph_open_kernel', 2)
        self.declare_parameter('morph_close_kernel', 15)  # 끊긴 차선 연결 강화
        self.declare_parameter('min_area', 100)
        self.declare_parameter('use_shape_filter', False)
        self.declare_parameter('min_width', 40)
        self.declare_parameter('max_height_ratio', 3.0)
        
        # Brightness/Reflection handling (NEW!)
        self.declare_parameter('use_clahe', True)          # CLAHE 사용
        self.declare_parameter('clahe_clip_limit', 3.5)   # 대비 제한 증가 (밝은 환경용)
        self.declare_parameter('clahe_tile_size', 8)      # 타일 크기 (8x8)
        self.declare_parameter('use_gamma_correction', False)  # 감마 보정
        self.declare_parameter('gamma_value', 1.2)        # 감마 값 (0.5~2.0)
        
        # PointCloud parameters
        self.declare_parameter('use_pointcloud', True)
        self.declare_parameter('cam_height', 0.35)
        self.declare_parameter('cam_pitch_deg', -40.0)
        self.declare_parameter('cloud_frame_id', 'camera_link')
        
        # Get parameters
        weights = self.get_parameter('weights').value
        yolop_path = self.get_parameter('yolop_path').value
        image_topic = self.get_parameter('image_topic').value
        camera_info_topic = self.get_parameter('camera_info_topic').value
        output_seg_topic = self.get_parameter('output_seg_topic').value
        output_overlay_topic = self.get_parameter('output_overlay_topic').value
        output_cloud_topic = self.get_parameter('output_cloud_topic').value
        device = self.get_parameter('device').value
        self.img_size = self.get_parameter('img_size').value
        
        # ROI parameters
        self.use_roi = self.get_parameter('use_roi').value
        self.roi_x_start = self.get_parameter('roi_x_start').value
        self.roi_y_start = self.get_parameter('roi_y_start').value
        self.roi_x_end = self.get_parameter('roi_x_end').value
        self.roi_y_end = self.get_parameter('roi_y_end').value
        
        # Noise removal parameters
        self.use_noise_removal = self.get_parameter('use_noise_removal').value
        self.use_gaussian_blur = self.get_parameter('use_gaussian_blur').value
        self.gaussian_kernel = self.get_parameter('gaussian_kernel').value
        self.morph_open_kernel = self.get_parameter('morph_open_kernel').value
        self.morph_close_kernel = self.get_parameter('morph_close_kernel').value
        self.min_area = self.get_parameter('min_area').value
        self.use_shape_filter = self.get_parameter('use_shape_filter').value
        self.min_width = self.get_parameter('min_width').value
        self.max_height_ratio = self.get_parameter('max_height_ratio').value
        
        # Brightness handling (NEW!)
        self.use_clahe = self.get_parameter('use_clahe').value
        self.clahe_clip_limit = self.get_parameter('clahe_clip_limit').value
        self.clahe_tile_size = self.get_parameter('clahe_tile_size').value
        self.use_gamma_correction = self.get_parameter('use_gamma_correction').value
        self.gamma_value = self.get_parameter('gamma_value').value
        
        # CLAHE 객체 생성
        if self.use_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=(self.clahe_tile_size, self.clahe_tile_size)
            )
        
        # PointCloud parameters
        self.use_pointcloud = self.get_parameter('use_pointcloud').value
        self.cam_height = self.get_parameter('cam_height').value
        self.cam_pitch_deg = self.get_parameter('cam_pitch_deg').value
        self.cloud_frame_id = self.get_parameter('cloud_frame_id').value
        
        self.cam_pitch_rad = np.radians(self.cam_pitch_deg)
        
        # Camera intrinsics
        self.fx = self.fy = self.cx = self.cy = None
        self.camera_height = None
        self.camera_width = None
        
        # CV Bridge
        self.bridge = CvBridge()
        self.device_torch = torch.device(device)
        
        # FPS 측정
        self.frame_count = 0
        self.fps_start_time = None
        self.last_fps_print = None
        
        # 프레임 스킵
        self.declare_parameter('frame_skip', 1)
        self.frame_skip = self.get_parameter('frame_skip').value
        self.frame_counter = 0
        
        # TF Broadcaster
        self.tf_broadcaster = StaticTransformBroadcaster(self)
        
        # Load YOLOP model
        self.get_logger().info(f'Loading YOLOP model from {weights}...')
        
        yolop_path = Path(yolop_path)
        if str(yolop_path) not in sys.path:
            sys.path.insert(0, str(yolop_path))
        
        from lib.models import get_net
        from lib.config import cfg
        from lib.utils import show_seg_result
        
        self.show_seg_result = show_seg_result
        
        self.model = get_net(cfg)
        checkpoint = torch.load(weights, map_location=self.device_torch)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to(self.device_torch)
        self.model.eval()
        
        self.get_logger().info(f'Model loaded successfully on {self.device_torch}')
        
        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )
        
        self.camerainfo_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            10
        )
        
        # Publishers
        self.seg_pub = self.create_publisher(Image, output_seg_topic, 10)
        self.overlay_pub = self.create_publisher(Image, output_overlay_topic, 10)
        self.cloud_pub = self.create_publisher(PointCloud2, output_cloud_topic, 10)
        
        self.get_logger().info(f'Subscribed to:')
        self.get_logger().info(f'  - Image: {image_topic}')
        self.get_logger().info(f'  - CameraInfo: {camera_info_topic}')
        self.get_logger().info(f'Publishing to:')
        self.get_logger().info(f'  - Segmentation: {output_seg_topic}')
        self.get_logger().info(f'  - Overlay: {output_overlay_topic}')
        self.get_logger().info(f'  - PointCloud2: {output_cloud_topic}')
        if self.use_roi:
            self.get_logger().info(f'ROI enabled: [{self.roi_x_start:.2f}, {self.roi_y_start:.2f}] to [{self.roi_x_end:.2f}, {self.roi_y_end:.2f}]')
        if self.use_pointcloud:
            self.get_logger().info(f'PointCloud enabled: height={self.cam_height}m, pitch={self.cam_pitch_deg}°')
        if self.use_clahe:
            self.get_logger().info(f'CLAHE enabled: clip={self.clahe_clip_limit}, tile={self.clahe_tile_size}')
        if self.use_gamma_correction:
            self.get_logger().info(f'Gamma correction enabled: gamma={self.gamma_value}')
        self.get_logger().info('YOLOP Lane Detection Node Ready!')
        
        self.publish_static_transforms()
    
    def publish_static_transforms(self):
        """Static TF 발행"""
        transforms = []
        
        t1 = TransformStamped()
        t1.header.stamp = self.get_clock().now().to_msg()
        t1.header.frame_id = 'base_link'
        t1.child_frame_id = 'ardu_cam_link'
        t1.transform.translation.x = 0.1
        t1.transform.translation.y = 0.0
        t1.transform.translation.z = float(self.cam_height)
        t1.transform.rotation.x = 0.0
        t1.transform.rotation.y = 0.0
        t1.transform.rotation.z = 0.0
        t1.transform.rotation.w = 1.0
        transforms.append(t1)
        
        if self.cloud_frame_id != 'ardu_cam_link':
            t2 = TransformStamped()
            t2.header.stamp = self.get_clock().now().to_msg()
            t2.header.frame_id = 'ardu_cam_link'
            t2.child_frame_id = self.cloud_frame_id
            t2.transform.translation.x = 0.0
            t2.transform.translation.y = 0.0
            t2.transform.translation.z = 0.0
            t2.transform.rotation.x = 0.0
            t2.transform.rotation.y = 0.0
            t2.transform.rotation.z = 0.0
            t2.transform.rotation.w = 1.0
            transforms.append(t2)
        
        self.tf_broadcaster.sendTransform(transforms)
        self.get_logger().info(f'Published static TF: base_link → ardu_cam_link → {self.cloud_frame_id}')
    
    def camera_info_callback(self, msg):
        """CameraInfo 콜백"""
        if self.fx is not None:
            return
        
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.camera_height = msg.height
        self.camera_width = msg.width
        
        self.get_logger().info(f'CameraInfo received: fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}')
    
    def use_default_camera_params(self):
        """CameraInfo가 없을 때 기본값"""
        if self.fx is None:
            self.camera_width = 1920
            self.camera_height = 1080
            self.fx = 1000.0
            self.fy = 1000.0
            self.cx = self.camera_width / 2.0
            self.cy = self.camera_height / 2.0
            self.get_logger().warn(f'CameraInfo not received! Using defaults')
            return True
        return False
    
    def enhance_image(self, image):
        """빛 반사/과다 노출 대응 + 차선 강조"""
        
        # ━━━ 1단계: HSV 차선 강조 (NEW!) ━━━
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # 흰색 차선 마스크 (밝은 부분)
        lower_white = np.array([0, 0, 200])  # H, S, V
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # 노란색 차선 마스크
        lower_yellow = np.array([15, 100, 200])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # 차선 마스크 합성
        lane_mask = cv2.bitwise_or(white_mask, yellow_mask)
        
        # 차선 영역 강조 (흰색으로)
        image_enhanced = image.copy()
        image_enhanced[lane_mask > 0] = [255, 255, 255]
        
        # ━━━ 2단계: CLAHE ━━━
        if self.use_clahe:
            lab = cv2.cvtColor(image_enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            lab = cv2.merge([l, a, b])
            image_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # ━━━ 3단계: 감마 보정 ━━━
        if self.use_gamma_correction:
            inv_gamma = 1.0 / self.gamma_value
            table = np.array([((i / 255.0) ** inv_gamma) * 255 
                             for i in range(256)]).astype("uint8")
            image_enhanced = cv2.LUT(image_enhanced, table)
        
        return image_enhanced
    
    def create_pointcloud2(self, header, points):
        """PointCloud2 메시지 생성"""
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        
        cloud_data = []
        for point in points:
            cloud_data.append(struct.pack('fff', point[0], point[1], point[2]))
        
        cloud_msg = PointCloud2()
        cloud_msg.header = header
        cloud_msg.height = 1
        cloud_msg.width = len(points)
        cloud_msg.fields = fields
        cloud_msg.is_bigendian = False
        cloud_msg.point_step = 12
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
        cloud_msg.is_dense = True
        cloud_msg.data = b''.join(cloud_data)
        
        return cloud_msg
    
    def mask_to_pointcloud(self, mask, original_shape):
        """마스크를 3D PointCloud로 변환"""
        self.use_default_camera_params()
        
        if self.fx is None:
            return []
        
        lane_pixels = np.where(mask > 0)
        
        if len(lane_pixels[0]) == 0:
            return []
        
        points_3d = []
        
        c = np.cos(self.cam_pitch_rad)
        s = np.sin(self.cam_pitch_rad)
        R_pitch = np.array([[1, 0, 0],
                           [0, c, -s],
                           [0, s, c]])
        
        step = 5  # 샘플링 간격 (픽셀) - 작을수록 밀도 높음
        for i in range(0, len(lane_pixels[0]), step):
            v = lane_pixels[0][i]
            u = lane_pixels[1][i]
            
            x_cam = (u - self.cx) / self.fx
            y_cam = (v - self.cy) / self.fy
            z_cam = 1.0
            
            direction = np.array([x_cam, y_cam, z_cam])
            direction_rotated = R_pitch @ direction
            
            if abs(direction_rotated[1]) > 1e-6:
                if direction_rotated[1] > 0:
                    t = self.cam_height / direction_rotated[1]
                    
                    x = direction_rotated[0] * t
                    y = 0.0
                    z = direction_rotated[2] * t
                    
                    distance = np.sqrt(z**2 + x**2)
                    if 0.1 < distance < 50.0:
                        point_3d = [z, -x, y]
                        points_3d.append(point_3d)
        
        return points_3d
    
    def apply_roi(self, image):
        h, w = image.shape[:2]
        x1 = int(w * self.roi_x_start)
        y1 = int(h * self.roi_y_start)
        x2 = int(w * self.roi_x_end)
        y2 = int(h * self.roi_y_end)
        roi_image = image[y1:y2, x1:x2]
        roi_info = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'original_shape': (h, w)}
        return roi_image, roi_info
    
    def restore_from_roi(self, mask, roi_info):
        h, w = roi_info['original_shape']
        x1, y1 = roi_info['x1'], roi_info['y1']
        x2, y2 = roi_info['x2'], roi_info['y2']
        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = mask
        return full_mask
    
    def remove_noise(self, mask):
        if not self.use_noise_removal:
            return mask
        
        if self.use_gaussian_blur and self.gaussian_kernel > 0:
            kernel_size = self.gaussian_kernel if self.gaussian_kernel % 2 == 1 else self.gaussian_kernel + 1
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        
        if self.morph_open_kernel > 0:
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.morph_open_kernel, self.morph_open_kernel))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        
        if self.min_area > 0:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            cleaned_mask = np.zeros_like(mask)
            
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]
                
                if area < self.min_area:
                    continue
                if self.use_shape_filter and width < self.min_width:
                    continue
                
                aspect_ratio = height / max(width, 1)
                if self.use_shape_filter and aspect_ratio > self.max_height_ratio:
                    continue
                
                bounding_area = width * height
                density = area / max(bounding_area, 1)
                if self.use_shape_filter and density < 0.2:
                    continue
                
                img_height = mask.shape[0]
                y_center = y + height / 2
                if y_center < img_height * 0.3 and area < self.min_area * 1.5:
                    continue

                cleaned_mask[labels == i] = 255
            
            mask = cleaned_mask
        
        if self.morph_close_kernel > 0:
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morph_close_kernel, self.morph_close_kernel))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.dilate(mask, kernel_dilate, iterations=1)
        
        return mask
        
    def preprocess(self, image):
        h0, w0 = image.shape[:2]
        r = self.img_size / max(h0, w0)
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            image_resized = cv2.resize(image, (int(w0 * r), int(h0 * r)), interpolation=interp)
        else:
            image_resized = image
            
        h, w = image_resized.shape[:2]
        dh = self.img_size - h
        dw = self.img_size - w
        top = dh // 2
        bottom = dh - top
        left = dw // 2
        right = dw - left
        
        image_padded = cv2.copyMakeBorder(image_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        image_rgb = cv2.cvtColor(image_padded, cv2.COLOR_BGR2RGB)
        img_tensor = transform(image_rgb).unsqueeze(0).to(self.device_torch)
        shapes = ((h0, w0), ((h / h0, w / w0), (left, top)))
        
        return img_tensor, image, shapes
    
    def image_callback(self, msg):
        try:
            # 프레임 스킵
            self.frame_counter += 1
            if self.frame_skip > 1 and self.frame_counter % self.frame_skip != 0:
                return
            
            # FPS 측정
            import time
            start_time = time.time()
            
            if self.fps_start_time is None:
                self.fps_start_time = start_time
                self.last_fps_print = start_time
            
            self.frame_count += 1
            
            # 이미지 수신
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            original_full_image = image.copy()
            
            # 빛 반사 대응 (NEW!)
            image = self.enhance_image(image)
            
            if self.use_roi:
                image, roi_info = self.apply_roi(image)
            
            img_tensor, original_img, shapes = self.preprocess(image)
            
            with torch.no_grad():
                det_out, da_seg_out, ll_seg_out = self.model(img_tensor)
            
            _, _, height, width = img_tensor.shape
            pad_w, pad_h = shapes[1][1]
            pad_w, pad_h = int(pad_w), int(pad_h)
            ratio = shapes[1][0][1]
            
            ll_predict = ll_seg_out[:, :, pad_h:(height-pad_h), pad_w:(width-pad_w)]
            ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=(1/ratio), mode='bilinear')
            _, ll_seg_mask = torch.max(ll_seg_mask, 1)
            ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
            
            da_predict = da_seg_out[:, :, pad_h:(height-pad_h), pad_w:(width-pad_w)]
            da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=(1/ratio), mode='bilinear')
            _, da_seg_mask = torch.max(da_seg_mask, 1)
            da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
            
            h0, w0 = shapes[0]
            ll_seg_mask_resized = cv2.resize(ll_seg_mask.astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST)
            da_seg_mask_resized = cv2.resize(da_seg_mask.astype(np.uint8), (w0, h0), interpolation=cv2.INTER_NEAREST)
            
            seg_binary = (ll_seg_mask_resized > 0).astype(np.uint8) * 255
            seg_binary = self.remove_noise(seg_binary)
            
            if self.use_roi:
                seg_binary = self.restore_from_roi(seg_binary, roi_info)
                ll_seg_mask_full = self.restore_from_roi(ll_seg_mask_resized, roi_info)
                da_seg_mask_full = self.restore_from_roi(da_seg_mask_resized, roi_info)
                overlay = self.show_seg_result(original_full_image, (da_seg_mask_full, ll_seg_mask_full), _, _, is_demo=True)
            else:
                overlay = self.show_seg_result(original_img.copy(), (da_seg_mask_resized, ll_seg_mask_resized), _, _, is_demo=True)
            
            timestamp = self.get_clock().now().to_msg()
            
            # 발행
            seg_msg = self.bridge.cv2_to_imgmsg(seg_binary, encoding='mono8')
            seg_msg.header.stamp = timestamp
            seg_msg.header.frame_id = msg.header.frame_id
            self.seg_pub.publish(seg_msg)
            
            overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
            overlay_msg.header.stamp = timestamp
            overlay_msg.header.frame_id = msg.header.frame_id
            self.overlay_pub.publish(overlay_msg)
            
            if self.use_pointcloud:
                points_3d = self.mask_to_pointcloud(seg_binary, original_full_image.shape[:2])
                
                if len(points_3d) > 0:
                    header = Header()
                    header.stamp = timestamp
                    header.frame_id = self.cloud_frame_id
                    
                    cloud_msg = self.create_pointcloud2(header, points_3d)
                    self.cloud_pub.publish(cloud_msg)
            
            # FPS
            end_time = time.time()
            processing_time = end_time - start_time
            
            if end_time - self.last_fps_print >= 1.0:
                elapsed = end_time - self.fps_start_time
                fps = self.frame_count / elapsed
                self.get_logger().info(
                    f'FPS: {fps:.2f} | Processing time: {processing_time*1000:.1f}ms | Total frames: {self.frame_count}'
                )
                self.last_fps_print = end_time
            
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