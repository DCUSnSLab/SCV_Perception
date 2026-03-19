#!/home/jay/anaconda3/envs/ros_pcdet/bin/python

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from tracking_msgs.msg import DetectedObject, DetectedObjectArray
import numpy as np
import torch
import os
from easydict import EasyDict
from sensor_msgs_py import point_cloud2 as pc2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Quaternion
import math

# OpenPCDet 라이브러리
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.datasets.processor.data_processor import DataProcessor

class PointPillarsNode(Node):
    def __init__(self):
        super().__init__('pointpillars_node')

        # [수정 1] QoS 프로파일 생성 (Best Effort = UDP처럼 동작)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT, # 늦으면 버림 (끊김 방지 핵심)
            history=HistoryPolicy.KEEP_LAST,
            depth=1 # 최신 1개만 유지 (버퍼 쌓임 방지)
        )
        # 1. 시각화용 퍼블리셔 (RViz)
        self.pub_markers = self.create_publisher(MarkerArray, '/detections/visual_markers', 10)
        # 2. 트래킹용 데이터 퍼블리셔 (AB3DMOT)
        self.pub = self.create_publisher(DetectedObjectArray, '/detected_objects_3d', 10)
        
        # 작업 경로 변경
        work_dir = '/home/jay/OpenPCDet/tools'
        if os.path.exists(work_dir):
            os.chdir(work_dir)
            self.get_logger().info(f"Changed working directory to: {os.getcwd()}")
        
        # Config & Checkpoint 설정
        self.cfg_file = 'cfgs/kitti_models/pv_rcnn.yaml'
        self.ckpt_file = 'pv_rcnn_8369.pth'
        
        # Config 로드
        self.get_logger().info('Loading Config...')
        cfg_from_yaml_file(self.cfg_file, cfg)

        # 클래스 이름 저장
        self.class_names = ['Car', 'Pedestrian', 'Cyclist']
        cfg.CLASS_NAMES = self.class_names

        # 가짜 데이터셋 설정
        demo_dataset = EasyDict(class_names=cfg.CLASS_NAMES)
        demo_dataset.point_feature_encoder = EasyDict(num_point_features=4)
        demo_dataset.depth_downsample_factor = None
        
        point_cloud_range = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
        voxel_size = np.array(cfg.DATA_CONFIG.DATA_PROCESSOR[2].VOXEL_SIZE)
        grid_size = (point_cloud_range[3:6] - point_cloud_range[0:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)
        
        demo_dataset.grid_size = grid_size
        demo_dataset.voxel_size = voxel_size
        demo_dataset.point_cloud_range = point_cloud_range

        # 모델 빌드
        self.get_logger().info('Building Model...')
        self.model = build_network(
            model_cfg=cfg.MODEL, 
            num_class=len(cfg.CLASS_NAMES), 
            dataset=demo_dataset
        )
        self.model.load_params_from_file(filename=self.ckpt_file, to_cpu=False, logger=self.get_logger())
        self.model.cuda()
        self.model.eval()
        self.get_logger().info('Model Loaded Successfully!')

        # 데이터 프로세서 초기화
        self.processor = DataProcessor(
            processor_configs=cfg.DATA_CONFIG.DATA_PROCESSOR,
            point_cloud_range=np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE),
            training=False,
            num_point_features=4
        )

        # ROS 통신 설정
        self.sub = self.create_subscription(
            PointCloud2,
            '/no_ground_points', 
            self.lidar_callback,
            qos_profile  # <--- 여기에 10 대신 qos_profile 넣기
        )

    def lidar_callback(self, msg):
        points = self.pointcloud2_to_array(msg)
        if points.shape[0] == 0:
            return
        
        # [진단] 현재 들어오는 데이터의 Z 평균을 찍어보세요.
        # avg_z = points[:, 2].mean()
        # self.get_logger().info(f"Avg Z: {avg_z}")

        # [해결] 만약 바닥이 0.0 근처라면, KITTI 기준(-1.73)으로 강제 이동
        # Ground Removal을 거쳤어도 모델 입력 좌표계는 센서 기준이어야 좋을 때가 많음

        points[:, 2] -= 1.73
        
        # Intensity 채우기
        if points.shape[1] < 4:
            points = np.hstack([points[:, :3], np.zeros((points.shape[0], 1))])
        
        # [옵션] Z축 높이 보정 (필요시 주석 해제)
        # points[:, 2] -= 1.73 

        input_dict = {
            'points': points,
            'frame_id': msg.header.frame_id,
            'use_lead_xyz': True
        }

        # 전처리 및 추론
        data_dict = self.processor.forward(input_dict)
        data_dict = self.prepare_data_for_inference(data_dict)
        
        with torch.no_grad():
            load_data_to_gpu(data_dict)
            pred_dicts, _ = self.model.forward(data_dict)
        
        # === 1. 트래킹 노드로 데이터 전송 (Threshold 0.1) ===
        # 원본 예측 결과를 그대로 넘김 (함수 안에서 0.1 필터링)
        self.publish_results(pred_dicts[0], msg.header)

        # === 2. RViz 시각화 (Threshold 0.3 ~ 0.5) ===
        pred_boxes = pred_dicts[0]['pred_boxes']
        pred_scores = pred_dicts[0]['pred_scores']
        pred_labels = pred_dicts[0]['pred_labels']

        # 시각화는 좀 더 확실한 것만 보여주기 위해 0.3 사용 (조절 가능)
        vis_thresh = 0.1
        mask = pred_scores > vis_thresh
        
        self.publish_markers(pred_boxes[mask], pred_scores[mask], pred_labels[mask], msg.header)

    def prepare_data_for_inference(self, data_dict):
        for key, val in data_dict.items():
            if key in ['voxels', 'voxel_num_points']:
                data_dict[key] = torch.from_numpy(val).float().cuda()
            elif key == 'voxel_coords':
                data_dict[key] = torch.from_numpy(val).int().cuda()
            elif key == 'points':
                data_dict[key] = torch.from_numpy(val).float().cuda()

        coords = data_dict['voxel_coords']
        batch_idx_voxel = torch.zeros((coords.shape[0], 1), dtype=coords.dtype).cuda()
        data_dict['voxel_coords'] = torch.cat([batch_idx_voxel, coords], dim=1)

        points = data_dict['points']
        batch_idx_points = torch.zeros((points.shape[0], 1), dtype=points.dtype).cuda()
        data_dict['points'] = torch.cat([batch_idx_points, points], dim=1)
        
        data_dict['batch_size'] = 1
        return data_dict

    def pointcloud2_to_array(self, msg):
        available_fields = [f.name for f in msg.fields]
        required_fields = ['x', 'y', 'z']
        intensity_name = None
        if 'intensity' in available_fields:
            intensity_name = 'intensity'
        elif 'i' in available_fields:
            intensity_name = 'i'
            
        if intensity_name:
            gen = pc2.read_points(msg, field_names=required_fields + [intensity_name], skip_nans=True)
            points_list = [list(p) for p in gen]
        else:
            gen = pc2.read_points(msg, field_names=required_fields, skip_nans=True)
            points_list = [list(p) + [0.0] for p in gen]

        if not points_list:
            return np.zeros((0, 4), dtype=np.float32)
        return np.array(points_list, dtype=np.float32)

    # [복구됨] 트래커 통신용 함수
    def publish_results(self, pred_dict, header):
        boxes = pred_dict['pred_boxes'].cpu().numpy()
        scores = pred_dict['pred_scores'].cpu().numpy()
        labels = pred_dict['pred_labels'].cpu().numpy()

        out_msg = DetectedObjectArray()
        out_msg.header = header

        for i in range(len(boxes)):
            # 트래커 성능을 위해 문턱을 낮게 설정 (0.1)
            if scores[i] < 0.: 
                continue
            
            obj = DetectedObject()
            obj.header = header
            obj.id = 0 
            obj.label = str(int(labels[i])) # 1, 2, 3
            obj.score = float(scores[i])
            
            obj.pose = [float(boxes[i][0]), float(boxes[i][1]), float(boxes[i][2])]
            obj.dimensions = [float(boxes[i][3]), float(boxes[i][4]), float(boxes[i][5])]
            obj.yaw = float(boxes[i][6])
            
            out_msg.objects.append(obj)
        
        self.pub.publish(out_msg)

    # 시각화용 함수
    def publish_markers(self, boxes, scores, labels, header):
        marker_array = MarkerArray()
        
        if hasattr(boxes, 'cpu'): boxes = boxes.cpu().numpy()
        if hasattr(scores, 'cpu'): scores = scores.cpu().numpy()
        if hasattr(labels, 'cpu'): labels = labels.cpu().numpy()
            
        for i, box in enumerate(boxes):
            # 1. 큐브 마커
            marker = Marker()
            marker.header = header
            marker.ns = "detections"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.lifetime.sec = 0
            marker.lifetime.nanosec = 200000000 
            
            marker.pose.position.x = float(box[0])
            marker.pose.position.y = float(box[1])
            marker.pose.position.z = float(box[2])
            
            yaw = float(box[6])
            marker.pose.orientation.z = math.sin(yaw / 2.0)
            marker.pose.orientation.w = math.cos(yaw / 2.0)
            
            marker.scale.x = float(box[3])
            marker.scale.y = float(box[4])
            marker.scale.z = float(box[5])
            
            marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0; marker.color.a = 0.5 
            marker_array.markers.append(marker)
            
            # 2. 텍스트 마커
            text_marker = Marker()
            text_marker.header = header
            text_marker.ns = "detection_info"
            text_marker.id = i
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.lifetime.sec = 0
            text_marker.lifetime.nanosec = 200000000
            
            text_marker.pose.position.x = float(box[0])
            text_marker.pose.position.y = float(box[1])
            text_marker.pose.position.z = float(box[2]) + float(box[5])/2.0 + 0.5
            
            text_marker.scale.z = 0.5
            text_marker.color.a = 1.0; text_marker.color.r = 1.0; text_marker.color.g = 1.0; text_marker.color.b = 1.0
            
            label_idx = int(labels[i]) - 1
            if 0 <= label_idx < len(self.class_names):
                cls_name = self.class_names[label_idx]
            else:
                cls_name = "Unknown"
            
            text_marker.text = f"{cls_name}: {float(scores[i]):.2f}"
            marker_array.markers.append(text_marker)
            
        self.pub_markers.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = PointPillarsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()