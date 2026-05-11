# #!/usr/bin/env python3
# import sys
# import os
# import time
# import math
# import warnings
# import numpy as np
# import torch
# import gc  # [추가] 가비지 컬렉터 (메모리 해제용)

# # ---------------- CONFIGURATION ----------------
# # OpenPCDet 루트 경로 (사용자 환경에 맞게 수정 필요)
# OPENPCDET_PATH = "/home/jay/OpenPCDet"
# sys.path.append(OPENPCDET_PATH)
# # -----------------------------------------------

# # PyTorch 경고 무시
# warnings.filterwarnings("ignore", category=UserWarning)

# import rclpy
# from rclpy.node import Node
# from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
# from rclpy.duration import Duration
# from sensor_msgs.msg import PointCloud2
# from visualization_msgs.msg import Marker, MarkerArray
# from geometry_msgs.msg import Point
# import sensor_msgs_py.point_cloud2 as pc2
# from tf2_ros import Buffer, TransformListener
# from scipy.spatial.transform import Rotation as R

# from pcdet.config import cfg, cfg_from_yaml_file
# from pcdet.models import build_network
# from pcdet.datasets.processor.data_processor import DataProcessor

# # 같은 폴더에 있는 tracker.py 임포트
# try:
#     from tracker import GlobalTracker
# except ImportError:
#     print("Error: 'tracker.py' not found in the same directory.")
#     sys.exit(1)

# class OpenPCDetNode(Node):
#     def __init__(self):
#         super().__init__('pcdet_ros2_node')
        
#         # 1. 파일 경로 설정 (nuScenes Model)
#         cur_path = os.path.dirname(os.path.realpath(__file__))
#         self.cfg_file = os.path.join(OPENPCDET_PATH, 'tools/cfgs/kitti_models/pv_rcnn_my_ver.yaml')
#         self.ckpt_file = os.path.join(OPENPCDET_PATH, 'output/coda32_allclass_bestoracle.pth')

#         # 2. 토픽 및 파라미터
#         self.lidar_topic = '/no_ground_points' 
        
#         # Z-Offset 보정
#         self.scv_sensor_height = 1.2
#         self.nusc_sensor_height = 1.84
#         self.z_offset = self.nusc_sensor_height - self.scv_sensor_height 

#         self.SCORE_THRESH = 0.2
#         # [수정포인트] 중복 박스가 여전히 겹치면 이 값을 0.8이나 1.0으로 키워주세요.
#         self.NMS_DIST_THRESH = 1.5
#         self.MAX_DETECTION_RANGE = 50.0 # m

#         self.logger_ros = self.get_logger()
        
#         # 3. Config 로드
#         if not os.path.exists(self.cfg_file):
#             self.logger_ros.error(f"Config file not found: {self.cfg_file}")
#             sys.exit(1)
#         cfg_from_yaml_file(self.cfg_file, cfg)
        
#         self.class_names = cfg.CLASS_NAMES # nuScenes Class List
#         pc_range_numpy = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE, dtype=np.float32)

#         # 4. Dummy Dataset
#         class DemoDataset:
#             def __init__(self):
#                 self.class_names = cfg.CLASS_NAMES
#                 self.point_feature_encoder = self.Encoder()
#                 self.point_cloud_range = pc_range_numpy
#                 self.depth_downsample_factor = None 
#                 self.voxel_size = None
#                 for processor in cfg.DATA_CONFIG.DATA_PROCESSOR:
#                     if processor['NAME'] == 'transform_points_to_voxels':
#                         self.voxel_size = processor['VOXEL_SIZE']
#                         break
#                 if self.voxel_size:
#                     self.grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(self.voxel_size)
#                     self.grid_size = np.round(self.grid_size).astype(np.int64)
            
#             class Encoder:
#                 def __init__(self): 
#                     self.num_point_features = 4

#         self.demo_dataset = DemoDataset()
#         self.data_processor = DataProcessor(
#             processor_configs=cfg.DATA_CONFIG.DATA_PROCESSOR, 
#             point_cloud_range=pc_range_numpy, training=False, num_point_features=4
#         )

#         # 5. 모델 빌드
#         self.logger_ros.info("Building Model...")
#         self.model = build_network(model_cfg=cfg.MODEL, num_class=len(self.class_names), dataset=self.demo_dataset)
        
#         if not os.path.exists(self.ckpt_file):
#             self.logger_ros.error(f"Checkpoint file not found: {self.ckpt_file}")
#             sys.exit(1)
            
#         self.logger_ros.info(f"Loading Checkpoint: {self.ckpt_file}")
#         self.model.load_params_from_file(filename=self.ckpt_file, logger=self.logger_ros, to_cpu=False)
#         self.model.cuda()
#         self.model.eval()

#         # 6. ROS Setup
#         qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
#         self.sub = self.create_subscription(PointCloud2, self.lidar_topic, self.lidar_callback, qos_profile)
#         self.pub_markers = self.create_publisher(MarkerArray, '/pcdet/detections', 10)
        
#         self.tf_buffer = Buffer()
#         self.tf_listener = TransformListener(self.tf_buffer, self)

#         # -----------------------------------------------------------------
#         # [핵심 수정 1] 트래커 파라미터 조정: 잔상은 짧게(max_age=3), 탐지되면 즉시 표시(min_hits=1)
#         # -----------------------------------------------------------------
#         self.tracker = GlobalTracker(max_age=3, min_hits=2, dist_threshold=2.0)
        
#         self.logger_ros.info(f'Node Ready! Listening to {self.lidar_topic}')

#     def cleanup_memory(self):
#         self.logger_ros.info("종료 시퀀스 시작: GPU 메모리를 해제합니다...")
#         if hasattr(self, 'model'):
#             del self.model
#         gc.collect()
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
#             torch.cuda.ipc_collect()
#         self.logger_ros.info("GPU 메모리 해제 완료.")

#     def lidar_callback(self, msg):
#         field_names = [f.name for f in msg.fields]
#         i_field = 'intensity' if 'intensity' in field_names else 'i' if 'i' in field_names else None
#         read_fields = ("x", "y", "z", i_field) if i_field else ("x", "y", "z")
        
#         gen = pc2.read_points(msg, field_names=read_fields, skip_nans=True)
        
#         try:
#             if i_field:
#                 points = np.array([[p[0], p[1], p[2], p[3]] for p in gen], dtype=np.float32)
#             else:
#                 points = np.array([[p[0], p[1], p[2], 0.0] for p in gen], dtype=np.float32)
                
#         except Exception as e:
#             self.get_logger().error(f"포인트 클라우드 변환 실패: {e}")
#             return
            
#         if points.shape[0] < 10: 
#             return

#         points[:, 2] -= self.z_offset
        
#         if points.shape[1] > 3 and points[:, 3].max() > 1.0:
#             points[:, 3] /= 255.0
            
#         input_dict = {
#             'points': points,
#             'frame_id': msg.header.frame_id
#         }
#         data_dict = self.prepare_data(input_dict)
        
#         for key, val in data_dict.items():
#             if isinstance(val, torch.Tensor):
#                 data_dict[key] = val.cuda()

#         with torch.no_grad():
#             pred_dicts, _ = self.model.forward(data_dict)
        
#         self.process_and_publish(pred_dicts[0], msg.header, points)

#     def prepare_data(self, input_dict):
#         input_dict['use_lead_xyz'] = True 
#         data_dict = self.data_processor.forward(data_dict=input_dict)
        
#         points = data_dict['points']
#         points_batch = np.zeros((points.shape[0], 5), dtype=np.float32)
#         points_batch[:, 0] = 0
#         points_batch[:, 1:] = points
#         data_dict['points'] = torch.from_numpy(points_batch).float()
        
#         if 'voxels' in data_dict:
#             data_dict['voxels'] = torch.from_numpy(data_dict['voxels']).float()
#             data_dict['voxel_num_points'] = torch.from_numpy(data_dict['voxel_num_points']).int()
#             coords = data_dict['voxel_coords']
#             coords_batch = np.zeros((coords.shape[0], 4), dtype=np.int32)
#             coords_batch[:, 0] = 0
#             coords_batch[:, 1:] = coords
#             data_dict['voxel_coords'] = torch.from_numpy(coords_batch).int()
            
#         data_dict['batch_size'] = 1
#         return data_dict

#     # -----------------------------------------------------------------
#     # [핵심 수정 2] 중복 박스를 제거하는 NMS 함수 추가
#     # -----------------------------------------------------------------
#     def apply_nms(self, boxes, scores, labels):
#         """중심점 거리를 기준으로 겹치는 박스 중 점수가 가장 높은 것만 남깁니다."""
#         if len(boxes) == 0: 
#             return boxes, scores, labels
#         order = scores.argsort()[::-1]
#         keep_indices = []
#         while order.size > 0:
#             i = order[0]
#             keep_indices.append(i)
#             xx1, yy1 = boxes[i, 0], boxes[i, 1]
#             rest = order[1:]
#             # XY 평면 거리 계산
#             dists = np.sqrt((xx1 - boxes[rest, 0])**2 + (yy1 - boxes[rest, 1])**2)
#             # NMS_DIST_THRESH 보다 멀리 떨어진 박스만 살림
#             inds = np.where(dists > self.NMS_DIST_THRESH)[0]
#             order = order[inds + 1]
#         return boxes[keep_indices], scores[keep_indices], labels[keep_indices]
#     # -----------------------------------------------------------------

#     # ... (생략: 상단 임포트 및 초기화 부분 동일)

#     def process_and_publish(self, pred_dict, header, raw_points):
#         # 1. 디텍션 결과 추출
#         pred_boxes = pred_dict['pred_boxes'].cpu().numpy()
#         pred_scores = pred_dict['pred_scores'].cpu().numpy()
#         pred_labels = pred_dict['pred_labels'].cpu().numpy()
        
#         # 2. 필터링 (임계값 상향 조정으로 노이즈 제거)
#         mask = (pred_scores > 0.4) # 0.2에서 0.4로 상향
#         pred_boxes = pred_boxes[mask]
#         pred_scores = pred_scores[mask]
#         pred_labels = pred_labels[mask]

#         if len(pred_boxes) > 0:
#             pred_boxes, pred_scores, pred_labels = self.apply_nms(pred_boxes, pred_scores, pred_labels)

#         # 3. 데이터 통합 [x, y, z, dx, dy, dz, yaw, label, score]
#         detections = np.zeros((len(pred_boxes), 9))
#         detections[:, 0:7] = pred_boxes[:, 0:7]
#         detections[:, 7] = pred_labels
#         detections[:, 8] = pred_scores

#         # 4. 중요: Ego-motion Compensation (TF Transform)
#         target_frame = 'odom' # 고정된 전역 좌표계
#         try:
#             # 현재 시점의 자차 위치(odom -> lidar) 정보 가져오기
#             trans = self.tf_buffer.lookup_transform(target_frame, header.frame_id, rclpy.time.Time())
#             t = trans.transform.translation
#             q = trans.transform.rotation
#             r = R.from_quat([q.x, q.y, q.z, q.w])
#             rot_mat = r.as_matrix()
            
#             # 모든 탐지 객체를 전역 좌표계(odom)로 변환
#             xyz = detections[:, 0:3]
#             detections[:, 0:3] = np.dot(xyz, rot_mat.T) + np.array([t.x, t.y, t.z])
#             detections[:, 6] += r.as_euler('zyx')[0] # Heading 보정
            
#             # 트래커는 이제 흔들리지 않는 'odom' 좌표계에서 객체를 추적함
#             tracked_objects = self.tracker.update(detections)
#             self.publish_markers(tracked_objects, target_frame, header.stamp)
            
#         except Exception as e:
#             self.logger_ros.warn(f"TF Lookup 실패: {e}")
#             # TF 실패 시 로컬 좌표계로라도 수행 (ID 유지는 어려움)
#             tracked_objects = self.tracker.update(detections)
#             self.publish_markers(tracked_objects, header.frame_id, header.stamp)


#     def publish_markers(self, tracked_objects, frame_id, stamp):
#         marker_array = MarkerArray()
        
#         del_marker = Marker()
#         del_marker.header.frame_id = frame_id
#         del_marker.action = Marker.DELETEALL
#         marker_array.markers.append(del_marker)

#         for obj in tracked_objects:
#             x, y, z, tid, yaw, label, dx, dy, dz, age = obj

#             x, y, z = float(x), float(y), float(z)
#             dx, dy, dz = float(dx), float(dy), float(dz)
#             yaw = float(yaw)
#             tid = int(tid)
            
#             label_idx = int(label) - 1 
#             if 0 <= label_idx < len(self.class_names):
#                 cls_name = self.class_names[label_idx]
#             else:
#                 cls_name = "Unknown"

#             r, g, b = 1.0, 1.0, 1.0
#             if 'car' in cls_name: r,g,b = 0.0, 1.0, 0.0
#             elif 'ped' in cls_name: r,g,b = 1.0, 1.0, 0.0
#             elif 'truck' in cls_name: r,g,b = 0.0, 0.5, 1.0
            
#             alpha = 0.5 if int(age) > 0 else 0.8 

#             time_since_update = obj[9] # 트래커에서 넘겨준 업데이트 경과 시간
            
#             # 고스트 박스 시각화: 오랫동안 미탐지된 객체는 투명하게 표시
#             if time_since_update > 0:
#                 alpha = max(0.1, 0.6 - (time_since_update * 0.1)) # 점점 투명해짐
#                 color_r, color_g, color_b = 0.5, 0.5, 0.5 # 고스트는 회색조
#             else:
#                 alpha = 0.8

#             marker = Marker()
#             marker.header.frame_id = frame_id
#             marker.header.stamp = stamp
#             marker.ns = "objects"
#             marker.id = int(tid)
#             marker.type = Marker.CUBE
#             marker.action = Marker.ADD
#             marker.pose.position.x = x
#             marker.pose.position.y = y
#             marker.pose.position.z = z
#             marker.pose.orientation.z = math.sin(yaw / 2.0)
#             marker.pose.orientation.w = math.cos(yaw / 2.0)
#             marker.scale.x = dx
#             marker.scale.y = dy
#             marker.scale.z = dz
#             marker.color.r, marker.color.g, marker.color.b, marker.color.a = r, g, b, alpha
#             marker.lifetime = Duration(seconds=0.2).to_msg()
#             marker_array.markers.append(marker)

#             text = Marker()
#             text.header.frame_id = frame_id
#             text.header.stamp = stamp
#             text.ns = "ids"
#             text.id = int(tid) + 10000
#             text.type = Marker.TEXT_VIEW_FACING
#             text.action = Marker.ADD
#             text.pose.position.x = x
#             text.pose.position.y = y
#             text.pose.position.z = z + dz/2.0 + 0.5
#             text.scale.z = 0.5
#             text.text = f"{cls_name} {int(tid)}"
#             text.color.r, text.color.g, text.color.b, text.color.a = 1.0, 1.0, 1.0, 1.0
#             text.lifetime = Duration(seconds=0.2).to_msg()
#             marker_array.markers.append(text)

#         self.pub_markers.publish(marker_array)

# def main(args=None):
#     rclpy.init(args=args)
#     node = OpenPCDetNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.cleanup_memory()
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()

#!/usr/bin/env python3
import sys
import os
import time
import csv
import math
import warnings
import numpy as np
import torch
import gc 

# ---------------- CONFIGURATION ----------------
# OpenPCDet 루트 경로 (사용자 환경에 맞게 수정 필요)
OPENPCDET_PATH = "/home/jay/OpenPCDet"
sys.path.insert(0, OPENPCDET_PATH)
# -----------------------------------------------

# PyTorch 경고 무시
warnings.filterwarnings("ignore", category=UserWarning)

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.duration import Duration
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
import sensor_msgs_py.point_cloud2 as pc2
from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation as R

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network
from pcdet.datasets.processor.data_processor import DataProcessor

class OpenPCDetNode(Node):
    def __init__(self):
        super().__init__('pcdet_ros2_node')
        
        # 1. 파일 경로 및 파라미터 설정
        self.cfg_file = os.path.join(OPENPCDET_PATH, 'tools/cfgs/kitti_models/pv_rcnn_my_ver.yaml')
        self.ckpt_file = os.path.join(OPENPCDET_PATH, 'output/coda32_allclass_bestoracle.pth')
        self.lidar_topic = '/no_ground_points' 
        self.z_offset = 0.64 # nuScenes(1.84) - SCV(1.2)
        self.SCORE_THRESH = 0.4
        self.NMS_DIST_THRESH = 1.5
        self.MAX_DETECTION_RANGE = 50.0

        self.logger_ros = self.get_logger()
        
        # 2. 성능 기록용 CSV 설정
        model_name = os.path.basename(self.cfg_file).split('.')[0]
        self.log_filename = f"perf_log_{model_name}_{time.strftime('%m%d_%H%M')}.csv"
        self.csv_file = open(self.log_filename, mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['frame_id', 'pre_proc_ms', 'inference_ms', 'post_proc_ms', 'total_ms', 'obj_count', 'avg_score'])

        # 3. Config 로드
        if not os.path.exists(self.cfg_file):
            self.logger_ros.error(f"Config file not found: {self.cfg_file}")
            sys.exit(1)
        cfg_from_yaml_file(self.cfg_file, cfg)
        self.class_names = cfg.CLASS_NAMES
        pc_range = np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE, dtype=np.float32)

        # 4. Data Processor 초기화
        self.data_processor = DataProcessor(
            processor_configs=cfg.DATA_CONFIG.DATA_PROCESSOR,
            point_cloud_range=pc_range,
            training=False,
            num_point_features=4
        )

        # 5. 모델 빌드 (PV-RCNN++ 호환용 Dummy Dataset 포함)
        self.logger_ros.info("Building Model...")
        self.model = build_network(
            model_cfg=cfg.MODEL, 
            num_class=len(self.class_names), 
            dataset=self.build_dummy_dataset(cfg, pc_range)
        )
        self.model.load_params_from_file(filename=self.ckpt_file, logger=self.logger_ros, to_cpu=False)
        self.model.cuda()
        self.model.eval()

        # 정적 지표: 파라미터 수 출력
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger_ros.info(f"Model Parameters: {total_params / 1e6:.2f}M")

        # 6. ROS Setup
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.sub = self.create_subscription(PointCloud2, self.lidar_topic, self.lidar_callback, qos)
        self.pub_markers = self.create_publisher(MarkerArray, '/pcdet/detections', 10)
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.frame_count = 0
        self.logger_ros.info(f'Node Ready! Logging to {self.log_filename}')

    def build_dummy_dataset(self, cfg, pc_range):
        class Dummy:
            def __init__(self):
                self.class_names = cfg.CLASS_NAMES
                self.point_feature_encoder = type('Encoder', (), {'num_point_features': 4})()
                self.point_cloud_range = pc_range
                self.depth_downsample_factor = None 
                self.dataset_cfg = cfg.DATA_CONFIG
                self.voxel_size = None
                for p in cfg.DATA_CONFIG.DATA_PROCESSOR:
                    if p['NAME'] == 'transform_points_to_voxels':
                        self.voxel_size = np.array(p['VOXEL_SIZE'], dtype=np.float32)
                        break
                self.grid_size = np.round((self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / self.voxel_size).astype(np.int64)
        return Dummy()

    def cleanup_memory(self):
        self.logger_ros.info("종료 중: 로그 파일을 저장하고 메모리를 해제합니다.")
        self.csv_file.close()
        if hasattr(self, 'model'): del self.model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def lidar_callback(self, msg):
        start_total = time.perf_counter()
        self.frame_count += 1

        # --- [1. Pre-processing] ---
        start_pre = time.perf_counter()
        gen = pc2.read_points(msg, field_names=("x", "y", "z", "intensity"), skip_nans=True)
        points = np.array([[p[0], p[1], p[2], p[3]] for p in gen], dtype=np.float32)
        
        if points.shape[0] < 10: return
        points[:, 2] -= self.z_offset
        if points[:, 3].max() > 1.0: points[:, 3] /= 255.0

        input_dict = {'points': points, 'frame_id': msg.header.frame_id, 'use_lead_xyz': True}
        data_dict = self.data_processor.forward(data_dict=input_dict)
        
        # Batching & GPU Transfer
        data_dict['points'] = torch.from_numpy(np.pad(data_dict['points'], ((0,0),(1,0)), mode='constant')).float().cuda()
        if 'voxels' in data_dict:
            for k in ['voxels', 'voxel_num_points', 'voxel_coords']:
                data_dict[k] = torch.from_numpy(data_dict[k]).cuda()
                if k == 'voxel_coords':
                    data_dict[k] = torch.nn.functional.pad(data_dict[k], (1,0), mode='constant', value=0)
        data_dict['batch_size'] = 1
        pre_time = (time.perf_counter() - start_pre) * 1000

        # --- [2. Inference] ---
        start_inf = time.perf_counter()
        with torch.no_grad():
            torch.cuda.synchronize()
            pred_dicts, _ = self.model.forward(data_dict)
            torch.cuda.synchronize()
        inf_time = (time.perf_counter() - start_inf) * 1000

        # --- [3. Post-processing & Logging] ---
        start_post = time.perf_counter()
        self.process_and_publish(pred_dicts[0], msg.header)
        post_time = (time.perf_counter() - start_post) * 1000
        
        total_time = (time.perf_counter() - start_total) * 1000
        
        # 통계 계산
        scores = pred_dicts[0]['pred_scores'].cpu().numpy()
        mask = scores > self.SCORE_THRESH
        obj_count = np.sum(mask)
        avg_score = np.mean(scores[mask]) if obj_count > 0 else 0.0

        # CSV 기록
        self.csv_writer.writerow([self.frame_count, pre_time, inf_time, post_time, total_time, obj_count, avg_score])
        if self.frame_count % 20 == 0:
            self.logger_ros.info(f"FPS: {1000/total_time:.1f} | Latency: {inf_time:.1f}ms | Objects: {obj_count}")

    def apply_nms(self, boxes, scores, labels):
        if len(boxes) == 0: return boxes, scores, labels
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]; keep.append(i)
            dists = np.linalg.norm(boxes[order[1:], :2] - boxes[i, :2], axis=1)
            order = order[np.where(dists > self.NMS_DIST_THRESH)[0] + 1]
        return boxes[keep], scores[keep], labels[keep]

    def process_and_publish(self, pred_dict, header):
        boxes = pred_dict['pred_boxes'].cpu().numpy()
        scores = pred_dict['pred_scores'].cpu().numpy()
        labels = pred_dict['pred_labels'].cpu().numpy()
        
        mask = (scores > self.SCORE_THRESH) & (np.linalg.norm(boxes[:, :2], axis=1) < self.MAX_DETECTION_RANGE)
        boxes, scores, labels = self.apply_nms(boxes[mask], scores[mask], labels[mask])
        detections = np.hstack([boxes, labels.reshape(-1,1), scores.reshape(-1,1)])

        # TF Transform
        target_frame = 'odom'
        try:
            trans = self.tf_buffer.lookup_transform(target_frame, header.frame_id, rclpy.time.Time())
            r = R.from_quat([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w])
            xyz = detections[:, 0:3]
            detections[:, 0:3] = np.dot(xyz, r.as_matrix().T) + np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
            detections[:, 6] += r.as_euler('zyx')[0]
            self.publish_markers(detections, target_frame, header.stamp)
        except:
            self.publish_markers(detections, header.frame_id, header.stamp)

    def publish_markers(self, detections, frame_id, stamp):
        ma = MarkerArray()
        dm = Marker()
        dm.header.frame_id, dm.header.stamp, dm.action = frame_id, stamp, Marker.DELETEALL
        ma.markers.append(dm)

        for i, det in enumerate(detections):
            x, y, z, dx, dy, dz, yaw, label, score = map(float, det)
            cls_name = self.class_names[int(label)-1] if 0 < int(label) <= len(self.class_names) else "Unknown"
            
            # Box
            m = Marker()
            m.header.frame_id, m.header.stamp = frame_id, stamp
            m.ns, m.id, m.type, m.action = "det", i, Marker.CUBE, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, z
            m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw/2), math.cos(yaw/2)
            m.scale.x, m.scale.y, m.scale.z = dx, dy, dz
            m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 1.0, 0.0, 0.8
            m.lifetime = Duration(seconds=0.2).to_msg()
            ma.markers.append(m)

            # Text
            t = Marker()
            t.header.frame_id, t.header.stamp = frame_id, stamp
            t.ns, t.id, t.type, t.action = "score", i+1000, Marker.TEXT_VIEW_FACING, Marker.ADD
            t.pose.position.x, t.pose.position.y, t.pose.position.z = x, y, z + dz/2 + 0.5
            t.scale.z, t.text = 0.5, f"{cls_name}: {score:.2f}"
            t.color.r, t.color.g, t.color.b, t.color.a = 1.0, 1.0, 1.0, 1.0
            ma.markers.append(t)
        self.pub_markers.publish(ma)

def main():
    rclpy.init()
    node = OpenPCDetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup_memory()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
