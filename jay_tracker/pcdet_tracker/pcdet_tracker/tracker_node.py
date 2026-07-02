import rclpy
from rclpy.node import Node
from tracking_msgs.msg import DetectedObject, DetectedObjectArray
import numpy as np
import sys
import os

# AB3DMOT 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from AB3DMOT_libs.model import AB3DMOT

class AB3DMOTNode(Node):
    def __init__(self):
        super().__init__('ab3dmot_node')
        
        # 1. Config 설정
        class Config:
            def __init__(self):
                self.max_age = 2
                self.min_hits = 1
                self.affinity_threshold = 0.1
                self.vis = False
                self.vis_dir = None
                self.ego_com = False
                self.affi_pro_type = 'iou'
                self.affi_pro = False
                self.dataset = 'KITTI'
                self.det_name = 'pointrcnn'
        
        cfg = Config()
        
        # --- [핵심 수정] 클래스별로 별도의 트래커 생성 ---
        self.target_classes = ['Car', 'Pedestrian', 'Cyclist']
        self.trackers = {}
        
        for cls_name in self.target_classes:
            self.trackers[cls_name] = AB3DMOT(cfg, cls_name)
            self.get_logger().info(f'Initialized tracker for {cls_name}')
        # ---------------------------------------------
        
        self.sub = self.create_subscription(
            DetectedObjectArray, '/detected_objects_3d', self.callback, 10
        )
        self.pub = self.create_publisher(DetectedObjectArray, '/tracked_objects_3d', 10)

    def callback(self, msg):
        # 1. 클래스별로 데이터 분류
        # dets_by_class = {'Car': [[...]], 'Pedestrian': [[...]]}
        dets_by_class = {cls: [] for cls in self.target_classes}
        info_by_class = {cls: [] for cls in self.target_classes}

        # [디버깅] 현재 들어온 객체 수와 라벨 확인
        if len(msg.objects) > 0:
            labels_in_frame = [obj.label for obj in msg.objects]
            # 너무 많이 뜨면 정신없으니 가끔만 출력하거나, 특정 클래스만 출력
            if '2' in labels_in_frame or 'Pedestrian' in labels_in_frame:
                self.get_logger().info(f"Received: {labels_in_frame}")
        
        # PointPillars 결과(라벨 숫자)를 이름으로 매핑 (1:Car, 2:Ped, 3:Cyc)
        # 모델마다 다를 수 있으나 보통 KITTI 학습 모델은 이 순서입니다.
        label_map = {1: 'Car', 2: 'Pedestrian', 3: 'Cyclist'}

        for obj in msg.objects:
            # 라벨 숫자를 이름으로 변환 (예: "1" -> "Car")
            try:
                label_idx = int(obj.label)
                class_name = label_map.get(label_idx, 'Unknown')
            except ValueError:
                class_name = obj.label # 이미 문자열인 경우

            if class_name not in self.target_classes:
                continue

            # 데이터 포맷팅 [h, w, l, x, y, z, theta, score]
            h, w, l = obj.dimensions[2], obj.dimensions[1], obj.dimensions[0]
            x, y, z = obj.pose[0], obj.pose[1], obj.pose[2]
            theta, score = obj.yaw, obj.score
            
            dets_by_class[class_name].append([h, w, l, x, y, z, theta, score])
            info_by_class[class_name].append({'score': score})

        # 2. 각 트래커 업데이트 및 결과 병합
        out_msg = DetectedObjectArray()
        out_msg.header = msg.header

        for cls_name in self.target_classes:
            dets = np.array(dets_by_class[cls_name])
            if len(dets) == 0:
                dets = np.empty((0, 8))
            
            # 해당 클래스 전용 트래커 실행
            try:
                trackers = self.trackers[cls_name].update(dets, info_by_class[cls_name])
            except Exception as e:
                continue

            # 결과 담기
            for trk in trackers:
                obj = DetectedObject()
                h, w, l, x, y, z, theta, trk_id = trk[0:8]
                
                obj.id = int(trk_id)
                obj.label = cls_name  # 라벨도 같이 저장
                obj.score = float(trk[8]) if len(trk) > 8 else 1.0
                
                obj.pose = [float(x), float(y), float(z)]
                obj.dimensions = [float(l), float(w), float(h)]
                obj.yaw = float(theta)
                
                out_msg.objects.append(obj)
            
        self.pub.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = AB3DMOTNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()