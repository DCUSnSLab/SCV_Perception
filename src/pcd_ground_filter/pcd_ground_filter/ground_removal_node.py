#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

import numpy as np
import math
import time

# Numba 필수 임포트
from numba import njit

# ==============================================================================
# [Numba 가속 함수] Slope Consistency(기울기 일관성) 기반 바닥 판별
# ==============================================================================
# 기존 파라미터에 'max_gap' 추가
@njit(cache=True, fastmath=True)
def run_ground_filter(x, y, z, r, bin_indices, num_bins, 
                      max_slope_rad, 
                      slope_tolerance_rad, 
                      max_height_jump, 
                      ground_level_z,
                      max_gap): # [New] 점 사이의 최대 허용 간격
    
    is_ground = np.zeros(len(x), dtype=np.bool_)

    for b in range(num_bins):
        idxs = np.where(bin_indices == b)[0]
        if len(idxs) < 2: continue

        sorted_arg = np.argsort(r[idxs])
        ray_idxs = idxs[sorted_arg]

        first_set = False
        prev_r = 0.0
        prev_z = ground_level_z 
        prev_slope = 0.0
        
        # 1. 시작점 찾기
        for i in range(min(10, len(ray_idxs))):
            idx = ray_idxs[i]
            if abs(z[idx] - ground_level_z) < 0.5:
                is_ground[idx] = True
                prev_r = r[idx]
                prev_z = z[idx]
                prev_slope = 0.0
                first_set = True
                break 
        
        if not first_set: continue

        # 2. 트래킹
        for k in range(1, len(ray_idxs)):
            cur_idx = ray_idxs[k]
            dr = r[cur_idx] - prev_r
            
            # [최적화 핵심] 
            # 점 사이 거리가 너무 멀면(희소하면) 더 볼 것도 없이 이 Ray 종료
            if dr > max_gap: 
                break 
            
            dz = z[cur_idx] - prev_z

            if dr < 0.05: continue

            # 현재 구간의 순간 기울기 계산
            current_slope = np.arctan2(dz, dr)

            # [핵심 로직 변경]
            
            # Case A: 절대 각도가 너무 크면 '벽'으로 의심
            if np.abs(current_slope) > max_slope_rad:
                # 하지만 높이 차이가 아주 작다면? -> 낮은 턱(Curb)이나 노이즈 -> 바닥 인정
                if np.abs(dz) < max_height_jump:
                    is_ground[cur_idx] = True
                    prev_r = r[cur_idx]
                    prev_z = z[cur_idx]
                    prev_slope = current_slope # 기울기 갱신
                else:
                    # 진짜 벽(장애물) -> prev 업데이트 안 함 (장애물 뒤의 바닥을 잡기 위해)
                    pass

            # Case B: 각도가 안전 범위 내라면? -> '오르막/내리막' 일관성 체크
            else:
                # 이전 기울기와 현재 기울기의 차이 (변화량)
                slope_change = np.abs(current_slope - prev_slope)
                
                # 조건 1: 기울기가 일정하게 유지되는가? (선형적 변화) -> 오르막/내리막 바닥
                is_consistent = slope_change < slope_tolerance_rad
                
                # 조건 2: 기울기가 변했더라도 높이 단차가 작은가? -> 경사로 진입부 or 작은 요철
                is_small_step = np.abs(dz) < max_height_jump

                # 둘 중 하나라도 만족하면 바닥
                if is_consistent or is_small_step:
                    is_ground[cur_idx] = True
                    prev_r = r[cur_idx]
                    prev_z = z[cur_idx]
                    prev_slope = current_slope # 현재 기울기를 '이전 기울기'로 저장 (학습)
                else:
                    # 기울기도 확 바뀌고 높이도 튀었다면 장애물
                    pass
                
    return is_ground

# ==============================================================================

def voxel_downsample(points_xyzi: np.ndarray, voxel=0.20):
    if points_xyzi.shape[0] == 0 or voxel <= 0:
        return points_xyzi
    
    xyz = points_xyzi[:, :3]
    keys = np.floor(xyz / voxel).astype(np.int32)
    _, keep_idx = np.unique(keys, axis=0, return_index=True)
    return points_xyzi[keep_idx]


def pointcloud2_to_xyzi(msg: PointCloud2):
    field_names = [f.name for f in msg.fields]
    target_fields = ['x', 'y', 'z']
    
    if 'intensity' in field_names:
        target_fields.append('intensity')
    elif 'i' in field_names:
        target_fields.append('i')
    
    pts_gen = point_cloud2.read_points(msg, field_names=target_fields, skip_nans=True)
    pts_list = [list(p) for p in pts_gen]
    
    if not pts_list:
        return np.zeros((0, 4), dtype=np.float32)
    
    data = np.array(pts_list, dtype=np.float32)
    
    if data.shape[1] == 3:
        intensity = np.zeros((data.shape[0], 1), dtype=np.float32)
        data = np.hstack([data, intensity])
        
    return data


def xyzi_to_pointcloud2(points_xyzi: np.ndarray, frame_id: str, stamp=None):
    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id

    fields = [
        PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    ]

    pc_iter = points_xyzi.astype(np.float32).tolist()
    pc_msg = point_cloud2.create_cloud(header, fields, pc_iter)
    return pc_msg


class GroundRemovalNode(Node):
    def __init__(self):
        super().__init__('ground_removal_node')

        self.declare_parameter('input_topic', '/velodyne_points')
        self.declare_parameter('output_topic', '/no_ground_points')

        # [중요] 센서 높이 (실측값 입력 필수, 예: 1.2m -> -1.2)
        self.declare_parameter('ground_level_z', -1.3) 
        
        # [수정] 벽 판단 각도: 60도 -> 25도 (이제 오르막은 이 각도가 아니라 consistency로 판단하므로 낮춰야 함)
        self.declare_parameter('max_slope_deg', 25.0) 
        
        # [추가] 기울기 변화 허용치 (5~10도 권장): 이 범위 내에서 각도가 변하면 오르막으로 인정
        self.declare_parameter('slope_tolerance_deg', 8.0) 

        self.declare_parameter('max_height_jump', 0.2) 
        self.declare_parameter('num_angular_bins', 720) 

        self.declare_parameter('min_range', 1.0) 
        self.declare_parameter('max_range', 15.0) 
        
        self.declare_parameter('use_voxel', False)
        self.declare_parameter('voxel_size', 0.1)

        self.declare_parameter('max_obj_height', 3.5)
        self.declare_parameter('min_obj_height', 1.0) 

        # [추가] 점 사이 간격이 1.5m를 넘어가면 그 뒤는 계산 포기 (희소 데이터 무시)
        self.declare_parameter('max_dist_gap', 1.5)

        in_topic  = self.get_parameter('input_topic').get_parameter_value().string_value
        out_topic = self.get_parameter('output_topic').get_parameter_value().string_value

        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.sub_ = self.create_subscription(PointCloud2, in_topic, self.cb_pointcloud, qos_profile)
        self.pub_ = self.create_publisher(PointCloud2, out_topic, 10)

        self.get_logger().info("Slope Consistency Ground Filter Ready.")

    def cb_pointcloud(self, msg: PointCloud2):
        t0 = time.time()

        # 1. 데이터 읽기
        xyzi = pointcloud2_to_xyzi(msg)
        if xyzi.shape[0] == 0: return

        # 2. Voxel Downsample
        if bool(self.get_parameter('use_voxel').get_parameter_value().bool_value):
            voxel = float(self.get_parameter('voxel_size').get_parameter_value().double_value)
            xyzi = voxel_downsample(xyzi, voxel)
            if xyzi.shape[0] == 0: return

        # 3. 좌표 분리
        x = xyzi[:, 0]
        y = xyzi[:, 1]
        z = xyzi[:, 2]

        r = np.sqrt(x*x + y*y)
        theta = np.arctan2(y, x)

        # 4. Range Filtering
        min_range = float(self.get_parameter('min_range').get_parameter_value().double_value)
        max_range = float(self.get_parameter('max_range').get_parameter_value().double_value)
        
        valid_mask = (r > min_range) & (r < max_range)
        xyzi = xyzi[valid_mask]
        x, y, z, r, theta = x[valid_mask], y[valid_mask], z[valid_mask], r[valid_mask], theta[valid_mask]

        if len(x) == 0: return

        # 5. Binning
        num_bins = int(self.get_parameter('num_angular_bins').get_parameter_value().integer_value)
        theta_wrap = (theta + math.pi) % (2.0 * math.pi)
        bin_indices = (theta_wrap / (2.0 * math.pi) * num_bins).astype(np.int32)
        bin_indices = np.clip(bin_indices, 0, num_bins-1)

        # 파라미터 준비
        max_slope_rad = math.radians(float(self.get_parameter('max_slope_deg').get_parameter_value().double_value))
        # [추가] 파라미터 읽기
        slope_tolerance_rad = math.radians(float(self.get_parameter('slope_tolerance_deg').get_parameter_value().double_value))
        max_gap = float(self.get_parameter('max_dist_gap').get_parameter_value().double_value)
        max_height_jump = float(self.get_parameter('max_height_jump').get_parameter_value().double_value)
        ground_level_z = float(self.get_parameter('ground_level_z').get_parameter_value().double_value)

        # Numba 실행 (인자 추가됨)
        is_ground = run_ground_filter(
            x, y, z, r, bin_indices, num_bins, 
            max_slope_rad, slope_tolerance_rad, max_height_jump, ground_level_z,
            max_gap # [New]
        )

        # 6. 결과 필터링
        no_ground_mask = ~is_ground
        filtered_xyzi = xyzi[no_ground_mask]
        
        # 7. Height Cutoff (천장/지하 제거)
        f_z = filtered_xyzi[:, 2] 
        # min_obj_height는 바닥을 0으로 뒀을 때의 상대 높이가 아니라 절대 좌표계 높이를 의미하는 것 같으므로 주의 필요
        # 보통 ground_level_z + min_obj_height 로 계산하는 것이 안전함.
        # 여기서는 단순히 z값 자체로 필터링하는 로직 유지 (사용자 원본 존중)
        
        # [수정 권장] min_obj_height 로직이 약간 애매하므로, 바닥면 기준 +0.1m 이상인 것만 남기도록 수정하면 더 깔끔함.
        # 일단은 원본 로직 유지하되, ground_level_z를 고려한 필터링으로 동작하게 둠.
        
        min_obj_height = float(self.get_parameter('min_obj_height').get_parameter_value().double_value)
        max_obj_height = float(self.get_parameter('max_obj_height').get_parameter_value().double_value)
        
        # 바닥면(ground_level_z)보다 min_obj_height만큼 위에 있는 것만 객체로 인정
        # 예: -1.2(바닥) + (-1.0) = -2.2?? -> 파라미터 의미가 '절대높이'인지 '상대높이'인지 중요.
        # 보통은 '바닥 위 10cm부터 객체' -> ground_level_z + 0.1
        # 사용자 코드 원본: f_z > (ground_level_z + min_obj_height) -> 이대로 둠
        
        keep_h = (f_z > (ground_level_z + min_obj_height)) & (f_z < max_obj_height)
        final_xyzi = filtered_xyzi[keep_h]

        # 8. 발행
        out_msg = xyzi_to_pointcloud2(final_xyzi, msg.header.frame_id, msg.header.stamp)
        self.pub_.publish(out_msg)

def main():
    rclpy.init()
    node = GroundRemovalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()