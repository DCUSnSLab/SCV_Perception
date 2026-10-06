#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

import numpy as np
import math
import time

# 파일 상단 import 아래에 추가
try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False


def voxel_downsample(points_xyz: np.ndarray, voxel=0.20):
    if points_xyz.shape[0] == 0 or voxel <= 0:
        return points_xyz
    keys = np.floor(points_xyz / voxel).astype(np.int32)
    _, keep_idx = np.unique(keys, axis=0, return_index=True)
    return points_xyz[keep_idx]


def pointcloud2_to_xyz(msg: PointCloud2):
    pts = []
    for p in point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
        pts.append([p[0], p[1], p[2]])
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(pts, dtype=np.float32)


def xyz_to_pointcloud2(points_xyz: np.ndarray, frame_id: str, stamp=None):
    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id

    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]

    pc_iter = points_xyz.astype(np.float32).tolist()
    pc_msg = point_cloud2.create_cloud(header, fields, pc_iter)
    return pc_msg


def ransac_plane(points_xyz: np.ndarray,
                 max_iterations: int = 200,
                 distance_threshold: float = 0.05,
                 min_inlier_ratio: float = 0.2):
    """
    단순 RANSAC 평면 피팅
    points_xyz: (N,3)
    """
    N = points_xyz.shape[0]
    if N < 3:
        return None, None, None

    best_inlier_count = 0
    best_normal = None
    best_d = None

    # 미리 transposed
    pts = points_xyz

    for _ in range(max_iterations):
        # 서로 다른 3점 샘플
        idx = np.random.choice(N, 3, replace=False)
        p1, p2, p3 = pts[idx[0]], pts[idx[1]], pts[idx[2]]

        # 법선 벡터 계산 (collinear 방지)
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal = normal / norm

        # 평면식: n·x + d = 0
        d = -np.dot(normal, p1)

        # 거리 계산
        # dist = |n·x + d|
        dists = np.abs(np.dot(pts, normal) + d)

        inliers = dists < distance_threshold
        inlier_count = np.count_nonzero(inliers)

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_normal = normal
            best_d = d

    if best_normal is None:
        return None, None, None

    if best_inlier_count < max(3, int(min_inlier_ratio * N)):
        # inlier가 너무 적으면 실패로 간주
        return None, None, None

    return best_normal, best_d, best_inlier_count


class GroundRemovalNode(Node):
    """
    Hybrid:
    1) polar ray 기반 local slope / height jump로 1차 ground 추정
    2) 그 결과를 seed로 RANSAC 평면 피팅 (옵션)
    3) 평면 근처 포인트를 ground로 재보정하여 더 안정적인 바닥 제거
    4) 최종 height band(ground_level_z ~ max_obj_height) 내 non-ground만 퍼블리시
    """

    def __init__(self):
        super().__init__('ground_removal_node')

        # 파라미터 선언
        self.declare_parameter('input_topic', '/velodyne_points')
        self.declare_parameter('output_topic', '/no_ground_points')
        self.declare_parameter('num_angular_bins', 360)      # theta 방향 쪼개는 수
        self.declare_parameter('max_slope_deg', 45.0)        # 허용 경사 (deg)
        self.declare_parameter('max_height_jump', 0.5)       # 바로 앞 지점 대비 z 점프 허용치 (m)
        self.declare_parameter('min_range', 0.05)            # 너무 가까운건 버림
        self.declare_parameter('max_range', 20.0)            # 너무 먼건 버림
        self.declare_parameter('use_voxel', True)
        self.declare_parameter('voxel_size', 0.10)           # 0.15~0.30 권장

        # 높이 관련
        self.declare_parameter('ground_level_z', -0.5)       # 대략 도로면 z
        self.declare_parameter('min_obj_height', 0.05)       # 바닥보다 최소 이 정도 떠있어야 장애물
        self.declare_parameter('max_obj_height', 1.5)        # 차량보다 약간 높은 높이까지만 고려

        # RANSAC 옵션
        self.declare_parameter('use_ransac', True)           # RANSAC 사용 여부
        self.declare_parameter('ransac_max_iterations', 200)
        self.declare_parameter('ransac_distance_threshold', 0.08)  # 평면으로부터 허용 거리(m)
        self.declare_parameter('ransac_min_inlier_ratio', 0.05)     # seed 중 최소 inlier 비율

        in_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        out_topic = self.get_parameter('output_topic').get_parameter_value().string_value

        self.sub_ = self.create_subscription(PointCloud2, in_topic, self.cb_pointcloud, 10)
        self.pub_ = self.create_publisher(PointCloud2, out_topic, 10)

        self.get_logger().info("ground_removal_node with RANSAC hybrid ready")

    def cb_pointcloud(self, msg: PointCloud2):
        t0 = time.time()

        xyz_all = pointcloud2_to_xyz(msg)
        if xyz_all.shape[0] == 0:
            self.get_logger().warn("Empty input cloud")
            return

        # ---- (1) 초기 height cutoff (도로보다 너무 낮은 포인트 제거, 연산량 줄이기) ----
        ground_level_z = float(self.get_parameter('ground_level_z').get_parameter_value().double_value)
        min_obj_height = float(self.get_parameter('min_obj_height').get_parameter_value().double_value)

        early_keep = np.ones(xyz_all.shape[0], dtype=bool)
        xyz = xyz_all[early_keep]
        if xyz.shape[0] == 0:
            self.get_logger().warn("No points after early height filter")
            return

        # ---- (2) 옵션: voxel downsample ----
        if bool(self.get_parameter('use_voxel').get_parameter_value().bool_value):
            voxel = float(self.get_parameter('voxel_size').get_parameter_value().double_value)
            xyz = voxel_downsample(xyz, voxel)
            if xyz.shape[0] == 0:
                self.get_logger().warn("No points after voxel downsample")
                return

        if xyz.shape[0] == 0:
            self.get_logger().warn("Empty cloud after preprocess")
            return

        # ---- (3) 극좌표 변환 및 range 필터 ----
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        r = np.sqrt(x * x + y * y)
        theta = np.arctan2(y, x)

        min_range = float(self.get_parameter('min_range').get_parameter_value().double_value)
        max_range = float(self.get_parameter('max_range').get_parameter_value().double_value)
        valid_mask = (r > min_range) & (r < max_range)

        x = x[valid_mask]
        y = y[valid_mask]
        z = z[valid_mask]
        r = r[valid_mask]
        theta = theta[valid_mask]

        if r.shape[0] == 0:
            self.get_logger().warn("No valid points after range filter")
            return

        # ---- (4) theta binning: 레이 별로 나누기 ----
        num_bins = int(self.get_parameter('num_angular_bins').get_parameter_value().integer_value)
        theta_wrap = (theta + math.pi) % (2.0 * math.pi)  # [-pi,pi] -> [0,2pi)
        bin_indices = (theta_wrap / (2.0 * math.pi) * num_bins).astype(np.int32)
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)

        max_slope_deg = float(self.get_parameter('max_slope_deg').get_parameter_value().double_value)
        max_slope_rad = math.radians(max_slope_deg)
        max_height_jump = float(self.get_parameter('max_height_jump').get_parameter_value().double_value)

        # slope 기반 ground 라벨
        N = r.shape[0]
        is_ground = np.zeros(N, dtype=bool)

        for b in range(num_bins):
            idxs = np.where(bin_indices == b)[0]
            if idxs.size < 2:
                continue

            order = np.argsort(r[idxs])
            ray_idxs = idxs[order]

            for b in range(num_bins):
                idxs = np.where(bin_indices == b)[0]
                if idxs.size < 2:
                    continue
                order = np.argsort(r[idxs])
                ray_idxs = idxs[order]

                # z가 ground_level_z 근처인 애들 중 가장 가까운 것만 seed
                seed_found = False
                for idx in ray_idxs:
                    if abs(z[idx] - ground_level_z) < 0.5:  # 상황에 맞게
                        is_ground[idx] = True
                        prev_idx = idx
                        prev_r = r[idx]
                        prev_z = z[idx]
                        seed_found = True
                        break
                if not seed_found:
                    continue
            prev_r = r[prev_idx]
            prev_z = z[prev_idx]

            for k in range(1, ray_idxs.size):
                cur_idx = ray_idxs[k]
                dr = r[cur_idx] - prev_r
                dz = z[cur_idx] - prev_z

                if dr > 1e-3:
                    slope = abs(math.atan2(dz, dr))
                else:
                    slope = 0.0

                # 급격한 변화면 non-ground로 유지
                if slope > max_slope_rad or abs(dz) > max_height_jump:
                    # non-ground
                    pass
                else:
                    # 완만하게 이어지면 ground로 편입
                    is_ground[cur_idx] = True
                    prev_idx = cur_idx
                    prev_r = r[cur_idx]
                    prev_z = z[cur_idx]

        # ---- (5) RANSAC로 global ground plane 보정 (옵션) ----
        use_ransac = bool(self.get_parameter('use_ransac').get_parameter_value().bool_value)
        if use_ransac:
            # slope 기반으로 뽑힌 ground seed
            seed_pts = np.stack([x[is_ground], y[is_ground], z[is_ground]], axis=1)
            if seed_pts.shape[0] >= 30:  # 최소 seed 개수 (상황에 맞게 조정)
                ransac_max_iter = int(self.get_parameter('ransac_max_iterations').get_parameter_value().integer_value)
                ransac_dist_th = float(self.get_parameter('ransac_distance_threshold').get_parameter_value().double_value)
                ransac_min_ratio = float(self.get_parameter('ransac_min_inlier_ratio').get_parameter_value().double_value)

                normal, d, inlier_count = ransac_plane(
                    seed_pts,
                    max_iterations=ransac_max_iter,
                    distance_threshold=ransac_dist_th,
                    min_inlier_ratio=ransac_min_ratio
                )

                if normal is not None:
                    # 전체 포인트에 대해 평면 거리 계산
                    # dist = |n·x + d|
                    all_pts = np.stack([x, y, z], axis=1)
                    dists = np.abs(np.dot(all_pts, normal) + d)

                    # 평면 가까운 포인트는 ground로 추가 인정
                    # (slope 결과와 OR 연산)
                    ransac_ground = dists < ransac_dist_th
                    before = np.count_nonzero(is_ground)
                    is_ground = np.logical_or(is_ground, ransac_ground)
                    after = np.count_nonzero(is_ground)

                    self.get_logger().info(
                        f"RANSAC plane applied: normal={normal}, inliers_seed={inlier_count}, "
                        f"ground_pts {before} -> {after}"
                    )
                else:
                    self.get_logger().info("RANSAC plane fitting failed or not reliable; using slope-only ground.")
            else:
                self.get_logger().info("Not enough ground seeds for RANSAC; using slope-only ground.")

        # ---- (6) 최종 non-ground 추출 ----
        no_ground_mask = ~is_ground
        filtered_x = x[no_ground_mask]
        filtered_y = y[no_ground_mask]
        filtered_z = z[no_ground_mask]

        # ---- (7) 높이 band로 바닥 찌꺼기 + 너무 높은 포인트 제거 ----
        ground_level_z = float(self.get_parameter('ground_level_z').get_parameter_value().double_value)
        min_obj_height = float(self.get_parameter('min_obj_height').get_parameter_value().double_value)
        max_obj_height = float(self.get_parameter('max_obj_height').get_parameter_value().double_value)

        keep_mask = (filtered_z > (ground_level_z + min_obj_height)) & (filtered_z < max_obj_height)
        filtered_x = filtered_x[keep_mask]
        filtered_y = filtered_y[keep_mask]
        filtered_z = filtered_z[keep_mask]

        if filtered_x.shape[0] == 0:
            self.get_logger().info("No obstacles above height threshold")
            filtered_xyz = np.zeros((0, 3), dtype=np.float32)
        else:
            filtered_xyz = np.stack([filtered_x, filtered_y, filtered_z], axis=1).astype(np.float32)

        # ---- (8) publish ----
        out_msg = xyz_to_pointcloud2(
            filtered_xyz,
            frame_id=msg.header.frame_id,
            stamp=msg.header.stamp
        )
        self.pub_.publish(out_msg)

        t1 = time.time()
        self.get_logger().info(
            f"input={xyz_all.shape[0]} "
            f"preproc={xyz.shape[0]} "
            f"valid={r.shape[0]} "
            f"nonground_pub={filtered_xyz.shape[0]} "
            f"elapsed={(t1 - t0) * 1000:.1f}ms"
        )


def main():
    rclpy.init()
    node = GroundRemovalNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
