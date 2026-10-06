"""Directional, per-scan terrain geometry. No ROS or learned traction model."""
from dataclasses import dataclass, asdict
from collections import deque
import math

import numpy as np

UNKNOWN, PASS, BLOCKED = 0, 1, 2


@dataclass
class TerrainConfig:
    resolution: float = 0.4
    radius: float = 6.0
    min_points: int = 3
    min_plane_cells: int = 6
    max_uphill_deg: float = 15.0
    max_downhill_deg: float = 12.0
    max_cross_slope_deg: float = 10.0
    max_step_up: float = 0.15
    max_drop: float = 0.12
    max_roughness: float = 0.08
    robot_width: float = 0.8
    robot_length: float = 1.2
    footprint_margin: float = 0.05
    lookahead: float = 3.0
    limits_calibrated: bool = False

    def __post_init__(self):
        numeric = [v for k, v in asdict(self).items()
                   if not isinstance(v, bool) and k != 'footprint_margin']
        if not math.isfinite(self.footprint_margin) or self.footprint_margin < 0:
            raise ValueError('footprint_margin must be finite and nonnegative')
        if not isinstance(self.min_points, int) or not isinstance(self.min_plane_cells, int):
            raise ValueError('Point/cell counts must be integers')
        if not all(math.isfinite(v) and v > 0 for v in numeric):
            raise ValueError('Terrain dimensions, limits and counts must be finite and positive')
        if self.min_points < 3 or self.min_plane_cells < 3:
            raise ValueError('At least three points/cells are required')
        if self.radius < self.lookahead + self.robot_length / 2 + self.resolution:
            raise ValueError('radius must include lookahead, half vehicle length and neighbor cells')
        if max(self.max_uphill_deg, self.max_downhill_deg, self.max_cross_slope_deg) >= 90:
            raise ValueError('Slope limits must be below 90 degrees')
        if math.ceil(self.radius / self.resolution) > 150:
            raise ValueError('Grid exceeds the supported per-scan size')


@dataclass
class TerrainResult:
    centers: np.ndarray  # local heading-aligned x,y and world z
    observed: np.ndarray
    slope_deg: np.ndarray
    cross_deg: np.ndarray
    roughness: np.ndarray
    step_up: np.ndarray  # direction (+heading,-heading), row, column
    drop: np.ndarray
    status: np.ndarray
    reasons: list
    config: TerrainConfig

    def corridor(self, direction=1):
        """Evaluate the complete vehicle-width corridor outside its current body."""
        c = self.config
        travel_x = self.centers[:, :, 0] * direction
        # Include every cell intersecting the rectangle, not just its center.
        selected = ((travel_x + c.resolution/2 > c.robot_length/2) &
                    (travel_x - c.resolution/2 < c.robot_length/2 + c.lookahead) &
                    (np.abs(self.centers[:, :, 1]) <
                     c.robot_width/2 + c.footprint_margin + c.resolution/2))
        index = 0 if direction == 1 else 1
        values = self.status[index][selected]
        count = int(selected.sum())
        state = BLOCKED if np.any(values == BLOCKED) else (
            PASS if count and np.all(values == PASS) else UNKNOWN)
        reason_counts = {}
        for row, col in np.argwhere(selected):
            for reason in self.reasons[index][row][col]:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        def maximum(array):
            v = array[selected]
            v = v[np.isfinite(v)]
            return float(v.max()) if len(v) else None
        return dict(status=['UNKNOWN', 'GEOMETRY_PASS', 'GEOMETRY_BLOCKED'][state],
                    calibrated=c.limits_calibrated, cells=count,
                    observed_fraction=float(self.observed[selected].mean()) if count else 0.0,
                    classified_fraction=float((values != UNKNOWN).mean()) if count else 0.0,
                    max_uphill_deg=maximum(np.maximum(self.slope_deg*direction, 0)),
                    max_downhill_deg=maximum(np.maximum(-self.slope_deg*direction, 0)),
                    max_cross_slope_deg=maximum(np.abs(self.cross_deg)),
                    max_step_up_m=maximum(self.step_up[index]),
                    max_drop_m=maximum(self.drop[index]), reasons=reason_counts)


def analyze_terrain(world_points, robot_x, robot_y, yaw, config=None):
    """Fit observed local surfaces and evaluate opposite travel directions.

    Each scan replaces the previous map. Missing returns remain UNKNOWN; no
    interpolation across missing cells and no assumption that they are cliffs.
    Z must already be in a gravity-aligned world frame.
    """
    c = config or TerrainConfig()
    points = np.asarray(world_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('Expected Nx3 world points')
    if not np.isfinite([robot_x, robot_y, yaw]).all():
        raise ValueError('Robot pose must be finite')
    points = points[np.isfinite(points).all(axis=1)].copy()
    dx, dy = points[:, 0]-robot_x, points[:, 1]-robot_y
    points[:, 0] = np.cos(yaw)*dx + np.sin(yaw)*dy
    points[:, 1] = -np.sin(yaw)*dx + np.cos(yaw)*dy
    half = int(np.ceil(c.radius/c.resolution))
    size = 2*half+1
    axis = np.arange(-half, half+1)*c.resolution
    xx, yy = np.meshgrid(axis, axis)
    centers = np.stack([xx, yy, np.full_like(xx, np.nan)], axis=-1)
    ij = np.floor(points[:, :2]/c.resolution+0.5).astype(int)+half
    valid = (np.abs(points[:, :2]) <= c.radius).all(axis=1)
    groups = {}
    for p, (col, row) in zip(points[valid], ij[valid]):
        groups.setdefault((row, col), []).append(p)
    observed = np.zeros((size, size), bool)
    for (row, col), group in list(groups.items()):
        group = groups[row, col] = np.asarray(group)
        if len(group) >= c.min_points:
            observed[row, col] = True
            centers[row, col, 2] = np.percentile(group[:, 2], 20)
    slope, cross, rough = [np.full((size, size), np.nan) for _ in range(3)]
    gradients = np.full((size, size, 2), np.nan)
    for row, col in np.argwhere(observed):
        lo_r, hi_r = max(0,row-1), min(size,row+2)
        lo_c, hi_c = max(0,col-1), min(size,col+2)
        patch = centers[lo_r:hi_r, lo_c:hi_c].reshape(-1,3)
        patch = patch[np.isfinite(patch[:,2])]
        if len(patch) < c.min_plane_cells:
            continue
        # Cell representatives provide a neighborhood plane. Actual point
        # residuals retain within-cell obstacles and uneven surfaces.
        design = np.column_stack([patch[:,:2]-centers[row,col,:2], np.ones(len(patch))])
        coeff, _, rank, _ = np.linalg.lstsq(design, patch[:,2], rcond=None)
        if rank < 3:
            continue
        a,b,z = coeff
        gradients[row,col] = [a,b]
        slope[row,col], cross[row,col] = np.degrees(np.arctan([a,b]))
        local = groups[row,col]
        residual = local[:,2] - ((local[:,:2]-centers[row,col,:2]) @ coeff[:2]+z)
        # Keep vertical discontinuities: a fitted ramp must not erase a step.
        rough[row,col] = np.percentile(np.abs(residual), 95)
    status = np.zeros((2,size,size),np.uint8)
    step, drop = [np.full((2,size,size), np.nan) for _ in range(2)]
    reasons = [[[[] for _ in range(size)] for _ in range(size)] for _ in range(2)]
    for d, direction in enumerate((1,-1)):
        for row in range(size):
            for col in range(size):
                why = reasons[d][row][col]
                if not observed[row,col]:
                    why.append('unobserved'); continue
                if not np.isfinite(slope[row,col]):
                    why.append('insufficient_plane_support'); continue
                along = direction*slope[row,col]
                if along > c.max_uphill_deg: why.append('uphill_limit')
                if -along > c.max_downhill_deg: why.append('downhill_limit')
                if abs(cross[row,col]) > c.max_cross_slope_deg: why.append('cross_slope_limit')
                if rough[row,col] > c.max_roughness: why.append('roughness_or_obstacle')
                nxt = col+direction
                neighbor_ok = 0 <= nxt < size and observed[row,nxt]
                if neighbor_ok:
                    # Remove expected slope only when both local gradients agree;
                    # otherwise retain the complete height transition conservatively.
                    gradient = gradients[row,col,0]
                    other = gradients[row,nxt,0]
                    correction = gradient*direction*c.resolution if (
                        np.isfinite(other) and abs(gradient-other) < 0.05) else 0.0
                    dz = centers[row,nxt,2]-centers[row,col,2]-correction
                    step[d,row,col] = max(0.0,dz)
                    drop[d,row,col] = max(0.0,-dz)
                    if dz > c.max_step_up: why.append('step_up_limit')
                    if -dz > c.max_drop: why.append('drop_limit')
                blocked = bool(why)
                if not neighbor_ok: why.append('unobserved_neighbor')
                status[d,row,col] = BLOCKED if blocked else (PASS if neighbor_ok else UNKNOWN)
    return TerrainResult(centers,observed,slope,cross,rough,step,drop,status,reasons,c)


class MotionEstimator:
    """Windowed odometry height change; explicitly reports its source."""
    def __init__(self, window_seconds=1.0, min_distance=0.1, slope_threshold_deg=2.0):
        self.window = window_seconds
        self.min_distance = min_distance
        self.threshold = slope_threshold_deg
        self.history = deque(maxlen=2000)

    def update(self, timestamp, x, y, z):
        if not np.isfinite([timestamp,x,y,z]).all():
            return dict(state='UNKNOWN', source='odometry_z', reason='invalid_pose')
        if self.history and timestamp <= self.history[-1][0]:
            self.history.clear()
        self.history.append((timestamp,x,y,z))
        while len(self.history)>2 and self.history[1][0] <= timestamp-self.window:
            self.history.popleft()
        old = self.history[0]
        dt = timestamp-old[0]
        result = dict(state='UNKNOWN', source='odometry_z', grade_deg=None, vertical_speed_mps=None)
        if dt < self.window*0.5:
            return result
        distance = math.hypot(x-old[1],y-old[2])
        result['vertical_speed_mps'] = (z-old[3])/dt
        if distance < self.min_distance:
            result['state'] = 'STATIONARY' if abs(z-old[3]) < 0.03 else 'UNKNOWN'
            return result
        grade = math.degrees(math.atan2(z-old[3],distance))
        result.update(grade_deg=grade, state=(
            'ASCENDING' if grade>self.threshold else 'DESCENDING' if grade < -self.threshold else 'LEVEL'))
        return result


def attitude_motion(roll, pitch, yaw, dx, dy, min_distance=0.1, threshold_deg=2.0):
    """Estimate travel grade from chassis attitude and actual displacement.

    This assumes the chassis support plane follows the terrain. Reverse travel
    changes grade sign. Pitch alone never establishes actual upward movement.
    """
    from .utils.transforms import rotation_matrix_from_euler
    result = dict(state='UNKNOWN', source='imu_attitude_and_odometry_direction', grade_deg=None)
    if not np.isfinite([roll, pitch, yaw, dx, dy]).all():
        return result
    distance = math.hypot(dx, dy)
    if distance < min_distance:
        result['state'] = 'STATIONARY'
        return result
    normal = rotation_matrix_from_euler(roll, pitch, yaw)[:, 2]
    if normal[2] <= 0.1:
        return result
    dz = -float(normal[0]*dx + normal[1]*dy)/normal[2]
    grade = math.degrees(math.atan2(dz, distance))
    result.update(grade_deg=grade, state=(
        'ASCENDING' if grade > threshold_deg else
        'DESCENDING' if grade < -threshold_deg else 'LEVEL'))
    return result
