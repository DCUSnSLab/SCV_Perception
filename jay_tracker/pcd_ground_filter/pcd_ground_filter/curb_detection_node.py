#!/usr/bin/env python3
"""Curb / negative-obstacle detection for sidewalk autonomy.

Motivation
----------
The existing ``ground_removal_node`` (and ``local_costmap``) intentionally treat
low curbs as ground and drop everything below ~0.15 m, so the boundary between a
sidewalk and the lower asphalt road never appears in the costmap.  When a planner
avoids an obstacle it can therefore steer across the curb onto the road, where the
15-25 cm drop can high-center or roll the vehicle and traps it off the sidewalk.

This node reconstructs that boundary.  For every angular bin of the LiDAR it walks
outward along the ground and flags radial locations where the ground elevation
steps up or down by a *curb-sized* amount ([curb_min_h, curb_max_h]) over a short
radial span -- i.e. a curb edge or a drop-off (negative obstacle).  At each flagged
edge it synthesises a short vertical "wall" of points, raised into the obstacle
height band, and republishes them merged with the original cloud.  Point
``local_costmap``'s ``point_cloud_topic`` at the output topic and the curb becomes a
lethal obstacle -- no C++ change required.

Only geometry is used, so it works at night (unlike the camera).
"""

import math
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header

from numba import njit


# ==============================================================================
# Core detector (Numba-accelerated).  Runs in the sensor frame.
# ==============================================================================
@njit(cache=True, fastmath=True)
def detect_curb_edges(x, y, z, r, bin_indices, num_bins,
                      ground_ref_z, z_band,
                      curb_min_h, curb_max_h,
                      max_step_dr, min_dr, max_gap):
    """Return (ex, ey, ebase_z): x/y of curb edges and the ground z to base a
    wall on.  Within each angular bin the near-ground returns are walked
    outward; an edge is a step in ground elevation of magnitude in
    [curb_min_h, curb_max_h] over a short radial span (or a comparable drop
    across an occlusion gap = drop-off).

    A single-ring elevation blip (up then immediately back down) is NOT a curb,
    so a candidate step is confirmed only if the new ground level *persists*
    into the next return.  This rejects the concentric ring artifacts that a
    naive radial walk produces on slightly uneven ground.  Points far from
    ground level are ignored so canopy / signboards do not create phantoms."""
    n = len(x)
    ex = np.empty(n, np.float32)
    ey = np.empty(n, np.float32)
    ebase = np.empty(n, np.float32)
    cnt = 0

    lo = ground_ref_z - z_band
    hi = ground_ref_z + z_band

    for b in range(num_bins):
        idxs = np.where(bin_indices == b)[0]
        if len(idxs) < 3:
            continue
        ray = idxs[np.argsort(r[idxs])]

        # compact the in-band (near-ground) returns, already sorted by range
        mb = 0
        for k in range(len(ray)):
            zc = z[ray[k]]
            if zc >= lo and zc <= hi:
                mb += 1
        if mb < 3:
            continue
        rr = np.empty(mb, np.float32)
        zz = np.empty(mb, np.float32)
        xx = np.empty(mb, np.float32)
        yy = np.empty(mb, np.float32)
        j = 0
        for k in range(len(ray)):
            ii = ray[k]
            zc = z[ii]
            if zc >= lo and zc <= hi:
                rr[j] = r[ii]; zz[j] = zc; xx[j] = x[ii]; yy[j] = y[ii]; j += 1

        for k in range(1, mb):
            dr = rr[k] - rr[k - 1]
            if dr < min_dr:
                continue
            dz = zz[k] - zz[k - 1]
            adz = dz if dz >= 0.0 else -dz

            if dr > max_gap:
                # occlusion gap; a lower continuation implies a drop-off edge
                if dz <= -curb_min_h and dz >= -curb_max_h:
                    ex[cnt] = 0.5 * (xx[k - 1] + xx[k])
                    ey[cnt] = 0.5 * (yy[k - 1] + yy[k])
                    ebase[cnt] = zz[k - 1]
                    cnt += 1
                continue

            is_curb = (adz >= curb_min_h and adz <= curb_max_h and dr <= max_step_dr)
            is_cliff = (dz < 0.0 and -dz > curb_max_h and dr <= max_step_dr)
            if not (is_curb or is_cliff):
                continue

            # persistence: the new level must not immediately revert (ring blip)
            persist = True
            if k + 1 < mb:
                dz2 = zz[k + 1] - zz[k]
                # opposite-sign curb-sized step back = blip -> reject
                if (dz2 * dz) < 0.0 and (dz2 if dz2 >= 0.0 else -dz2) >= curb_min_h:
                    persist = False
            if not persist:
                continue

            ex[cnt] = 0.5 * (xx[k - 1] + xx[k])
            ey[cnt] = 0.5 * (yy[k - 1] + yy[k])
            ebase[cnt] = zz[k - 1] if zz[k - 1] > zz[k] else zz[k]
            cnt += 1

    return ex[:cnt], ey[:cnt], ebase[:cnt]


@njit(cache=True, fastmath=True)
def detect_curb_grid(x, y, z, half_size, cell,
                     ground_ref_z, z_band,
                     curb_min_h, curb_max_h, min_pts_cell):
    """2.5D grid height-gradient curb detector (Cartesian, sensor frame).

    Builds a ground-height grid (low-z per cell) over a square window, then
    flags a cell as a curb where the horizontal height step to a neighbour is
    curb-sized ([curb_min_h, curb_max_h]).  Because it operates on the
    reconstructed ground surface rather than a per-ray radial walk, it does not
    produce the ego-centred ring arcs the radial method suffers from, and real
    curbs come out as continuous lines.  Returns (ex, ey, ebase_z)."""
    n = len(x)
    ncell = int((2.0 * half_size) / cell) + 1
    INF = np.float32(1e9)
    gmin = np.full((ncell, ncell), INF, np.float32)
    cnt = np.zeros((ncell, ncell), np.int32)

    lo = ground_ref_z - z_band
    hi = ground_ref_z + z_band

    for i in range(n):
        zc = z[i]
        if zc < lo or zc > hi:
            continue
        gx = int((x[i] + half_size) / cell)
        gy = int((y[i] + half_size) / cell)
        if gx < 0 or gx >= ncell or gy < 0 or gy >= ncell:
            continue
        cnt[gx, gy] += 1
        if zc < gmin[gx, gy]:
            gmin[gx, gy] = zc

    # 4-neighbour height gradient
    max_out = ncell * ncell
    ex = np.empty(max_out, np.float32)
    ey = np.empty(max_out, np.float32)
    eb = np.empty(max_out, np.float32)
    k = 0
    for gx in range(1, ncell - 1):
        for gy in range(1, ncell - 1):
            if cnt[gx, gy] < min_pts_cell:
                continue
            g0 = gmin[gx, gy]
            best = np.float32(0.0)
            bhi = g0
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    if cnt[gx + dx, gy + dy] < min_pts_cell:
                        continue
                    gn = gmin[gx + dx, gy + dy]
                    d = g0 - gn
                    if d < 0.0:
                        d = -d
                    if d > best:
                        best = d
                        bhi = g0 if g0 > gn else gn
            if best >= curb_min_h and best <= curb_max_h:
                ex[k] = (gx + 0.5) * cell - half_size
                ey[k] = (gy + 0.5) * cell - half_size
                eb[k] = bhi
                k += 1
    return ex[:k], ey[:k], eb[:k]


@njit(cache=True, fastmath=True)
def below_grade_detect(x, y, z, half_size, cell,
                       plane_a, plane_c, z_band,
                       drop_min, raise_min, raise_max,
                       min_pts_cell, max_mark_range):
    """Below-grade region masking — the validated primary detector.

    The sparse-edge approaches fail here because the LiDAR rings run nearly
    parallel to the curb, so a single frame yields only a handful of edge
    crossings.  Instead we flag the entire region whose ground level is a
    curb-drop BELOW the sidewalk plane the robot is on (the road: a large,
    continuous, densely-sampled surface ~20 cm down), plus cells slightly
    RAISED above the plane (kerb tops / islands below the costmap's 0.15 m
    positive threshold).  The flagged region's boundary IS the curb line.

    The sidewalk plane z = plane_a*x + plane_c is fitted on the strip dead
    ahead and absorbs sensor pitch and walkway grade.
    Returns (ex, ey): flagged cell centres."""
    n = len(x)
    ncell = int((2.0 * half_size) / cell) + 1
    INF = np.float32(1e9)
    gmin = np.full((ncell, ncell), INF, np.float32)
    cnt = np.zeros((ncell, ncell), np.int32)

    for i in range(n):
        xi = x[i]
        exp_z = plane_a * xi + plane_c
        zc = z[i]
        if zc < exp_z - z_band or zc > exp_z + z_band:
            continue
        gx = int((xi + half_size) / cell)
        gy = int((y[i] + half_size) / cell)
        if gx < 0 or gx >= ncell or gy < 0 or gy >= ncell:
            continue
        cnt[gx, gy] += 1
        if zc < gmin[gx, gy]:
            gmin[gx, gy] = zc

    ex = np.empty(ncell * ncell, np.float32)
    ey = np.empty(ncell * ncell, np.float32)
    k = 0
    for gx in range(ncell):
        for gy in range(ncell):
            if cnt[gx, gy] < min_pts_cell:
                continue
            cx = (gx + 0.5) * cell - half_size
            cy = (gy + 0.5) * cell - half_size
            if cx * cx + cy * cy > max_mark_range * max_mark_range:
                continue
            exp_z = plane_a * cx + plane_c
            dev = gmin[gx, gy] - exp_z          # <0: below the sidewalk plane
            if dev <= -drop_min:
                ex[k] = cx; ey[k] = cy; k += 1
            elif dev >= raise_min and dev <= raise_max:
                ex[k] = cx; ey[k] = cy; k += 1
    return ex[:k], ey[:k]


def fit_ahead_plane(x, y, z, c_min=-2.0, c_max=-0.3):
    """Fit z = a*x + c on the narrow strip the robot is driving on.

    Robustness (validated against calibration bags with a car parked 5 m
    ahead): an object inside the strip dominates mid percentiles and hijacks
    the fit, so we anchor on the LOWEST surface (5th percentile band — the
    ground is always below whatever stands on it) and then require the fitted
    height to be physically plausible for the sensor mount ([c_min, c_max]).
    Returns (a, c, ok) — ok=False when the strip is obstructed / implausible
    and the fit must not be trusted (caller keeps the previous EMA plane)."""
    m = (x > 1.0) & (x < 6.0) & (np.abs(y) < 0.8) & (np.abs(z) < 3.0)
    if m.sum() < 30:
        return 0.0, -0.9, False
    xs = x[m]; zs = z[m]
    z0 = np.percentile(zs, 5)
    mm = np.abs(zs - z0) < 0.15
    if mm.sum() >= 20:
        xs, zs = xs[mm], zs[mm]
    A = np.stack([xs, np.ones_like(xs)], axis=1)
    sol, *_ = np.linalg.lstsq(A, zs, rcond=None)
    a, c = float(sol[0]), float(sol[1])
    if not (c_min <= c <= c_max) or abs(a) > 0.15:
        return a, c, False
    return a, c, True


def pointcloud2_to_xyz(msg: PointCloud2):
    s = point_cloud2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
    if s.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    out = np.empty((s.shape[0], 3), dtype=np.float32)
    out[:, 0] = s['x']; out[:, 1] = s['y']; out[:, 2] = s['z']
    return out


def xyzi_to_pointcloud2(pts, frame_id, stamp):
    pts = np.ascontiguousarray(pts, dtype=np.float32)
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    m = PointCloud2()
    m.header = Header(stamp=stamp, frame_id=frame_id)
    m.height = 1
    m.width = int(pts.shape[0])
    m.fields = fields
    m.is_bigendian = False
    m.point_step = 16
    m.row_step = m.point_step * m.width
    m.is_dense = True
    m.data = pts.tobytes()
    return m


def build_curb_walls(ex, ey, ebase, offsets, intensity):
    """Turn each edge into a short vertical stack of points raised into the
    obstacle height band, so the height-filtering costmap marks it lethal."""
    if ex.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)
    k = len(offsets)
    n = ex.shape[0]
    out = np.empty((n * k, 4), dtype=np.float32)
    for j, off in enumerate(offsets):
        s = slice(j * n, (j + 1) * n)
        out[s, 0] = ex
        out[s, 1] = ey
        out[s, 2] = ebase + off
        out[s, 3] = intensity
    return out


class CurbDetectionNode(Node):
    def __init__(self):
        super().__init__('curb_detection_node')

        self.declare_parameter('input_topic', '/velodyne_points')
        self.declare_parameter('output_topic', '/velodyne_points_curb')
        self.declare_parameter('curb_topic', '/curb_points')  # debug-only cloud

        # detector: 'below_grade' (validated default: masks the whole road
        # region below the sidewalk plane), 'grid' (2.5D height-gradient
        # edges) or 'radial' (per-ray walk)
        self.declare_parameter('method', 'below_grade')
        self.declare_parameter('grid_cell', 0.30)         # grid cell size (m)
        self.declare_parameter('grid_min_pts', 1)         # min returns per cell

        # below_grade params (relative to the fitted sidewalk plane)
        self.declare_parameter('drop_min', 0.10)      # below-plane => road
        self.declare_parameter('raise_min', 0.08)     # raised kerb band lower
        self.declare_parameter('raise_max', 0.45)     # raised kerb band upper
        self.declare_parameter('plane_z_band', 0.8)   # ground band around plane
        self.declare_parameter('max_mark_range', 12.0)
        self.declare_parameter('plane_ema_alpha', 0.3)   # temporal smoothing
        self.declare_parameter('plane_max_jump', 0.25)   # reject fits deviating more
        self.declare_parameter('plane_c_min', -2.0)      # plausible ground height range
        self.declare_parameter('plane_c_max', -0.3)      #   (sensor is c above ground)

        # sensor height: ground sits near this z in the sensor frame
        self.declare_parameter('ground_ref_z', -1.5)
        self.declare_parameter('ground_z_band', 0.6)

        self.declare_parameter('curb_min_height', 0.10)   # ignore < 10 cm (ground noise/ring)
        self.declare_parameter('curb_max_height', 0.5)    # taller -> wall, not curb
        self.declare_parameter('max_step_dr', 0.35)       # step must be radially abrupt (real curb face); rejects far ring-gap arcs
        self.declare_parameter('min_dr', 0.04)
        self.declare_parameter('max_gap', 1.0)            # occlusion gap for drop-offs

        self.declare_parameter('num_angular_bins', 720)
        self.declare_parameter('min_range', 1.0)
        self.declare_parameter('max_range', 15.0)

        # vertical stack of synthetic wall points, metres above local ground
        self.declare_parameter('wall_offsets', [0.20, 0.35, 0.50])
        self.declare_parameter('wall_intensity', 250.0)
        self.declare_parameter('merge_original', True)    # republish raw + curbs

        gp = self.get_parameter
        self.method = str(gp('method').value)
        self.grid_cell = float(gp('grid_cell').value)
        self.grid_min_pts = int(gp('grid_min_pts').value)
        self.drop_min = float(gp('drop_min').value)
        self.raise_min = float(gp('raise_min').value)
        self.raise_max = float(gp('raise_max').value)
        self.plane_z_band = float(gp('plane_z_band').value)
        self.max_mark_range = float(gp('max_mark_range').value)
        self.plane_ema_alpha = float(gp('plane_ema_alpha').value)
        self.plane_max_jump = float(gp('plane_max_jump').value)
        self.plane_c_min = float(gp('plane_c_min').value)
        self.plane_c_max = float(gp('plane_c_max').value)
        self.plane_a = 0.0            # EMA state of the sidewalk plane
        self.plane_c = None
        self.in_topic = gp('input_topic').value
        self.out_topic = gp('output_topic').value
        self.curb_topic = gp('curb_topic').value
        self.ground_ref_z = float(gp('ground_ref_z').value)
        self.z_band = float(gp('ground_z_band').value)
        self.curb_min = float(gp('curb_min_height').value)
        self.curb_max = float(gp('curb_max_height').value)
        self.max_step_dr = float(gp('max_step_dr').value)
        self.min_dr = float(gp('min_dr').value)
        self.max_gap = float(gp('max_gap').value)
        self.num_bins = int(gp('num_angular_bins').value)
        self.min_range = float(gp('min_range').value)
        self.max_range = float(gp('max_range').value)
        self.wall_offsets = [float(v) for v in gp('wall_offsets').value]
        self.wall_intensity = float(gp('wall_intensity').value)
        self.merge_original = bool(gp('merge_original').value)

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.sub_ = self.create_subscription(PointCloud2, self.in_topic,
                                             self.cb, qos)
        self.pub_ = self.create_publisher(PointCloud2, self.out_topic, 10)
        self.curb_pub_ = self.create_publisher(PointCloud2, self.curb_topic, 10)

        self._logged = 0
        self.get_logger().info(
            f"Curb detection ready: {self.in_topic} -> {self.out_topic} "
            f"(curb {self.curb_min:.2f}-{self.curb_max:.2f} m, bins={self.num_bins})")

    def _publish_passthrough(self, xyz, msg):
        """Fail-safe: forward the original cloud unmodified so positive
        obstacles always reach the costmap even when curb augmentation is
        not possible for this frame."""
        orig = np.empty((xyz.shape[0], 4), dtype=np.float32)
        orig[:, :3] = xyz
        orig[:, 3] = 0.0
        self.pub_.publish(
            xyzi_to_pointcloud2(orig, msg.header.frame_id, msg.header.stamp))

    def cb(self, msg: PointCloud2):
        t0 = time.perf_counter()
        xyz = pointcloud2_to_xyz(msg)
        if xyz.shape[0] == 0:
            return

        x = xyz[:, 0]; y = xyz[:, 1]; z = xyz[:, 2]
        r = np.sqrt(x * x + y * y)
        m = (r > self.min_range) & (r < self.max_range)
        xf, yf, zf, rf = x[m], y[m], z[m], r[m]
        if xf.shape[0] == 0:
            self._publish_passthrough(xyz, msg)
            return

        if self.method == 'below_grade':
            a, c, ok = fit_ahead_plane(xf, yf, zf, self.plane_c_min, self.plane_c_max)
            if self.plane_c is None:
                if not ok:
                    # No trustworthy plane yet (e.g. parked facing a wall).
                    # CRITICAL: still republish the raw cloud — downstream
                    # local_costmap consumes ONLY our output topic, so an
                    # early return here would starve the whole obstacle
                    # pipeline and the vehicle would drive blind.
                    self._publish_passthrough(xyz, msg)
                    self.get_logger().warn(
                        'below_grade plane not initialised — passthrough '
                        '(no curb walls yet)', throttle_duration_sec=5.0)
                    return
                self.plane_a, self.plane_c = a, c
            elif ok and abs(c - self.plane_c) <= self.plane_max_jump:
                al = self.plane_ema_alpha
                self.plane_a = (1 - al) * self.plane_a + al * a
                self.plane_c = (1 - al) * self.plane_c + al * c
            # else: keep previous EMA plane (strip obstructed / bad fit)
            ex, ey = below_grade_detect(
                xf, yf, zf, self.max_range, self.grid_cell,
                np.float32(self.plane_a), np.float32(self.plane_c),
                np.float32(self.plane_z_band),
                np.float32(self.drop_min),
                np.float32(self.raise_min), np.float32(self.raise_max),
                self.grid_min_pts, np.float32(self.max_mark_range))
            # anchor walls on the SIDEWALK plane (not the lower road ground)
            # so they stay inside the costmap's odom-frame [0.15, 2.0] band
            eb = self.plane_a * ex + self.plane_c
        elif self.method == 'grid':
            ex, ey, eb = detect_curb_grid(
                xf, yf, zf, self.max_range, self.grid_cell,
                self.ground_ref_z, self.z_band,
                self.curb_min, self.curb_max, self.grid_min_pts)
        else:
            theta = np.arctan2(yf, xf)
            tw = (theta + math.pi) % (2.0 * math.pi)
            bins = np.clip((tw / (2.0 * math.pi) * self.num_bins).astype(np.int32),
                           0, self.num_bins - 1)
            ex, ey, eb = detect_curb_edges(
                xf, yf, zf, rf, bins, self.num_bins,
                self.ground_ref_z, self.z_band,
                self.curb_min, self.curb_max,
                self.max_step_dr, self.min_dr, self.max_gap)

        walls = build_curb_walls(ex, ey, eb, self.wall_offsets, self.wall_intensity)

        # debug cloud: curb walls only
        self.curb_pub_.publish(
            xyzi_to_pointcloud2(walls, msg.header.frame_id, msg.header.stamp))

        # costmap-facing cloud: original + curb walls
        if self.merge_original:
            orig = np.empty((xyz.shape[0], 4), dtype=np.float32)
            orig[:, :3] = xyz
            orig[:, 3] = 0.0
            out = np.vstack((orig, walls)) if walls.shape[0] else orig
        else:
            out = walls
        self.pub_.publish(
            xyzi_to_pointcloud2(out, msg.header.frame_id, msg.header.stamp))

        self._logged += 1
        if self._logged <= 3 or self._logged % 50 == 0:
            dt = (time.perf_counter() - t0) * 1000.0
            self.get_logger().info(
                f"curb edges={ex.shape[0]} wall_pts={walls.shape[0]} "
                f"in={xyz.shape[0]} took={dt:.1f}ms")


def main():
    rclpy.init()
    node = CurbDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
