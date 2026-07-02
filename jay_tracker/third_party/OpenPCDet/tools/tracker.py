import numpy as np
from scipy.optimize import linear_sum_assignment


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def box_corners_bev(center_x, center_y, length, width, yaw):
    half_l = length * 0.5
    half_w = width * 0.5
    corners = np.array([
        [half_l, half_w],
        [half_l, -half_w],
        [-half_l, -half_w],
        [-half_l, half_w],
    ], dtype=np.float32)

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rot = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float32)
    return corners @ rot.T + np.array([center_x, center_y], dtype=np.float32)


def polygon_area(poly):
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def inside_edge(point, edge_start, edge_end):
    return ((edge_end[0] - edge_start[0]) * (point[1] - edge_start[1]) -
            (edge_end[1] - edge_start[1]) * (point[0] - edge_start[0])) >= 0.0


def segment_intersection(p1, p2, q1, q2):
    s = np.vstack([p1, p2, q1, q2]).astype(np.float32)
    h = np.hstack([s, np.ones((4, 1), dtype=np.float32)])
    line1 = np.cross(h[0], h[1])
    line2 = np.cross(h[2], h[3])
    x, y, z = np.cross(line1, line2)
    if abs(z) < 1e-6:
        return p2
    return np.array([x / z, y / z], dtype=np.float32)


def polygon_clip(subject, clipper):
    output = subject.copy()
    for i in range(len(clipper)):
        edge_start = clipper[i]
        edge_end = clipper[(i + 1) % len(clipper)]
        input_list = output
        if len(input_list) == 0:
            break
        output = []
        prev = input_list[-1]
        for curr in input_list:
            curr_inside = inside_edge(curr, edge_start, edge_end)
            prev_inside = inside_edge(prev, edge_start, edge_end)
            if curr_inside:
                if not prev_inside:
                    output.append(segment_intersection(prev, curr, edge_start, edge_end))
                output.append(curr)
            elif prev_inside:
                output.append(segment_intersection(prev, curr, edge_start, edge_end))
            prev = curr
        output = np.asarray(output, dtype=np.float32)
    return output


def bev_iou(box_a, box_b):
    poly_a = box_corners_bev(box_a[0], box_a[1], box_a[2], box_a[3], box_a[4])
    poly_b = box_corners_bev(box_b[0], box_b[1], box_b[2], box_b[3], box_b[4])
    inter_poly = polygon_clip(poly_a, poly_b)
    inter_area = polygon_area(inter_poly)
    if inter_area <= 0.0:
        return 0.0
    area_a = polygon_area(poly_a)
    area_b = polygon_area(poly_b)
    union = max(area_a + area_b - inter_area, 1e-6)
    return float(inter_area / union)


class Track:
    _next_id = 0

    def __init__(self, det, timestamp, default_dt=0.1, n_init=3):
        self.id = Track._next_id
        Track._next_id += 1

        self.default_dt = default_dt
        self.last_timestamp = timestamp
        self.n_init = n_init

        # Motion state: [x, y, vx, vy]
        self.x = np.array([det[0], det[1], 0.0, 0.0], dtype=np.float32).reshape(4, 1)
        self.P = np.eye(4, dtype=np.float32)
        self.F = np.eye(4, dtype=np.float32)
        self.H = np.zeros((2, 4), dtype=np.float32)
        self.H[0, 0] = self.H[1, 1] = 1.0
        self.Q = np.eye(4, dtype=np.float32) * 0.01
        self.R = np.eye(2, dtype=np.float32) * 0.1

        self.yaw = float(det[6])
        self.z = float(det[2])
        self.dim = det[3:6].astype(np.float32).copy()
        self.cls_id = int(det[7])
        self.score = float(det[8]) if len(det) > 8 else 0.0
        self.class_scores = {}
        if self.cls_id > 0:
            self.class_scores[self.cls_id] = self.score

        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.state = 'confirmed' if self.hits >= self.n_init else 'tentative'
        self.max_yaw_jump = np.deg2rad(35.0)
        self.min_speed_for_yaw = 0.75
        self.yaw_smoothing = 0.2
        self.early_yaw_smoothing = 0.45
        self.min_iou_for_static_match = 0.05
        self.stationary_anchor = det[0:2].astype(np.float32).copy()
        self.dim_smoothing = 0.15
        self.max_dim_change = 0.2
        self.z_smoothing = 0.2
        self.extent_history = []
        self.history_size = 5
        self._push_extent(det)

    def _push_extent(self, det):
        self.extent_history.append({
            'yaw': float(det[6]),
            'z': float(det[2]),
            'dim': det[3:6].astype(np.float32).copy(),
        })
        if len(self.extent_history) > self.history_size:
            self.extent_history.pop(0)

    def _apply_extent_history(self):
        if not self.extent_history:
            return
        yaw_values = np.array([item['yaw'] for item in self.extent_history], dtype=np.float32)
        yaw_unit = np.exp(1j * yaw_values)
        self.yaw = wrap_angle(np.angle(np.mean(yaw_unit)))
        z_values = np.array([item['z'] for item in self.extent_history], dtype=np.float32)
        self.z = float(np.median(z_values))
        dim_values = np.stack([item['dim'] for item in self.extent_history], axis=0)
        self.dim = np.median(dim_values, axis=0).astype(np.float32)

    def _update_semantics(self, cls_id, score):
        if cls_id <= 0:
            return
        score = float(score)
        self.class_scores[cls_id] = self.class_scores.get(cls_id, 0.0) * 0.8 + score
        best_cls, best_score = max(self.class_scores.items(), key=lambda item: item[1])
        self.cls_id = int(best_cls)
        self.score = float(best_score)

    def _set_dt(self, dt):
        self.F = np.eye(4, dtype=np.float32)
        self.F[0, 2] = dt
        self.F[1, 3] = dt

        self.Q = np.eye(4, dtype=np.float32) * 0.01
        self.Q[0, 0] = self.Q[1, 1] = max(0.01, 0.05 * dt)
        self.Q[2, 2] = self.Q[3, 3] = max(0.01, 0.2 * dt)

    def predict(self, timestamp):
        if self.last_timestamp is None or timestamp is None:
            dt = self.default_dt
        else:
            dt = float(timestamp - self.last_timestamp)
            if dt <= 0.0:
                dt = self.default_dt
            dt = min(max(dt, 1e-2), 1.0)

        self._set_dt(dt)
        speed = float(np.linalg.norm(self.x[2:4, 0]))
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        if speed < self.min_speed_for_yaw:
            self.yaw = wrap_angle(self.yaw)
        else:
            vel_yaw = np.arctan2(self.x[3, 0], self.x[2, 0])
            self.yaw = wrap_angle(0.85 * self.yaw + 0.15 * vel_yaw)
        self.age += 1
        self.time_since_update += 1
        self.last_timestamp = timestamp

    def update(self, det, timestamp):
        z = np.array([det[0], det[1]], dtype=np.float32).reshape(2, 1)

        diff = wrap_angle(float(det[6] - self.yaw))
        if abs(diff) > np.pi / 2:
            flip_diff = diff - np.sign(diff) * np.pi
            if abs(flip_diff) < abs(diff):
                diff = flip_diff
        speed = float(np.linalg.norm(self.x[2:4, 0]))
        allow_early_yaw = self.hits < self.n_init + 1 and abs(diff) <= np.deg2rad(70.0)
        use_yaw_measurement = (speed >= self.min_speed_for_yaw and abs(diff) <= self.max_yaw_jump) or allow_early_yaw

        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(4, dtype=np.float32) - K @ self.H) @ self.P

        if use_yaw_measurement:
            yaw_alpha = self.early_yaw_smoothing if allow_early_yaw and speed < self.min_speed_for_yaw else self.yaw_smoothing
            self.yaw = wrap_angle(self.yaw + yaw_alpha * diff)
        if speed >= self.min_speed_for_yaw:
            vel_yaw = np.arctan2(self.x[3, 0], self.x[2, 0])
            self.yaw = wrap_angle(0.8 * self.yaw + 0.2 * vel_yaw)

        self.hits += 1
        self.time_since_update = 0
        self.last_timestamp = timestamp
        measured_dim = det[3:6].astype(np.float32).copy()
        dim_ratio = measured_dim / np.maximum(self.dim, 1e-3)
        dim_ratio = np.clip(dim_ratio, 1.0 - self.max_dim_change, 1.0 + self.max_dim_change)
        measured_dim = self.dim * dim_ratio
        det_for_history = det.copy()
        det_for_history[3:6] = ((1.0 - self.dim_smoothing) * self.dim) + (self.dim_smoothing * measured_dim)
        det_for_history[2] = (1.0 - self.z_smoothing) * self.z + self.z_smoothing * float(det[2])
        det_for_history[6] = self.yaw
        self._push_extent(det_for_history)
        self._apply_extent_history()
        self._update_semantics(int(det[7]), float(det[8]) if len(det) > 8 else 0.0)
        anchor_alpha = 0.05 if speed < self.min_speed_for_yaw else 0.2
        self.stationary_anchor = ((1.0 - anchor_alpha) * self.stationary_anchor) + (anchor_alpha * self.x[:2, 0])
        if self.hits >= self.n_init:
            self.state = 'confirmed'

    def mark_missed(self, max_age):
        if self.time_since_update > max_age:
            self.state = 'deleted'

    @property
    def is_confirmed(self):
        return self.state == 'confirmed'

    @property
    def is_deleted(self):
        return self.state == 'deleted'


class GlobalTracker:
    def __init__(
        self,
        max_age=15,
        min_hits=3,
        dist_threshold=4.0,
        score_threshold=0.4,
        cost_weights=None,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.dist_threshold = dist_threshold
        self.score_threshold = score_threshold
        self.cost_weights = cost_weights or {
            'xy': 1.0,
            'z': 0.5,
            'yaw': 0.75,
            'size': 0.5,
            'iou': 1.5,
        }
        self.tracks = []
        self.archived_tracks = []
        self.reid_max_age = max_age * 4

    def _deduplicate_tracks(self):
        if len(self.tracks) < 2:
            return

        keep = [True] * len(self.tracks)
        for i in range(len(self.tracks)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(self.tracks)):
                if not keep[j]:
                    continue

                track_a = self.tracks[i]
                track_b = self.tracks[j]
                if track_a.cls_id > 0 and track_b.cls_id > 0 and track_a.cls_id != track_b.cls_id:
                    continue

                xy_dist = np.linalg.norm(track_a.x[:2, 0] - track_b.x[:2, 0])
                iou = bev_iou(
                    [track_a.x[0, 0], track_a.x[1, 0], track_a.dim[0], track_a.dim[1], track_a.yaw],
                    [track_b.x[0, 0], track_b.x[1, 0], track_b.dim[0], track_b.dim[1], track_b.yaw],
                )
                if xy_dist > self.dist_threshold * 0.5 and iou < 0.2:
                    continue

                a_rank = (track_a.is_confirmed, track_a.hits, -track_a.time_since_update, track_a.score)
                b_rank = (track_b.is_confirmed, track_b.hits, -track_b.time_since_update, track_b.score)
                if a_rank >= b_rank:
                    keep[j] = False
                else:
                    keep[i] = False
                    break

        self.tracks = [track for idx, track in enumerate(self.tracks) if keep[idx]]

    def _should_spawn_track(self, det):
        for track in self.tracks:
            det_cls = int(det[7])
            if track.is_deleted:
                continue
            if track.cls_id > 0 and det_cls > 0 and track.cls_id != det_cls:
                continue

            xy_dist = np.linalg.norm(track.x[:2, 0] - det[:2])
            iou = bev_iou(
                [track.x[0, 0], track.x[1, 0], track.dim[0], track.dim[1], track.yaw],
                [det[0], det[1], det[3], det[4], det[6]],
            )
            if xy_dist < max(0.9, self.dist_threshold * 0.4) and iou > 0.1:
                return False
        return True

    def _archive_deleted_tracks(self):
        alive_tracks = []
        for track in self.tracks:
            if track.is_deleted:
                self.archived_tracks.append(track)
            else:
                alive_tracks.append(track)
        self.tracks = alive_tracks
        self.archived_tracks = [
            track for track in self.archived_tracks
            if track.time_since_update <= self.reid_max_age
        ]

    def _reassociate_archived_track(self, det, timestamp):
        ego_speed = 0.0
        ego_yaw_rate = 0.0
        if hasattr(self, 'current_ego_motion') and self.current_ego_motion is not None:
            ego_speed = abs(float(self.current_ego_motion.get('speed', 0.0)))
            ego_yaw_rate = abs(float(self.current_ego_motion.get('yaw_rate', 0.0)))
        best_track = None
        best_cost = np.inf
        for track in self.archived_tracks:
            det_cls = int(det[7])
            if track.cls_id > 0 and det_cls > 0 and track.cls_id != det_cls:
                continue
            if track.time_since_update > self.reid_max_age:
                continue

            xy_dist = np.linalg.norm(track.stationary_anchor[:2] - det[:2])
            iou = bev_iou(
                [track.stationary_anchor[0], track.stationary_anchor[1], track.dim[0], track.dim[1], track.yaw],
                [det[0], det[1], det[3], det[4], det[6]],
            )
            size_diff = np.linalg.norm(track.dim - det[3:6]) / max(np.linalg.norm(track.dim), 1e-3)
            yaw_diff = abs(wrap_angle(float(track.yaw - det[6])))

            reid_xy_gate = max(1.5, self.dist_threshold + 0.75 * ego_speed + 2.0 * ego_yaw_rate)
            if xy_dist > reid_xy_gate:
                continue
            if iou < 0.02 and xy_dist > 1.0:
                continue
            if size_diff > 0.5 or yaw_diff > np.deg2rad(70.0):
                continue

            cost = xy_dist + 0.75 * size_diff + 0.5 * (1.0 - iou)
            if cost < best_cost:
                best_cost = cost
                best_track = track

        if best_track is None:
            return None

        self.archived_tracks = [track for track in self.archived_tracks if track.id != best_track.id]
        best_track.state = 'confirmed'
        best_track.time_since_update = 0
        best_track.update(det, timestamp)
        return best_track

    def _match_cost(self, track, det):
        det_cls = int(det[7])
        if track.cls_id > 0 and det_cls > 0 and track.cls_id != det_cls:
            return np.inf

        ego_speed = 0.0
        ego_yaw_rate = 0.0
        if hasattr(self, 'current_ego_motion') and self.current_ego_motion is not None:
            ego_speed = abs(float(self.current_ego_motion.get('speed', 0.0)))
            ego_yaw_rate = abs(float(self.current_ego_motion.get('yaw_rate', 0.0)))

        speed = float(np.linalg.norm(track.x[2:4, 0]))
        xy_dist = np.linalg.norm(track.x[:2, 0] - det[:2])
        base_dist_threshold = self.dist_threshold + 0.35 * ego_speed + 1.5 * ego_yaw_rate
        if xy_dist > base_dist_threshold:
            return np.inf

        z_diff = abs(float(track.z - det[2]))
        yaw_diff = abs(wrap_angle(float(track.yaw - det[6])))
        size_diff = np.linalg.norm(track.dim - det[3:6]) / max(np.linalg.norm(track.dim), 1e-3)
        anchor_dist = np.linalg.norm(track.stationary_anchor[:2] - det[:2])
        iou = bev_iou(
            [track.x[0, 0], track.x[1, 0], track.dim[0], track.dim[1], track.yaw],
            [det[0], det[1], det[3], det[4], det[6]],
        )

        if z_diff > 2.0 or yaw_diff > np.pi * 0.75:
            return np.inf
        if xy_dist < self.dist_threshold * 0.5 and iou < 0.01:
            return np.inf
        if size_diff > 0.45:
            return np.inf

        static_track = track.is_confirmed and speed < track.min_speed_for_yaw
        if static_track:
            static_xy_gate = min(
                base_dist_threshold,
                max(0.8, 0.35 * np.linalg.norm(track.dim[:2])) + 0.3 * ego_speed + 1.25 * ego_yaw_rate
            )
            if xy_dist > static_xy_gate or anchor_dist > max(1.0, static_xy_gate + 0.5 * ego_yaw_rate):
                return np.inf
            if iou < track.min_iou_for_static_match:
                return np.inf
            if yaw_diff > np.deg2rad(50.0):
                return np.inf

        return (
            self.cost_weights['xy'] * xy_dist
            + self.cost_weights['z'] * z_diff
            + self.cost_weights['yaw'] * yaw_diff
            + self.cost_weights['size'] * size_diff
            + self.cost_weights['iou'] * (1.0 - iou)
            + (1.25 * anchor_dist if static_track else 0.25 * anchor_dist)
        )

    def update(self, dets, timestamp=None, ego_motion=None):
        dets = np.asarray(dets, dtype=np.float32)
        if dets.size == 0:
            dets = np.empty((0, 9), dtype=np.float32)
        self.current_ego_motion = ego_motion or {'speed': 0.0, 'yaw_rate': 0.0}

        for track in self.tracks:
            track.predict(timestamp)

        matched = []
        unmatched_dets = list(range(len(dets)))
        unmatched_trks = list(range(len(self.tracks)))

        if self.tracks and len(dets) > 0:
            cost_matrix = np.full((len(self.tracks), len(dets)), np.inf, dtype=np.float32)
            for i, track in enumerate(self.tracks):
                for j, det in enumerate(dets):
                    cost_matrix[i, j] = self._match_cost(track, det)

            finite_mask = np.isfinite(cost_matrix)
            if finite_mask.any():
                safe_cost = cost_matrix.copy()
                safe_cost[~finite_mask] = 1e6
                row_ind, col_ind = linear_sum_assignment(safe_cost)

                used_rows, used_cols = set(), set()
                for r, c in zip(row_ind, col_ind):
                    if not np.isfinite(cost_matrix[r, c]):
                        continue
                    matched.append((r, c))
                    used_rows.add(r)
                    used_cols.add(c)

                unmatched_trks = [i for i in range(len(self.tracks)) if i not in used_rows]
                unmatched_dets = [i for i in range(len(dets)) if i not in used_cols]

        for t_idx, d_idx in matched:
            self.tracks[t_idx].update(dets[d_idx], timestamp)

        for t_idx in unmatched_trks:
            self.tracks[t_idx].mark_missed(self.max_age)

        for d_idx in unmatched_dets:
            det = dets[d_idx]
            if float(det[8]) < self.score_threshold:
                continue
            if not self._should_spawn_track(det):
                continue
            recovered_track = self._reassociate_archived_track(det, timestamp)
            if recovered_track is not None:
                self.tracks.append(recovered_track)
                continue
            self.tracks.append(Track(det, timestamp, n_init=self.min_hits))

        self._archive_deleted_tracks()
        self._deduplicate_tracks()

        ret = []
        for track in self.tracks:
            recently_seen_tentative = (not track.is_confirmed) and track.hits >= max(1, self.min_hits - 1) and track.time_since_update <= 1
            if not track.is_confirmed and not recently_seen_tentative:
                continue
            ret.append([
                track.x[0, 0],
                track.x[1, 0],
                track.z,
                track.id,
                track.yaw,
                track.cls_id,
                track.dim[0],
                track.dim[1],
                track.dim[2],
                track.time_since_update,
            ])
        return ret
