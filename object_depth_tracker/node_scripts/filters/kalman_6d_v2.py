# object_depth_tracker/filters/kalman_6d.py
import numpy as np
import itertools

class Track:
    _ids = itertools.count()
    def __init__(self, xyz: np.ndarray, cls_id: int, stamp: float):
        self.id      = next(self._ids)
        self.cls_id  = cls_id
        # 상태 벡터: [x, y, z, vx, vy, vz]
        self.x       = np.hstack([xyz, [0., 0., 0.]])
        # 상태 추정 오차 공분산
        self.P       = np.eye(6) * 0.1
        # 마지막 실제 업데이트 시각 (초)
        self.last_t  = stamp

    @property
    def xyz(self) -> np.ndarray:
        """현재 위치 추정치 (x, y, z)"""
        return self.x[:3]

    @property
    def velocity(self) -> np.ndarray:
        """현재 속도 추정치 (vx, vy, vz)"""
        return self.x[3:]

class Filter:
    def __init__(self,
                 process_var: float = 1e-2,
                 meas_var:    float = 1e-1,
                 dist_thresh: float = 1.0,
                 max_age:     float = 2.0):
        self.tracks      = []
        self.Q           = np.eye(6) * process_var
        self.R           = np.eye(3) * meas_var
        self.H           = np.hstack([np.eye(3), np.zeros((3,3))])
        self.dist_thresh = dist_thresh
        self.max_age     = max_age

    def _predict(self, trk: Track, dt: float):
        F = np.eye(6)
        F[0,3] = dt; F[1,4] = dt; F[2,5] = dt
        trk.x = F @ trk.x
        trk.P = F @ trk.P @ F.T + self.Q

    def _update(self, trk: Track, z: np.ndarray):
        x_pred = trk.x
        P_pred = trk.P
        y      = z - (self.H @ x_pred)
        S      = self.H @ P_pred @ self.H.T + self.R
        K      = P_pred @ self.H.T @ np.linalg.inv(S)
        trk.x  = x_pred + (K @ y)
        trk.P  = (np.eye(6) - K @ self.H) @ P_pred

    def update(self, measurements: list, stamp: float):
        """
        measurements: List of (xyz: np.ndarray(shape=(3,)), cls_id: int)
        stamp: 현재 시간 (초)
        반환: 이번 프레임에서 실제 매칭(업데이트 또는 신규 생성)된 트랙 리스트
        """
        # 예측
        for trk in self.tracks:
            dt = stamp - trk.last_t
            if dt > 0:
                self._predict(trk, dt)

        matched = []
        unassigned = []

        # 연관 및 갱신
        for z, cls_id in measurements:
            best_trk, best_dist = None, float('inf')
            for trk in self.tracks:
                if trk.cls_id != cls_id:
                    continue
                dist = np.linalg.norm(z - trk.xyz)
                if dist < best_dist:
                    best_dist, best_trk = dist, trk

            if best_trk and best_dist <= self.dist_thresh:
                self._update(best_trk, z)
                best_trk.last_t = stamp
                matched.append(best_trk)
            else:
                unassigned.append((z, cls_id))

        # 신규 트랙 생성 및 반환 목록에 추가
        for z, cls_id in unassigned:
            trk = Track(z, cls_id, stamp)
            self.tracks.append(trk)
            matched.append(trk)

        # 오래된 트랙 제거
        self.tracks = [
            trk for trk in self.tracks
            if (stamp - trk.last_t) <= self.max_age
        ]
        return matched
