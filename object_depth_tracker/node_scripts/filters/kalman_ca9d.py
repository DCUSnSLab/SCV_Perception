# object_depth_tracker/filters/kalman_ca9d.py
import numpy as np
import itertools
from typing import Tuple
from scipy.optimize import linear_sum_assignment   # Hungarian

__all__ = ["Filter"]

# ────────────────────────────────────────────────────────────────
class Track:
    _ids = itertools.count()

    def __init__(self, xyz: np.ndarray, cls_id: int, stamp: float,
                 proc_var: float, meas_var: float):
        """
        상태벡터: [x y z  vx vy vz  ax ay az]  (9차)
        """
        self.id      = next(self._ids)
        self.cls_id  = cls_id

        self.x       = np.hstack([xyz, np.zeros(6)])
        self.P       = np.eye(9) * 0.1

        # 공분산 행렬 초기값
        self.Q_base  = np.eye(9) * proc_var
        self.R       = np.eye(3) * meas_var
        self.H       = np.hstack([np.eye(3), np.zeros((3,6))])

        self.last_t  = stamp

        # ── 상태관리 ─────────────────────
        self.hits    = 1         # 연속 매칭 횟수
        self.misses  = 0         # 연속 미매칭
        self.age     = 1         # 총 업데이트 수
        self.state   = "tentative"   # tentative | confirmed

    # ── 프로퍼티 ─────────────────────────
    @property
    def xyz(self) -> np.ndarray:
        return self.x[:3]

# ────────────────────────────────────────────────────────────────
class Filter:
    def __init__(self,
                 process_var: float = 1e-2,
                 meas_var:    float = 1e-1,
                 max_age:     float = 2.0,
                 miss_thresh: int   = 5,
                 conf_hits:   int   = 3,
                 gate_prob:   float = 0.995):
        self.tracks   = []

        # 공통 파라미터
        self.process_var = process_var
        self.meas_var    = meas_var
        self.max_age     = max_age
        self.miss_thresh = miss_thresh
        self.conf_hits   = conf_hits

        # χ²(3, 0.995) ≈ 12.8  (3 자유도)
        from scipy.stats import chi2
        self.gate_thresh = chi2.ppf(gate_prob, df=3)

    # ─────────────────────────────────────────────────────────
    # 내부 메서드
    def _build_F_Q(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """상수가속 모델용 F, Q 생성"""
        I3 = np.eye(3)
        O3 = np.zeros((3,3))

        # F
        F = np.block([
            [I3, dt*I3, 0.5*dt**2*I3],
            [O3,   I3 ,     dt*I3   ],
            [O3,   O3 ,      I3     ]
        ])

        # Q (σ_a² * G Gᵀ)  — 단순화: proc_var · I₉ 로도 충분
        Q = np.eye(9) * self.process_var
        return F, Q

    def _predict(self, trk: Track, dt: float):
        F, Q = self._build_F_Q(dt)
        trk.x = F @ trk.x
        trk.P = F @ trk.P @ F.T + Q

    def _update(self, trk: Track, z: np.ndarray):
        H = trk.H
        R = trk.R

        y  = z - (H @ trk.x)
        S  = H @ trk.P @ H.T + R
        K  = trk.P @ H.T @ np.linalg.inv(S)

        trk.x = trk.x + K @ y
        trk.P = (np.eye(9) - K @ H) @ trk.P

    # ─────────────────────────────────────────────────────────
    # 외부 호출 메서드
    def update(self, measurements: list, stamp: float):
        """
        measurements : List[(np.ndarray(3,), cls_id)]
        """
        # 0) 모든 트랙 예측
        for trk in self.tracks:
            dt = stamp - trk.last_t
            if dt > 0:
                self._predict(trk, dt)

        # 1) 비용행렬 (Mahalanobis²) 생성
        nT = len(self.tracks)
        nZ = len(measurements)
        cost = np.full((nT, nZ), np.inf)

        for i, trk in enumerate(self.tracks):
            H = trk.H
            S = H @ trk.P @ H.T + trk.R
            S_inv = np.linalg.inv(S)

            for j, (z, cls_id) in enumerate(measurements):
                if cls_id != trk.cls_id:
                    continue
                y   = z - (H @ trk.x)
                d2  = float(y.T @ S_inv @ y)
                if d2 <= self.gate_thresh:
                    cost[i, j] = d2                     # 게이트 통과한 항목만

        # 2) Hungarian 할당
        row_ind = np.array([], dtype=int)
        col_ind = np.array([], dtype=int)

        if nT and nZ and np.isfinite(cost).any():
            # (a) finite 값이 하나라도 있는 행·열만 선택
            valid_rows = np.isfinite(cost).any(axis=1)
            valid_cols = np.isfinite(cost).any(axis=0)

            if valid_rows.any() and valid_cols.any():
                cost_sub = cost[np.ix_(valid_rows, valid_cols)]
                r_sub, c_sub = linear_sum_assignment(cost_sub)

                # (b) 원래 인덱스로 복원
                row_ind = np.where(valid_rows)[0][r_sub]
                col_ind = np.where(valid_cols)[0][c_sub]

        assigned_t = set()
        assigned_z = set()
        matched    = []

        # 3) 매칭 처리
        for i, j in zip(row_ind, col_ind):
            if cost[i, j] == np.inf:    # 게이트 탈락
                continue
            trk = self.tracks[i]
            z, _ = measurements[j]

            self._update(trk, z)
            trk.last_t = stamp
            trk.hits  += 1
            trk.misses = 0
            trk.age   += 1
            if trk.state == "tentative" and trk.hits >= self.conf_hits:
                trk.state = "confirmed"
            matched.append(trk)

            assigned_t.add(i)
            assigned_z.add(j)

        # 4) 미매칭 트랙 → miss++
        for idx, trk in enumerate(self.tracks):
            if idx in assigned_t:
                continue
            trk.misses += 1
            trk.age    += 1
            # 삭제 조건
            if trk.misses > self.miss_thresh or (stamp - trk.last_t) > self.max_age:
                trk.state = "deleted"

        # 5) 미매칭 측정 → 신규 트랙
        for j, (z, cls_id) in enumerate(measurements):
            if j in assigned_z:
                continue
            new_trk = Track(z, cls_id, stamp,
                            proc_var=self.process_var,
                            meas_var=self.meas_var)
            self.tracks.append(new_trk)
            matched.append(new_trk)

        # 6) deleted 제거
        self.tracks = [trk for trk in self.tracks if trk.state != "deleted"]

        # confirmed + tentative 모두 반환 (RViz용)
        return matched
