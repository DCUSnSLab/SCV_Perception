# object_depth_tracker/filters/kalman_6d.py
import numpy as np, itertools

class Track:
    _ids = itertools.count()
    def __init__(self, xyz, cls_id, stamp):
        self.id, self.cls_id = next(self._ids), cls_id
        self.x = np.hstack([xyz, [0,0,0]])             # [x y z vx vy vz]
        self.P = np.eye(6)*0.1
        self.last_t = stamp

class Filter:
    def __init__(self):


    def _predict(self, trk, dt):


    def _update(self, trk, z):


    def update(self, meas, stamp):

