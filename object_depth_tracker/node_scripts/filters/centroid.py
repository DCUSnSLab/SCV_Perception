# object_depth_tracker/filters/centroid.py
import itertools

class Track:
    _ids = itertools.count()
    def __init__(self, xyz, cls_id):
        self.id     = next(self._ids)
        self.xyz    = xyz
        self.cls_id = cls_id

class Filter:
    def __init__(self):
        self.tracks = []

    def update(self, meas, stamp):
        """
        meas: list of (xyz, cls_id) this frame
        """
        self.tracks = [Track(x, c) for x, c in meas]   # 매 프레임 새로
        return self.tracks
