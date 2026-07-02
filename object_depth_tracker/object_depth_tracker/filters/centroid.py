# object_depth_tracker/filters/centroid.py
import itertools

class Track:
    _ids = itertools.count()
    def __init__(self, xyz, cls_id, bbox_3d=None, class_name="unknown"):
        self.id     = next(self._ids)
        self.xyz    = xyz
        self.cls_id = cls_id
        self.class_name = class_name
        self.bbox_3d = bbox_3d if bbox_3d is not None else [1.0, 1.0, 1.0]

class Filter:
    def __init__(self):
        self.tracks = []

    def update(self, meas, stamp):
        """
        meas: list of (xyz, cls_id) this frame
        """
        tracks = []
        for measurement_data in meas:
            if len(measurement_data) == 2:
                x, c = measurement_data
                trk = Track(x, c)
            elif len(measurement_data) == 4:
                x, c, bbox_3d, class_name = measurement_data
                trk = Track(x, c, bbox_3d, class_name)
            else:  # 5 values
                x, c, bbox_3d, class_name, bbox_info = measurement_data
                trk = Track(x, c, bbox_3d, class_name)
                if bbox_info is not None and 'point_count' in bbox_info:
                    trk.point_count = bbox_info['point_count']
            tracks.append(trk)
        self.tracks = tracks
        return self.tracks