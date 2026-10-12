"""Build the sensor inspection region from a detection or predicted track."""

import numpy as np


def expanded_box(box, margin_xy=0.0, margin_z=0.0):
    region = np.asarray(box[:7], dtype=np.float32).copy()
    region[3] = max(1e-3, region[3] + 2.0 * float(margin_xy))
    region[4] = max(1e-3, region[4] + 2.0 * float(margin_xy))
    region[5] = max(1e-3, region[5] + 2.0 * float(margin_z))
    return region
