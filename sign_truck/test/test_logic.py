import numpy as np

from sign_truck.detector import COLORS, SignTruckDetector
from sign_truck.logic import GREEN, RED, UNKNOWN, assign_lane_states, state_at_anchor


def test_lane_assignment_and_auto_anchor():
    detections = [(35.0, 'red_sign', 0.9), (52.0, 'green_sign', 0.8), (68.0, 'red_sign', 0.95)]
    states = assign_lane_states(detections, {'left': 35.0, 'right': 68.0}, 12.0)

    assert states == {'left': RED, 'right': RED}
    assert state_at_anchor(detections, 50.0, 12.0) == GREEN
    assert state_at_anchor([], 50.0, 12.0) == UNKNOWN


def test_visualization_keeps_roi_size():
    roi = np.zeros((160, 640, 3), dtype=np.uint8)
    result = SignTruckDetector.draw_visualization(
        None,
        roi,
        [(10, 10, 20, 20, 'green_sign', 0.9), (30, 30, 40, 40, 'red_sign', 0.8)],
    )

    assert result.shape == roi.shape
    assert tuple(result[10, 10]) == COLORS[GREEN]
    assert tuple(result[30, 30]) == COLORS[RED]
