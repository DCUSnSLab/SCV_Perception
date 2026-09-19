import numpy as np
from std_msgs.msg import Header

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


def test_observations_keep_image_stamp_coordinates_and_each_frame_count():
    header = Header(frame_id='panorama')
    header.stamp.sec = 12
    header.stamp.nanosec = 345
    detections = [(10, 20, 30, 40, 'green_sign', 0.9),
                  (50, 20, 70, 40, 'red_sign', 0.8),
                  (90, 20, 110, 40, 'red_sign', 0.7)]
    for count in (3, 2, 0):
        msg = SignTruckDetector.make_observations(header, detections[:count], 100)
        assert msg.header == header
        assert len(msg.detections) == count  # Never pad or reuse missing panels.
        for index, detection in enumerate(msg.detections):
            assert detection.header == header
            assert detection.bbox.center.position.x == 120 + index * 40
            assert detection.bbox.center.position.y == 30
            assert detection.bbox.size_x == detection.bbox.size_y == 20
            assert detection.results[0].hypothesis.class_id == ('green' if index == 0 else 'red')
            assert detection.results[0].hypothesis.score == detections[index][-1]
