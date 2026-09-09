import cv2
import numpy as np
import pytest

from mando_tools.black_box_color_bits import BlackBoxColorDetector
from mando_tools.black_box_color_bits import DetectorConfig


@pytest.mark.parametrize('colors, expected', [
    ([(0, 0, 230)], [0]),
    ([(0, 230, 0), (0, 0, 230)], [1, 0]),
    ([(0, 0, 230), (0, 230, 0), (0, 0, 230)], [0, 1, 0]),
])
def test_filled_square_displays_with_default_config(colors, expected):
    frame = np.full((400, 640, 3), 180, dtype=np.uint8)
    for index, color in enumerate(colors):
        left = 80 + index * 120
        cv2.rectangle(frame, (left, 40), (left + 60, 100), (20, 20, 20), -1)
        cv2.rectangle(frame, (left + 22, 62), (left + 38, 78), color, -1)
    bits, observations, _ = BlackBoxColorDetector().process(frame, now_ns=0)
    assert bits == expected
    assert len(observations) == len(colors)


def test_black_border_without_black_fill_is_rejected():
    frame = np.full((400, 640, 3), 180, dtype=np.uint8)
    cv2.rectangle(frame, (80, 40), (140, 100), (20, 20, 20), -1)
    cv2.rectangle(frame, (87, 47), (133, 93), (0, 230, 0), -1)
    bits, observations, _ = BlackBoxColorDetector().process(frame, now_ns=0)
    assert bits == []
    assert observations == []


def test_unlit_display_never_initializes_as_red():
    frame = np.full((400, 640, 3), 180, dtype=np.uint8)
    cv2.rectangle(frame, (80, 40), (140, 100), (20, 20, 20), -1)
    bits, observations, _ = BlackBoxColorDetector().process(frame, now_ns=0)
    assert bits == []
    assert len(observations) == 1
    assert observations[0].stable_bit is None


def make_frame(colors, height=240, width=640):
    frame = np.full((height, width, 3), (145, 145, 145), dtype=np.uint8)
    for x, color in colors:
        cv2.rectangle(frame, (x, 35), (x + 64, 65), (20, 20, 20), -1)
        cv2.rectangle(frame, (x + 14, 42), (x + 50, 58), color, -1)
    return frame


def detector(**overrides):
    values = {
        'roi_bottom_ratio': 0.50,
        'max_aspect_ratio': 3.0,
        'min_box_width_px': 20,
        'min_box_height_px': 10,
        'min_box_area_px': 100,
        'max_box_width_ratio': 0.25,
        'max_box_height_ratio': 0.50,
        'black_v_max': 85,
        'morphology_open_iterations': 0,
        'color_score_threshold': 0.04,
        'color_hysteresis_delta': 0.08,
        'color_ema_alpha': 0.45,
        'hold_timeout_s': 0.5,
    }
    values.update(overrides)
    return BlackBoxColorDetector(DetectorConfig(**values))


def test_detects_left_to_right_red_green_red_bits():
    node = detector()
    frame = make_frame([
        (80, (0, 0, 220)),
        (180, (0, 220, 0)),
        (280, (0, 0, 220)),
    ])

    bits, observations, _ = node.process(frame, now_ns=0)

    assert bits == [0, 1, 0]
    assert [observation.bbox[0] for observation in observations] == [80, 180, 280]


def test_ignores_black_box_outside_upper_roi():
    node = detector()
    frame = make_frame([(80, (0, 220, 0))])
    cv2.rectangle(frame, (300, 180), (364, 210), (20, 20, 20), -1)
    cv2.rectangle(frame, (314, 187), (350, 203), (0, 0, 220), -1)

    bits, observations, _ = node.process(frame, now_ns=0)

    assert bits == [1]
    assert len(observations) == 1


def test_rejects_oversized_dark_window_like_region():
    node = detector()
    frame = make_frame([])
    cv2.rectangle(frame, (80, 35), (280, 95), (20, 20, 20), -1)
    cv2.rectangle(frame, (125, 52), (235, 78), (0, 220, 0), -1)

    bits, observations, _ = node.process(frame, now_ns=0)

    assert bits == []
    assert observations == []


def test_red_hue_wraparound_is_classified_as_zero():
    node = detector()
    frame = make_frame([(80, (10, 0, 150))])

    bits, _, _ = node.process(frame, now_ns=0)

    assert bits == [0]


def test_unknown_color_keeps_previous_bit_then_clears_after_timeout():
    node = detector()
    first_frame = make_frame([(80, (0, 220, 0))])
    unknown_frame = make_frame([(80, (105, 105, 105))])
    empty_frame = np.full_like(first_frame, (145, 145, 145))

    first_bits, _, _ = node.process(first_frame, now_ns=0)
    held_bits, _, _ = node.process(unknown_frame, now_ns=200_000_000)
    cleared_bits, _, _ = node.process(empty_frame, now_ns=800_000_000)

    assert first_bits == [1]
    assert held_bits == [1]
    assert cleared_bits == []


def test_ema_hysteresis_rejects_one_frame_color_flip():
    node = detector(color_ema_alpha=0.30, color_hysteresis_delta=0.20)
    green_frame = make_frame([(80, (0, 220, 0))])
    red_frame = make_frame([(80, (0, 0, 220))])

    assert node.process(green_frame, now_ns=0)[0] == [1]
    assert node.process(red_frame, now_ns=100_000_000)[0] == [1]
