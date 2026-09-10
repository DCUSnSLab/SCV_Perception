from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from mando_tools.black_box_color_bits import DetectorConfig
from mando_tools.yolo_box_color_bits import YoloBoxColorDetector


def fake_model(rows):
    data = np.asarray(rows, dtype=np.float32).reshape(-1, 6)
    tensor = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: data))
    model = Mock()
    model.names = {0: 'green_sign', 1: 'red_sign', 2: 'truck'}
    model.task = 'detect'
    model.predict.return_value = [SimpleNamespace(boxes=SimpleNamespace(data=tensor))]
    return model


def test_yolo_boxes_use_hsv_not_class_and_restore_coordinates():
    model = fake_model([
        [120, 20, 150, 50, .9, 0],
        [20, 20, 50, 50, .9, 1],
        [70, 20, 100, 50, .9, 2],
    ])
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    frame[20:50, 120:150] = (0, 230, 0)
    frame[20:50, 220:250] = (0, 0, 230)
    detector = YoloBoxColorDetector(DetectorConfig(), '', model=model)
    bits, observations, bounds = detector.process(frame, now_ns=1)
    assert bits == [1, 0]
    assert [item.bbox for item in observations] == [(120, 20, 150, 50), (220, 20, 250, 50)]
    assert bounds == (100, 0, 300, 100)
    assert model.predict.call_args.kwargs['source'].shape == (100, 200, 3)
    assert model.predict.call_args.kwargs['classes'] == [0, 1]


def test_duplicate_invalid_low_confidence_boxes_are_filtered():
    model = fake_model([
        [20, 20, 50, 50, .9, 0], [20, 20, 50, 50, .8, 1],
        [70, 20, 100, 50, .1, 0], [80, 20, 70, 50, .9, 0],
        [float('nan'), 20, 50, 50, .9, 0],
    ])
    detector = YoloBoxColorDetector(DetectorConfig(), '', model=model)
    bits, observations, _ = detector.process(np.zeros((200, 400, 3), dtype=np.uint8), now_ns=1)
    assert bits == []
    assert len(observations) == 1


def test_unknown_color_and_missing_boxes_hold_then_expire():
    model = fake_model([[20, 20, 50, 50, .9, 0]])
    detector = YoloBoxColorDetector(DetectorConfig(), '', model=model)
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    frame[20:50, 120:150] = (0, 230, 0)
    assert detector.process(frame, now_ns=1)[0] == [1]
    frame[20:50, 120:150] = (0, 230, 230)
    assert detector.process(frame, now_ns=100_000_001)[0] == [1]
    model.predict.return_value = [SimpleNamespace(boxes=None)]
    assert detector.process(frame, now_ns=200_000_001)[0] == [1]
    assert detector.process(frame, now_ns=700_000_001)[0] == []


def test_model_without_supported_classes_fails():
    model = fake_model([])
    model.names = {0: 'truck'}
    with pytest.raises(ValueError, match='green_sign'):
        YoloBoxColorDetector(DetectorConfig(), '', model=model)


def test_clipped_small_box_and_red_hue_wrap():
    model = fake_model([[-5, -5, 2, 2, .9, 1]])
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    frame[:2, 100:102] = (10, 0, 150)
    detector = YoloBoxColorDetector(DetectorConfig(), '', model=model)
    bits, observations, _ = detector.process(frame, now_ns=1)
    assert bits == [0]
    assert observations[0].bbox == (100, 0, 102, 2)
