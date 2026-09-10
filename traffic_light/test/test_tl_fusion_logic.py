from collections import deque
from dataclasses import replace
from pathlib import Path
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mando_tools import tl_fusion
from mando_tools.tl_fusion import DetectionCandidate
from mando_tools.tl_fusion import STATE_GREEN
from mando_tools.tl_fusion import STATE_LEFT_ARROW
from mando_tools.tl_fusion import STATE_RED
from mando_tools.tl_fusion import STATE_UNKNOWN
from mando_tools.tl_fusion import TLFusionNode

import numpy as np
import torch
from sensor_msgs.msg import Image
from ultralytics.engine.results import Boxes


class RecordingPublisher:
    def __init__(self) -> None:
        self.messages = []
        self.subscription_count = 0
        self.subscription_queries = 0

    def publish(self, message) -> None:
        self.messages.append(message)

    def get_subscription_count(self) -> int:
        self.subscription_queries += 1
        return self.subscription_count


class RecordingLogger:
    def __init__(self) -> None:
        self.errors = []

    def error(self, message: str) -> None:
        self.errors.append(message)


@pytest.fixture
def fusion_node(monkeypatch):
    model = Mock()
    model.names = {
        0: 'vehicular_red',
        1: 'vehicular_green',
        2: 'vehicular_green_arrow(down)',
        3: 'traffic light',
    }
    monkeypatch.setattr(tl_fusion.Node, '__init__', lambda self, name: None)
    monkeypatch.setattr(tl_fusion, 'YOLO', Mock(return_value=model))
    monkeypatch.setattr(tl_fusion, 'resolve_inference_device', lambda device: 'cpu')
    overrides = {
        'model_path': str(Path(__file__)),
        'detect_left_ratio': 0.0, 'detect_right_ratio': 1.0,
        'detect_top_ratio': 0.0, 'detect_bottom_ratio': 1.0,
    }
    monkeypatch.setattr(
        TLFusionNode, '_declare_param',
        lambda self, name, default: overrides.get(name, default),
    )
    monkeypatch.setattr(TLFusionNode, '_now_ns', lambda self: 1_000_000_000)
    monkeypatch.setattr(TLFusionNode, 'get_logger', lambda self: Mock())
    monkeypatch.setattr(TLFusionNode, 'create_subscription', Mock())
    monkeypatch.setattr(TLFusionNode, 'create_timer', Mock())
    monkeypatch.setattr(
        TLFusionNode, 'create_publisher',
        lambda self, *args: RecordingPublisher(),
    )
    node = TLFusionNode()
    node._log_processing_error = Mock(side_effect=AssertionError('Frame processing failed'))
    return node


def make_candidate(confidence=0.4, state=STATE_UNKNOWN, class_name='traffic light'):
    return DetectionCandidate(
        box=(50, 40, 100, 90),
        conf=confidence,
        class_id=3,
        class_name=class_name,
        model_state=state,
        model_resolved=state != STATE_UNKNOWN,
    )


def color_frame(red_pixels, yellow_pixels, green_pixels):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    pixels = frame.reshape(-1, 3)
    pixels[:red_pixels] = (0, 0, 255)
    pixels[red_pixels:red_pixels + yellow_pixels] = (0, 255, 255)
    pixels[red_pixels + yellow_pixels:red_pixels + yellow_pixels + green_pixels] = (0, 255, 0)
    return frame


@pytest.fixture
def color_node(fusion_node):
    fusion_node._expanded_crop = lambda frame, box: frame
    fusion_node._enhance_crop = lambda crop: crop
    fusion_node._clean_mask = lambda mask: mask
    fusion_node._largest_component = Mock(wraps=fusion_node._largest_component)
    return fusion_node


def test_model_class_state_mapping() -> None:
    node = object.__new__(TLFusionNode)

    assert node._state_from_class_name('vehicular_red') == (STATE_RED, True)
    assert node._state_from_class_name('vehicular_green') == (STATE_GREEN, True)
    assert node._state_from_class_name('vehicular_green_arrow') == (
        STATE_LEFT_ARROW,
        True,
    )
    assert node._state_from_class_name('vehicular_red_and_green_arrow') == (
        STATE_LEFT_ARROW,
        True,
    )
    assert node._state_from_class_name('vehicular_green_arrow(down)') == (
        STATE_UNKNOWN,
        False,
    )


def test_image_timeout_publishes_unknown_and_invalid() -> None:
    node = object.__new__(TLFusionNode)
    node.input_timeout_s = 3.0
    node.last_image_received_monotonic_ns = time.monotonic_ns() - 4_000_000_000
    node.input_timeout_active = False
    node.current_state = STATE_GREEN
    node.current_source = 'model'
    node.current_reason = 'vehicular_green'
    node.state_history = deque([STATE_GREEN], maxlen=5)
    node.last_candidate_box = (1, 2, 3, 4)
    node.last_overlay_candidate = object()
    node.image_topic = '/panorama/image_raw'
    node.state_topic = '/tl/state_id'
    node.input_valid_pub = RecordingPublisher()
    node.state_pub = RecordingPublisher()
    node.state_label_pub = RecordingPublisher()
    node.state_reason_pub = RecordingPublisher()
    logger = RecordingLogger()
    node.get_logger = lambda: logger

    node._publish_timeout_if_needed()

    assert node.current_state == STATE_UNKNOWN
    assert node.input_valid_pub.messages[-1].data is False
    assert node.state_pub.messages[-1].data == STATE_UNKNOWN
    assert 'input_timeout' in node.state_reason_pub.messages[-1].data
    assert len(logger.errors) == 1


@pytest.mark.parametrize('show_windows', [False, True])
@pytest.mark.parametrize('subscribers', [0, 1])
@pytest.mark.parametrize('candidate_kind', ['absent', 'high_confidence', 'fallback'])
def test_frame_debug_demand(fusion_node, monkeypatch, show_windows, subscribers, candidate_kind):
    node = fusion_node
    node.show_windows = show_windows
    node.debug_pub.subscription_count = subscribers
    node.last_state_change_ns = 0
    frame = np.full((160, 320, 3), (0, 0, 255), dtype=np.uint8)
    candidate = make_candidate()
    if candidate_kind == 'high_confidence':
        candidate = make_candidate(0.9, STATE_RED, 'vehicular_red')
    detections = [] if candidate_kind == 'absent' else [candidate]
    node._detect_candidates = Mock(return_value=(detections, (0, 0, 320, 160)))
    node.bridge = SimpleNamespace(imgmsg_to_cv2=Mock(return_value=frame))
    for method_name in (
        '_highlight_masks', '_build_debug_image', '_numpy_to_image_msg',
        '_analyze_selected_candidate', '_draw_debug_inset',
    ):
        setattr(node, method_name, Mock(wraps=getattr(node, method_name)))
    show_image = Mock()
    wait_key = Mock()
    monkeypatch.setattr(tl_fusion.cv2, 'imshow', show_image)
    monkeypatch.setattr(tl_fusion.cv2, 'waitKey', wait_key)
    message = Image()
    message.header.stamp.sec = 1
    node.latest_msg = message

    node._process_latest_frame()

    render_debug = show_windows or subscribers > 0
    has_color_analysis = candidate_kind == 'fallback'
    assert node.debug_pub.subscription_queries == 1
    assert node._build_debug_image.call_count == int(render_debug)
    assert node._highlight_masks.call_count == int(render_debug and has_color_analysis)
    assert node._draw_debug_inset.call_count == int(render_debug and has_color_analysis)
    assert node._analyze_selected_candidate.call_count == int(candidate_kind != 'high_confidence')
    assert node._numpy_to_image_msg.call_count == int(subscribers > 0)
    assert len(node.debug_pub.messages) == int(subscribers > 0)
    assert show_image.call_count == int(show_windows)
    assert wait_key.call_count == int(show_windows)
    if show_windows and subscribers:
        assert show_image.call_args.args[1] is node._numpy_to_image_msg.call_args.args[0]
    if subscribers:
        assert node.debug_pub.messages[0].header == message.header
    expected_state = STATE_UNKNOWN if candidate_kind == 'absent' else STATE_RED
    assert [message.data for message in node.state_pub.messages] == [expected_state]
    assert [message.data for message in node.input_valid_pub.messages] == [True]
    assert len(node.state_label_pub.messages) == len(node.state_reason_pub.messages) == 1
    expected_reasons = {
        'absent': 'UNKNOWN none no_candidate',
        'high_confidence': 'RED model vehicular_red:0.90',
        'fallback': 'RED color_fallback color_red:score=1.00',
    }
    assert node.state_reason_pub.messages[0].data == expected_reasons[candidate_kind]
    assert node.latest_msg is None
    assert node.processing is False
    assert node.processed_frames == 1
    node._log_processing_error.assert_not_called()


def test_low_confidence_fast_mode_skips_color_analysis(fusion_node):
    node = fusion_node
    node.enable_low_confidence_color_fallback = False
    node.show_windows = False
    node.debug_pub.subscription_count = 0
    node.last_state_change_ns = 0
    frame = np.zeros((160, 320, 3), dtype=np.uint8)
    candidate = make_candidate(0.4, STATE_RED, 'vehicular_red')
    node._detect_candidates = Mock(return_value=([candidate], (0, 0, 320, 160)))
    node._analyze_selected_candidate = Mock(
        side_effect=AssertionError('low-confidence fast mode ran color analysis')
    )
    node.bridge = SimpleNamespace(imgmsg_to_cv2=Mock(return_value=frame))
    node._publish_outputs = Mock()
    message = Image()
    message.header.stamp.sec = 1
    node.latest_msg = message

    node._process_latest_frame()

    node._analyze_selected_candidate.assert_not_called()
    assert node._publish_outputs.call_args.args[3].source == 'model_low_conf'
    node._log_processing_error.assert_not_called()


@pytest.mark.parametrize('render_debug', [False, True])
def test_absent_candidate_has_no_placeholder_image(fusion_node, monkeypatch, render_debug):
    draw_text = Mock()
    monkeypatch.setattr(tl_fusion.cv2, 'putText', draw_text)
    analysis = fusion_node._analyze_selected_candidate(
        np.zeros((10, 10, 3), dtype=np.uint8), None, render_debug=render_debug,
    )
    assert analysis == fusion_node._empty_analysis('no_candidate')
    assert analysis.highlighted is None
    draw_text.assert_not_called()


@pytest.mark.parametrize('rows, expected_states', [
    ([], []),
    ([[15.9, 12.9, 40.9, 45.9, 0.8, 0]], [STATE_RED]),
    ([[15, 12, 40, 45, 0.4, 0], [15, 12, 40, 45, 0.8, 1]], [STATE_RED, STATE_GREEN]),
    ([[15, 12, 40, 45, 0.8, 0], [15, 12, 40, 45, 0.8, 1]], [STATE_RED, STATE_GREEN]),
    ([[2, 12, 40, 45, 0.8, 0], [15, 12, 198, 45, 0.8, 1]], []),
    ([[15, 2, 40, 45, 0.8, 0], [15, 12, 40, 98, 0.8, 1]], []),
    ([[15, 12, 19, 45, 0.8, 0], [15, 12, 20, 17, 0.8, 1]], []),
    ([[15, 12, 15, 45, 0.8, 0], [15, 12, 40, 12, 0.8, 1]], []),
    ([[15, 12, 40, 45, 0.9, 2], [15, 12, 40, 45, 0.4, 1]], [STATE_GREEN]),
])
@pytest.mark.parametrize('storage', ['torch', 'numpy'])
def test_detection_bulk_transfer_preserves_filtering(fusion_node, rows, expected_states, storage):
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    data = np.asarray(rows, dtype=np.float32).reshape(-1, 6)
    if storage == 'torch':
        data = torch.from_numpy(data)
    cpu_boxes = Boxes(data, frame.shape[:2])
    device_boxes = Mock(spec=['cpu'])
    device_boxes.cpu.return_value = cpu_boxes
    fusion_node.model.predict.return_value = [SimpleNamespace(boxes=device_boxes)]

    detections, window = fusion_node._detect_candidates(frame)

    device_boxes.cpu.assert_called_once_with()
    assert window == (0, 0, 200, 100)
    assert [candidate.model_state for candidate in detections] == expected_states
    assert all(candidate.box == (15, 12, 40, 45) for candidate in detections)
    arguments = fusion_node.model.predict.call_args.kwargs
    assert arguments['imgsz'] == 640
    assert arguments['conf'] == 0.10
    assert arguments['iou'] == 0.45
    assert arguments['max_det'] == 50
    selected = fusion_node._select_candidate(detections, frame.shape)
    if detections:
        assert selected is max(detections, key=lambda candidate: candidate.conf)
    else:
        assert selected is None


def test_detection_without_boxes(fusion_node):
    fusion_node.model.predict.return_value = [SimpleNamespace(boxes=None)]
    assert fusion_node._detect_candidates(np.zeros((100, 200, 3), dtype=np.uint8)) == (
        [], (0, 0, 200, 100),
    )


@pytest.mark.parametrize('width,height,expected', [(1254, 370, (123, 627)), (1878, 555, (185, 939))])
def test_default_center_roi_reaches_model(fusion_node, monkeypatch, width, height, expected):
    monkeypatch.setattr(TLFusionNode, '_declare_param', lambda self, name, default: str(Path(__file__)) if name == 'model_path' else default)
    node = TLFusionNode()
    node.model.predict.return_value = [SimpleNamespace(boxes=None)]
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    _, bounds = node._detect_candidates(frame)
    source = node.model.predict.call_args.kwargs['source']
    assert source.shape[:2] == expected
    assert np.shares_memory(frame, source)
    assert bounds == (int(width * .25), 0, int(width * .75), int(height / 3))


@pytest.mark.parametrize('width,height', [(1254, 370), (1878, 555), (40, 30)])
def test_cropped_debug_and_inset_fit(fusion_node, width, height):
    node = fusion_node
    node.detect_left_ratio, node.detect_right_ratio = .25, .75
    node.detect_bottom_ratio = 1.0 / 3.0
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    original = frame.copy()
    left = int(width * .25)
    overlay = tl_fusion.OverlayCandidate((left+5, 5, left+25, 30), 'RED', (0, 0, 255), True)
    analysis = replace(node._empty_analysis('test'), highlighted=np.ones((100, 100, 3), dtype=np.uint8))
    debug = node._build_debug_image(frame, [overlay], make_candidate(), analysis)
    assert debug.shape == (int(height / 3), int(width*.75)-left, 3)
    np.testing.assert_array_equal(frame, original)
    assert not np.shares_memory(frame, debug)
    if height > 30:
        assert debug[25, 5].tolist() == [0, 0, 255]
    assert overlay.box == (left+5, 5, left+25, 30)


def test_fallback_expansion_stays_inside_detection_roi(fusion_node):
    node = fusion_node
    node.detect_left_ratio, node.detect_right_ratio = .25, .75
    node.detect_bottom_ratio = 1.0 / 3.0
    node.fallback_min_margin_px = 1000
    frame = np.zeros((370, 1254, 3), dtype=np.uint8)
    frame[:123, 313:940] = 80
    crop = node._expanded_crop(frame, (320, 10, 350, 40))
    assert crop.shape == (123, 627, 3)
    assert np.all(crop == 80)


def test_detection_preserves_crop_offsets(fusion_node):
    fusion_node.detect_left_ratio = 0.25
    fusion_node.detect_right_ratio = 0.75
    fusion_node.detect_top_ratio = 0.20
    fusion_node.detect_bottom_ratio = 0.80
    boxes = Boxes(np.array([[5.9, 5.9, 25.9, 25.9, 0.8, 0]], dtype=np.float32), (60, 100))
    fusion_node.model.predict.return_value = [SimpleNamespace(boxes=boxes)]
    detections, window = fusion_node._detect_candidates(np.zeros((100, 200, 3), dtype=np.uint8))
    assert window == (50, 20, 150, 80)
    assert detections[0].box == (55, 25, 75, 45)


@pytest.mark.parametrize('red_pixels, yellow_pixels, green_pixels, expected_state', [
    (0, 0, 0, STATE_UNKNOWN),
    (11, 0, 0, STATE_UNKNOWN),
    (12, 0, 0, STATE_RED),
    (70, 0, 30, STATE_LEFT_ARROW),
    (30, 0, 70, STATE_LEFT_ARROW),
    (29, 0, 71, STATE_GREEN),
    (82, 0, 18, STATE_LEFT_ARROW),
    (83, 0, 17, STATE_RED),
    (58, 12, 30, STATE_LEFT_ARROW),
    (57, 13, 30, STATE_RED),
])
def test_color_analysis_preserves_results_with_debug(
    color_node, red_pixels, yellow_pixels, green_pixels, expected_state,
):
    frame = color_frame(red_pixels, yellow_pixels, green_pixels)
    analysis = color_node._analyze_selected_candidate(frame, make_candidate())
    if expected_state == STATE_LEFT_ARROW:
        color_node._largest_component.assert_not_called()
    elif expected_state != STATE_UNKNOWN:
        color_node._largest_component.assert_called_once()
    rendered_analysis = color_node._analyze_selected_candidate(
        frame, make_candidate(), render_debug=True,
    )
    assert analysis.state == expected_state
    assert analysis.decisive == (expected_state != STATE_UNKNOWN)
    assert analysis.highlighted is None
    assert rendered_analysis.highlighted.shape == frame.shape
    assert replace(rendered_analysis, highlighted=None) == analysis


@pytest.mark.parametrize('parameter', [
    'fallback_score_threshold', 'fallback_score_gap',
    'fallback_min_valid_pixels', 'fallback_min_component_pixels',
])
@pytest.mark.parametrize('direction', [-1, 0, 1])
def test_color_threshold_boundaries(color_node, parameter, direction):
    frame = color_frame(60, 40, 0)
    baseline = color_node._analyze_selected_candidate(frame, make_candidate())
    boundaries = {
        'fallback_score_threshold': baseline.top_score,
        'fallback_score_gap': baseline.score_gap,
        'fallback_min_valid_pixels': 100,
        'fallback_min_component_pixels': 60,
    }
    boundary = boundaries[parameter]
    if isinstance(boundary, int):
        threshold = boundary + direction
    else:
        threshold = boundary if direction == 0 else np.nextafter(boundary, direction * np.inf)
    setattr(color_node, parameter, threshold)
    analysis = color_node._analyze_selected_candidate(frame, make_candidate())
    assert analysis.state == (STATE_RED if direction <= 0 else STATE_UNKNOWN)
    assert analysis.scores == baseline.scores
    assert analysis.valid_pixels == baseline.valid_pixels


def test_fixed_time_state_transitions_and_missing_candidates(fusion_node):
    sequence = [
        (0, STATE_RED, STATE_UNKNOWN),
        (249, STATE_RED, STATE_UNKNOWN),
        (250, STATE_RED, STATE_RED),
        (300, STATE_GREEN, STATE_RED),
        (400, STATE_GREEN, STATE_RED),
        (500, STATE_GREEN, STATE_GREEN),
        (600, None, STATE_GREEN),
        (900, None, STATE_GREEN),
        (1000, None, STATE_GREEN),
        (1100, None, STATE_UNKNOWN),
        (1700, None, STATE_UNKNOWN),
    ]
    for elapsed_ms, state, expected_state in sequence:
        timestamp_ns = 1_000_000_000 + elapsed_ms * 1_000_000
        fusion_node._now_ns = lambda: timestamp_ns
        candidate = None
        if state is not None:
            label = tl_fusion.STATE_LABELS[state].lower()
            candidate = make_candidate(0.9, state, f'vehicular_{label}')
            fusion_node.last_seen_candidate_ns = timestamp_ns
            fusion_node.last_candidate_box = candidate.box
        decision = fusion_node._decide_state(candidate, fusion_node._empty_analysis('test'))
        stable_state = fusion_node._update_stable_state(decision.proposed_state, candidate is not None)
        assert stable_state == expected_state
        assert decision.source == ('none' if candidate is None else 'model')
        expected_reason = 'no_candidate' if candidate is None else f'{candidate.class_name}:0.90'
        assert decision.reason == expected_reason
    assert fusion_node.last_candidate_box is None
