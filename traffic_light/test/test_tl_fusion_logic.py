from dataclasses import replace
from pathlib import Path
import time
from threading import Event, Thread
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
    fusion_node._expanded_crop_and_mask = lambda frame, box: (
        frame,
        np.full(frame.shape[:2], 255, dtype=np.uint8),
    )
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
    node.last_candidate_box = (1, 2, 3, 4)
    node.last_overlay_candidate = object()
    node.image_topic = '/panorama/image_raw'
    node.state_topic = '/tl/state_id'
    node.state_pub = RecordingPublisher()
    node.detection_pub = RecordingPublisher()
    node.get_clock = Mock()
    node.get_clock.return_value.now.return_value.to_msg.return_value = Image().header.stamp
    logger = RecordingLogger()
    node.get_logger = lambda: logger

    node._publish_timeout_if_needed()

    assert node.current_state == STATE_UNKNOWN
    assert node.state_pub.messages[-1].data == STATE_UNKNOWN
    assert node.detection_pub.messages[-1].detections == []
    assert len(logger.errors) == 1


@pytest.mark.parametrize('show_windows', [False, True])
@pytest.mark.parametrize('subscribers', [0, 1])
@pytest.mark.parametrize('publish_debug_image', [False, True])
@pytest.mark.parametrize('candidate_kind', ['absent', 'high_confidence', 'fallback'])
def test_frame_debug_demand(
    fusion_node,
    monkeypatch,
    show_windows,
    subscribers,
    publish_debug_image,
    candidate_kind,
):
    node = fusion_node
    node.state_confirm_ms = 0
    node.show_windows = show_windows
    node.publish_debug_image = publish_debug_image
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

    debug_requested = publish_debug_image and subscribers > 0
    render_debug = show_windows or debug_requested
    has_color_analysis = candidate_kind != 'absent'
    assert node.debug_pub.subscription_queries == int(publish_debug_image)
    assert node._build_debug_image.call_count == int(render_debug)
    assert node._highlight_masks.call_count == int(render_debug and has_color_analysis)
    assert node._draw_debug_inset.call_count == int(render_debug and has_color_analysis)
    assert node._analyze_selected_candidate.call_count == 1
    assert node._numpy_to_image_msg.call_count == int(debug_requested)
    assert len(node.debug_pub.messages) == int(debug_requested)
    assert show_image.call_count == int(show_windows)
    assert wait_key.call_count == int(show_windows)
    if show_windows and debug_requested:
        assert show_image.call_args.args[1] is node._numpy_to_image_msg.call_args.args[0]
    if debug_requested:
        assert node.debug_pub.messages[0].header == message.header
    expected_state = STATE_UNKNOWN if candidate_kind == 'absent' else STATE_RED
    assert [message.data for message in node.state_pub.messages] == [expected_state]
    assert len(node.detection_pub.messages) == 1
    detection_array = node.detection_pub.messages[0]
    assert detection_array.header == message.header
    assert len(detection_array.detections) == len(detections)
    if detections:
        result = detection_array.detections[0].results[0]
        assert result.hypothesis.class_id == detections[0].class_name
        assert result.hypothesis.score == detections[0].conf
    assert node.latest_msg is None
    assert node.processing is False
    assert node.processed_frames == 1
    node._log_processing_error.assert_not_called()


def test_color_analysis_remains_enabled_when_legacy_switch_is_false(fusion_node):
    node = fusion_node
    node.enable_low_confidence_color_fallback = False
    node.show_windows = False
    node.debug_pub.subscription_count = 0
    node.last_state_change_ns = 0
    frame = np.zeros((160, 320, 3), dtype=np.uint8)
    candidate = make_candidate(0.4, STATE_RED, 'vehicular_red')
    node._detect_candidates = Mock(return_value=([candidate], (0, 0, 320, 160)))
    node._analyze_selected_candidate = Mock(
        return_value=node._empty_analysis('test_color_analysis')
    )
    node.bridge = SimpleNamespace(imgmsg_to_cv2=Mock(return_value=frame))
    node._publish_outputs = Mock()
    message = Image()
    message.header.stamp.sec = 1
    node.latest_msg = message

    node._process_latest_frame()

    node._analyze_selected_candidate.assert_called_once()
    assert node._publish_outputs.call_args.args[4].source == 'model_low_conf'
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
    assert arguments['imgsz'] == 960
    assert arguments['conf'] == 0.05
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


@pytest.mark.parametrize('width,height,expected', [(1254, 370, (123, 753)), (1878, 555, (185, 1127))])
def test_default_center_roi_reaches_model(fusion_node, monkeypatch, width, height, expected):
    monkeypatch.setattr(TLFusionNode, '_declare_param', lambda self, name, default: str(Path(__file__)) if name == 'model_path' else default)
    node = TLFusionNode()
    node.model.predict.return_value = [SimpleNamespace(boxes=None)]
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    _, bounds = node._detect_candidates(frame)
    source = node.model.predict.call_args.kwargs['source']
    assert source.shape[:2] == expected
    assert np.shares_memory(frame, source)
    assert bounds == (int(width * .20), 0, int(width * .80), int(height / 3))


def test_default_model_path_is_fixed_best_pt():
    assert tl_fusion.default_tl_model_path() == (
        '/home/ki/SSC/src/perception/traffic_light/model/best.pt'
    )


@pytest.mark.parametrize('width,height', [(1254, 370), (1878, 555), (40, 30)])
def test_cropped_debug_and_inset_fit(fusion_node, width, height):
    node = fusion_node
    node.debug_image_max_side_px = 0
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


def test_debug_inset_is_aligned_to_left(fusion_node):
    image = np.zeros((120, 240, 3), dtype=np.uint8)
    inset = np.full((30, 50, 3), 80, dtype=np.uint8)

    fusion_node._draw_debug_inset(image, inset, 'Color Mask')

    assert image[9, 9].tolist() == [10, 12, 16]
    assert image[9, 190].tolist() == [0, 0, 0]


def test_debug_image_is_downsampled_without_changing_source(fusion_node):
    node = fusion_node
    node.debug_image_max_side_px = 100
    frame = np.zeros((160, 320, 3), dtype=np.uint8)
    original = frame.copy()
    analysis = replace(node._empty_analysis('test'), highlighted=None)

    debug = node._build_debug_image(frame, [], None, analysis)

    assert max(debug.shape[:2]) <= 100
    np.testing.assert_array_equal(frame, original)


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
    (70, 0, 30, STATE_RED),
    (30, 0, 70, STATE_GREEN),
    (29, 0, 71, STATE_GREEN),
    (82, 0, 18, STATE_RED),
    (83, 0, 17, STATE_RED),
    (58, 12, 30, STATE_RED),
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


def test_color_analysis_ignores_color_outside_signal_box(fusion_node):
    node = fusion_node
    node.use_torch_color_fallback = False
    node._enhance_crop = lambda crop: crop
    node._clean_mask = lambda mask: mask
    frame = np.full((120, 120, 3), (0, 0, 255), dtype=np.uint8)
    frame[40:90, 50:100] = (0, 255, 0)

    analysis = node._analyze_selected_candidate(frame, make_candidate())

    assert analysis.state == STATE_GREEN


def test_color_analysis_rejects_orange_as_red(fusion_node):
    node = fusion_node
    node.use_torch_color_fallback = False
    node._enhance_crop = lambda crop: crop
    node._clean_mask = lambda mask: mask
    orange = np.full((10, 10, 3), (20, 80, 220), dtype=np.uint8)

    analysis = node._analyze_selected_candidate(orange, make_candidate())

    assert analysis.state == STATE_UNKNOWN
    assert analysis.scores['red'] == 0.0


def test_color_analysis_rejects_blue_as_green(color_node):
    hsv_pixel = np.array([[[95, 220, 180]]], dtype=np.uint8)
    bgr_pixel = tl_fusion.cv2.cvtColor(hsv_pixel, tl_fusion.cv2.COLOR_HSV2BGR)[0, 0]
    blue = np.tile(bgr_pixel, (10, 10, 1))

    analysis = color_node._analyze_selected_candidate(blue, make_candidate())

    assert analysis.state == STATE_UNKNOWN
    assert analysis.scores['green'] == 0.0


def test_color_analysis_recovers_dim_green_in_middle_band(color_node):
    hsv_pixel = np.array([[[60, 180, 60]]], dtype=np.uint8)
    bgr_pixel = tl_fusion.cv2.cvtColor(hsv_pixel, tl_fusion.cv2.COLOR_HSV2BGR)[0, 0]
    dim_green = np.tile(bgr_pixel, (10, 10, 1))

    analysis = color_node._analyze_selected_candidate(dim_green, make_candidate())

    assert analysis.state == STATE_GREEN
    assert analysis.decisive


def test_color_analysis_accepts_slightly_shifted_red(color_node):
    node = color_node
    node.use_torch_color_fallback = False
    node._enhance_crop = lambda crop: crop
    node._clean_mask = lambda mask: mask
    hsv_pixel = np.array([[[8, 220, 180]]], dtype=np.uint8)
    bgr_pixel = tl_fusion.cv2.cvtColor(hsv_pixel, tl_fusion.cv2.COLOR_HSV2BGR)[0, 0]
    shifted_red = np.tile(bgr_pixel, (10, 10, 1))

    analysis = node._analyze_selected_candidate(shifted_red, make_candidate())

    assert analysis.state == STATE_RED
    assert analysis.decisive


def test_color_analysis_accepts_lower_brightness_red(color_node):
    hsv_pixel = np.array([[[8, 220, 90]]], dtype=np.uint8)
    bgr_pixel = tl_fusion.cv2.cvtColor(hsv_pixel, tl_fusion.cv2.COLOR_HSV2BGR)[0, 0]
    dim_red = np.tile(bgr_pixel, (10, 10, 1))

    analysis = color_node._analyze_selected_candidate(dim_red, make_candidate())

    assert analysis.state == STATE_RED
    assert analysis.decisive


def test_color_analysis_rejects_dim_red_orange_as_active_red(fusion_node):
    node = fusion_node
    node.use_torch_color_fallback = False
    node._enhance_crop = lambda crop: crop
    node._clean_mask = lambda mask: mask
    dim_orange = np.full((10, 10, 3), (0, 20, 80), dtype=np.uint8)

    analysis = node._analyze_selected_candidate(dim_orange, make_candidate())

    assert analysis.state == STATE_UNKNOWN
    assert analysis.scores['red'] == 0.0


def test_color_analysis_accepts_dim_yellow_green(color_node):
    hsv_pixel = np.array([[[60, 100, 90]]], dtype=np.uint8)
    bgr_pixel = tl_fusion.cv2.cvtColor(hsv_pixel, tl_fusion.cv2.COLOR_HSV2BGR)[0, 0]
    frame = np.tile(bgr_pixel, (10, 10, 1))

    analysis = color_node._analyze_selected_candidate(frame, make_candidate())

    assert analysis.state == STATE_GREEN
    assert analysis.decisive


def test_color_vertical_weights_favor_middle_of_candidate(color_node):
    signal_mask = np.ones((10, 10), dtype=np.uint8) * 255

    weights = color_node._color_vertical_weights(signal_mask, signal_mask.shape)

    assert np.allclose(weights[0], 0.20)
    assert np.allclose(weights[3:6], 1.50)
    assert np.allclose(weights[-1], 0.20)


def test_middle_mask_amplification_stays_inside_middle_band(color_node):
    signal_mask = np.ones((9, 9), dtype=np.uint8) * 255
    mask = np.zeros((9, 9), dtype=np.uint8)
    mask[4, 4] = 255

    amplified = color_node._amplify_middle_mask(mask, signal_mask)

    assert np.count_nonzero(amplified) > np.count_nonzero(mask)
    assert np.count_nonzero(amplified[:3]) == 0
    assert np.count_nonzero(amplified[6:]) == 0


@pytest.mark.parametrize('edge_pixel, central_pixel, edge_name, central_name', [
    ((0, 255, 0), (0, 0, 255), 'green', 'red'),
    ((0, 255, 0), (0, 255, 255), 'green', 'yellow'),
    ((0, 0, 255), (0, 255, 0), 'red', 'green'),
])
def test_color_vertical_weight_favors_middle_for_all_colors(
    color_node, edge_pixel, central_pixel, edge_name, central_name,
):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    frame[:3] = edge_pixel
    frame[3:6] = central_pixel

    analysis = color_node._analyze_selected_candidate(frame, make_candidate())

    assert analysis.scores[central_name] > analysis.scores[edge_name]


def test_color_vertical_weight_matches_torch_score(color_node):
    color_node.color_fallback_device = 'cpu'
    color_node._clean_mask = lambda mask: mask
    color_node._torch_clean_mask = lambda mask: mask
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    frame[:5] = (0, 255, 0)
    frame[5:] = (0, 255, 255)
    signal_mask = np.full(frame.shape[:2], 255, dtype=np.uint8)

    cpu_analysis = color_node._analyze_selected_candidate(frame, make_candidate())
    torch_raw_scores, _, _ = color_node._torch_color_measurements(
        frame, signal_mask, frame,
    )
    torch_total = sum(torch_raw_scores.values())
    torch_scores = {
        name: score / torch_total for name, score in torch_raw_scores.items()
    }

    assert cpu_analysis.scores == pytest.approx(torch_scores, abs=1.0e-5)


def test_color_mask_debug_marks_green_weight_boundaries(color_node):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    analysis = color_node._analyze_selected_candidate(
        frame,
        make_candidate(),
        render_debug=True,
    )

    assert analysis.highlighted is not None
    assert np.all(analysis.highlighted[3] == (255, 255, 255))
    assert np.all(analysis.highlighted[6] == (255, 255, 255))


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
        'fallback_min_valid_pixels': baseline.valid_pixels,
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


def test_uncertain_candidate_temporarily_keeps_last_state(fusion_node):
    node = fusion_node
    clock_ns = [1_000_000_000]
    node._now_ns = lambda: clock_ns[0]
    node.current_state = STATE_RED
    node.pending_state = STATE_RED
    node.last_state_change_ns = clock_ns[0]
    node.last_seen_candidate_ns = clock_ns[0]
    node.state_confirm_ms = 0.0
    node.hold_ms = 0
    node.uncertain_hold_ms = 300.0

    assert node._update_stable_state(STATE_UNKNOWN, True, uncertain=True) == STATE_RED

    clock_ns[0] += 250_000_000
    assert node._update_stable_state(STATE_UNKNOWN, True, uncertain=True) == STATE_RED

    clock_ns[0] += 100_000_000
    assert node._update_stable_state(STATE_UNKNOWN, True, uncertain=True) == STATE_UNKNOWN


def test_high_confidence_model_recovers_from_unknown_immediately(fusion_node):
    node = fusion_node
    node.state_confirm_ms = 200.0
    node.hold_ms = 250
    node.current_state = STATE_UNKNOWN
    node.last_state_change_ns = 1_000_000_000
    node.last_seen_candidate_ns = 1_000_000_000
    node._now_ns = lambda: 1_000_000_001

    assert node._update_stable_state(STATE_RED, True, immediate=True) == STATE_RED


@pytest.mark.parametrize('age_ms,reason', [(0, None), (500, None), (501, 'invalid_image_stale'), (-51, 'invalid_image_future')])
def test_frame_age_boundaries(fusion_node, age_ms, reason):
    message = Image()
    stamp_ns = 1_000_000_000 - age_ms * 1_000_000
    message.header.stamp.sec, message.header.stamp.nanosec = divmod(stamp_ns, 1_000_000_000)
    result = fusion_node._validate_input_timestamp(message)
    assert result is None if reason is None else result.startswith(reason)


def test_zero_stamp_rejected_with_age_check(fusion_node):
    fusion_node.require_image_header_stamp = False
    assert fusion_node._validate_input_timestamp(Image()) == 'invalid_image_stamp_zero'


@pytest.mark.parametrize('period_ms', [25, 50, 100])
def test_confirmation_independent_of_frame_count(fusion_node, period_ms):
    fusion_node.hold_ms = 0
    for elapsed_ms in range(0, 201, period_ms):
        fusion_node._now_ns = lambda: 1_000_000_000 + elapsed_ms * 1_000_000
        state = fusion_node._update_stable_state(STATE_RED, True)
        assert state == (STATE_RED if elapsed_ms == 200 else STATE_UNKNOWN)


def test_long_observation_gap_restarts_confirmation(fusion_node):
    fusion_node.hold_ms = 0
    fusion_node._update_stable_state(STATE_GREEN, True)
    fusion_node._now_ns = lambda: 2_000_000_000
    assert fusion_node._update_stable_state(STATE_GREEN, True) == STATE_UNKNOWN
    fusion_node._now_ns = lambda: 2_200_000_000
    assert fusion_node._update_stable_state(STATE_GREEN, True) == STATE_GREEN


def test_receive_continues_during_inference_and_retains_latest(fusion_node):
    entered = Event()
    release = Event()
    def slow_detect(frame):
        entered.set()
        assert release.wait(5)
        return [], (0, 0, 20, 20)
    fusion_node._detect_candidates = slow_detect
    fusion_node.bridge.imgmsg_to_cv2 = Mock(return_value=np.zeros((20, 20, 3), dtype=np.uint8))
    message = Image()
    message.header.stamp.sec = 1
    fusion_node._image_callback(message)
    worker = Thread(target=fusion_node._process_latest_frame)
    worker.start()
    try:
        assert entered.wait(5)
        for index in range(3):
            newest = Image()
            newest.header.stamp.sec = 1
            newest.header.stamp.nanosec = index
            fusion_node._image_callback(newest)
        assert fusion_node.latest_msg is newest
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert fusion_node.latest_msg is newest
    assert fusion_node.receive_group is not fusion_node.process_group


def test_frame_expiring_during_inference_is_not_published_valid(fusion_node):
    message = Image()
    message.header.stamp.sec = 1
    fusion_node._image_callback(message)
    fusion_node.bridge.imgmsg_to_cv2 = Mock(return_value=np.zeros((20, 20, 3), dtype=np.uint8))
    def expire(frame):
        fusion_node._now_ns = lambda: 1_600_000_000
        return [], (0, 0, 20, 20)
    fusion_node._detect_candidates = expire
    fusion_node._process_latest_frame()
    assert fusion_node.processed_frames == 0
    assert fusion_node.state_pub.messages[-1].data == STATE_UNKNOWN
    assert fusion_node.detection_pub.messages[-1].detections == []


def test_processing_error_clears_previous_state_and_detections(fusion_node):
    fusion_node.current_state = STATE_GREEN
    fusion_node.last_candidate_box = (1, 2, 3, 4)
    fusion_node.bridge.imgmsg_to_cv2 = Mock(side_effect=RuntimeError('decode failed'))
    fusion_node._log_processing_error = Mock()
    message = Image()
    message.header.stamp.sec = 1
    fusion_node._image_callback(message)

    fusion_node._process_latest_frame()

    assert fusion_node.current_state == STATE_UNKNOWN
    assert fusion_node.state_pub.messages[-1].data == STATE_UNKNOWN
    assert fusion_node.detection_pub.messages[-1].detections == []
    assert fusion_node.last_candidate_box is None


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
