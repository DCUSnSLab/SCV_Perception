import cv2
import numpy as np
import pytest

from mando_tools.black_box_color_bits import BlackBoxColorDetector
from mando_tools.black_box_color_bits import DetectorConfig
from mando_tools.black_box_color_bits import BlackBoxColorBitsNode, BoxObservation
from types import SimpleNamespace
from cv_bridge import CvBridge


@pytest.mark.parametrize('colors, expected', [
    ([(0, 0, 230)], [0]),
    ([(0, 230, 0), (0, 0, 230)], [1, 0]),
    ([(0, 0, 230), (0, 230, 0), (0, 0, 230)], [0, 1, 0]),
])
def test_color_regions_without_black_background(colors, expected):
    frame = np.full((400, 640, 3), 180, dtype=np.uint8)
    for index, color in enumerate(colors):
        left = 80 + index * 120
        cv2.rectangle(frame, (left + 22, 62), (left + 38, 78), color, -1)
    config = DetectorConfig(roi_left_ratio=0.0, roi_right_ratio=1.0, surround_min_dark_ratio=0.0)
    bits, observations, _ = BlackBoxColorDetector(config).process(frame, now_ns=0)
    assert bits == expected
    assert len(observations) == len(colors)


def test_color_region_does_not_require_black_fill():
    frame = np.full((400, 640, 3), 180, dtype=np.uint8)
    cv2.rectangle(frame, (80, 40), (140, 100), (20, 20, 20), -1)
    cv2.rectangle(frame, (87, 47), (133, 93), (0, 230, 0), -1)
    bits, observations, _ = BlackBoxColorDetector(DetectorConfig(roi_left_ratio=0.0, roi_right_ratio=1.0)).process(frame, now_ns=0)
    assert bits == [1]
    assert len(observations) == 1


def test_unlit_display_never_initializes_as_red():
    frame = np.full((400, 640, 3), 180, dtype=np.uint8)
    cv2.rectangle(frame, (80, 40), (140, 100), (20, 20, 20), -1)
    bits, observations, _ = BlackBoxColorDetector(DetectorConfig(roi_left_ratio=0.0, roi_right_ratio=1.0)).process(frame, now_ns=0)
    assert bits == []
    assert observations == []


def make_frame(colors, height=240, width=640):
    frame = np.full((height, width, 3), (145, 145, 145), dtype=np.uint8)
    for x, color in colors:
        cv2.rectangle(frame, (x, 35), (x + 64, 65), (20, 20, 20), -1)
        cv2.rectangle(frame, (x + 14, 42), (x + 50, 58), color, -1)
    return frame


def detector(**overrides):
    values = {
        'roi_left_ratio': 0.0,
        'roi_right_ratio': 1.0,
        'roi_bottom_ratio': 0.50,
        'min_box_width_px': 20,
        'min_box_height_px': 10,
        'min_box_area_px': 100,
        'max_box_width_ratio': 0.25,
        'max_box_height_ratio': 0.50,
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
    assert [observation.bbox[0] for observation in observations] == [94, 194, 294]


def test_ignores_black_box_outside_upper_roi():
    node = detector()
    frame = make_frame([(80, (0, 220, 0))])
    cv2.rectangle(frame, (300, 180), (364, 210), (20, 20, 20), -1)
    cv2.rectangle(frame, (314, 187), (350, 203), (0, 0, 220), -1)

    bits, observations, _ = node.process(frame, now_ns=0)

    assert bits == [1]
    assert len(observations) == 1


def test_rejects_oversized_color_region():
    node = detector()
    frame = make_frame([])
    cv2.rectangle(frame, (80, 35), (280, 95), (0, 220, 0), -1)

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


@pytest.mark.parametrize('width,height,expected', [(1254, 370, (185, 627)), (1878, 555, (277, 939))])
def test_default_roi_dimensions_and_view(width, height, expected):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    roi, bounds = BlackBoxColorDetector()._extract_roi(frame)
    assert roi.shape[:2] == expected
    assert bounds == (int(width * .25), 0, int(width * .75), int(height / 2))
    assert np.shares_memory(frame, roi)


def test_default_roi_detects_only_center_top_and_preserves_global_coordinates():
    frame = np.full((370, 1254, 3), 180, dtype=np.uint8)
    for left, top, color in [(360, 45, (0, 0, 230)), (520, 45, (0, 230, 0)), (680, 45, (0, 0, 230)), (100, 45, (0, 230, 0)), (1000, 45, (0, 230, 0)), (520, 200, (0, 230, 0))]:
        cv2.rectangle(frame, (left, top), (left + 60, top + 60), (20, 20, 20), -1)
        cv2.rectangle(frame, (left + 22, top + 22), (left + 38, top + 38), color, -1)
    original = frame.copy()
    bits, observations, _ = BlackBoxColorDetector().process(frame, now_ns=1)
    assert bits == [0, 1, 0]
    assert [item.bbox[0] for item in observations] == [382, 542, 702]
    np.testing.assert_array_equal(frame, original)


def test_debug_crop_global_to_local_and_input_unchanged():
    frame = np.zeros((370, 1254, 3), dtype=np.uint8)
    observation = BoxObservation((360, 45, 421, 106), 1.0, 0.0, stable_bit=0)
    drawer = SimpleNamespace(_put_debug_text=BlackBoxColorBitsNode._put_debug_text)
    debug = BlackBoxColorBitsNode._draw_debug(drawer, frame, [observation], (313, 0, 940, 185), [0])
    assert debug.shape == (185, 627, 3)
    np.testing.assert_array_equal(debug[-1], frame[184, 313:940])
    assert not np.any(np.all(debug == (255, 0, 255), axis=2))
    assert debug[70, 47].tolist() == [0, 0, 255]
    assert not frame.any()
    assert not np.shares_memory(frame, debug)


@pytest.mark.parametrize('subscriber_count', [0, 1])
def test_callback_preserves_debug_header(subscriber_count):
    bridge = CvBridge()
    message = bridge.cv2_to_imgmsg(np.full((370, 1254, 3), 180, dtype=np.uint8), encoding='bgr8')
    message.header.frame_id = 'panorama_optical_frame'
    message.header.stamp.sec = 123
    message.header.stamp.nanosec = 456
    published = []
    published_bits = []
    drawn = []
    fake = SimpleNamespace(
        processing=False, latest_msg=message, bridge=bridge,
        detector=BlackBoxColorDetector(), publish_debug_image=True,
        debug_pub=SimpleNamespace(publish=published.append, get_subscription_count=lambda: subscriber_count),
        _publish_bits=published_bits.append,
        _put_debug_text=BlackBoxColorBitsNode._put_debug_text,
    )
    def draw_debug(*args):
        drawn.append(True)
        return BlackBoxColorBitsNode._draw_debug(fake, *args)
    fake._draw_debug = draw_debug
    BlackBoxColorBitsNode._process_latest_frame(fake)
    assert published_bits == [[]]
    if subscriber_count == 0:
        assert published == []
        assert drawn == []
        fake.debug_pub.get_subscription_count = lambda: 1
        fake.latest_msg = message
        BlackBoxColorBitsNode._process_latest_frame(fake)
    assert len(published) == 1
    assert published[0].header == message.header
    assert (published[0].width, published[0].height) == (627, 185)
    assert drawn == [True]


@pytest.mark.parametrize('margin', [1, 4, 30])
def test_surround_ratio_matches_ring_mask(margin):
    rng = np.random.default_rng(23)
    value = rng.integers(0, 256, (40, 60), dtype=np.uint8)
    detector = BlackBoxColorDetector(DetectorConfig(surround_margin_px=margin))
    for _ in range(40):
        left, right = sorted(rng.choice(61, 2, replace=False).tolist())
        top, bottom = sorted(rng.choice(41, 2, replace=False).tolist())
        outer_left, outer_top = max(0, left-margin), max(0, top-margin)
        outer_right, outer_bottom = min(60, right+margin), min(40, bottom+margin)
        ring = np.zeros(value.shape, dtype=bool)
        ring[outer_top:outer_bottom, outer_left:outer_right] = True
        ring[top:bottom, left:right] = False
        pixels = np.count_nonzero(ring)
        expected = float(np.count_nonzero((value <= 70) & ring)/pixels) if pixels else 0.0
        assert detector._surrounding_dark_ratio(value, (left, top, right, bottom)) == expected


@pytest.mark.parametrize('shape', [(1, 1, 3), (15, 40, 3), (185, 627, 3)])
def test_debug_text_fits_crop(shape, monkeypatch):
    calls = []
    monkeypatch.setattr(cv2, 'putText', lambda *args: calls.append(args))
    BlackBoxColorBitsNode._put_debug_text(np.zeros(shape, dtype=np.uint8), '0 R:0 r=1.00 g=0.00', (shape[1]-1, -5), (0, 0, 255))
    for args in calls:
        (width, height), baseline = cv2.getTextSize(args[1], args[3], args[4], args[6])
        assert args[2][0] >= 0 and args[2][0] + width < shape[1]
        assert args[2][1] - height >= 0 and args[2][1] + baseline < shape[0]


def test_new_upper_half_band_is_detected_but_lower_half_is_excluded():
    frame = np.full((600, 1200, 3), 180, dtype=np.uint8)
    for left, top in [(400, 220), (600, 320)]:
        cv2.rectangle(frame, (left, top), (left + 60, top + 60), (20, 20, 20), -1)
        cv2.rectangle(frame, (left + 22, top + 22), (left + 38, top + 38), (0, 230, 0), -1)
    bits, observations, bounds = BlackBoxColorDetector().process(frame, now_ns=1)
    assert bounds == (300, 0, 900, 300)
    assert bits == [1]
    assert [item.bbox for item in observations] == [(422, 242, 439, 259)]


def test_touching_different_colors_remain_separate_bits():
    frame = np.full((370, 1254, 3), 180, dtype=np.uint8)
    frame[50:80, 400:430] = (0, 0, 230)
    frame[50:80, 430:460] = (0, 230, 0)
    bits, observations, _ = BlackBoxColorDetector(DetectorConfig(surround_min_dark_ratio=0.0)).process(frame, now_ns=1)
    assert bits == [0, 1]
    assert len(observations) == 2


def test_yellow_gray_and_tiny_color_noise_are_ignored():
    frame = np.full((370, 1254, 3), 180, dtype=np.uint8)
    frame[50:80, 400:430] = (0, 230, 230)
    frame[50:80, 500:530] = (90, 90, 90)
    frame[50:52, 600:602] = (0, 230, 0)
    assert BlackBoxColorDetector().process(frame, now_ns=1)[0] == []


def test_irregular_color_shape_does_not_require_rectangularity():
    frame = np.full((370, 1254, 3), 180, dtype=np.uint8)
    cv2.circle(frame, (450, 80), 22, (0, 230, 0), 7)
    config = DetectorConfig(surround_min_dark_ratio=0.0)
    assert BlackBoxColorDetector(config).process(frame, now_ns=1)[0] == [1]


def led_frame(offset_x=0, offset_y=0, dark_surround=True):
    frame = np.full((600, 1200, 3), 180, dtype=np.uint8)
    if dark_surround:
        frame[150 + offset_y:190 + offset_y, 410 + offset_x:460 + offset_x] = 20
    for left, top in [(430, 165), (435, 165), (430, 171), (435, 171)]:
        frame[top + offset_y:top + offset_y + 3, left + offset_x:left + offset_x + 2] = (0, 55, 0)
    return frame


def test_dim_led_fragments_group_into_one_bit():
    bits, observations, _ = BlackBoxColorDetector().process(led_frame(), now_ns=1)
    assert bits == [1]
    assert len(observations) == 1
    assert observations[0].bbox == (430, 165, 437, 174)


def test_identical_color_on_bright_surround_is_rejected():
    assert BlackBoxColorDetector().process(led_frame(dark_surround=False), now_ns=1)[0] == []

def test_moving_led_group_keeps_track_identity():
    detector = BlackBoxColorDetector()
    bits, first, _ = detector.process(led_frame(), now_ns=1)
    moved_bits, moved, _ = detector.process(led_frame(8, 6), now_ns=100_000_001)
    assert bits == moved_bits == [1]
    assert first[0].track_id == moved[0].track_id
    assert moved[0].bbox == (438, 171, 445, 180)

def test_grouping_does_not_invent_color_pixels():
    frame = np.full((600, 1200, 3), 20, dtype=np.uint8)
    frame[165:168, 430:432] = (0, 230, 0)
    assert BlackBoxColorDetector().process(frame, now_ns=1)[0] == []

def test_separated_led_groups_are_not_merged():
    frame = led_frame()
    second = led_frame(80)
    frame[:, 480:550] = second[:, 480:550]
    bits, observations, _ = BlackBoxColorDetector().process(frame, now_ns=1)
    assert bits == [1, 1]
    assert [item.bbox[0] for item in observations] == [430, 510]

def test_dark_ring_at_roi_edge_uses_available_pixels_only():
    detector = BlackBoxColorDetector()
    value = np.full((20, 20), 20, dtype=np.uint8)
    value[:5, :5] = 230
    assert detector._surrounding_dark_ratio(value, (0, 0, 5, 5)) == 1.0
    assert detector._surrounding_dark_ratio(value, (0, 0, 20, 20)) == 0.0


def test_grouping_kernel_is_reused_between_frames(monkeypatch):
    detector = BlackBoxColorDetector()
    monkeypatch.setattr(cv2, 'getStructuringElement', lambda *args: pytest.fail('Kernel recreated during processing'))
    for timestamp in (1, 100_000_001):
        assert detector.process(led_frame(), now_ns=timestamp)[0] == [1]
