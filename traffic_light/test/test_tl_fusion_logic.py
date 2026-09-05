from collections import deque
import time

from mando_tools.tl_fusion import STATE_GREEN
from mando_tools.tl_fusion import STATE_LEFT_ARROW
from mando_tools.tl_fusion import STATE_RED
from mando_tools.tl_fusion import STATE_UNKNOWN
from mando_tools.tl_fusion import TLFusionNode


class RecordingPublisher:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, message) -> None:
        self.messages.append(message)


class RecordingLogger:
    def __init__(self) -> None:
        self.errors = []

    def error(self, message: str) -> None:
        self.errors.append(message)


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
