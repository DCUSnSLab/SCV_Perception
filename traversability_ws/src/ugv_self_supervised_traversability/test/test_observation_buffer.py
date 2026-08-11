import numpy as np

from ugv_self_supervised_traversability.observation_buffer import Observation, ObservationBuffer


def make_observation(timestamp_ns: int) -> Observation:
    return Observation(timestamp_ns, 'camera', np.zeros((2, 2, 3), np.uint8), None,
                       np.eye(3), np.eye(4), {}, np.zeros((2, 2), np.uint8))


def test_time_and_frame_bounded_buffer() -> None:
    buffer = ObservationBuffer(duration_seconds=1.0, max_frames=2)
    assert buffer.append(make_observation(0)) == []
    buffer.append(make_observation(500_000_000))
    evicted = buffer.append(make_observation(800_000_000))
    assert [item.timestamp_ns for item in evicted] == [0]
    expired = buffer.pop_expired(1_600_000_000)
    assert [item.timestamp_ns for item in expired] == [500_000_000]
