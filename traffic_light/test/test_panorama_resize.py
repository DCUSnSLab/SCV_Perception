import numpy as np
import pytest
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from mando_tools.panorama_resize import resize_message


@pytest.mark.parametrize('encoding,shape,dtype', [
    ('bgr8', (555, 1878, 3), np.uint8),
    ('rgb8', (555, 1878, 3), np.uint8),
    ('mono8', (555, 1878), np.uint8),
    ('mono16', (555, 1878), np.uint16),
])
def test_resize_preserves_header_encoding_and_input(encoding, shape, dtype):
    bridge = CvBridge()
    frame = np.full(shape, 42, dtype=dtype)
    message = bridge.cv2_to_imgmsg(frame, encoding=encoding)
    message.header.frame_id = 'panorama_optical_frame'
    message.header.stamp.sec = 1788670745
    message.header.stamp.nanosec = 123456789
    result = resize_message(message, 1254, 370, bridge)
    assert (result.width, result.height) == (1254, 370)
    assert result.header == message.header
    assert result.encoding == encoding
    assert len(result.data) == result.step * result.height
    assert np.all(bridge.imgmsg_to_cv2(result) == 42)
    assert (message.width, message.height) == (1878, 555)


def test_same_dimensions_return_original_message():
    bridge = CvBridge()
    message = bridge.cv2_to_imgmsg(np.zeros((370, 1254, 3), dtype=np.uint8), encoding='bgr8')
    assert resize_message(message, 1254, 370, bridge) is message


@pytest.mark.parametrize('width,height', [(0, 370), (1254, -1)])
def test_invalid_output_size(width, height):
    with pytest.raises(ValueError):
        resize_message(Image(), width, height, CvBridge())


def test_empty_input():
    with pytest.raises(ValueError):
        resize_message(Image(), 1254, 370, CvBridge())
