import cv2
import numpy as np

from mando_tools.green_down_arrow import GREEN_ARROW
from mando_tools.green_down_arrow import RED_X
from mando_tools.green_down_arrow import classify_panel
from mando_tools.green_down_arrow import find_three_panel_rig


def test_color_only_three_panel_detection() -> None:
    image = np.zeros((300, 800, 3), np.uint8)
    centers = ((280, 130), (380, 130), (480, 130))
    for center in centers:
        cv2.rectangle(
            image,
            (center[0] - 35, center[1] - 35),
            (center[0] + 35, center[1] + 35),
            (20, 20, 20),
            -1,
        )
    for center in (centers[0], centers[2]):
        cv2.line(
            image,
            (center[0] - 22, center[1] - 22),
            (center[0] + 22, center[1] + 22),
            (0, 0, 255),
            10,
        )
        cv2.line(
            image,
            (center[0] + 22, center[1] - 22),
            (center[0] - 22, center[1] + 22),
            (0, 0, 255),
            10,
        )
    cv2.rectangle(image, (374, 102), (386, 136), (0, 255, 0), -1)
    cv2.fillConvexPoly(
        image,
        np.array([[354, 130], [406, 130], [380, 160]], np.int32),
        (0, 255, 0),
    )

    rig = find_three_panel_rig(image, (0, 0, 800, 250))

    assert rig is not None
    decisions = [classify_panel(image, box).class_id for box in rig.boxes]
    assert decisions == [RED_X, GREEN_ARROW, RED_X]


def test_blank_image_has_no_rig() -> None:
    blank = np.zeros((300, 800, 3), np.uint8)
    assert find_three_panel_rig(blank, (0, 0, 800, 250)) is None
