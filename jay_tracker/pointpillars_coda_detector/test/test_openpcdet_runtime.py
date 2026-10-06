import torch

from pointpillars_coda_detector.openpcdet_runtime import opencv_rotated_nms


def test_rotated_nms_suppresses_overlapping_boxes():
    boxes = torch.tensor([
        [0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0],
        [0.1, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0],
        [10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0],
    ])
    scores = torch.tensor([0.9, 0.8, 0.7])
    kept, selected_scores = opencv_rotated_nms(
        boxes, scores, thresh=0.1)
    assert kept.tolist() == [0, 2]
    assert selected_scores is None


def test_rotated_nms_empty_input_preserves_device():
    boxes = torch.empty((0, 7))
    scores = torch.empty((0,))
    kept, _ = opencv_rotated_nms(boxes, scores, thresh=0.1)
    assert kept.dtype == torch.long
    assert kept.device == boxes.device
