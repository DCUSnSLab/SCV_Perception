import numpy as np

from observation_tracker.evaluate_sustech_tracking import Box, aggregate_metrics, evaluate
from observation_tracker.evaluate_tracking_extended import (
    clear_3d_sequence, hota_sequence, identity_3d_sequence, iou_3d,
)


def test_ids_are_associated_by_position_not_literal_value():
    gt = {
        0: [Box(0, "1", "Car", 0.0, 0.0, 0.0)],
        1: [Box(1, "1", "Car", 1.0, 0.0, 0.0)],
        2: [Box(2, "1", "Car", 2.0, 0.0, 0.0)],
    }
    predictions = {
        0: [Box(0, "77", "Car", 0.1, 0.0, 0.0)],
        1: [Box(1, "77", "Car", 1.1, 0.0, 0.0)],
        2: [Box(2, "88", "Car", 2.1, 0.0, 0.0)],
    }

    metrics, events, mappings = evaluate(
        "test", gt, predictions, {"Car": 2.0, "Pedestrian": 1.0},
    )

    assert metrics["tp"] == 3
    assert metrics["id_switches"] == 1
    assert events[0]["old_pred_id"] == "77"
    assert events[0]["new_pred_id"] == "88"
    assert mappings[0]["gt_id"] == "1"
    assert mappings[0]["dominant_pred_id"] == "77"


def test_aggregate_recomputes_identity_metrics_from_counts():
    base = {
        "method": "tracker", "class_name": "All", "gt_boxes": 10,
        "pred_boxes": 10, "tp": 8, "fp": 2, "fn": 2, "id_switches": 1,
        "fragments": 1, "idtp": 7, "idfp": 3, "idfn": 3,
        "motp_center_m": 0.1, "center_rmse_m": 0.2, "yaw_mae_deg": 5.0,
        "ass_a": 0.8,
    }
    result = aggregate_metrics([base, base])[0]

    assert result["sequence"] == "ALL"
    assert result["id_switches"] == 2
    assert result["idf1"] == 0.7


def test_oriented_3d_iou_and_hota():
    box = Box(0, "gt", "Car", 0.0, 0.0, 0.0, 0.0, 4.0, 2.0, 2.0)
    same = Box(0, "different-id", "Car", 0.0, 0.0, 0.0, 0.0, 4.0, 2.0, 2.0)
    far = Box(0, "far", "Car", 10.0, 0.0, 0.0, 0.0, 4.0, 2.0, 2.0)

    assert iou_3d(box, same) == 1.0
    assert iou_3d(box, far) == 0.0
    result = hota_sequence({0: [box]}, {0: [same]})
    assert np.allclose(result["hota"], 1.0)
    assert identity_3d_sequence({0: [box]}, {0: [same]})["idf1"] == 1.0
    assert clear_3d_sequence({0: [box]}, {0: [same]}) == {
        "tp": 1, "fp": 0, "fn": 0, "idsw": 0, "fragmentation": 0,
    }
