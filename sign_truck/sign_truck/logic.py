from __future__ import annotations


UNKNOWN = 'UNKNOWN'
GREEN = 'GREEN'
RED = 'RED'


def class_state(class_name: str) -> str:
    if class_name == 'green_sign':
        return GREEN
    if class_name == 'red_sign':
        return RED
    return UNKNOWN


def state_at_anchor(
    detections: list[tuple[float, str, float]],
    anchor_x: float,
    max_distance: float,
) -> str:
    candidates = [
        (abs(center_x - anchor_x), -confidence, class_state(class_name))
        for center_x, class_name, confidence in detections
        if abs(center_x - anchor_x) <= max_distance
    ]
    return min(candidates, default=(0.0, 0.0, UNKNOWN))[2]


def assign_lane_states(
    detections: list[tuple[float, str, float]],
    anchors: dict[str, float],
    max_distance: float,
) -> dict[str, str]:
    states = {lane: UNKNOWN for lane in anchors}
    used_lanes: set[str] = set()
    used_detections: set[int] = set()
    candidates = sorted(
        (
            abs(center_x - anchor_x),
            -confidence,
            lane,
            index,
            class_state(class_name),
        )
        for lane, anchor_x in anchors.items()
        for index, (center_x, class_name, confidence) in enumerate(detections)
        if abs(center_x - anchor_x) <= max_distance
    )
    for _, _, lane, index, state in candidates:
        if lane not in used_lanes and index not in used_detections:
            states[lane] = state
            used_lanes.add(lane)
            used_detections.add(index)
    return states
