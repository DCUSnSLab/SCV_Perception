import numpy as np

from terrain_seg.depth_geometry import (
    CURB, ELEVATED_GROUND, GROUND, OBSTACLE, GeometryConfig,
    classify_depth_geometry, fuse_cityscapes_geometry,
)


def synthetic_curb_scene():
    road_x, road_y = np.meshgrid(
        np.arange(0.0, 2.0, 0.05), np.arange(-1.0, 1.0, 0.05))
    sidewalk_x, sidewalk_y = np.meshgrid(
        np.arange(2.05, 4.0, 0.05), np.arange(-1.0, 1.0, 0.05))
    road = np.column_stack((road_x.ravel(), road_y.ravel(),
                            np.full(road_x.size, -0.9)))
    sidewalk = np.column_stack((
        sidewalk_x.ravel(), sidewalk_y.ravel(),
        np.full(sidewalk_x.size, -0.75)))
    curb_y, curb_z = np.meshgrid(
        np.arange(-1.0, 1.0, 0.05), np.arange(-0.9, -0.74, 0.025))
    curb = np.column_stack((np.full(curb_y.size, 2.0),
                            curb_y.ravel(), curb_z.ravel()))
    obstacle = np.asarray([[1.0, 0.0, -0.5], [1.02, 0.0, -0.4]])
    return np.vstack((road, sidewalk, curb, obstacle)).astype(np.float32), (
        len(road), len(sidewalk), len(curb))


def test_depth_geometry_finds_ground_elevation_curb_and_obstacle():
    points, (road_end, sidewalk_count, curb_count) = synthetic_curb_scene()
    result = classify_depth_geometry(
        points, GeometryConfig(expected_ground_z=-0.9), seed=3)
    sidewalk_end = road_end + sidewalk_count
    curb_end = sidewalk_end + curb_count
    assert np.mean(result.labels[:road_end] == GROUND) > 0.95
    assert np.mean(result.labels[road_end:sidewalk_end] ==
                   ELEVATED_GROUND) > 0.85
    assert np.any(result.labels[sidewalk_end:curb_end] == CURB)
    assert np.all(result.labels[-2:] == OBSTACLE)
    assert np.allclose(result.plane, [0.0, 0.0, -0.9], atol=.02)


def test_fusion_gives_geometry_curb_and_obstacle_priority():
    semantic = np.asarray([0, 1, 0, 1], dtype=np.uint8)
    semantic_confidence = np.full(4, .7, dtype=np.float32)
    geometry = np.asarray([GROUND, ELEVATED_GROUND, CURB, OBSTACLE])
    geometry_confidence = np.asarray([.8, .8, .9, .95], dtype=np.float32)
    labels, confidence = fuse_cityscapes_geometry(
        semantic, semantic_confidence, geometry, geometry_confidence)
    assert labels.tolist() == [1, 2, 3, 4]
    assert np.allclose(confidence, [.7, .7, .9, .95])
