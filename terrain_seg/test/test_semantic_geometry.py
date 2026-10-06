import numpy as np

from terrain_seg.semantic_geometry import (
    colorize_labels, depth_to_meters, scale_intrinsics,
    rasterize_camera_depth, select_projected_semantics,
    semantic_rgbd_points, transform_xyz,
)


def test_depth_to_meters():
    depth = np.asarray([[0, 1000, 2500]], dtype=np.uint16)
    assert np.allclose(depth_to_meters(depth, '16UC1'), [[0.0, 1.0, 2.5]])


def test_scale_intrinsics():
    k = [600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0]
    assert scale_intrinsics(k, (640, 480), (320, 240)) == (
        300.0, 300.0, 160.0, 120.0)


def test_semantic_rgbd_points_filters_classes_and_confidence():
    depth = np.ones((2, 2), dtype=np.float32)
    labels = np.asarray([[0, 1], [2, 0]], dtype=np.uint8)
    confidence = np.asarray([[.9, .8], [.99, .2]], dtype=np.float32)
    bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr[0, 0] = (3, 2, 1)
    xyz, rgb, out_labels, out_confidence = semantic_rgbd_points(
        depth, labels, confidence, bgr, (1.0, 1.0, 0.0, 0.0),
        stride=1, min_confidence=.5, included_class_ids=(0, 1))
    assert xyz.shape == (2, 3)
    assert np.allclose(xyz, [[0, 0, 1], [1, 0, 1]])
    assert rgb.tolist() == [0x010203, 0]
    assert out_labels.tolist() == [0, 1]
    assert np.allclose(out_confidence, [.9, .8])


def test_palette_and_transform():
    colors = colorize_labels(np.asarray([[0, 1]], dtype=np.uint8))
    assert colors.tolist() == [[[128, 64, 128], [232, 35, 244]]]
    xyz = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    transformed = transform_xyz(xyz, np.eye(3), [2.0, 0.0, -1.0])
    assert np.allclose(transformed, [[3.0, 2.0, 2.0]])


def test_project_semantics_rejects_occluded_and_outside_points():
    labels = np.asarray([[0, 1], [2, 0]], dtype=np.uint8)
    confidence = np.full((2, 2), .9, dtype=np.float32)
    bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr[0, 0] = (3, 2, 1)
    # Projects to (0,0), (1,0), outside, and (0,0) behind measured depth.
    xyz_camera = np.asarray([
        [0, 0, 1], [1, 0, 1], [3, 0, 1], [0, 0, 2],
    ], dtype=np.float32)
    depth = np.ones((2, 2), dtype=np.float32)
    indices, rgb, labels_out, confidence_out = select_projected_semantics(
        xyz_camera, labels, confidence, bgr, (1, 1, 0, 0),
        included_class_ids=(0, 1), aligned_depth_m=depth,
        occlusion_tolerance=.25)
    assert indices.tolist() == [0, 1]
    assert rgb.tolist() == [0x010203, 0]
    assert labels_out.tolist() == [0, 1]
    assert np.allclose(confidence_out, [.9, .9])


def test_rasterize_camera_depth_keeps_nearest_surface():
    xyz = np.asarray([
        [0, 0, 2], [0, 0, 1], [1, 0, 1], [4, 0, 1],
    ], dtype=np.float32)
    depth = rasterize_camera_depth(xyz, (1, 1, 0, 0), (2, 2))
    assert np.allclose(depth, [[1, 1], [0, 0]])
