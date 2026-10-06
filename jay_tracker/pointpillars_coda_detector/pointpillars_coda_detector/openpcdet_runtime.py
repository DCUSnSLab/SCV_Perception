"""Load OpenPCDet with CUDA extensions compatible with the active PyTorch.

The complete OpenPCDet Python tree on this robot contains CUDA 11 extension
binaries, while the workspace also contains CUDA 12 binaries built for the
active Python/PyTorch ABI.  Import the compatible binaries under OpenPCDet's
expected module names before OpenPCDet imports its extension wrappers.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


CUDA_EXTENSION_PATHS = {
    "pcdet.ops.roiaware_pool3d.roiaware_pool3d_cuda":
        "roiaware_pool3d/roiaware_pool3d_cuda.cpython-310-x86_64-linux-gnu.so",
    "pcdet.ops.iou3d_nms.iou3d_nms_cuda":
        "iou3d_nms/iou3d_nms_cuda.cpython-310-x86_64-linux-gnu.so",
    "pcdet.ops.pointnet2.pointnet2_stack.pointnet2_stack_cuda":
        "pointnet2/pointnet2_stack/pointnet2_stack_cuda.cpython-310-x86_64-linux-gnu.so",
    "pcdet.ops.pointnet2.pointnet2_batch.pointnet2_batch_cuda":
        "pointnet2/pointnet2_batch/pointnet2_batch_cuda.cpython-310-x86_64-linux-gnu.so",
    "pcdet.ops.roipoint_pool3d.roipoint_pool3d_cuda":
        "roipoint_pool3d/roipoint_pool3d_cuda.cpython-310-x86_64-linux-gnu.so",
    "pcdet.ops.ingroup_inds.ingroup_inds_cuda":
        "ingroup_inds/ingroup_inds_cuda.cpython-310-x86_64-linux-gnu.so",
}


def _load_extension(module_name: str, path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"OpenPCDet CUDA extension is missing: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create extension loader for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise


def _disable_unused_bevfusion() -> None:
    """Prevent an unused CUDA-11 bev_pool extension from being imported.

    The upstream detector registry eagerly imports every detector, including
    BEVFusion. PointPillars does not use BEVFusion or bev_pool, so a clear stub
    keeps that optional dependency out of this inference runtime.
    """

    module_name = "pcdet.models.detectors.bevfusion"
    if module_name in sys.modules:
        return
    stub = types.ModuleType(module_name)

    class BevFusion:  # pragma: no cover - only guards accidental configuration
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "BEVFusion is disabled in the PointPillars CUDA overlay")

    stub.BevFusion = BevFusion
    sys.modules[module_name] = stub


def prepare_openpcdet_runtime(openpcdet_path: Path,
                              cuda_ops_path: Path) -> list[str]:
    """Prepare and return the names of preloaded CUDA extension modules."""

    source = Path(openpcdet_path).expanduser().resolve()
    ops = Path(cuda_ops_path).expanduser().resolve()
    if not (source / "pcdet" / "__init__.py").is_file():
        raise FileNotFoundError(f"Invalid OpenPCDet source path: {source}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

    # Importing only the root package does not load any CUDA extension.
    import pcdet  # noqa: F401, PLC0415

    loaded = []
    for module_name, relative_path in CUDA_EXTENSION_PATHS.items():
        _load_extension(module_name, ops / relative_path)
        loaded.append(module_name)
    _disable_unused_bevfusion()
    return loaded


def opencv_rotated_nms(boxes, scores, thresh, pre_maxsize=None, **kwargs):
    """OpenPCDet-compatible rotated NMS without its incompatible CUDA op.

    OpenCV performs exact rotated-rectangle intersection in compiled code.
    Only the reduced, score-filtered NMS candidates cross to CPU; returned
    indices are moved back to the input tensor's device.
    """

    import cv2  # noqa: PLC0415
    import torch  # noqa: PLC0415

    if boxes.ndim != 2 or boxes.shape[1] < 7:
        raise ValueError("NMS boxes must have shape (N, >=7)")
    if boxes.shape[0] == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device), None

    order = torch.argsort(scores, descending=True)
    if pre_maxsize is not None:
        order = order[:int(pre_maxsize)]
    selected_boxes = boxes[order, :7].detach().float().cpu().numpy()
    selected_scores = scores[order].detach().float().cpu().numpy()

    rectangles = [
        (
            (float(box[0]), float(box[1])),
            (max(float(box[3]), 1e-4), max(float(box[4]), 1e-4)),
            float(np.degrees(box[6])),
        )
        for box in selected_boxes
    ]
    kept = cv2.dnn.NMSBoxesRotated(
        rectangles,
        selected_scores.tolist(),
        score_threshold=0.0,
        nms_threshold=float(thresh),
    )
    kept = np.asarray(kept, dtype=np.int64).reshape(-1)
    kept_tensor = torch.as_tensor(
        kept, dtype=torch.long, device=order.device)
    return order[kept_tensor].contiguous(), None


def install_pointpillars_nms_fallback() -> str:
    """Replace only NMS; PointPillars' remaining GPU path stays unchanged."""

    from pcdet.ops.iou3d_nms import iou3d_nms_utils  # noqa: PLC0415

    iou3d_nms_utils.nms_gpu = opencv_rotated_nms
    iou3d_nms_utils.nms_normal_gpu = opencv_rotated_nms
    return "opencv_rotated_nms"
