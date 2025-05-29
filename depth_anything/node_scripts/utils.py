#!/usr/bin/env python


import numpy as np
import cv2
from typing import Optional
INPUT_WIDTH, INPUT_HEIGHT = 518, 518
RGB_MEAN = [0.485, 0.456, 0.406]
RGB_STD = [0.229, 0.224, 0.225]


def preprocess(image: np.ndarray):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    image = cv2.resize(image, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_CUBIC)
    image = (image - RGB_MEAN) / RGB_STD
    image = image.transpose(2, 0, 1)[None].astype("float32")
    return image

# def postprocess_depth(depth_raw: np.ndarray,
#                       rgb: Optional[np.ndarray] = None,
#                       d_min: float = 0.2,
#                       d_max: float = 40.0,
#                       guide_radius: int = 5,
#                       guide_eps: float = 1e-3,
#                       unsharp_gain: float = 0.6) -> np.ndarray:
#     # ---------- 1) 작은 블랙홀 메우기 ----------
#     depth_u16 = cv2.normalize(depth_raw, None, 0, 65535,
#                               cv2.NORM_MINMAX).astype(np.uint16)
#     kernel = np.ones((3, 3), np.uint8)
#     depth_closed = cv2.morphologyEx(depth_u16,
#                                     cv2.MORPH_CLOSE,
#                                     kernel)
#     depth_closed = depth_closed.astype(np.float32) / 65535. * depth_raw.max()

#     # ---------- 2) Bilateral(에지 보존 1차) ----------
#     depth_bi = cv2.bilateralFilter(depth_closed,
#                                    d=5,
#                                    sigmaColor=0.1,
#                                    sigmaSpace=3)

#     # ---------- 3) Guided Filter ----------
#     gf = depth_bi  # fallback
#     # ---------- 4) Un-sharp Mask ----------
#     lap = cv2.Laplacian(gf, cv2.CV_32F, ksize=3)
#     depth_sharp = cv2.addWeighted(gf, 1.0, lap, unsharp_gain, 0)

#     # ---------- 5) 클리핑 ----------
#     depth_out = np.clip(depth_sharp, d_min, d_max).astype(np.float32)
#     return depth_out

def postprocess_depth(depth_raw: np.ndarray,
                      d_min: float = 0.2,
                      d_max: float = 40.0) -> np.ndarray:
    """
    depth_raw : (H,W) float32, 단위 m
    1) morphology closing → 작은 구멍 채움
    2) bilateral filter → 에지 살리고 잡음 제거
    3) 값 클리핑
    """
    # 1) 작은 블랙홀(pit) 메우기
    depth_u16 = cv2.normalize(depth_raw, None, 0, 65535,
                              cv2.NORM_MINMAX).astype(np.uint16)
    kernel = np.ones((3, 3), np.uint8)
    depth_closed = cv2.morphologyEx(depth_u16,
                                    cv2.MORPH_CLOSE,
                                    kernel)
    depth_closed = depth_closed.astype(np.float32) / 65535. * depth_raw.max()

    # 2) 에지-보존 블러
    depth_blur = cv2.bilateralFilter(depth_closed,
                                     d=5,         # 필터 지름
                                     sigmaColor=0.1,
                                     sigmaSpace=3)

    # 3) 범위 클립
    depth_clipped = np.clip(depth_blur, d_min, d_max).astype(np.float32)

    return depth_clipped