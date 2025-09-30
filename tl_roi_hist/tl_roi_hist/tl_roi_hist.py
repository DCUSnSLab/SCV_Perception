#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Int32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
from collections import deque

def clamp(v, lo, hi): return max(lo, min(hi, v))

def pad_to_3x1_keep_scale(img, pad_color=(0, 0, 0)):
    """원본 스케일 유지, 3:1 레터박스 패딩."""
    if img is None or img.size == 0:
        return img
    h0, w0 = img.shape[:2]
    if h0 == 0 or w0 == 0:
        return img
    k = max(h0, int(math.ceil(w0 / 3.0)))
    H, W = k, 3 * k
    canvas = np.full((H, W, 3), pad_color, dtype=img.dtype)
    y = (H - h0) // 2
    x = (W - w0) // 2
    canvas[y:y + h0, x:x + w0] = img
    return canvas

def hue_hist_and_scores(bgr, s_min=60, v_min=80, h_bins=180,
                        red1=(0,5), red2=(168,179),
                        yel=(15,30), grn=(35,85)):
    """
    zoom(BGR)에서 Hue 히스토그램(0..179) 정규화와 R/Y/G 구간 합계, 유효픽셀수를 반환.
    returns: hist_norm(180,), (r,y,g), valid_count
    """
    if bgr is None or bgr.size == 0:
        return np.zeros((h_bins,), np.float32), (0.0,0.0,0.0), 0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(hsv)
    mask = cv2.inRange(hsv, (0, s_min, v_min), (179, 255, 255))
    hvals = H[mask > 0].astype(np.float32)
    valid_count = int(hvals.size)

    hist = np.zeros((h_bins,), dtype=np.float32)
    if valid_count > 0:
        hist, _ = np.histogram(hvals, bins=h_bins, range=(0, 180))
        hist = hist.astype(np.float32)

    s = hist.sum()
    hist_norm = hist / s if s > 0 else hist

    r = hist_norm[red1[0]:red1[1]+1].sum() + hist_norm[red2[0]:red2[1]+1].sum()
    y = hist_norm[yel[0]:yel[1]+1].sum()
    g = hist_norm[grn[0]:grn[1]+1].sum()
    return hist_norm, (float(r), float(y), float(g)), valid_count

def draw_hue_hist_image(hist_norm,
                        h_bins=180, height=180, width=360,
                        red1=(0,5), red2=(168,179),
                        yel=(10,35), grn=(40,85)):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    if hist_norm.size == 0:
        return img
    def bx(h): return int(h * (width / h_bins))
    def fill(h0,h1,bgr,a=0.15):
        x0,x1 = bx(h0), bx(h1)
        overlay = img.copy()
        cv2.rectangle(overlay, (x0,0), (x1,height), bgr, -1)
        cv2.addWeighted(overlay, a, img, 1-a, 0, img)
    # 색 구간 밴드
    fill(red1[0], red1[1], (0,0,255))
    fill(red2[0], red2[1], (0,0,255))
    fill(yel[0],  yel[1],  (0,255,255))
    fill(grn[0],  grn[1],  (0,255,0))
    cv2.rectangle(img,(0,0),(width-1,height-1),(64,64,64),1)
    for h in range(h_bins):
        x = bx(h)
        bar = int(hist_norm[h]*(height-10))
        cv2.line(img, (x, height-1), (x, height-1-bar), (220,220,220), 1)
    cv2.putText(img, "Hue Hist (0..179)", (8,16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220,220,220), 1, cv2.LINE_AA)
    return img

class TLCropOnlyNode(Node):
    def __init__(self):
        super().__init__('tl_crop_only')
        self.bridge = CvBridge()

        # ---------------- Params ----------------
        # I/O
        self.declare_parameter('model_path', 'model/yolo11s.pt')
        self.declare_parameter('image_topic', '/zed/zed_node/left/image_rect_color')
        self.declare_parameter('pub_hist_image', False)
        self.declare_parameter('show_windows', False)

        # ROI
        self.declare_parameter('roi_top_ratio',    0.00)
        self.declare_parameter('roi_bottom_ratio', 0.50)
        self.declare_parameter('roi_left_ratio',   0.25)
        self.declare_parameter('roi_right_ratio',  0.75)

        # YOLO 박스 필터
        self.declare_parameter('min_side_px', 10)
        self.declare_parameter('min_area_px', 150)
        self.declare_parameter('aspect_min', 1.15)
        self.declare_parameter('aspect_max', 3.00)
        self.declare_parameter('edge_margin_px', 2)

        # Hue/마스크
        self.declare_parameter('s_min', 60)
        self.declare_parameter('v_min', 70)
        self.declare_parameter('min_valid_pixels', 1)
        self.declare_parameter('score_threshold', 0.20)

        # 모폴로지
        self.declare_parameter('morph_close_iter', 1)
        self.declare_parameter('morph_open_iter',  0)
        self.declare_parameter('dilate_iter',      1)
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        # 스무딩/전이 제어
        self.declare_parameter('ema_tau_s', 0.25)
        self.declare_parameter('majority_k', 5)
        self.declare_parameter('hold_ms', 300)
        self.declare_parameter('hysteresis_delta', 0.08)
        self.declare_parameter('missing_timeout_ms', 400)

        # Hue 구간
        self.declare_parameter('red_lo1', 0)
        self.declare_parameter('red_hi1', 5)
        self.declare_parameter('red_lo2', 168)
        self.declare_parameter('red_hi2', 179)
        self.declare_parameter('yel_lo', 10)
        self.declare_parameter('yel_hi', 35)
        self.declare_parameter('grn_lo', 40)
        self.declare_parameter('grn_hi', 90)

        self.declare_parameter('max_h_w_ratio', 1.5)
        self.max_h_w_ratio = float(self.get_parameter('max_h_w_ratio').value)

        # --- 좌회전(R+G) 룰 파라미터 ---
        self.declare_parameter('lt_rule_enable', True)
        self.declare_parameter('lt_r_min', 0.50)
        self.declare_parameter('lt_g_min', 0.03)
        self.declare_parameter('lt_y_max', 0.03)
        self.declare_parameter('lt_valid_min', 20)

        self.lt_rule_enable = bool(self.get_parameter('lt_rule_enable').value)
        self.lt_r_min  = float(self.get_parameter('lt_r_min').value)
        self.lt_g_min  = float(self.get_parameter('lt_g_min').value)
        self.lt_y_max  = float(self.get_parameter('lt_y_max').value)
        self.lt_valid_min = int(self.get_parameter('lt_valid_min').value)

        # Fetch params
        self.model_path  = self.get_parameter('model_path').value
        self.image_topic = self.get_parameter('image_topic').value
        self.pub_hist_image = bool(self.get_parameter('pub_hist_image').value)
        self.show_windows   = bool(self.get_parameter('show_windows').value)

        self.roi_top     = float(self.get_parameter('roi_top_ratio').value)
        self.roi_bottom  = float(self.get_parameter('roi_bottom_ratio').value)
        self.roi_left    = float(self.get_parameter('roi_left_ratio').value)
        self.roi_right   = float(self.get_parameter('roi_right_ratio').value)

        self.min_side_px = int(self.get_parameter('min_side_px').value)
        self.min_area_px = int(self.get_parameter('min_area_px').value)
        self.aspect_min  = float(self.get_parameter('aspect_min').value)
        self.aspect_max  = float(self.get_parameter('aspect_max').value)
        self.edge_margin = int(self.get_parameter('edge_margin_px').value)

        self.s_min = int(self.get_parameter('s_min').value)
        self.v_min = int(self.get_parameter('v_min').value)
        self.min_valid_pixels = int(self.get_parameter('min_valid_pixels').value)
        self.score_threshold  = float(self.get_parameter('score_threshold').value)

        self.morph_close_iter = int(self.get_parameter('morph_close_iter').value)
        self.morph_open_iter  = int(self.get_parameter('morph_open_iter').value)
        self.dilate_iter      = int(self.get_parameter('dilate_iter').value)

        self.ema_tau_s  = float(self.get_parameter('ema_tau_s').value)
        self.majority_k = int(self.get_parameter('majority_k').value)
        self.hold_ms    = int(self.get_parameter('hold_ms').value)
        self.delta_hys  = float(self.get_parameter('hysteresis_delta').value)
        self.missing_timeout_ms = int(self.get_parameter('missing_timeout_ms').value)

        self.red_lo1 = int(self.get_parameter('red_lo1').value)
        self.red_hi1 = int(self.get_parameter('red_hi1').value)
        self.red_lo2 = int(self.get_parameter('red_lo2').value)
        self.red_hi2 = int(self.get_parameter('red_hi2').value)
        self.yel_lo  = int(self.get_parameter('yel_lo').value)
        self.yel_hi  = int(self.get_parameter('yel_hi').value)
        self.grn_lo  = int(self.get_parameter('grn_lo').value)
        self.grn_hi  = int(self.get_parameter('grn_hi').value)

        # ---------------- Pub/Sub ----------------
        qos = QoSProfile(
            depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST
        )
        self.sub = self.create_subscription(Image, self.image_topic, self.cb, qos)
        self.pub_roi_dbg = self.create_publisher(Image, '/tl/debug_image', 10)
        self.pub_zoom    = self.create_publisher(Image, '/tl/zoom_image', 10)
        self.pub_state   = self.create_publisher(Int32, '/tl/state', 10)
        self.pub_hist    = self.create_publisher(Image, '/tl/hist_image', 10) # del

        # ---------------- YOLO ----------------
        self.get_logger().info(f'Loading YOLO model: {self.model_path}')
        self.model = YOLO(self.model_path)

        # ---------------- State Vars ----------------
        self.margin = 3
        self.class_names = None
        try:
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'names'):
                self.class_names = self.model.model.names
        except Exception:
            pass

        self.last_time_ns = self.now_ns()
        self.ema = np.zeros(3, dtype=np.float32)          # [r,y,g]
        self.majority_buf = deque(maxlen=self.majority_k)
        self.current_state = 0
        self.last_change_ns = self.now_ns()
        self.last_seen_box_ns = self.now_ns()

    # ---------- Utilities ----------
    def now_ns(self):
        return int(self.get_clock().now().nanoseconds)

    def ns_to_ms(self, ns):
        return ns / 1e6

    def enhance_light_with_morph(self, img_bgr):
        if img_bgr is None or img_bgr.size == 0:
            return img_bgr
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, self.s_min, self.v_min), (179, 255, 255))
        if self.morph_close_iter > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel,
                                    iterations=self.morph_close_iter)
        if self.morph_open_iter > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.morph_kernel,
                                    iterations=self.morph_open_iter)
        if self.dilate_iter > 0:
            mask = cv2.dilate(mask, self.morph_kernel, iterations=self.dilate_iter)
        fg = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
        return fg

    def _yolo_boxes_filtered(self, roi, results):
        H, W = roi.shape[:2]
        boxes = []
        for r in results:
            if r.boxes is None:
                continue
            for b in r.boxes:
                if b.xyxy is None:
                    continue
                xA, yA, xB, yB = b.xyxy[0].cpu().numpy().astype(int)
                w = xB - xA
                h = yB - yA
                if w <= 0 or h <= 0:
                    continue
                if min(w, h) < self.min_side_px:
                    continue
                if w * h < self.min_area_px:
                    continue
                if xA <= self.edge_margin or yA <= self.edge_margin \
                   or (W - xB) <= self.edge_margin or (H - yB) <= self.edge_margin:
                    continue
                conf = float(b.conf[0]) if b.conf is not None else 0.0
                boxes.append((conf, (xA, yA, xB, yB)))
        return boxes

    def _select_box(self, roi, results):
        boxes = self._yolo_boxes_filtered(roi, results)
        if not boxes:
            return None

        H, W = roi.shape[:2]

        upper = []   # 상반(위쪽) 후보
        scored = []  # 전체 후보 (스코어 포함)

        for conf, (xA, yA, xB, yB) in boxes:
            w = max(1, xB - xA)
            h = max(1, yB - yA)

            # 세로 위치 가중치: 위일수록 큼
            cy = 0.5 * (yA + yB)
            cy_n = float(cy) / float(max(1, H))   # 0(위) ~ 1(아래)
            vy = 1.0 - cy_n                       # 위쪽일수록 1에 가까움

            # 크기 가중치: 너무 작으면 패널티
            area_n = (w * h) / float(max(1, W * H))  # 0~1

            # 최종 점수: conf * (위쪽 선호) * (크기 선호)
            score = float(conf) * (0.5 + 0.5 * vy) * (0.5 + 0.5 * area_n)

            item = (score, (xA, yA, xB, yB))
            scored.append(item)
            if cy_n <= 0.5:  # 상반에 있으면 upper 리스트에 별도로 보관
                upper.append(item)

        # 상반에 후보가 하나라도 있으면 상반에서만 선택, 없으면 전체에서 선택
        candidates = upper if upper else scored
        candidates.sort(key=lambda t: t[0], reverse=True)
        return candidates[0][1]


    # ---------- Main Callback ----------
    def cb(self, msg: Image):
        t_ns = self.now_ns()
        dt_s = max(1e-6, (t_ns - self.last_time_ns) / 1e9)
        self.last_time_ns = t_ns

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge error: {e}')
            return

        H, W = frame.shape[:2]
        x0 = int(W * self.roi_left);   x1 = int(W * self.roi_right)
        y0 = int(H * self.roi_top);    y1 = int(H * self.roi_bottom)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        if x1 <= x0 or y1 <= y0:
            return

        roi = frame[y0:y1, x0:x1].copy()
        roi_dbg = roi.copy()

        try:
            results = self.model.predict(roi, classes=[9], verbose=False)
        except Exception as e:
            self.get_logger().warn(f'YOLO predict error: {e}')
            return

        sel = self._select_box(roi, results)
        has_box = sel is not None
        if has_box:
            self.last_seen_box_ns = t_ns
            xA, yA, xB, yB = sel
            cv2.rectangle(roi_dbg, (xA, yA), (xB, yB), (0, 255, 0), 2)

            xA = clamp(xA + self.margin, 0, roi.shape[1]-1)
            yA = clamp(yA + self.margin, 0, roi.shape[0]-1)
            xB = clamp(xB - self.margin, 0, roi.shape[1]-1)
            yB = clamp(yB - self.margin, 0, roi.shape[0]-1)
            if xB <= xA or yB <= yA:
                has_box = False

        if has_box:
            zoom = roi[yA:yB, xA:xB].copy()
            zoom = self.enhance_light_with_morph(zoom)
            zoom = pad_to_3x1_keep_scale(zoom, pad_color=(0,0,0))

            hist_norm, (r_score, y_score, g_score), valid = hue_hist_and_scores(
                zoom,
                s_min=self.s_min, v_min=self.v_min,
                h_bins=180,
                red1=(self.red_lo1, self.red_hi1), red2=(self.red_lo2, self.red_hi2),
                yel=(self.yel_lo, self.yel_hi), grn=(self.grn_lo, self.grn_hi)
            )
        else:
            zoom = np.zeros((64, 64, 3), dtype=np.uint8)
            cv2.putText(zoom, "NO BOX", (5, 32), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1, cv2.LINE_AA)
            hist_norm = np.zeros((180,), np.float32)
            r_score = y_score = g_score = 0.0
            valid = 0

        hist_img = draw_hue_hist_image(
            hist_norm, h_bins=180, height=180, width=360,
            red1=(self.red_lo1, self.red_hi1), red2=(self.red_lo2, self.red_hi2),
            yel=(self.yel_lo, self.yel_hi), grn=(self.grn_lo, self.grn_hi)
        )
        if self.pub_hist_image:
            try:
                self.pub_hist.publish(self.bridge.cv2_to_imgmsg(hist_img, 'bgr8'))
            except Exception as e:
                self.get_logger().warn(f'pub hist_image: {e}')
        elif self.show_windows:
            cv2.imshow("Hue Histogram", hist_img)
            cv2.waitKey(1)

        print(f"R:{r_score:.2f}  Y:{y_score:.2f}  G:{g_score:.2f}  valid:{valid}  box:{has_box}")

        scores = np.array([r_score, y_score, g_score], dtype=np.float32)
        if self.ema_tau_s <= 1e-6:
            alpha = 1.0
        else:
            alpha = 1.0 - math.exp(-dt_s / self.ema_tau_s)
            alpha = float(clamp(alpha, 0.0, 1.0))
        self.ema = (1.0 - alpha) * self.ema + alpha * scores

        # ---------- 상태 판정 ----------
        proposed = 0  # 0=UNK, 1=R, 2=Y, 3=G, 4=R+G

        missing_ms = self.ns_to_ms(t_ns - self.last_seen_box_ns)
        if not has_box and missing_ms < self.missing_timeout_ms:
            proposed = self.current_state
        else:
            # --- 좌회전 룰 ---
            if (self.lt_rule_enable and has_box and
                valid >= self.lt_valid_min and
                r_score >= self.lt_r_min and
                g_score >= self.lt_g_min and
                y_score <= self.lt_y_max):
                proposed = 4
            else:
                if valid >= self.min_valid_pixels and float(scores.max()) >= self.score_threshold:
                    idx = int(np.argmax(self.ema))
                    proposed = [1,2,3][idx]
                else:
                    proposed = 0

        self.majority_buf.append(proposed)
        if len(self.majority_buf) > 0:
            vals, counts = np.unique(self.majority_buf, return_counts=True)
            maj = int(vals[np.argmax(counts)])
        else:
            maj = proposed

        new_state = self.current_state
        time_since_change_ms = self.ns_to_ms(t_ns - self.last_change_ns)

        if maj != self.current_state:
            can_change = (time_since_change_ms >= self.hold_ms)
            if can_change:
                if maj == 0:
                    new_state = 0
                else:
                    cur_idx = {1:0, 2:1, 3:2}.get(self.current_state, None)
                    new_idx = {1:0, 2:1, 3:2, 4:None}.get(maj, None)
                    cur_val = self.ema[cur_idx] if cur_idx is not None else 0.0
                    new_val = self.ema[new_idx] if (new_idx is not None) else 0.0
                    if maj == 4:
                        new_state = 4
                    elif (new_val - cur_val) >= self.delta_hys:
                        new_state = maj

        if new_state != self.current_state:
            self.current_state = new_state
            self.last_change_ns = t_ns

        try:
            self.pub_roi_dbg.publish(self.bridge.cv2_to_imgmsg(roi_dbg, 'bgr8'))
        except Exception as e:
            self.get_logger().warn(f'pub debug_image: {e}')
        try:
            self.pub_zoom.publish(self.bridge.cv2_to_imgmsg(zoom, 'bgr8'))
        except Exception as e:
            self.get_logger().warn(f'pub zoom_image: {e}')
        try:
            self.pub_state.publish(Int32(data=int(self.current_state)))
        except Exception as e:
            self.get_logger().warn(f'pub state_id: {e}')

def main():
    rclpy.init()
    node = TLCropOnlyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    if node.show_windows:
        try:
            cv2.destroyAllWindows()
        except:
            pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
