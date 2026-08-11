#!/usr/bin/python3
import os
import time
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge

# --------- OPTIONAL linear assignment (SciPy -> LAP -> greedy fallback) ---------
def hungarian_pairs(cost):
    # cost: np.ndarray (N, M)
    try:
        from scipy.optimize import linear_sum_assignment
        ri, ci = linear_sum_assignment(cost)
        return list(zip(ri, ci))
    except Exception:
        try:
            from lap import lapjv
            _, x, _ = lapjv(cost)
            return [(i, j) for i, j in enumerate(x) if j != -1]
        except Exception:
            # greedy fallback
            pairs = []
            used_r, used_c = set(), set()
            idx = [(i, j, cost[i, j]) for i in range(cost.shape[0]) for j in range(cost.shape[1])]
            idx.sort(key=lambda t: t[2])
            for i, j, v in idx:
                if v >= 1e6:  # gated-out
                    continue
                if i in used_r or j in used_c:
                    continue
                used_r.add(i); used_c.add(j); pairs.append((i, j))
            return pairs

# ----------------- Utils -----------------
def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    au = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bu = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / (au + bu - inter + 1e-6))

def mask_iou(m1, m2):
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)

def xyxy_to_uvsr(b):
    x1, y1, x2, y2 = b
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    u = x1 + w / 2.0
    v = y1 + h / 2.0
    s = w * h
    r = w / h
    return np.array([u, v, s, r], dtype=float)

def uvsr_to_xyxy(x):
    u, v, s, r = x[0], x[1], max(1e-6, x[2]), max(1e-6, x[3])
    w = np.sqrt(s * r); h = s / w
    return np.array([u - w/2, v - h/2, u + w/2, v + h/2], dtype=float)

# ----------------- Kalman (CV, SORT 스타일) -----------------
class KFBox:
    """x=[u,v,s,r, du,dv,ds]; z=[u,v,s,r]"""
    def __init__(self, z, dt=1.0, q_scale=1e-2, r_scale=1e-1):
        self.x = np.zeros((7,1)); self.x[0:4,0] = z.reshape(4,)
        self.P = np.eye(7); self.P[4:,4:] *= 100.0
        self.F = np.eye(7); self.H = np.zeros((4,7))
        self.H[0,0] = self.H[1,1] = self.H[2,2] = self.H[3,3] = 1.0
        self.set_dt(dt)
        self.Q = np.diag([1,1,1,0.1,10,10,10]).astype(float) * q_scale
        self.R = np.diag([1,1,10,10]).astype(float) * r_scale
    def set_dt(self, dt):
        self.F[:] = np.eye(7)
        self.F[0,4]=dt; self.F[1,5]=dt; self.F[2,6]=dt
    def predict(self, dt=None):
        if dt is not None: self.set_dt(dt)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x
    def update(self, z):
        z = z.reshape(4,1)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(7); self.P = (I - K @ self.H) @ self.P
    def bbox(self):
        return uvsr_to_xyxy(self.x[:,0])

class Track:
    _next_id = 0
    def __init__(self, z_uvsr, dt, q_scale=1e-2, r_scale=1e-1, cls_id=-1):
        self.kf = KFBox(z_uvsr, dt=dt, q_scale=q_scale, r_scale=r_scale)
        self.id = Track._next_id; Track._next_id += 1
        self.cls_id = cls_id
        self.time_since_update = 0
        self.hits = 1
        self.age = 0
        self.last_mask = None
    def predict(self, dt):
        self.kf.predict(dt); self.age += 1; self.time_since_update += 1
        return self.kf.bbox()
    def update(self, z_uvsr):
        self.kf.update(z_uvsr); self.time_since_update = 0; self.hits += 1
    def bbox(self): return self.kf.bbox()

class MemorySORT:
    def __init__(self, iou_thr_active=0.3, iou_thr_memory=0.2,
                 max_age_active=0, min_hits=3, memory_ttl_frames=30,
                 q_scale=1e-2, r_scale=1e-1, class_match=False,
                 output_coasting=True, coast_inflate=0.05,
                 use_mask_assoc=False, output_masks=False):
        self.iou_thr_active = float(iou_thr_active)
        self.iou_thr_memory = float(iou_thr_memory)
        self.max_age_active = int(max_age_active)
        self.min_hits = int(min_hits)
        self.memory_ttl_frames = int(memory_ttl_frames)
        self.q_scale = float(q_scale)
        self.r_scale = float(r_scale)
        self.class_match = bool(class_match)
        self.output_coasting = bool(output_coasting)
        self.coast_inflate = float(coast_inflate)
        self.use_mask_assoc = bool(use_mask_assoc)
        self.output_masks = bool(output_masks)
        self.active = []
        self.dormant = []

    def _assoc(self, tracks, dets, det_classes, thr, class_match, det_masks=None, use_mask=False):
        N, M = len(tracks), len(dets)
        if N == 0 or M == 0:
            return [], list(range(N)), list(range(M))
        C = np.ones((N, M), dtype=float) * 1e6
        for i, t in enumerate(tracks):
            tb = t.bbox()
            tcls = t.cls_id
            tmask = getattr(t, 'last_mask', None)
            for j, db in enumerate(dets):
                if class_match:
                    dcls = det_classes[j]
                    if tcls != -1 and dcls != -1 and tcls != dcls:
                        continue
                if use_mask and (tmask is not None) and (det_masks is not None) and (det_masks[j] is not None):
                    iou = mask_iou(tmask, det_masks[j])
                else:
                    iou = iou_xyxy(tb, db)
                if iou >= thr:
                    C[i, j] = 1.0 - iou
        pairs = hungarian_pairs(C)
        matched, used_t, used_d = [], set(), set()
        for i, j in pairs:
            if C[i, j] < 1e6:
                matched.append((i, j)); used_t.add(i); used_d.add(j)
        ut = [i for i in range(N) if i not in used_t]
        ud = [j for j in range(M) if j not in used_d]
        return matched, ut, ud

    def update(self, dets_xyxy_cls, dt):
        dets_xyxy = [d[:4] for d in dets_xyxy_cls] if dets_xyxy_cls else []
        det_classes = [int(d[5]) for d in dets_xyxy_cls] if dets_xyxy_cls else []
        det_masks = [(d[6] if (len(d) > 6) else None) for d in dets_xyxy_cls] if dets_xyxy_cls else []

        # 1) predict
        for t in self.active:  t.predict(dt)
        for k in range(len(self.dormant)):
            tr, ttl, ttl0 = self.dormant[k]
            tr.kf.x[6, 0] = 0.0
            tr.kf.x[5, 0] *= 0.85
            tr.predict(dt)
            self.dormant[k] = (tr, ttl - 1, ttl0)

        # 2) active ↔ det
        matched, ut_idx, ud_idx = self._assoc(
            self.active, dets_xyxy, det_classes,
            self.iou_thr_active, self.class_match,
            det_masks=det_masks, use_mask=self.use_mask_assoc
        )

        # 3) update matched
        for ia, jd in matched:
            z = xyxy_to_uvsr(dets_xyxy[jd])
            self.active[ia].update(z)
            self.active[ia].last_mask = det_masks[jd]
            if self.active[ia].cls_id == -1:
                self.active[ia].cls_id = det_classes[jd]

        # 4) unmatched active → dormant
        new_active, moved_to_dormant = [], []
        for idx, t in enumerate(self.active):
            if idx in ut_idx: moved_to_dormant.append(t)
            else:             new_active.append(t)
        self.active = new_active
        for t in moved_to_dormant:
            self.dormant.append((t, self.memory_ttl_frames, self.memory_ttl_frames))

        # 5) dormant ↔ remaining det
        dormant_tracks = [tr for (tr, ttl, ttl0) in self.dormant if ttl > 0]
        matched2 = []
        if len(ud_idx) and len(dormant_tracks):
            dets_left = [dets_xyxy[j] for j in ud_idx]
            dcls_left = [det_classes[j] for j in ud_idx]
            dm_left  = [det_masks[j]   for j in ud_idx]
            m2, ut2, ud2 = self._assoc(
                dormant_tracks, dets_left, dcls_left,
                self.iou_thr_memory, self.class_match,
                det_masks=dm_left, use_mask=self.use_mask_assoc
            )
            for di, dj in m2:
                tr = dormant_tracks[di]
                real_j = ud_idx[dj]
                z = xyxy_to_uvsr(dets_xyxy[real_j])
                tr.update(z)
                tr.last_mask = det_masks[real_j]
                if tr.cls_id == -1: tr.cls_id = det_classes[real_j]
                self.active.append(tr)
                matched2.append((tr.id, real_j))
            revived_ids = set([tid for (tid, _) in matched2])
            new_dormant = []
            for (tr, ttl, ttl0) in self.dormant:
                if ttl > 0 and tr.id not in revived_ids:
                    new_dormant.append((tr, ttl, ttl0))
            self.dormant = new_dormant
            ud_idx = [ud_idx[j] for j in ud2]

        # 6) TTL expire
        self.dormant = [(tr, ttl, ttl0) for (tr, ttl, ttl0) in self.dormant if ttl > 0]

        # 7) new tracks
        for j in ud_idx:
            z = xyxy_to_uvsr(dets_xyxy[j])
            cls_id = det_classes[j] if det_classes else -1
            t = Track(z, dt=dt, q_scale=self.q_scale, r_scale=self.r_scale, cls_id=cls_id)
            t.last_mask = det_masks[j]
            self.active.append(t)

        # 8/9) output
        out = []
        for t in self.active:
            if (t.hits >= self.min_hits) or (t.age <= self.min_hits):
                x1,y1,x2,y2 = t.bbox()
                if self.output_masks:
                    out.append([float(x1),float(y1),float(x2),float(y2), int(t.id), int(t.cls_id), 0, t.last_mask])
                else:
                    out.append([float(x1),float(y1),float(x2),float(y2), int(t.id), int(t.cls_id), 0])
        if self.output_coasting:
            for (tr, ttl, ttl0) in self.dormant:
                x1,y1,x2,y2 = tr.bbox()
                cx=(x1+x2)/2; cy=(y1+y2)/2; w=max(1e-6,x2-x1); h=max(1e-6,y2-y1)
                coast_age = max(0, ttl0-ttl); scale = 1.0 + self.coast_inflate*coast_age
                w2=w*scale; h2=h*scale
                x1p=cx-w2/2; y1p=cy-h2/2; x2p=cx+w2/2; y2p=cy+h2/2
                if self.output_masks:
                    out.append([float(x1p),float(y1p),float(x2p),float(y2p), int(tr.id), int(tr.cls_id), 1, tr.last_mask])
                else:
                    out.append([float(x1p),float(y1p),float(x2p),float(y2p), int(tr.id), int(tr.cls_id), 1])
        return out

# ----------------- Detectors -----------------
class BaseDetector:
    def __init__(self, weights, imgsz=640, conf=0.30, iou=0.45, classes=None, device=None):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.imgsz = imgsz; self.conf = conf; self.iou = iou
        self.classes = set(classes) if classes else None
        self.device = device
    @staticmethod
    def _class_filter_ok(cls, allowed):
        return (allowed is None) or (cls in allowed)

class YOLOv11SegDetector(BaseDetector):
    def detect(self, frame_bgr):
        H,W = frame_bgr.shape[:2]
        r = self.model.predict(source=frame_bgr, imgsz=self.imgsz,
                               conf=self.conf, iou=self.iou, device=self.device, verbose=False)[0]
        dets=[]
        n = len(r.boxes) if r.boxes is not None else 0
        for i in range(n):
            x1,y1,x2,y2 = map(float, r.boxes.xyxy[i].tolist())
            conf = float(r.boxes.conf[i].item())
            cls  = int(r.boxes.cls[i].item())
            if not self._class_filter_ok(cls, self.classes): 
                continue
            m = None
            if r.masks is not None:
                mi = r.masks.data[i].cpu().numpy().astype(np.uint8)
                m = cv2.resize(mi, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            dets.append([x1,y1,x2,y2,conf,cls,m])
        return dets

class YOLOv8SegDetector(YOLOv11SegDetector):
    pass

class YOLOv11Detector(BaseDetector):
    def detect(self, frame_bgr):
        r = self.model.predict(source=frame_bgr, imgsz=self.imgsz,
                               conf=self.conf, iou=self.iou, device=self.device, verbose=False)[0]
        dets=[]
        if r.boxes is not None and len(r.boxes)>0:
            for b in r.boxes:
                x1,y1,x2,y2 = map(float, b.xyxy[0].tolist())
                conf=float(b.conf[0]); cls=int(b.cls[0])
                if not self._class_filter_ok(cls, self.classes):
                    continue
                dets.append([x1,y1,x2,y2,conf,cls])
        return dets

class YOLOv8Detector(YOLOv11Detector):
    pass

# ----------------- ROS2 Node -----------------
class MemorySortNode(Node):
    def __init__(self):
        super().__init__('memory_sort_node')
        self.bridge = CvBridge()

        # ---- declare params ----
        p = self.declare_parameter
        self.image_topic   = p('image_topic', '/camera/color/image_raw').value
        self.det           = p('det', 'v11').value                 # v11, v8, v11seg, v8seg
        self.weights       = p('weights', 'yolo11n.pt').value
        self.classes_str   = p('classes', 'person').value
        self.conf          = float(p('conf', 0.25).value)
        self.diou          = float(p('diou', 0.45).value)
        self.imgsz         = int(p('imgsz', 640).value)
        self.device        = p('device', None).value

        self.assoc_mask    = bool(p('assoc_mask', False).value)
        self.output_masks  = bool(p('output_masks', False).value)
        self.overlay_mask  = bool(p('overlay_mask', False).value)
        self.mask_alpha    = float(p('mask_alpha', 0.40).value)

        self.iou_thr_active= float(p('iou_thr_active', 0.2).value)
        self.iou_thr_memory= float(p('iou_thr_memory', 0.1).value)
        self.max_age_active= int(p('max_age_active', 0).value)
        self.min_hits      = int(p('min_hits', 1).value)
        self.memory_ttl_sec= float(p('memory_ttl_sec', 2.0).value)
        self.q_scale       = float(p('q_scale', 1e-2).value)
        self.r_scale       = float(p('r_scale', 1e-1).value)
        self.class_match   = bool(p('class_match', False).value)

        self.show_window   = bool(p('show_window', False).value)
        self.publish_overlay = bool(p('publish_overlay', True).value)
        self.out_fps       = float(p('out_fps', 0.0).value)
        self.save_path     = p('save_path', '').value

        # class filter
        self.COCO = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
                     'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse',
                     'sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase',
                     'frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard',
                     'surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana',
                     'apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
                     'potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone',
                     'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear',
                     'hair drier','toothbrush']
        self.class_filter = None
        if self.classes_str.strip():
            names = [s.strip() for s in self.classes_str.split(',') if s.strip()]
            cf = []
            for n in names:
                if n.isdigit(): cf.append(int(n))
                elif n in self.COCO: cf.append(self.COCO.index(n))
            self.class_filter = set(cf) if cf else None

        # detector
        self.detector = self._make_detector()

        # tracker (TTL=초→프레임)
        self.prev_stamp = None
        self.fps_est = 30.0
        self.video_writer = None
        self.vw_w = self.vw_h = None

        self.tracker = None  # 만들 때 FPS 필요

        # pubs/subs
        qos = QoSProfile(depth=1)
        qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
        qos.history = QoSHistoryPolicy.KEEP_LAST

        self.sub = self.create_subscription(Image, self.image_topic, self.cb_image, qos)
        self.pub_overlay = self.create_publisher(Image, 'overlay', 1) if self.publish_overlay else None

        self.get_logger().info(f'Listening: {self.image_topic}')
        self.get_logger().info(f'Detector: {self.det} weights={self.weights} classes={self.classes_str}')
        self.get_logger().info(f'show_window={self.show_window} save_path={self.save_path}')

    # ---- build detector ----
    def _make_detector(self):
        # 세그 가중치 자동 치환
        if self.det in ('v11seg','v8seg') and not self.weights.endswith('-seg.pt'):
            self.weights = self.weights.replace('.pt', '-seg.pt')
        kwargs = dict(weights=self.weights, imgsz=self.imgsz,
                      conf=self.conf, iou=self.diou,
                      classes=self.class_filter, device=self.device)
        if self.det == 'v11':
            return YOLOv11Detector(**kwargs)
        if self.det == 'v8':
            return YOLOv8Detector(**kwargs)
        if self.det == 'v11seg':
            return YOLOv11SegDetector(**kwargs)
        if self.det == 'v8seg':
            return YOLOv8SegDetector(**kwargs)
        raise RuntimeError(f'Unknown det type: {self.det}')

    def _ensure_tracker(self, fps_now):
        if self.tracker is not None:
            return
        ttl_frames = int(round(self.memory_ttl_sec * fps_now))
        self.tracker = MemorySORT(
            iou_thr_active=self.iou_thr_active,
            iou_thr_memory=self.iou_thr_memory,
            max_age_active=self.max_age_active,
            min_hits=self.min_hits,
            memory_ttl_frames=ttl_frames,
            q_scale=self.q_scale, r_scale=self.r_scale,
            class_match=self.class_match,
            output_coasting=True,
            coast_inflate=0.05,
            use_mask_assoc=self.assoc_mask,
            output_masks=self.output_masks
        )

    def cb_image(self, msg: Image):
        # dt 추정
        stamp = msg.header.stamp
        if self.prev_stamp is not None:
            dt = (stamp.sec - self.prev_stamp.sec) + (stamp.nanosec - self.prev_stamp.nanosec) * 1e-9
            if 0.001 < dt < 1.0:
                self.fps_est = 0.9*self.fps_est + 0.1*(1.0/max(dt,1e-6))
        self.prev_stamp = stamp

        # tracker 준비
        self._ensure_tracker(self.fps_est)
        dt_use = 1.0 / max(self.fps_est, 1e-6)

        # to cv2
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge convert failed: {e}')
            return
        H, W = frame.shape[:2]

        # VideoWriter 준비
        if self.save_path and (self.video_writer is None):
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps_out = self.out_fps if self.out_fps > 0 else max(1.0, min(240.0, self.fps_est))
            self.video_writer = cv2.VideoWriter(self.save_path, fourcc, fps_out, (W, H))
            self.vw_w, self.vw_h = W, H

        # detector
        dets = self.detector.detect(frame)
        if self.class_filter is not None:
            dets = [d for d in dets if int(d[5]) in self.class_filter]
        dets = [d for d in dets if d[4] >= self.conf]

        # tracker
        tracks = self.tracker.update(dets, dt=dt_use)

        # draw
        for item in tracks:
            # unpack basic data
            if len(item) >= 7:
                x1, y1, x2, y2, track_id, cls_id, is_pred = item[:7]
            else:
                x1, y1, x2, y2, track_id, cls_id = item
                is_pred = 0

            mask = item[7] if len(item) >= 8 else None

            # convert coordinates
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # color depending on prediction state
            if is_pred == 1:
                color = (128, 128, 128)   # gray for predicted (no detection)
                label = f'ID {int(track_id)} (pred)'
            else:
                color = (0, 255, 0)       # green for detection + tracking
                label = f'ID {int(track_id)}'

            # add class name (if available)
            if 0 <= int(cls_id) < len(self.COCO):
                label += f' {self.COCO[int(cls_id)]}'

            # draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # optional segmentation mask overlay
            if self.overlay_mask and isinstance(mask, np.ndarray) and mask.shape[:2] == frame.shape[:2]:
                overlay = frame.copy()
                rng = np.random.default_rng(int(cls_id) + 12345)
                mask_color = tuple(int(c) for c in rng.integers(80, 255, size=3))
                overlay[mask] = mask_color
                frame[:] = cv2.addWeighted(overlay, self.mask_alpha, frame, 1.0 - self.mask_alpha, 0)

        # publish overlay
        if self.pub_overlay is not None:
            out_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            out_msg.header = Header()
            out_msg.header.stamp = msg.header.stamp
            out_msg.header.frame_id = msg.header.frame_id
            self.pub_overlay.publish(out_msg)

        # save
        if self.video_writer is not None:
            if (W, H) != (self.vw_w, self.vw_h):
                # safety: size changed (rare)
                pass
            else:
                self.video_writer.write(frame)

        # window
        if self.show_window:
            if not hasattr(self, '_win_init'):
                cv2.namedWindow('Memory-SORT', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Memory-SORT', min(W, 1280), min(H, 720))
                self._win_init = True
            cv2.imshow('Memory-SORT', frame)
            cv2.waitKey(1)

    def destroy_node(self):
        if self.video_writer is not None:
            self.video_writer.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main():
    print("✅ memory_sort_node started")
    rclpy.init()
    node = MemorySortNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()