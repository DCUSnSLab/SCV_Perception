#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ---- hard switch: make cv_bridge bind to system NumPy (1.x), then restore user site ----
import sys, os, glob

MAJOR, MINOR = sys.version_info[:2]
HOME = os.path.expanduser("~")
USR_BASE = os.path.join(HOME, ".local", "lib", f"python{MAJOR}.{MINOR}")
USR_SITE = os.path.join(USR_BASE, "site-packages")
SYS_DIST = "/usr/lib/python3/dist-packages"   # Ubuntu/ROS Humble의 기본 dist-packages

# 1) 시스템 dist-packages 우선
if SYS_DIST not in sys.path:
    sys.path.insert(0, SYS_DIST)

# 2) 사용자 site 경로 일단 전부 제거 (NumPy 2.x 차단)
_removed_user_paths = []
for p in list(sys.path):
    if p.startswith(USR_BASE):
        _removed_user_paths.append(p)
        sys.path.remove(p)

# 3) 이제 cv_bridge를 로드 (NumPy 1.x에 결속)
from cv_bridge import CvBridge  # <-- 여기서 numpy 1.x을 사용하게 강제

# 4) 사용자 site 경로 복원 (YOLO/torch는 ~/.local 것을 쓰게)
for p in [USR_SITE] + sorted(set(_removed_user_paths) - {USR_SITE}):
    if p not in sys.path:
        sys.path.append(p)

# --------------------------------------------------------------------
import rclpy
from rclpy.node import Node
try:
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
except ImportError:
    # Humble에서의 백워드 호환
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy as ReliabilityPolicy, QoSHistoryPolicy as HistoryPolicy
    try:
        from rclpy.qos import QoSDurabilityPolicy as DurabilityPolicy
    except Exception:
        DurabilityPolicy = None

from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Int32
from ultralytics import YOLO
import torch, cv2

try:
    from ament_index_python.packages import get_package_share_directory
except Exception:
    get_package_share_directory = None

# ---------------------- 라벨 정규화 ----------------------
def norm_label(s: str) -> str:
    if not s:
        return ""
    t = s.strip().lower().replace('-', '_').replace(' ', '')
    if 'red' in t: return 'red'
    if 'yellow' in t or 'amber' in t: return 'yellow'
    if 'green' in t or 'blue' in t: return 'green'
    if 'left' in t and ('arrow' in t or 'turn' in t): return 'left'
    if t == 'left': return 'left'
    return t

class RoiInfer(Node):
    def __init__(self):
        super().__init__('roi_infer')

        # -------- default model path --------
        default_model_path = 'best.pt'
        try:
            if get_package_share_directory is not None:
                share_dir = get_package_share_directory('tl_roi_infer')
                cand = os.path.join(share_dir, 'models', 'best.pt')
                if os.path.exists(cand):
                    default_model_path = cand
        except Exception:
            pass

        # -------- params --------
        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('image_topic', '/zed_node/left/image_rect_color')
        self.declare_parameter('debug_topic', '/tl/debug_image')
        self.declare_parameter('state_id_topic', '/tl/state_id')

        self.declare_parameter('imgsz', 1088)
        self.declare_parameter('conf', 0.25)
        self.declare_parameter('iou', 0.50)
        self.declare_parameter('agnostic_nms', False)
        self.declare_parameter('half', False)

        # ROI (비율)
        self.declare_parameter('roi_top_ratio', 0.00)
        self.declare_parameter('roi_bottom_ratio', 0.35)
        self.declare_parameter('roi_left_ratio', 0.25)
        self.declare_parameter('roi_right_ratio', 0.75)

        default_device = 0 if torch.cuda.is_available() else 'cpu'
        self.declare_parameter('device', default_device)

        # ----- 시간 저역통과 + 히스테리시스/데바운스 -----
        self.declare_parameter('lp_alpha_rise', 0.30)
        self.declare_parameter('lp_alpha_decay', 0.06)
        self.declare_parameter('lp_tau_hi', 0.60)
        self.declare_parameter('lp_tau_lo', 0.45)
        self.declare_parameter('lp_hold_ms', 400)
        self.declare_parameter('lp_margin', 0.10)
        self.declare_parameter('lp_unknown_timeout_ms', 1500)
        self.declare_parameter('lp_acquire_ms', 400)

        # ----- 이미지 LPF 옵션 -----
        self.declare_parameter('img_lpf_enable', False)
        self.declare_parameter('img_lpf_type', 'gaussian')
        self.declare_parameter('img_lpf_ksize', 3)
        self.declare_parameter('img_lpf_sigma', 0.8)
        self.declare_parameter('img_lpf_roi_only', True)
        self.declare_parameter('img_lpf_bilateral_sigma_color', 50.0)
        self.declare_parameter('img_lpf_bilateral_sigma_space', 3.0)

        # ----- (NEW) 디버그 경량화 옵션 -----
        self.declare_parameter('debug_rate_hz', 8.0)         # 0=매프레임
        self.declare_parameter('debug_scale', 1.0)           # ROI 디버그 리사이즈
        self.declare_parameter('debug_draw_boxes', True)
        self.declare_parameter('debug_draw_labels', False)

        # ----- (NEW) 상태 퍼블리시 제어(타이머/온체인지/킵얼라이브) -----
        self.declare_parameter('state_pub_rate_hz', 20.0)    # 타이머 틱 주기(Hz)
        self.declare_parameter('state_pub_on_change', True)  # 변경 시에만 전송
        self.declare_parameter('state_keepalive_ms', 1000)   # 변경 없을 때 keepalive 간격(ms), 0이면 keepalive 없음

        g = lambda k: self.get_parameter(k).value
        self.model_path = str(g('model_path'))
        self.image_topic = str(g('image_topic'))
        self.debug_topic = str(g('debug_topic'))
        self.state_id_topic = str(g('state_id_topic'))
        self.imgsz = int(g('imgsz'))
        self.conf = float(g('conf')); self.iou = float(g('iou'))
        self.agnostic_nms = bool(g('agnostic_nms')); self.half = bool(g('half'))

        # ROI clamp
        self.r_top    = max(0.0, min(float(g('roi_top_ratio')),    1.0))
        self.r_bottom = max(0.0, min(float(g('roi_bottom_ratio')), 1.0))
        self.r_left   = max(0.0, min(float(g('roi_left_ratio')),   1.0))
        self.r_right  = max(0.0, min(float(g('roi_right_ratio')),  1.0))

        dev = g('device')
        self.device = dev if isinstance(dev, str) else (dev if (dev >= 0 and torch.cuda.is_available()) else 'cpu')

        # 필터 파라미터
        self.lp_alpha_rise = float(g('lp_alpha_rise'))
        self.lp_alpha_decay = float(g('lp_alpha_decay'))
        self.lp_tau_hi = float(g('lp_tau_hi'))
        self.lp_tau_lo = float(g('lp_tau_lo'))
        self.lp_hold_ms = int(g('lp_hold_ms'))
        self.lp_unknown_timeout_ms = int(g('lp_unknown_timeout_ms'))
        self.lp_margin = float(g('lp_margin'))
        self.lp_acquire_ms = int(g('lp_acquire_ms'))

        # 이미지 LPF 파라미터
        self.img_lpf_enable   = bool(g('img_lpf_enable'))
        self.img_lpf_type     = str(g('img_lpf_type')).lower()
        self.img_lpf_ksize    = int(g('img_lpf_ksize'))
        self.img_lpf_sigma    = float(g('img_lpf_sigma'))
        self.img_lpf_roi_only = bool(g('img_lpf_roi_only'))
        self.img_lpf_bilat_sc = float(g('img_lpf_bilateral_sigma_color'))
        self.img_lpf_bilat_ss = float(g('img_lpf_bilateral_sigma_space'))

        # 디버그 옵션
        self.debug_rate_hz     = float(g('debug_rate_hz'))
        self.debug_scale       = float(g('debug_scale'))
        self.debug_draw_boxes  = bool(g('debug_draw_boxes'))
        self.debug_draw_labels = bool(g('debug_draw_labels'))
        self._dbg_period_ns = int(1e9 / self.debug_rate_hz) if self.debug_rate_hz > 0 else 0
        self._dbg_last_pub  = self.get_clock().now()

        # 상태 퍼블리시 제어
        self.state_pub_rate_hz   = float(g('state_pub_rate_hz'))
        self.state_pub_on_change = bool(g('state_pub_on_change'))
        self.state_keepalive_ms  = int(g('state_keepalive_ms'))

        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)

        # 카메라 구독 QoS (BEST_EFFORT)
        qos_sub = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                             history=HistoryPolicy.KEEP_LAST, depth=5)

        # 토픽 타입 자동 판별
        types = dict(self.get_topic_names_and_types()).get(self.image_topic, [])
        if 'sensor_msgs/msg/CompressedImage' in types:
            MsgT = CompressedImage
            self.decode = lambda m: self.bridge.compressed_imgmsg_to_cv2(m)
            self.get_logger().info(f'{self.image_topic} -> CompressedImage')
        else:
            MsgT = Image
            self.decode = lambda m: self.bridge.imgmsg_to_cv2(m, desired_encoding='bgr8')
            self.get_logger().info(f'{self.image_topic} -> Image (RAW assumed)')

        self.sub = self.create_subscription(MsgT, self.image_topic, self.cb, qos_sub)

        # 디버그 퍼블리셔
        self.pub_dbg = self.create_publisher(Image, self.debug_topic, 10)

        # 상태 퍼블리셔 (RELIABLE, (가능하면) TRANSIENT_LOCAL)
        qos_state = QoSProfile(depth=10)
        try:
            qos_state.reliability = ReliabilityPolicy.RELIABLE
            if DurabilityPolicy:
                qos_state.durability = DurabilityPolicy.TRANSIENT_LOCAL
        except Exception:
            pass
        self.pub_state_id = self.create_publisher(Int32, self.state_id_topic, qos_state)

        # 상태 ID 매핑
        self.label2id = {'red': 1, 'yellow': 2, 'green': 3, 'left': 4}

        # 상태 메모리
        self.scores = {k: 0.0 for k in self.label2id.keys()}
        self.cur_label = 'none'
        now = self.get_clock().now()
        self.last_change = now
        self.last_seen = now

        # 최초 확정 후보
        self.cand_label = None
        self.cand_since = None

        # 상태 타이머용 버퍼
        self._state_last_id   = 0
        self._state_changed   = True     # 시작 시 한 번 내보내도록
        self._state_last_sent = self.get_clock().now()

        # 상태 타이머 시작
        period = 1.0 / self.state_pub_rate_hz if self.state_pub_rate_hz > 0 else 0.05
        self.create_timer(period, self._state_timer_cb)

        devname = f'cuda:{self.device}' if isinstance(self.device, int) else self.device
        self.get_logger().info(f'Loaded: {self.model_path} | device={devname} imgsz={self.imgsz} conf={self.conf} iou={self.iou} half={self.half}')
        self.get_logger().info(f'ROI ratios t/b/l/r = {self.r_top}/{self.r_bottom}/{self.r_left}/{self.r_right}')
        self.get_logger().info(f'Publishing debug to: {self.debug_topic}')
        self.get_logger().info(f'Publishing state_id to: {self.state_id_topic}')

    # -------- 상태 타이머 콜백: 온체인지 + 킵얼라이브 --------
    def _state_timer_cb(self):
        now = self.get_clock().now()
        # 온체인지 모드면 변화 없을 땐 스킵 (단, keepalive 주기마다 1회 전송)
        if self.state_pub_on_change and not self._state_changed:
            if self.state_keepalive_ms <= 0:
                return
            if (now - self._state_last_sent).nanoseconds < self.state_keepalive_ms * 1e6:
                return

        m_id = Int32()
        m_id.data = self._state_last_id
        self.pub_state_id.publish(m_id)
        self._state_last_sent = now
        self._state_changed = False

    # -------- 이미지 LPF 유틸 --------
    def _odd(self, k: int) -> int:
        k = int(max(1, k))
        return k if (k % 2 == 1) else (k + 1)

    def _apply_lpf(self, img):
        if not self.img_lpf_enable:
            return img
        t = self.img_lpf_type
        k = self._odd(self.img_lpf_ksize)
        if k < 3 and t != 'gaussian':
            return img
        if t == 'gaussian':
            k = max(3, k)
            return cv2.GaussianBlur(img, (k, k), self.img_lpf_sigma)
        elif t == 'median':
            return cv2.medianBlur(img, k)
        elif t == 'bilateral':
            return cv2.bilateralFilter(img, k, self.img_lpf_bilat_sc, self.img_lpf_bilat_ss)
        elif t == 'box':
            return cv2.blur(img, (k, k))
        else:
            return img

    # -------- 카메라 콜백 --------
    def cb(self, msg):
        frame = self.decode(msg)
        if frame is None:
            return
        H, W = frame.shape[:2]

        # ROI 자르기
        x1 = int(W * self.r_left);  x2 = int(W * self.r_right)
        y1 = int(H * self.r_top);   y2 = int(H * self.r_bottom)
        x1 = max(0, min(x1, W-1));  x2 = max(x1+1, min(x2, W))
        y1 = max(0, min(y1, H-1));  y2 = max(y1+1, min(y2, H))
        roi = frame[y1:y2, x1:x2]

        # ROI에만 LPF(옵션)
        if self.img_lpf_enable:
            if self.img_lpf_roi_only:
                roi_in = self._apply_lpf(roi)
            else:
                filt = self._apply_lpf(frame)
                roi_in = filt[y1:y2, x1:x2]
        else:
            roi_in = roi

        # 추론
        r = self.model(
            roi_in,
            imgsz=self.imgsz, conf=self.conf, iou=self.iou,
            agnostic_nms=self.agnostic_nms, device=self.device,
            half=(self.half and self.device != 'cpu'),
            verbose=False
        )[0]

        # --- 프레임 점수 집계 ---
        frame_score = {k: 0.0 for k in self.scores.keys()}
        if len(r.boxes):
            names = r.names
            items = []
            for b in r.boxes:
                lab_raw = names[int(b.cls)]
                lab = norm_label(lab_raw)
                cf = float(b.conf)
                items.append(f"{lab_raw}:{cf:.2f}")
                if lab in frame_score:
                    frame_score[lab] = max(frame_score[lab], cf)
            self.get_logger().info("Det: " + ", ".join(items))
            self.last_seen = self.get_clock().now()

        # --- 상승/하강 분리 EWMA ---
        for k in self.scores:
            a = self.lp_alpha_rise if frame_score[k] >= self.scores[k] else self.lp_alpha_decay
            self.scores[k] = (1.0 - a) * self.scores[k] + a * frame_score[k]

        # --- 상태 결정 ---
        best_lab, best_val = max(self.scores.items(), key=lambda kv: kv[1])
        now = self.get_clock().now()
        ms_since_change = (now - self.last_change).nanoseconds / 1e6
        ms_since_seen   = (now - self.last_seen).nanoseconds / 1e6

        # 1) 관측 끊김 → unknown
        if self.cur_label != 'none':
            if best_val < self.lp_tau_lo and ms_since_seen >= self.lp_unknown_timeout_ms:
                self.get_logger().info(f"-> UNKNOWN (timeout). best={best_lab}:{best_val:.2f}, last_seen={ms_since_seen:.0f}ms")
                self.cur_label = 'none'
                self.last_change = now
                self.cand_label = None
                self.cand_since = None

        # 2) none → 후보 지속 확인 후 확정
        if self.cur_label == 'none':
            if best_val >= self.lp_tau_hi:
                if self.cand_label == best_lab:
                    ms_cand = (now - self.cand_since).nanoseconds / 1e6 if self.cand_since else 0.0
                    if ms_cand >= self.lp_acquire_ms:
                        self.cur_label = best_lab
                        self.last_change = now
                        self.get_logger().info(f"ACQUIRE {self.cur_label} (sustained {ms_cand:.0f}ms, val={best_val:.2f})")
                        self.cand_label = None
                        self.cand_since = None
                else:
                    self.cand_label = best_lab
                    self.cand_since = now
                    self.get_logger().info(f"candidate: {self.cand_label} start (val={best_val:.2f})")
            else:
                if self.cand_label is not None:
                    self.get_logger().info("candidate reset (below tau_hi)")
                self.cand_label = None
                self.cand_since = None

        # 3) 라벨 보유 중 → 홀드/마진/임계 충족 시 전환
        else:
            cur_val = self.scores.get(self.cur_label, 0.0)
            if best_lab != self.cur_label and ms_since_change >= self.lp_hold_ms \
               and best_val >= self.lp_tau_hi and (best_val >= cur_val + self.lp_margin):
                self.get_logger().info(f"SWITCH {self.cur_label} -> {best_lab} (best={best_val:.2f} >= cur+margin={cur_val+self.lp_margin:.2f})")
                self.cur_label = best_lab
                self.last_change = now
                self.cand_label = None
                self.cand_since = None

        # ---- 상태 ID 버퍼 업데이트 (타이머가 퍼블리시) ----
        new_id = self.label2id.get(self.cur_label, 0)
        if new_id != self._state_last_id:
            self._state_last_id = new_id
            self._state_changed = True

        # ======= 디버그: ROI만 경량 박스/텍스트 =======
        if self._dbg_period_ns == 0 or (now - self._dbg_last_pub).nanoseconds >= self._dbg_period_ns:
            dbg = roi_in  # in-place
            if self.debug_draw_boxes and len(r.boxes):
                def _color(lab):
                    return {
                        'red':   (0, 0, 255),
                        'yellow':(0, 255, 255),
                        'green': (0, 255, 0),
                        'left':  (255, 0, 0),
                    }.get(lab, (255, 255, 255))
                names = r.names
                xyxy = r.boxes.xyxy
                cls  = r.boxes.cls
                conf = r.boxes.conf
                xyxy = xyxy.cpu().numpy() if hasattr(xyxy, 'cpu') else xyxy
                cls  = cls.cpu().numpy().astype(int) if hasattr(cls, 'cpu') else cls
                conf = conf.cpu().numpy() if hasattr(conf, 'cpu') else conf
                h, w = dbg.shape[:2]
                for i in range(len(xyxy)):
                    x1b, y1b, x2b, y2b = xyxy[i].astype(int).tolist()
                    x1b = max(0, min(x1b, w-1));  x2b = max(0, min(x2b, w-1))
                    y1b = max(0, min(y1b, h-1));  y2b = max(0, min(y2b, h-1))
                    lab_raw = names[cls[i]] if 0 <= cls[i] < len(names) else ""
                    lab = norm_label(lab_raw)
                    color = _color(lab)
                    cv2.rectangle(dbg, (x1b, y1b), (x2b, y2b), color, 2)
                    if self.debug_draw_labels:
                        score = conf[i] if conf is not None else None
                        txt = lab if lab else (lab_raw or "")
                        if score is not None:
                            txt = f"{txt}:{score:.2f}" if txt else f"{score:.2f}"
                        if txt:
                            ytxt = max(0, y1b - 5)
                            cv2.putText(dbg, txt, (x1b, ytxt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            if self.debug_scale and abs(self.debug_scale - 1.0) > 1e-3:
                new_w = max(1, int(dbg.shape[1] * self.debug_scale))
                new_h = max(1, int(dbg.shape[0] * self.debug_scale))
                dbg = cv2.resize(dbg, (new_w, new_h), interpolation=cv2.INTER_AREA)

            out = self.bridge.cv2_to_imgmsg(dbg, encoding='bgr8')
            out.header = msg.header
            self.pub_dbg.publish(out)
            self._dbg_last_pub = now

def main():
    rclpy.init()
    node = RoiInfer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
