#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rock_loop.py - บอทวนทดสอบ: จุดมองหา -> หาหิน -> ตี -> จุดมองหา -> กล่อง(บ้าน) -> จุดมองหา -> วนซ้ำ

ขั้นตอน (ตามที่กำหนด):
  1) เดินไปใกล้ "พิกัดจุดมองหา" แล้วมองหาหิน (เลือกโทนสีหินตามช่วงเวลาในเกม จากรูปในโฟลเดอร์ icons)
  2) เจอหิน -> เดินเข้าไปชิดตัวละคร -> ตี
  3) ตีแล้วชื่อบนแถบตรงกับแร่ -> พิมพ์ "ตีสำเร็จ" -> กลับไปจุดมองหา
  4) เดินไปใกล้ "พิกัดกล่อง" -> ถ้าเห็นรูปมือขึ้น -> พิมพ์ "กลับบ้านสำเร็จ"
  5) กลับไปจุดมองหา -> วนอยู่อย่างนั้น

เป้าหมายของไฟล์นี้คือทดสอบ: อ่านเวลา/เลือกรูปหินตามช่วงเวลาถูกไหม (เช้า/เย็น)
ทุกครั้งที่หาหินจะพิมพ์ผลต่อโปรไฟล์ + เซฟ debug/time_log.csv ไว้ดูย้อนหลัง

ใส่พิกัดที่หัวไฟล์: SEARCH_COORD และ BOX_COORD (ถ้ายังเป็น None: ใช้ตำแหน่งปัจจุบันเป็นจุดมองหา / ข้ามขั้นไปกล่อง)

คำสั่ง:
  python rock_loop.py run                       # เริ่มวน
  python rock_loop.py run --max-rounds 3        # วนแค่ 3 รอบแล้วหยุด
  python rock_loop.py calibrate                 # จับภาพ 1 เฟรม -> debug/calibrate.png + พิมพ์สิ่งที่บอทอ่านได้
  python rock_loop.py calibrate --image ภาพ.png --landscape
  python rock_loop.py test-input joystick|hold|camera
"""
import argparse
import difflib
import glob
import json
import math
import os
import random
import subprocess
import sys
import time
from collections import Counter

import cv2
import numpy as np

try:
    import pytesseract
    HAS_TESS = True
except Exception:          # ยังไม่ได้ลง pytesseract/tesseract: บอททำงานได้ แต่อ่านชื่อ/นาฬิกาไม่ได้
    pytesseract = None
    HAS_TESS = False

# =============================================================================
# CONFIG  (แก้ที่นี่)
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
DEBUG_DIR = os.path.join(BASE_DIR, "debug")

# พิกัด UI ทั้งหมดเขียนเป็น "พิกเซลอ้างอิง" ของภาพแนวนอน 1210x720 (เกมบนเครื่อง cloud 720x1280 หลังหมุนกลับ)
REF_W, REF_H = 1210, 720

# screencap ของเครื่อง cloud เป็นจอตั้ง แต่เกมถูกหมุนตามเข็ม 90° อยู่ในจอ
SCREEN_ROTATED = "auto"      # "auto" | True | False
VIEW_HEIGHT = 1210           # ความสูงส่วนเกมในภาพจอตั้ง (ตัดแถบปุ่ม back/home) None = ประมาณเอง

ANCHORS = {
    "joy_center":    (227, 540),    # กลางจอยสติ๊ก
    "joy_radius":    65,
    "sword":         (1041, 539),   # ปุ่มดาบ (กดค้างเพื่อตี)
    "hand_guess":    (835, 478),    # ปุ่มมือเก็บของ
    "char_feet":     (605, 548),    # เท้าตัวละคร
    "camera_start":  (929, 360),    # จุดเริ่มลากกล้อง (เท่ากับที่ลากจริง)
    "pickaxe_slot":  (277, 674),    # ช่องของที่ 1 = ที่ขุด
}
ROI = {                              # (x1, y1, x2, y2)
    "clock":      (0, 4, 64, 32),
    "durability": (292, 652, 320, 710),   # แถบความทนทานข้างไอคอนที่ขุด
    "bag_ring":   (968, 636, 1046, 712),  # วงแหวนรอบไอคอนกระเป๋า
    "target_bar": (240, 0, 970, 110),     # แถบชื่อ+เลือดเป้าหมายกลางบน
    "dialog":     (330, 290, 880, 560),   # หน้าต่าง Pick All
    "repair":     (380, 40, 840, 640),    # หน้าต่างซ่อมที่ขุด
}
UI_EXCLUDE = [                        # โซน UI ที่ไม่มองหาหิน
    (0, 0, 195, 205), (30, 205, 260, 280), (195, 0, 480, 50), (200, 55, 425, 230),
    (0, 0, 1210, 60), (880, 0, 1210, 170), (0, 420, 380, 720), (900, 380, 1210, 720),
    (230, 575, 1050, 720), (930, 180, 1010, 280),
]
CHAR_EXCLUDE = (572, 300, 642, 572)   # ตัวละครเรา (ผม/เสื้อผ้าไม่ให้นับเป็นหิน)
ROCK_MIN_Y = 150
ROCK_V_FLOOR = 40

# ช่วงเวลาในเกม (นาทีของวัน): เช้ามืด 05:20-06:20 / กลางวัน 06:20-19:50 / กลางคืน 19:50-05:20
PERIODS = {"dawn": (5 * 60 + 20, 6 * 60 + 20), "day": (6 * 60 + 20, 19 * 60 + 50), "night": (19 * 60 + 50, 5 * 60 + 20)}
TWILIGHT_BLEND_MIN = 40       # ใกล้รอยต่อ ±นาที ใช้โทนม่วง 'dawn' ร่วมด้วย (แสงเปลี่ยนไม่ตรงเวลาเป๊ะ)
CLOCK_READ_INTERVAL = 4.0     # อ่านนาฬิกาทุกกี่วินาที
CLOCK_FALLBACK_VISUAL = True  # อ่านไม่ได้เลย -> เดาจากแสงของฉาก

# ชื่อบนแถบเป้าหมาย: ต้องมีคำเหล่านี้ (ตัวพิมพ์เล็ก) จึงถือว่าเป็นแร่ / ชื่อที่ห้ามตี
ORE_NAME_KEYWORDS = {"dawn": ["mine", "iron", "ore"], "day": ["mine", "iron", "ore"], "night": ["mine", "iron", "ore"]}
NAME_BLOCKLIST = ["cow", "chicken", "pig", "sheep", "goblin"]

# การสั่งเครื่อง
SHELL_PREFIX = ["su", "-c"]            # ไม่ได้รูท: ["sh", "-c"]
TOUCH_BACKEND = {
    "hold": "input",                   # กดค้างดาบ: "input" | "sendevent"
    "joystick": "input",
    "camera": "none",                  # "none" = ไม่หมุนกล้อง ใช้จอยเดินหาแทน | "input" | "sendevent"
}
TOUCH_EVENT_DEVICE = "/dev/input/event0"   # อุปกรณ์สัมผัสของเครื่องนี้ (จาก getevent)

# จังหวะ/เกณฑ์
HOLD_MS = 2500                # กดค้างดาบครั้งละ
HIT_TIMEOUT = 150             # ตีหินก้อนเดียวไม่เกินกี่วินาที
HP_NO_DROP_RETRY = 2          # เลือดไม่ลดกี่รอบติดกัน -> ขยับเข้าไปอีกนิดแล้วกดใหม่
TIRED_REST_SEC = 12           # รอสแตมิน่าฟื้น
STEP_MS_MIN, STEP_MS_MAX = 350, 900
NEAR_DIST = 105               # ระยะ (px อ้างอิง) เท้า->ฐานหิน ที่ลองตีได้
APPROACH_MAX_STEPS = 40
ENGAGE_TRIES = 4
SEARCH_MAX_ROUNDS = 25
AFTER_PICK_WAIT = 1.5
STUCK_FRAME_DIFF = 2.0
BLACKLIST_SEC = 90
BAG_FULL_RATIO = 0.25         # สัดส่วนสีเหลือง/ส้ม/แดงในวงแหวนกระเป๋าที่ถือว่า 'เต็ม'
REPAIR_MAX_FAILS = 3
COUNTDOWN_SEC = 10

# โปรไฟล์สีหินสำรอง (ถ้าไม่มีโฟลเดอร์ templates/rocks จะใช้ชุดนี้)
# ถ้ามีรูปใน templates/rocks/<dawn|day|night>/ บอทจะสร้างโปรไฟล์ใหม่จากรูปเองทุกครั้งที่เริ่ม
DEFAULT_PROFILES = {
    "dawn": {"s": [28, 158], "v": [40, 141], "hue": [[0, 9], [151, 179]]},
    "day": {"s": [0, 83], "v": [40, 172], "any_hue": True},
    "night": {"s": [4, 173], "v": [40, 143], "hue": [[99, 119]]},
}

DAY_MIN = 24 * 60
DEBUG = False


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# =============================================================================
# เครื่อง: จับภาพ + แปลงพิกัด + สั่งแตะ
# =============================================================================
class Frame:
    """หนึ่งเฟรมในแนวนอน + ตัวแปลงพิกัด"""

    def __init__(self, img, raw_shape, rotated):
        self.img = img
        self.h, self.w = img.shape[:2]
        self.raw_shape = raw_shape
        self.rotated = rotated
        self.sx, self.sy = self.w / REF_W, self.h / REF_H

    def A(self, name):
        x, y = ANCHORS[name]
        return int(x * self.sx), int(y * self.sy)

    def box(self, b):
        x1, y1, x2, y2 = ROI[b] if isinstance(b, str) else b
        return int(x1 * self.sx), int(y1 * self.sy), int(x2 * self.sx), int(y2 * self.sy)

    def crop(self, b):
        x1, y1, x2, y2 = self.box(b)
        return self.img[max(0, y1):y2, max(0, x1):x2]

    def to_touch(self, xl, yl):
        """พิกัดบนภาพแนวนอน -> พิกัดสัมผัสบนจอจริง (จอตั้ง: x = กว้าง-1-yl, y = xl)"""
        if self.rotated:
            return int(round(self.raw_shape[1] - 1 - yl)), int(round(xl))
        return int(round(xl)), int(round(yl))


def make_frame(raw):
    h, w = raw.shape[:2]
    rotated = (h > w) if SCREEN_ROTATED == "auto" else bool(SCREEN_ROTATED)
    if rotated:
        vh = min(h, VIEW_HEIGHT or int(h * 1210 / 1280))
        land = np.rot90(raw[:vh], 1).copy()        # หมุนทวนเข็มคืนเป็นแนวนอน
    else:
        land = raw
    return Frame(land, (h, w), rotated)


class Device:
    def __init__(self, dry=False):
        self.dry = dry
        self.last = None

    def sh(self, cmd, binary=False, timeout=25):
        if self.dry:
            log(f"[dry] {cmd}")
            return b"" if binary else ""
        p = subprocess.run(SHELL_PREFIX + [cmd], capture_output=True, timeout=timeout)
        return p.stdout if binary else p.stdout.decode("utf-8", "ignore")

    def root_ok(self):
        out = self.sh("id")
        return "uid=0" in out, out.strip()

    def grab(self):
        raw = self.sh("screencap -p", binary=True)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR) if arr.size else None
        if img is not None:
            self.last = make_frame(img)
        return self.last

    # ---- แตะ / กดค้าง
    def tap(self, xl, yl):
        x, y = self.last.to_touch(xl, yl)
        log(f"  แตะ ({int(xl)},{int(yl)}) -> touch ({x},{y})")
        self.sh(f"input tap {x} {y}")

    def hold(self, xl, yl, ms):
        x, y = self.last.to_touch(xl, yl)
        if TOUCH_BACKEND["hold"] == "sendevent":
            self._events([(x, y)], ms)
        else:
            self.sh(f"input swipe {x} {y} {x} {y} {int(ms)}", timeout=ms / 1000 + 12)

    def _swipe(self, p1, p2, ms, backend):
        x1, y1 = self.last.to_touch(*p1)
        x2, y2 = self.last.to_touch(*p2)
        if backend == "sendevent":
            n = max(6, int(ms / 25))
            pts = [(int(x1 + (x2 - x1) * i / n), int(y1 + (y2 - y1) * i / n)) for i in range(n + 1)]
            self._events(pts, ms)
        else:
            self.sh(f"input swipe {x1} {y1} {x2} {y2} {int(ms)}", timeout=ms / 1000 + 12)

    def joystick(self, dx, dy, ms):
        """ดันจอยไปทิศ (dx,dy) บนภาพแนวนอน (ขึ้น = dy ลบ) นาน ms"""
        n = math.hypot(dx, dy)
        if n < 1e-6:
            return
        ux, uy = dx / n, dy / n
        cx, cy = self.last.A("joy_center")
        r = ANCHORS["joy_radius"] * self.last.sy
        self._swipe((cx + ux * r * 0.45, cy + uy * r * 0.45), (cx + ux * r * 0.85, cy + uy * r * 0.85),
                    ms, TOUCH_BACKEND["joystick"])

    def camera_available(self):
        return TOUCH_BACKEND["camera"] != "none"

    def camera_drag(self, dx_px, dy_px=0, ms=260):
        if not self.camera_available():
            return False
        sx, sy = self.last.A("camera_start")
        self._swipe((sx, sy), (sx + dx_px, sy + dy_px), ms, TOUCH_BACKEND["camera"])
        return True

    def _events(self, pts, ms):
        """ยิงสัญญาณแตะเข้าอุปกรณ์ตรงๆ ตามลำดับที่ getevent บันทึกจากนิ้วจริง (ทดลอง)"""
        dev = TOUCH_EVENT_DEVICE
        tid = 3000 + int(time.time()) % 5000
        dt = max(0.01, ms / 1000.0 / max(1, len(pts)))
        ev = lambda t, c, v: f"sendevent {dev} {t} {c} {v};"
        s = ""
        for i, (x, y) in enumerate(pts):
            s += ev(3, 57, tid) + ev(3, 53, x) + ev(3, 54, y)
            if i == 0:
                s += ev(1, 330, 1)
            s += ev(0, 2, 0) + ev(0, 0, 0) + f"sleep {dt:.3f};"
        s += ev(3, 57, 4294967295) + ev(0, 2, 0) + ev(1, 330, 0) + ev(0, 0, 0)
        self.sh(s, timeout=ms / 1000 + 25)


# =============================================================================
# มองหาหิน: โปรไฟล์สีตามช่วงเวลา
# =============================================================================
def _rock_pixels(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(int)
    H, S, V = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    grass = (H >= 33) & (H <= 95) & (S > 70)
    tan = (H >= 3) & (H <= 28) & (S > 70) & (V > 90)
    return hsv[~grass & ~tan]


def _hue_ranges(h, min_frac=0.015, pad=6):
    hist = np.bincount(h, minlength=180).astype(float)
    ext = np.concatenate([hist[-4:], hist, hist[:4]])
    sm = np.convolve(ext, np.ones(5) / 5.0, mode="same")[4:-4]
    on = sm >= min_frac * hist.sum()
    rng, i = [], 0
    while i < 180:
        if on[i]:
            j = i
            while j + 1 < 180 and on[j + 1]:
                j += 1
            rng.append([max(0, i - pad), min(179, j + pad)])
            i = j + 1
        else:
            i += 1
    merged = []
    for r in sorted(rng):
        if merged and r[0] <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], r[1])
        else:
            merged.append(r)
    return merged


def build_profiles(verbose=True):
    """สร้างโปรไฟล์สีหินจากรูปใน templates/rocks/<ช่วงเวลา>/ (ตัดภาพหินเป็นหลัก ขอบมีหญ้านิดหน่อยได้)"""
    profiles = {}
    for p in PERIODS:
        files = sorted(glob.glob(os.path.join(TEMPLATE_DIR, "rocks", p, "*.*")))
        pix = [_rock_pixels(im) for im in (cv2.imread(f) for f in files) if im is not None]
        if not pix:
            if p in DEFAULT_PROFILES:
                profiles[p] = DEFAULT_PROFILES[p]
            continue
        a = np.vstack(pix)
        s_lo, s_hi = np.percentile(a[:, 1], [2, 97])
        v_lo, v_hi = np.percentile(a[:, 2], [2, 97])
        prof = {"s": [int(max(0, s_lo - 12)), int(min(255, s_hi + 14))],
                "v": [int(max(ROCK_V_FLOOR, v_lo - 12)), int(min(255, v_hi + 18))],
                "images": len(files)}
        if float((a[:, 1] < 35).mean()) > 0.4:
            prof["any_hue"] = True
        else:
            prof["hue"] = _hue_ranges(a[a[:, 1] >= 35, 0])
        profiles[p] = prof
        if verbose:
            log(f"โปรไฟล์ {p}: {len(files)} รูป -> {json.dumps({k: v for k, v in prof.items() if k != 'images'})}")
    return profiles or dict(DEFAULT_PROFILES)


def rock_mask(img, names, profiles):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    out = np.zeros(H.shape, np.uint8)
    for n in names:
        p = profiles.get(n)
        if not p:
            continue
        hm = np.ones(H.shape, bool) if p.get("any_hue") else np.zeros(H.shape, bool)
        if not p.get("any_hue"):
            for h0, h1 in p["hue"]:
                hm |= (H >= h0) & (H <= h1)
        m = hm & (S >= p["s"][0]) & (S <= p["s"][1]) & (V >= p["v"][0]) & (V <= p["v"][1])
        out |= m.astype(np.uint8) * 255
    return out


def find_rocks(fr, names, profiles):
    """-> (รายการหินเรียงใกล้->ไกล, มาสก์) แต่ละก้อน {'base','bbox','area','dist'}"""
    m = rock_mask(fr.img, names, profiles)
    for b in list(UI_EXCLUDE) + [CHAR_EXCLUDE]:
        x1, y1, x2, y2 = fr.box(b)
        m[max(0, y1):y2, max(0, x1):x2] = 0
    k = max(3, int(5 * fr.sy))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    feet = fr.A("char_feet")
    rocks = []
    for i in range(1, n):
        x, y, w, h, a = [int(v) for v in st[i]]
        if a < 260 * fr.sx * fr.sy or a > 90000 * fr.sx * fr.sy:
            continue
        if h < 0.45 * w or h > 2.0 * w:                 # หินสูงพอๆ กับกว้าง (ตัดลำต้น/เสา)
            continue
        if (x <= 1 or x + w >= fr.w - 1) and w < 70 * fr.sx:
            continue
        if a / float(w * h) < 0.28:
            continue
        base = (x + w // 2, y + h)
        if base[1] < ROCK_MIN_Y * fr.sy:
            continue
        rocks.append({"base": base, "bbox": (x, y, w, h), "area": a,
                      "dist": float(math.hypot(base[0] - feet[0], base[1] - feet[1]))})
    rocks.sort(key=lambda r: r["dist"])
    return rocks, m


# =============================================================================
# นาฬิกาในเกม -> ช่วงเวลา -> โปรไฟล์
# =============================================================================
def fmt_min(m):
    return f"{int(m) // 60:02d}:{int(m) % 60:02d}"


def minutes_to_period(m):
    m = int(m) % DAY_MIN
    for name, (a, b) in PERIODS.items():
        if (a <= m < b) if a <= b else (m >= a or m < b):
            return name
    return "day"


def profiles_for(m):
    """ใกล้รอยต่อของช่วง ใช้โทนม่วง 'dawn' ร่วมด้วย"""
    m = int(m) % DAY_MIN
    names = [minutes_to_period(m)]
    for bnd in [a for a, _ in PERIODS.values()] + [b for _, b in PERIODS.values()]:
        if min((m - bnd) % DAY_MIN, (bnd - m) % DAY_MIN) <= TWILIGHT_BLEND_MIN and "dawn" not in names:
            names.append("dawn")
    return names


def visual_period(img):
    """เดาช่วงเวลาจากแสงพื้นหญ้าหน้าตัวละคร (ทดสอบกับภาพจริง 30 ภาพ) -> day | dawn(รวมค่ำ) | night"""
    h, w = img.shape[:2]
    reg = np.concatenate([img[int(h * .50):int(h * .62), int(w * .30):int(w * .45)].reshape(-1, 3),
                          img[int(h * .50):int(h * .62), int(w * .58):int(w * .78)].reshape(-1, 3)])
    b, g, r = [float(x) for x in reg.mean(0)]
    if r - b < 12:
        return "night"
    return "day" if max(b, g, r) >= 112 else "dawn"


class ClockReader:
    """อ่านเวลาในเกม: OCR หลายแบบ+โหวต -> กรองความสมเหตุสมผล -> เดินนาฬิกาเองด้วยอัตราที่วัดได้ -> สำรอง: ดูแสงฉาก"""

    def __init__(self):
        self.last_min = self.last_real = None
        self.pairs, self.rate, self.unwrapped = [], None, 0.0
        self.source, self._t = "none", 0.0
        self.override = None

    @staticmethod
    def _variants(roi):
        g = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
        _, o = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        a = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, -8)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        w = cv2.resize(cv2.inRange(hsv, (0, 0, 170), (180, 90, 255)), None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
        _, w = cv2.threshold(cv2.GaussianBlur(w, (5, 5), 0), 127, 255, cv2.THRESH_BINARY)
        return [255 - o, o, 255 - a, 255 - w]

    def _ocr(self, fr):
        if not HAS_TESS:
            return []
        roi = fr.crop("clock")
        if roi.size == 0:
            return []
        out = []
        for v in self._variants(roi):
            v = cv2.copyMakeBorder(v, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=255)
            try:
                t = pytesseract.image_to_string(v, config="--psm 7 -c tessedit_char_whitelist=0123456789:")
            except Exception:
                continue
            d = "".join(c for c in t if c.isdigit())
            if len(d) == 4 and int(d[:2]) < 24 and int(d[2:]) < 60:
                out.append(int(d[:2]) * 60 + int(d[2:]))
        return out

    def _consistent(self, cand, now):
        if self.last_min is None:
            return False
        start, width = ((self.last_min + self.rate * (now - self.last_real)) % DAY_MIN, 25.0) if self.rate \
            else (self.last_min, 180.0)
        return (cand - start) % DAY_MIN <= width or (start - cand) % DAY_MIN <= 6

    def _accept(self, minutes, now):
        if self.last_min is not None:
            delta = (minutes - self.last_min) % DAY_MIN
            if delta > DAY_MIN / 2:                    # ถอยหลัง = อ่านเพี้ยน
                return
            self.unwrapped += delta
        self.last_min, self.last_real = minutes, now
        self.pairs = (self.pairs + [(now, self.unwrapped)])[-12:]
        old = [p for p in self.pairs if now - p[0] >= 25]
        if old and now > old[0][0] and self.unwrapped > old[0][1]:
            self.rate = (self.unwrapped - old[0][1]) / (now - old[0][0])

    def update(self, fr, force=False):
        now = time.time()
        if not force and now - self._t < CLOCK_READ_INTERVAL:
            return
        self._t = now
        cands = self._ocr(fr)
        if not cands:
            return
        best, n = Counter(cands).most_common(1)[0]
        if n >= 2 and (self.last_min is None or self._consistent(best, now) or n >= 3):
            self._accept(best, now)
            self.source = "ocr"
        elif n == 1:
            if self.last_min is not None:
                ok = [c for c in set(cands) if self._consistent(c, now)]
                if ok:
                    self._accept(ok[0], now)
                    self.source = "ocr?"
            elif visual_period(fr.img) in profiles_for(best):     # ครั้งแรก: เชื่อเมื่อตรงกับแสงฉาก
                self._accept(best, now)
                self.source = "ocr?"

    def now_minutes(self):
        if self.last_min is None:
            return None
        if self.rate:
            return (self.last_min + self.rate * (time.time() - self.last_real)) % DAY_MIN
        return self.last_min

    def current(self, fr):
        """-> (โปรไฟล์ที่ใช้, ช่วงหลัก, ข้อความเวลา, แหล่งที่มา)"""
        if self.override:
            return [self.override], self.override, "--:--", "override"
        self.update(fr)
        m = self.now_minutes()
        if m is not None:
            return profiles_for(m), minutes_to_period(m), fmt_min(m), self.source
        if CLOCK_FALLBACK_VISUAL:
            p = visual_period(fr.img)
            return ([p, "dawn"] if p != "dawn" else ["dawn"]), p, "--:--", "visual"
        return ["day"], "day", "--:--", "default"


# =============================================================================
# ตัวตรวจ UI (สี/รูปทรง ไม่พึ่งแสงกลางวัน-กลางคืน)
# =============================================================================
CYAN = ((82, 90, 170), (102, 255, 255))                # เส้นฟ้าใต้แถบเป้าหมาย
RED_LO, RED_HI = ((0, 110, 130), (9, 255, 255)), ((171, 110, 130), (179, 255, 255))
GREEN_BTN = ((40, 70, 120), (88, 255, 235))            # Pick All / Fix
YELLOW_BTN = ((15, 130, 170), (32, 255, 255))          # Equip Now


def hsv_range(img, lo, hi):
    return cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), np.array(lo, np.uint8), np.array(hi, np.uint8))


def largest_blob(mask, min_area=50, min_aspect=0.0, max_aspect=1e9, min_fill=0.0):
    n, _, st, cen = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best = None
    for i in range(1, n):
        x, y, w, h, a = [int(v) for v in st[i]]
        if a < min_area or not (min_aspect <= w / float(max(1, h)) <= max_aspect) or a / float(w * h) < min_fill:
            continue
        if best is None or a > best[4]:
            best = (x + w // 2, y + h // 2, w, h, a)
    return best


def frame_diff(a, b):
    """ค่าต่างเฉลี่ยของสองเฟรม ใช้ตรวจว่าเดินแล้ว 'ติด' หรือไม่"""
    if a is None or b is None or a.shape != b.shape:
        return 99.0
    sa = cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.int16)
    sb = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.int16)
    return float(np.mean(np.abs(sa - sb)))


def find_button(fr, kind, roi, min_area_ref, min_aspect=1.8, max_aspect=6.0):
    x1, y1, x2, y2 = fr.box(roi)
    x1, y1 = max(0, x1), max(0, y1)
    sub = fr.img[y1:y2, x1:x2]
    m = cv2.morphologyEx(hsv_range(sub, *(GREEN_BTN if kind == "green" else YELLOW_BTN)),
                         cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    b = largest_blob(m, min_area_ref * fr.sx * fr.sy, min_aspect, max_aspect, 0.6)
    return (x1 + b[0], y1 + b[1]) if b else None


def read_target_bar(fr):
    """-> dict(engaged, hp, x0, x1, y)  engaged = แถบเป้าหมายขึ้น (ตีโดนแล้ว), hp = เลือดที่เหลือ 0..1 (0 = หมด)
    หาแถบจากเส้นสีฟ้ายาวๆ ใต้ชื่อ (ชิ้นส่วนที่ต่อเนื่องยาวสุด) แล้ววัดความยาวสีแดงเหนือเส้น"""
    out = {"engaged": False, "hp": 1.0, "x0": 0, "x1": 0, "y": 0}
    x1, y1, x2, y2 = fr.box("target_bar")
    x1, y1 = max(0, x1), max(0, y1)
    strip = fr.img[y1:y2, x1:x2]
    if strip.size == 0:
        return out
    cy = hsv_range(strip, *CYAN) > 0
    rows = cy.sum(axis=1)
    cand = np.where(rows >= 120 * fr.sx)[0]
    if cand.size == 0:
        return out
    ly = int(cand[np.argmax(rows[cand])])
    line = cy[max(0, ly - 1):ly + 2].any(axis=0)
    best, i = (0, 0), 0                                   # ช่วงต่อเนื่องยาวสุด (ยอมขาดได้ 3px)
    while i < len(line):
        if line[i]:
            j, gap = i, 0
            while j + 1 < len(line) and (line[j + 1] or gap < 3):
                gap = 0 if line[j + 1] else gap + 1
                j += 1
            e = j - gap
            if e - i > best[1] - best[0]:
                best = (i, e)
            i = j + 1
        else:
            i += 1
    xa, xb = best
    if xb - xa < 150 * fr.sx:
        return out
    ya, yb = max(0, ly - int(28 * fr.sy)), max(1, ly - int(2 * fr.sy))
    band = strip[ya:yb, xa:xb + 1]
    red = hsv_range(band, *RED_LO) | hsv_range(band, *RED_HI)
    hit = np.where((red > 0).sum(axis=0) >= max(2, int(3 * fr.sy)))[0]
    hp = (float(hit.max()) + 1.0) / float(xb - xa + 1) if hit.size else 0.0
    out.update(engaged=True, hp=0.0 if hp < 0.03 else min(1.0, hp), x0=x1 + xa, x1=x1 + xb, y=y1 + ly)
    return out


def read_target_name(fr, bar):
    """อ่านชื่อเป้าหมายบนแถบ (ตัวพิมพ์เล็ก) ต้องมี pytesseract"""
    if not bar["engaged"] or not HAS_TESS:
        return ""
    ya, yb = max(0, bar["y"] - int(58 * fr.sy)), max(1, bar["y"] - int(6 * fr.sy))
    w = bar["x1"] - bar["x0"]
    roi = fr.img[ya:yb, bar["x0"] + int(w * 0.18):bar["x1"] - int(w * 0.18)]
    if roi.size == 0:
        return ""
    g = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    texts = []
    for inv in (False, True):
        _, t = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = cv2.copyMakeBorder(255 - t if inv else t, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
        try:
            texts.append(pytesseract.image_to_string(t, config="--psm 7").strip().lower())
        except Exception:
            pass
    return " | ".join(x for x in texts if x)


def name_verdict(text, period):
    """-> 'ore' | 'bad' | 'unknown' ตาม ORE_NAME_KEYWORDS / NAME_BLOCKLIST (เทียบแบบเผื่อ OCR เพี้ยน)"""
    if not text:
        return "unknown"
    words = [w for w in text.replace("|", " ").split() if w]
    close = lambda k, thr: k in text or any(difflib.SequenceMatcher(None, k, w).ratio() >= thr for w in words)
    if any(close(b, 0.75) for b in NAME_BLOCKLIST):
        return "bad"
    if any(close(k, 0.7) for k in ORE_NAME_KEYWORDS.get(period, [])):
        return "ore"
    return "unknown"


def find_pick_all(fr):
    """ปุ่ม Pick All (เขียว) + ต้องมีปุ่ม Cancel สีแดงอยู่ซ้ายมือแนวเดียวกัน -> (x, y)"""
    p = find_button(fr, "green", "dialog", 3500)
    if not p:
        return None
    x, y = p
    sub = fr.img[max(0, int(y - 35 * fr.sy)):int(y + 35 * fr.sy), max(0, int(x - 330 * fr.sx)):max(1, int(x - 80 * fr.sx))]
    if sub.size == 0:
        return None
    red = hsv_range(sub, (165, 90, 150), (179, 255, 255)) | hsv_range(sub, (0, 90, 150), (8, 255, 255))
    return p if largest_blob(red, 2500 * fr.sx * fr.sy, 1.8, 6.0, 0.6) else None


def find_close_x(fr):
    x1, y1, x2, y2 = fr.box("repair")
    x1, y1 = max(0, x1), max(0, y1)
    sub = fr.img[y1:y2, x1:x2]
    m = hsv_range(sub, (0, 80, 120), (10, 255, 255)) | hsv_range(sub, (165, 80, 120), (179, 255, 255))
    b = largest_blob(m, 300 * fr.sx * fr.sy, 0.6, 1.5, 0.35)       # วงกลมแดง (มี X สีขาวอยู่ข้างใน)
    return (x1 + b[0], y1 + b[1]) if b else None


def find_fix(fr):
    return find_button(fr, "green", "repair", 6000) if find_close_x(fr) else None


def find_equip(fr):
    return find_button(fr, "yellow", "repair", 6000) if find_close_x(fr) else None


def banner_present(fr):
    """แถบข้อความมืดโปร่งแสงพาดกลางจอ (เช่น 'ฉันเหนื่อยแล้ว ต้องพักสักครู่')"""
    h, w = fr.h, fr.w
    y0, y1 = int(h * 0.25), int(h * 0.52)
    v = cv2.cvtColor(fr.img[y0:y1, int(w * 0.04):int(w * 0.96)], cv2.COLOR_BGR2HSV)[..., 2].astype(np.float32).mean(axis=1)
    k, lo, hi = max(2, int(10 * fr.sy)), int(30 * fr.sy), int(75 * fr.sy)
    for top in range(k, len(v) - lo - k):
        for hh in (lo, (lo + hi) // 2, hi):
            bot = top + hh
            if bot + k > len(v):
                continue
            inside, above, below = v[top + 3:bot - 3].mean(), v[top - k:top - 1].mean(), v[bot + 1:bot + k].mean()
            if inside < 0.68 * above and inside < 0.68 * below and above > 40 and below > 40 \
                    and v[top + 3:bot - 3].std() < 0.55 * inside + 12:
                return True
    return False


def pickaxe_state(fr):
    """'ok' = แถบเขียว | 'broken' = แดง/หมด | 'unknown'"""
    roi = fr.crop("durability")
    if roi.size == 0:
        return "unknown"
    H, S, V = [cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[..., i] for i in range(3)]
    if ((H >= 40) & (H <= 90) & (S > 130) & (V > 120)).sum() >= 6:
        return "ok"
    if ((((H <= 9) | (H >= 171)) & (S > 130) & (V > 130)).sum()) >= 6:
        return "broken"
    return "unknown"


def bag_state(fr):
    """'ok' = วงแหวนเขียว/ฟ้า | 'full' = เหลือง-ส้ม-แดง (ตามที่กำหนด: เหลือง = หยุด) | 'unknown'"""
    roi = fr.crop("bag_ring")
    if roi.size == 0:
        return "unknown"
    h, w = roi.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - w / 2.0, yy - h / 2.0)
    ring = (r >= 0.30 * w) & (r <= 0.48 * w)              # เฉพาะวงแหวน ไม่นับไอคอนทองตรงกลาง
    H, S, V = [cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[..., i] for i in range(3)]
    warm = ring & (S > 110) & (V > 140) & (((H >= 10) & (H <= 34)) | (H <= 6) | (H >= 172))
    cool = ring & (S > 90) & (V > 120) & (H >= 45) & (H <= 100)
    nw, nc = int(warm.sum()), int(cool.sum())
    if nw + nc < 20:
        return "unknown"
    return "full" if nw / float(nw + nc) >= BAG_FULL_RATIO else "ok"


# =============================================================================
# CONFIG ของบอทวนทดสอบ (ใส่พิกัดที่นี่)
# =============================================================================
SEARCH_COORD = None            # (x, y) พิกัดจุดมองหาหิน   เช่น (6345, 14165)   None = ใช้ตำแหน่งตอนเริ่มเป็นจุดมองหา
BOX_COORD = None               # (x, y) พิกัดกล่อง(บ้าน)   เช่น (6300, 14100)   None = ข้ามขั้นตอนไปกล่อง
GOTO_TOL = 5                   # ถึงพิกัดเมื่อห่างไม่เกินกี่หน่วย
BOX_TOL = 4
GOTO_MAX_STEPS = 60
LEASH = 45                     # หาหินแล้วเดินหลุดจากจุดมองหาเกินกี่หน่วย -> กลับไปจุดมองหาก่อน
CAL_MS = 1200                  # เดินทดสอบเพื่อเรียนรู้ทิศ (พิกัด<->จอย) ครั้งละกี่ ms
COORD_SETTLE = 1.2             # รอพิกัดบนจอ "ดีเล" ก่อนอ่านหลังเดิน (วินาที)
ROCKS_PER_VISIT = 6            # ไม่เจอหินกี่รอบ(เดินสุ่ม) แล้วกลับไปจุดมองหาใหม่
ROCK_TRIES_PER_ROUND = 5       # ตีไม่สำเร็จ/ไม่ใช่แร่ ลองหินกี่ก้อนต่อรอบ
SEARCH_LOOK_WAIT = 0.6

# ไอคอนที่ผู้ใช้เก็บไว้ (รูปหินตามช่วงเวลา + รูปมือ) ชื่อไฟล์ไม่ต้องบอกช่วงเวลา บอทจัดกลุ่มจากสีให้เอง
ICONS_DIR = os.path.expanduser("~/UtopiaFarmAI/icons")
ROCK_ICON_PATTERNS = ["*rock*", "*stone*", "*หิน*", "*ore*", "*mine*"]    # ไฟล์ใน ICONS_DIR ที่ถือเป็นรูปหิน
ROCK_ICON_PERIOD = {}          # บังคับช่วงเวลาของไฟล์ เช่น {"rock_night.png": "night"}
HAND_ICON_PATTERNS = ["*hand*", "*มือ*", "*pickup*"]
HAND_THRESHOLD = 0.6           # คะแนนจับไอคอนมือ (ทดสอบ: เห็นมือ 0.87-1.0 / ไม่เห็น <= 0.51)
HAND_SCALES = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0, 1.15]
COORD_MAX_JUMP = 120           # พิกัดที่อ่านได้กระโดดเกินนี้จากค่าก่อน ถือว่าอ่านผิด (ยกเว้นยืนยันซ้ำ)
ROI["coords"] = (40, 240, 290, 292)
TIME_LOG = os.path.join(DEBUG_DIR, "time_log.csv")


# =============================================================================
# รูปหินจาก icons -> โปรไฟล์สีตามช่วงเวลา
# =============================================================================
def classify_icon(img):
    """จัดรูปหินเข้ากลุ่มช่วงเวลาจากสี: ฟ้า=night, ม่วง(อิ่มสี)=dawn, เทา(อิ่มสีต่ำ)=day"""
    a = _rock_pixels(img)
    if len(a) < 50:
        return "day"
    H, S = a[:, 0], a[:, 1]
    if float(((H >= 95) & (H <= 125) & (S > 90)).mean()) >= 0.35:
        return "night"
    if float(np.median(S)) < 40:
        return "day"
    purple = ((H >= 145) | (H <= 12)) & (S >= 30)
    return "dawn" if float(purple.mean()) >= 0.4 else "day"


def _match_files(folder, patterns):
    out = []
    if os.path.isdir(folder):
        for p in patterns:
            out += glob.glob(os.path.join(folder, p))
            out += glob.glob(os.path.join(folder, p.upper()))
    seen, res = set(), []
    for f in out:
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")) and f not in seen:
            seen.add(f)
            res.append(f)
    return sorted(res)


def _profile_from_pixels(a, n_images):
    s_lo, s_hi = np.percentile(a[:, 1], [2, 97])
    v_lo, v_hi = np.percentile(a[:, 2], [2, 97])
    prof = {"s": [int(max(0, s_lo - 12)), int(min(255, s_hi + 14))],
            "v": [int(max(ROCK_V_FLOOR, v_lo - 12)), int(min(255, v_hi + 18))], "images": n_images}
    if float((a[:, 1] < 35).mean()) > 0.4:
        prof["any_hue"] = True
    else:
        prof["hue"] = _hue_ranges(a[a[:, 1] >= 35, 0])
    return prof


def load_profiles(verbose=True):
    """รวมรูปหินจาก ICONS_DIR (จัดกลุ่มตามสีอัตโนมัติ) + templates/rocks/<ช่วงเวลา>/ -> โปรไฟล์ต่อช่วงเวลา"""
    groups = {p: [] for p in PERIODS}
    names = {p: [] for p in PERIODS}
    for f in _match_files(ICONS_DIR, ROCK_ICON_PATTERNS):
        im = cv2.imread(f)
        if im is None:
            continue
        p = ROCK_ICON_PERIOD.get(os.path.basename(f)) or classify_icon(im)
        groups[p].append(_rock_pixels(im))
        names[p].append(os.path.basename(f))
    for p in PERIODS:
        for f in sorted(glob.glob(os.path.join(TEMPLATE_DIR, "rocks", p, "*.*"))):
            im = cv2.imread(f)
            if im is not None:
                groups[p].append(_rock_pixels(im))
                names[p].append("templates/" + os.path.basename(f))
    profiles = {}
    for p in PERIODS:
        if groups[p]:
            profiles[p] = _profile_from_pixels(np.vstack(groups[p]), len(groups[p]))
        elif p in DEFAULT_PROFILES:
            profiles[p] = DEFAULT_PROFILES[p]
            names[p].append("(ค่าสำรองในโค้ด)")
        if verbose:
            log(f"รูปหินช่วง {p:5s}: {len(names[p])} รูป {names[p]}")
    if verbose and not _match_files(ICONS_DIR, ROCK_ICON_PATTERNS):
        log(f"หมายเหตุ: ไม่พบรูปหินใน {ICONS_DIR} (ชื่อไฟล์ต้องตรง {ROCK_ICON_PATTERNS}) -> ใช้รูปใน templates/rocks")
    return profiles


# =============================================================================
# ไอคอนมือ (ขึ้นเมื่อเข้าใกล้กล่อง/ของที่เก็บได้)
# =============================================================================
def _lowS_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return (((hsv[..., 1] < 55) & (hsv[..., 2] > 115) & (hsv[..., 2] < 225)) * 255).astype(np.uint8)


_HAND_MASKS = None


def hand_masks():
    global _HAND_MASKS
    if _HAND_MASKS is None:
        files = _match_files(ICONS_DIR, HAND_ICON_PATTERNS) + glob.glob(os.path.join(TEMPLATE_DIR, "ui", "hand*.png"))
        _HAND_MASKS = []
        for f in files:
            im = cv2.imread(f)
            if im is not None and im.shape[0] >= 20:
                _HAND_MASKS.append(_lowS_mask(im))
    return _HAND_MASKS


def hand_present(fr):
    """-> (เห็นรูปมือไหม, คะแนน)  ค้นรอบตำแหน่งปุ่มมือ (ปุ่มขึ้นตรงเดิมเสมอ) หลายขนาด"""
    cx, cy = fr.A("hand_guess")
    rx, ry = int(130 * fr.sx), int(110 * fr.sy)
    roi = fr.img[max(0, cy - ry):cy + ry, max(0, cx - rx):cx + rx]
    if roi.size == 0:
        return False, 0.0
    m = _lowS_mask(roi).astype(np.float32)
    best = 0.0
    for t in hand_masks():
        for s in HAND_SCALES:
            tt = cv2.resize(t, None, fx=s, fy=s, interpolation=cv2.INTER_AREA).astype(np.float32)
            if tt.shape[0] >= m.shape[0] or tt.shape[1] >= m.shape[1] or tt.std() < 1e-3:
                continue
            best = max(best, float(cv2.matchTemplate(m, tt, cv2.TM_CCOEFF_NORMED).max()))
    return best >= HAND_THRESHOLD, best


# =============================================================================
# พิกัดบนจอ (OCR) + เดินไปตามพิกัด
# =============================================================================
def _coord_variants(roi):
    g = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    _, o = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    a = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -10)
    w = cv2.resize(cv2.inRange(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV), (0, 0, 185), (180, 70, 255)),
                   None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    _, w = cv2.threshold(cv2.GaussianBlur(w, (5, 5), 0), 127, 255, cv2.THRESH_BINARY)
    return [255 - o, o, 255 - a, 255 - w]


def _parse_coord(text):
    import re
    m = re.findall(r"\d+", text)
    if len(m) >= 2 and 3 <= len(m[0]) <= 5 and 4 <= len(m[1]) <= 6:
        return int(m[0]), int(m[1])
    d = "".join(m)
    if len(m) == 1 and len(d) == 9:
        return int(d[:4]), int(d[4:])
    return None


def read_coords_once(fr):
    """อ่านพิกัด 'x, y' ใต้ชื่อโซนด้วยหลายแบบ + โหวต -> (x, y) หรือ None"""
    if not HAS_TESS:
        return None
    roi = fr.crop("coords")
    if roi.size == 0:
        return None
    votes = []
    for v in _coord_variants(roi):
        v = cv2.copyMakeBorder(v, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        try:
            t = pytesseract.image_to_string(v, config="--psm 7 -c tessedit_char_whitelist=0123456789,. ")
        except Exception:
            continue
        c = _parse_coord(t)
        if c:
            votes.append(c)
    if not votes:
        return None
    best, n = Counter(votes).most_common(1)[0]
    return best if n >= 2 else (best if len(set(votes)) == 1 else None)


class Nav:
    """เดินไปตามพิกัด: เรียนรู้ก่อนว่า 'ดันจอยขึ้น/ขวา' พิกัดเปลี่ยนทางไหนเท่าไร (กล้องไม่หมุน ทิศจึงคงที่)"""

    def __init__(self, dev):
        self.dev = dev
        self.last = None
        self.M = None            # คอลัมน์ f (เดินหน้า), r (เดินขวา) หน่วยพิกัดต่อการดัน CAL_MS
        self.speed = None        # หน่วยพิกัดต่อ ms

    def grab(self):
        fr = self.dev.grab()
        if fr is None:
            raise Stop("จับภาพหน้าจอไม่ได้ (เช็คสิทธิ์รูท / su)")
        return fr

    def read(self, tries=4):
        for _ in range(tries):
            c = read_coords_once(self.grab())
            if c and (self.last is None or math.hypot(c[0] - self.last[0], c[1] - self.last[1]) <= COORD_MAX_JUMP):
                self.last = c
                return c
            time.sleep(0.4)
        return None

    def read_stable(self, timeout=5.0):
        """พิกัดบนจอดีเล: อ่านซ้ำจนสองครั้งติดกันเท่ากัน"""
        t0, prev = time.time(), None
        time.sleep(COORD_SETTLE)
        while time.time() - t0 < timeout:
            c = self.read(tries=2)
            if c and c == prev:
                return c
            prev = c or prev
            time.sleep(0.5)
        return prev

    def calibrate(self):
        p0 = self.read_stable()
        if not p0:
            log("[พิกัด] อ่านพิกัดไม่ได้ (เช็ค ROI['coords'] ด้วย calibrate / ติดตั้ง tesseract)")
            return False
        self.dev.joystick(0, -1, CAL_MS)
        p1 = self.read_stable()
        self.dev.joystick(1, 0, CAL_MS)
        p2 = self.read_stable()
        if not p1 or not p2:
            return False
        f = (p1[0] - p0[0], p1[1] - p0[1])
        r = (p2[0] - p1[0], p2[1] - p1[1])
        nf, nr = math.hypot(*f), math.hypot(*r)
        det = f[0] * r[1] - r[0] * f[1]
        log(f"[พิกัด] เรียนรู้ทิศ: เดินหน้า -> พิกัดเปลี่ยน {f} | เดินขวา -> {r}")
        if nf < 2 or nr < 2 or abs(det) < 0.25 * nf * nr:
            log("[พิกัด] เรียนรู้ทิศไม่สำเร็จ (ติดสิ่งกีดขวาง/ขยับน้อย) จะลองใหม่ที่จุดอื่น")
            return False
        self.M = (f, r)
        self.speed = ((nf + nr) / 2.0) / CAL_MS
        return True

    def goto(self, target, tol, label):
        """เดินไปใกล้พิกัด target -> True เมื่อถึง"""
        log(f"[พิกัด] เดินไป{label} {target}")
        bad = 0
        best = None
        for step in range(GOTO_MAX_STEPS):
            pos = self.read_stable()
            if not pos:
                log("[พิกัด] อ่านพิกัดไม่ได้")
                return False
            d = (target[0] - pos[0], target[1] - pos[1])
            dist = math.hypot(*d)
            log(f"[พิกัด] ก้าว {step + 1}: ตอนนี้ {pos} -> เป้า {target} ห่าง {dist:.1f}")
            if dist <= tol:
                log(f"[พิกัด] ถึง{label}แล้ว {pos}")
                return True
            if self.M is None and not self.calibrate():
                self.dev.joystick(random.uniform(-1, 1), random.uniform(-1, 1), 700)   # ขยับหนีที่ติดแล้วลองใหม่
                continue
            (f, r) = self.M
            det = f[0] * r[1] - r[0] * f[1]
            a = (d[0] * r[1] - r[0] * d[1]) / det           # ส่วนที่ต้องเดินหน้า
            b = (f[0] * d[1] - d[0] * f[1]) / det           # ส่วนที่ต้องเดินขวา
            ms = int(min(1800, max(350, 0.8 * dist / max(self.speed, 1e-6))))
            self.dev.joystick(b, -a, ms)
            if best is not None and dist >= best - 0.5:
                bad += 1
                if bad >= 3:
                    log("[พิกัด] ระยะไม่ลดลง -> เรียนรู้ทิศใหม่ (อาจติดสิ่งกีดขวาง/กล้องขยับ)")
                    self.M = None
                    bad = 0
            else:
                bad = 0
            best = dist if best is None else min(best, dist)
        log(f"[พิกัด] เดินไป{label}ไม่ถึงภายใน {GOTO_MAX_STEPS} ก้าว")
        return False


# =============================================================================
# บันทึกเวลา/โปรไฟล์ เพื่อตรวจว่าอ่านเวลา/เลือกรูปหินถูกช่วงไหม
# =============================================================================
def log_time_row(row):
    os.makedirs(DEBUG_DIR, exist_ok=True)
    new = not os.path.exists(TIME_LOG)
    with open(TIME_LOG, "a", encoding="utf-8") as f:
        if new:
            f.write("เวลาจริง,เวลาในเกม,ช่วง,แหล่งเวลา,โปรไฟล์ที่ใช้,แสงฉากบอกว่า,หินต่อโปรไฟล์(dawn/day/night),พิกัด,เหตุการณ์\n")
        f.write(",".join(str(x) for x in row) + "\n")


class Stop(Exception):
    pass


# =============================================================================
# ตัวบอทวน
# =============================================================================
class RockLoop:
    def __init__(self, dev, profiles, period_override=None):
        self.dev, self.profiles = dev, profiles
        self.clock = ClockReader()
        self.clock.override = period_override
        self.nav = Nav(dev)
        self.fr = None
        self.names, self.period, self._tstr, self._src = ["day"], "day", "--:--", "none"
        self.black = []
        self.search_pt = SEARCH_COORD
        self.box_pt = BOX_COORD
        self.wander_ang, self.wander_left = 0.0, 0
        self.stats = {"ตีสำเร็จ": 0, "กลับบ้านสำเร็จ": 0, "รอบ": 0}

    # ---------------------------------------------------------------- พื้นฐาน
    def frame(self):
        fr = self.dev.grab()
        if fr is None:
            raise Stop("จับภาพหน้าจอไม่ได้ (เช็คสิทธิ์รูท / su)")
        self.fr = fr
        self.names, self.period, self._tstr, self._src = self.clock.current(fr)
        return fr

    def blacklisted(self, rock):
        now = time.time()
        self.black = [(p, e) for p, e in self.black if e > now]
        return any(math.hypot(rock["base"][0] - p[0], rock["base"][1] - p[1]) < 60 * self.fr.sx for p, _ in self.black)

    def report_time(self, event):
        """พิมพ์/บันทึก: เวลาเกมที่อ่านได้, โปรไฟล์ที่ใช้, และหินที่ 'แต่ละโปรไฟล์' มองเห็นในภาพนี้"""
        fr = self.fr
        counts = {p: len(find_rocks(fr, [p], self.profiles)[0]) for p in PERIODS}
        vis = visual_period(fr.img)
        best = max(counts, key=counts.get) if any(counts.values()) else "-"
        msg = (f"เวลาเกม {self._tstr} ช่วง={self.period} (แหล่ง {self._src}) ใช้โปรไฟล์ {self.names} | แสงฉาก={vis} | "
               f"หินที่เห็น dawn={counts['dawn']} day={counts['day']} night={counts['night']}")
        log("[เวลา] " + msg)
        if best != "-" and best not in self.names and counts[best] >= 2 and counts[best] > 2 * max(
                [counts[n] for n in self.names if n in counts] + [0]):
            log(f"[เวลา] ⚠️ โปรไฟล์ที่ใช้ ({self.names}) ไม่ตรงกับภาพ: '{best}' เจอหินมากกว่ามาก -> "
                f"อ่านเวลาเกมผิด หรือรูปหินช่วง {self.period} ยังไม่พอ (เพิ่มรูปใน icons)")
        log_time_row([time.strftime("%Y-%m-%d %H:%M:%S"), self._tstr, self.period, self._src, "+".join(self.names), vis,
                      f"{counts['dawn']}/{counts['day']}/{counts['night']}", self.nav.last or "", event])

    # ---------------------------------------------------------------- ไปจุดต่างๆ
    def go_search(self):
        if not self.search_pt:
            return True
        return self.nav.goto(self.search_pt, GOTO_TOL, "จุดมองหา")

    def go_box(self):
        if not self.box_pt:
            log("[กล่อง] ยังไม่ได้ตั้ง BOX_COORD -> ข้ามขั้นไปกล่อง")
            return None
        ok = self.nav.goto(self.box_pt, BOX_TOL, "กล่อง")
        for k in range(6):                     # ใกล้กล่องแล้วรูปมือจะขึ้น ถ้ายังไม่ขึ้นขยับเข้าใกล้อีกนิด
            fr = self.frame()
            seen, score = hand_present(fr)
            log(f"[กล่อง] ตรวจรูปมือ คะแนน {score:.2f} (เกณฑ์ {HAND_THRESHOLD}) {'เห็น' if seen else 'ยังไม่เห็น'}")
            if seen:
                self.stats["กลับบ้านสำเร็จ"] += 1
                log("🏠 กลับบ้านสำเร็จ (เห็นรูปมือที่กล่อง)")
                self.report_time("กลับบ้านสำเร็จ")
                return True
            self.dev.joystick(random.uniform(-1, 1), random.uniform(-1, 1), 450)
            time.sleep(0.8)
        log("[กล่อง] ถึงพิกัดกล่องแล้วแต่ไม่เห็นรูปมือ -> เช็คพิกัดกล่อง หรือ HAND_THRESHOLD/ตำแหน่ง hand_guess (ใช้ calibrate)")
        self.report_time("กลับบ้านไม่สำเร็จ")
        return False

    # ---------------------------------------------------------------- มองหา / เดินเข้าหา / ตี
    def explore(self):
        if self.wander_left <= 0:
            self.wander_ang = random.uniform(-math.pi * 0.8, math.pi * 0.8)
            self.wander_left = random.randint(2, 3)
        self.wander_left -= 1
        self.dev.joystick(math.sin(self.wander_ang), -math.cos(self.wander_ang), 800)
        time.sleep(0.3)

    def find_target(self):
        """ยืนมองจากจุดนี้: เจอหินก็คืนก้อนที่ใกล้สุด ไม่เจอเดินสุ่มสั้นๆ แล้วมองใหม่"""
        for rnd in range(ROCKS_PER_VISIT):
            fr = self.frame()
            rocks, _ = find_rocks(fr, self.names, self.profiles)
            rocks = [r for r in rocks if not self.blacklisted(r)]
            self.report_time(f"มองหาหิน รอบ{rnd + 1} เจอ{len(rocks)}")
            if rocks:
                log(f"[หา] เจอหิน {len(rocks)} ก้อน ใกล้สุดฐาน={rocks[0]['base']}")
                return rocks[0]
            log(f"[หา] ไม่เจอหิน ({rnd + 1}/{ROCKS_PER_VISIT}) -> เดินสุ่มแล้วมองใหม่")
            self.explore()
            if self.search_pt and self.nav.last:      # กันเดินไกลจากจุดมองหาเกินไป
                c = read_coords_once(self.dev.grab())
                if c and math.hypot(c[0] - self.search_pt[0], c[1] - self.search_pt[1]) > LEASH:
                    log("[หา] เดินไกลจากจุดมองหาเกินกำหนด -> กลับไปจุดมองหา")
                    self.go_search()
        return None

    def approach(self, rock):
        d, stuck, lost, last = self.dev, 0, 0, rock["base"]
        for step in range(APPROACH_MAX_STEPS):
            fr = self.frame()
            rocks, _ = find_rocks(fr, self.names, self.profiles)
            rocks = [r for r in rocks if not self.blacklisted(r)]
            same = [r for r in rocks if math.hypot(r["base"][0] - last[0], r["base"][1] - last[1]) < 170 * fr.sx]
            tgt = (same or rocks or [None])[0]
            if tgt is None:
                lost += 1
                if lost >= 3:
                    log("[เดิน] หินหลุดจอ")
                    return None
                d.joystick(0, -1, 400)
                continue
            lost, last = 0, tgt["base"]
            feet = fr.A("char_feet")
            dx, dy = tgt["base"][0] - feet[0], tgt["base"][1] - feet[1]
            dist = math.hypot(dx, dy)
            if dist <= NEAR_DIST * fr.sy:
                log(f"[เดิน] ชิดตัวละครแล้ว (ห่าง {dist:.0f}px)")
                return tgt
            ms = int(min(STEP_MS_MAX, max(STEP_MS_MIN, dist * 1.6)))
            log(f"[เดิน] ก้าว {step + 1}: dx={dx:+.0f} dy={dy:+.0f} ห่าง {dist:.0f}px นาน {ms}ms")
            before = fr.img
            d.joystick(dx, dy, ms)
            time.sleep(0.15)
            fr2 = d.grab()
            if fr2 is not None and frame_diff(before, fr2.img) < STUCK_FRAME_DIFF:
                stuck += 1
                if stuck >= 2:
                    log("[เดิน] ติดสิ่งกีดขวาง -> เบี่ยงข้าง")
                    d.joystick(random.choice([-1, 1]), 0.2, 700)
                    stuck = 0
        return None

    def hit_once(self, rock):
        """ตีแล้วดูแถบชื่อ -> 'ore' (ตีสำเร็จ) | 'unknown' (ตีโดนแต่ชื่อไม่ชัด) | 'bad' (ไม่ใช่แร่) | 'miss'"""
        d = self.dev
        for t in range(ENGAGE_TRIES):
            fr = self.frame()
            d.hold(*fr.A("sword"), 1000)
            time.sleep(0.4)
            fr = self.frame()
            bar = read_target_bar(fr)
            if bar["engaged"]:
                name = read_target_name(fr, bar)
                v = name_verdict(name, self.period)
                log(f"[ตี] แถบเป้าหมายขึ้น ชื่อที่อ่านได้ '{name}' เลือด {bar['hp']:.2f} -> {v}")
                return v
            log(f"[ตี] ยังไม่โดนเป้า ({t + 1}/{ENGAGE_TRIES}) -> ขยับเข้าใกล้อีกนิด")
            rocks, _ = find_rocks(fr, self.names, self.profiles)
            if rocks:
                feet = fr.A("char_feet")
                d.joystick(rocks[0]["base"][0] - feet[0], rocks[0]["base"][1] - feet[1], 300)
            time.sleep(0.2)
        return "miss"

    def mine_one(self):
        for k in range(ROCK_TRIES_PER_ROUND):
            rock = self.find_target()
            if rock is None:
                return False
            rock = self.approach(rock)
            if rock is None:
                continue
            res = self.hit_once(rock)
            if res == "ore":
                self.stats["ตีสำเร็จ"] += 1
                log("✅ ตีสำเร็จ (ชื่อบนแถบตรงกับแร่)")
                self.report_time("ตีสำเร็จ")
                return True
            if res == "unknown":
                self.stats["ตีสำเร็จ"] += 1
                log("✅ ตีสำเร็จ (ตีโดนเป้า แต่อ่านชื่อไม่ชัด/ไม่มี tesseract - ยังไม่ยืนยันชื่อ)")
                self.report_time("ตีสำเร็จ(ชื่อไม่ยืนยัน)")
                return True
            log(f"[ตี] ไม่ใช่แร่/ตีไม่โดน ({res}) -> จำก้อนนี้ไว้แล้วหาก้อนใหม่")
            self.black.append((rock["base"], time.time() + BLACKLIST_SEC))
            self.report_time(f"ตีไม่สำเร็จ:{res}")
        return False

    # ---------------------------------------------------------------- ลูปหลัก
    def setup(self):
        self.frame()
        log(f"เวลาในเกมตอนเริ่ม: {self._tstr} ช่วง={self.period} (แหล่ง {self._src})")
        if not self.search_pt:
            c = self.nav.read_stable()
            if c:
                self.search_pt = c
                log(f"ไม่ได้ตั้ง SEARCH_COORD -> ใช้ตำแหน่งตอนเริ่ม {c} เป็นจุดมองหา")
            else:
                log("อ่านพิกัดตอนเริ่มไม่ได้ -> ทำงานแบบไม่เดินตามพิกัด (หาหินรอบตัวอย่างเดียว)")
        if not self.box_pt:
            log("ไม่ได้ตั้ง BOX_COORD -> จะข้ามขั้นตอนไปกล่อง (ตั้งที่หัวไฟล์ภายหลัง)")

    def run(self, max_rounds=0):
        self.setup()
        while True:
            self.stats["รอบ"] += 1
            r = self.stats["รอบ"]
            log("=" * 14 + f" รอบที่ {r} | ตีสำเร็จ {self.stats['ตีสำเร็จ']} | กลับบ้านสำเร็จ {self.stats['กลับบ้านสำเร็จ']} " + "=" * 14)
            log("[1] ไปจุดมองหา")
            self.go_search()
            log("[2] มองหาหิน -> เดินชิด -> ตี")
            ok = self.mine_one()
            if ok:
                log("[3] กลับไปจุดมองหา")
                self.go_search()
                log("[4] ไปกล่อง")
                self.go_box()
            else:
                log("[!] รอบนี้ยังไม่ได้ตีสำเร็จ -> กลับจุดมองหาแล้วเริ่มใหม่")
            log("[5] กลับไปจุดมองหา แล้ววนต่อ")
            self.go_search()
            if max_rounds and r >= max_rounds:
                return


class ImageDevice(Device):
    """ใช้ทดสอบ: ป้อนภาพนิ่งแทนการจับจากเครื่อง และไม่ส่งคำสั่งจริง"""

    def __init__(self, frame):
        super().__init__(dry=True)
        self.fixed = frame
        self.cmds = []

    def sh(self, cmd, binary=False, timeout=25):
        self.cmds.append(cmd)
        return b"" if binary else ""

    def grab(self):
        self.last = self.fixed
        return self.fixed


# =============================================================================
# คำสั่งเสริม
# =============================================================================
def countdown(sec, msg):
    log(msg)
    for s in range(sec, 0, -1):
        print(f"  เริ่มใน {s}...   ", end="\r", flush=True)
        time.sleep(1)
    print()


def cmd_calibrate(args):
    profiles = load_profiles()
    if args.image:
        raw = cv2.imread(args.image)
        if raw is None:
            sys.exit(f"เปิดไฟล์ภาพไม่ได้: {args.image}")
        fr = Frame(cv2.resize(raw, (REF_W, REF_H)), raw.shape[:2], True) if args.landscape else make_frame(raw)
    else:
        dev = Device(dry=args.dry)
        ok, who = dev.root_ok()
        log(f"รูท: {'OK' if ok else 'ไม่ใช่ -> ' + who}")
        fr = dev.grab()
        if fr is None:
            sys.exit("จับภาพไม่ได้")
    clk = ClockReader()
    clk.update(fr, force=True)
    names, period, tstr, src = clk.current(fr)
    log(f"ภาพดิบ {fr.raw_shape[1]}x{fr.raw_shape[0]} -> ภาพแนวนอน {fr.w}x{fr.h} (หมุน={fr.rotated}) tesseract={'มี' if HAS_TESS else 'ไม่มี'}")
    log(f"เวลาในเกม: {tstr} ช่วง={period} โปรไฟล์ที่ใช้={names} แหล่ง={src} แสงฉาก={visual_period(fr.img)}")
    log(f"พิกัดบนจอ: {read_coords_once(fr)}")
    seen, score = hand_present(fr)
    log(f"รูปมือ: {'เห็น' if seen else 'ไม่เห็น'} คะแนน {score:.2f} (ไอคอนมือที่ใช้ {len(hand_masks())} แบบ)")
    bar = read_target_bar(fr)
    log(f"แถบเป้าหมาย: ขึ้น={bar['engaged']} เลือด={bar['hp']:.2f} ชื่อ='{read_target_name(fr, bar)}'")
    counts = {p: len(find_rocks(fr, [p], profiles)[0]) for p in PERIODS}
    log(f"หินที่แต่ละโปรไฟล์เห็น: {counts}")
    rocks, _ = find_rocks(fr, names, profiles)
    ov = fr.img.copy()
    for name, c in (("joy_center", (0, 255, 0)), ("sword", (0, 255, 0)), ("hand_guess", (255, 0, 255)), ("char_feet", (255, 255, 0))):
        cv2.circle(ov, fr.A(name), 10, c, 2)
        cv2.putText(ov, name, (fr.A(name)[0] + 12, fr.A(name)[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
    for name in ("coords", "clock", "target_bar"):
        x1, y1, x2, y2 = fr.box(name)
        cv2.rectangle(ov, (x1, y1), (x2, y2), (200, 200, 0), 1)
        cv2.putText(ov, name, (x1 + 2, y1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)
    for r in rocks:
        x, y, w, h = r["bbox"]
        cv2.rectangle(ov, (x, y), (x + w, y + h), (0, 0, 255), 2)
    os.makedirs(DEBUG_DIR, exist_ok=True)
    out = os.path.join(DEBUG_DIR, "calibrate.png")
    cv2.imwrite(out, ov)
    log(f"เซฟภาพตรวจ: {out}  (กรอบแดง=หินที่เห็น, วงกลม=ปุ่มที่บอทใช้, กรอบเหลือง=โซนอ่านพิกัด/เวลา/แถบเป้าหมาย)")


def cmd_test_input(args):
    dev = Device(dry=args.dry)
    ok, who = dev.root_ok()
    log(f"รูท: {'OK' if ok else 'ไม่ใช่ -> ' + who}")
    countdown(5, "สลับเข้าเกมแล้วรอ (ตัวละครอย่าอยู่ใกล้มอน)")
    a = dev.grab()
    if a is None:
        sys.exit("จับภาพไม่ได้")
    if args.what == "joystick":
        dev.joystick(0, -1, 1500)
    elif args.what == "hold":
        dev.hold(*a.A("sword"), 1500)
    else:
        if not dev.camera_available():
            TOUCH_BACKEND["camera"] = args.backend
        dev.camera_drag(-300, 0, 500)
    time.sleep(1.0)
    b = dev.grab()
    os.makedirs(DEBUG_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{args.what}_before.png"), a.img)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{args.what}_after.png"), b.img)
    log(f"ภาพต่างกัน {frame_diff(a.img, b.img):.1f} (เกิน ~6 = ฉากขยับ/กล้องหมุนจริง) เซฟภาพก่อน-หลังใน debug/")


def main():
    ap = argparse.ArgumentParser(description="บอทวนทดสอบ: จุดมองหา -> หาหิน -> ตี -> กล่อง -> วน")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--max-rounds", type=int, default=0, help="วนกี่รอบแล้วหยุด (0 = ไม่หยุด)")
    r.add_argument("--period", choices=list(PERIODS), help="บังคับช่วงเวลา (ทดสอบ)")
    r.add_argument("--no-countdown", action="store_true")
    r.add_argument("--dry", action="store_true", help="ไม่ส่งคำสั่งจริง")
    r.add_argument("--image", help="ทดสอบกับภาพนิ่ง (ภาพแนวนอนที่หมุนกลับแล้ว) ไม่ต่อเครื่อง")
    c = sub.add_parser("calibrate")
    c.add_argument("--image")
    c.add_argument("--landscape", action="store_true")
    c.add_argument("--dry", action="store_true")
    t = sub.add_parser("test-input")
    t.add_argument("what", choices=["joystick", "hold", "camera"])
    t.add_argument("--backend", choices=["input", "sendevent"], default="sendevent")
    t.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    if args.cmd == "calibrate":
        cmd_calibrate(args)
        return
    if args.cmd == "test-input":
        cmd_test_input(args)
        return
    if args.image:
        raw = cv2.imread(args.image)
        if raw is None:
            sys.exit(f"เปิดไฟล์ภาพไม่ได้: {args.image}")
        dev = ImageDevice(Frame(cv2.resize(raw, (REF_W, REF_H)), raw.shape[:2], True))
    else:
        dev = Device(dry=args.dry)
        ok, who = dev.root_ok()
        log(f"รูท: {'OK' if ok else 'ไม่ใช่รูท -> ' + who}")
    if not HAS_TESS:
        log("เตือน: ไม่พบ pytesseract/tesseract -> อ่านชื่อแร่/นาฬิกา/พิกัดไม่ได้ (บอทจะเดินตามพิกัดไม่ได้)")
    profiles = load_profiles()
    bot = RockLoop(dev, profiles, args.period)
    if not (args.no_countdown or args.image):
        countdown(COUNTDOWN_SEC, "สลับเข้าเกม Utopia: Origin ได้เลย!")
    try:
        bot.run(args.max_rounds)
    except Stop as e:
        log(f"[หยุด] {e}")
    except KeyboardInterrupt:
        log("[หยุด] ผู้ใช้กด Ctrl+C")
    log(f"สรุป: {bot.stats}  (ดูบันทึกเวลา/โปรไฟล์ใน {TIME_LOG})")


if __name__ == "__main__":
    main()
