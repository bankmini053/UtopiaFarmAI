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
  python rock_loop.py test-input joystick|hold|camera [--backend input|sendevent]
  python rock_loop.py camera-cal                # วัดว่ากล้องหมุนไหม/ทิศไหน แล้วเซฟค่าไว้ใช้
  python rock_loop.py doctor                    # ตรวจ root/tesseract/จับภาพ/อ่านเวลา-พิกัด-ชื่อ
  python rock_loop.py stop                      # สั่งหยุดบอทที่รันอยู่ (หรือกด q + Enter / Ctrl+C ที่หน้าต่างบอท)
"""
import argparse
import difflib
import glob
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import Counter

import cv2
import numpy as np

try:
    import pytesseract
    HAS_TESS = shutil.which("tesseract") is not None      # ต้องมีโปรแกรม tesseract จริง ไม่ใช่แค่ pytesseract
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
NAME_STRICT = True            # อ่านชื่อได้แต่ไม่ใช่แร่ (เช่น ต้นไม้/มอน) = ไม่ใช่หิน ให้ออกจากเป้า (อ่านไม่ออกเลย = ถือว่าตีได้)
NAME_BLOCKLIST = ["cow", "chicken", "pig", "sheep", "goblin"]

# การสั่งเครื่อง
SHELL_PREFIX = ["su", "-c"]            # ไม่ได้รูท: ["sh", "-c"]
TOUCH_BACKEND = {
    "hold": "input",                   # กดค้างดาบ: "input" | "sendevent"
    "joystick": "input",
    "camera": "input",                 # "input" | "sendevent" | "none" = ไม่หมุนกล้อง ใช้จอยเดินหาแทน
}
TOUCH_EVENT_DEVICE = "/dev/input/event0"   # อุปกรณ์สัมผัสของเครื่องนี้ (จาก getevent)

# จังหวะ/เกณฑ์
HOLD_MS = 1200                # กดค้างดาบครั้งละ
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
NEAR_BOX = (430, 330, 790, 640)     # รอบตัวละคร: ถ้ามีสีหินเยอะ = หินอยู่ติดตัวแล้ว (หินใหญ่จนบังตัว/ถูกตัดทิ้ง)
NEAR_PIX = 9000                     # พื้นที่สีหิน (px อ้างอิง) ในกรอบด้านบนที่ถือว่า 'ถึงหินแล้ว'
LOST_NEAR_DIST = 330                # หินหายตอนห่างไม่เกินนี้ = เดินชนติดแล้ว (ไม่ใช่หลุดจอ) -> ตีเลย
PROGRESS_MIN = 10                   # เดินแล้วระยะต้องลดอย่างน้อยกี่ px ไม่งั้นนับว่า 'ไม่คืบหน้า'
UNSTICK_MAX = 4                     # ติดสิ่งกีดขวางกี่ครั้งติดกันแล้วยอมข้ามหินก้อนนี้
PERIOD_TH = {"dawn": "เช้ามืด", "day": "กลางวัน", "night": "กลางคืน"}
BLACKLIST_SEC = 90
BAG_FULL_RATIO = 0.25         # สัดส่วนสีเหลือง/ส้ม/แดงในวงแหวนกระเป๋าที่ถือว่า 'เต็ม'
REPAIR_MAX_FAILS = 3
COUNTDOWN_SEC = 10

# กล้อง: ลากนิ้วที่ camera_start ไปซ้าย/ขวา (จอแนวนอน) ระบบวัดเองว่าลาก 1px แล้วฉากขยับกี่px (มีเครื่องหมาย)
CAM_DRAG_PX = 260             # ระยะลากต่อครั้งตอนกวาด
CAM_DRAG_MS = 380
CAM_SWEEP_STEPS = 14          # กวาดซ้ายกี่ครั้งต่อ 1 รอบการมองหา
CAM_CENTER_BAND = 85          # หินอยู่ห่างกลางจอไม่เกินนี้ (px อ้างอิง) = 'อยู่กลาง' เดินตรงได้
CAM_ALIGN_MAX = 4             # หมุนปรับเข้าหาหินไม่เกินกี่ครั้งต่อก้าว
CAM_MAX_SHIFT = 260           # ฉากขยับต่อครั้งไม่เกินนี้ (px) กันวัดสลับรอบ (ROI กว้าง ~750)
CAM_MIN_SHIFT = 6.0           # ฉากต้องขยับอย่างน้อยกี่px ถึงนับว่ากล้องหมุน
CAM_FILE = os.path.join(DEBUG_DIR, "camera_cal.json")
CAM_SCENE_ROI = (250, 70, 1000, 290)     # ส่วนฉากไกล (ท้องฟ้า/ต้นไม้) ใช้วัดการหมุน ไม่โดน UI/ตัวละคร

# เลย์เอาต์ HUD: "A" = นาฬิกา/พิกัดมุมซ้ายบน (เครื่อง cloud ที่ทดสอบ)  "B" = นาฬิกาขวาบนข้างมินิแมพ + พิกัดใต้มินิแมพ (เครื่องจริง)  "auto" = ลองทั้งคู่
LAYOUT = "auto"
LAYOUT_ROI = {
    "A": {"clock": (0, 4, 64, 32), "coords": (40, 246, 290, 276)},
    "B": {"clock": (888, 3, 958, 25), "coords": (1072, 130, 1205, 160)},
}

# โปรไฟล์สีหินสำรอง (ถ้าไม่มีโฟลเดอร์ templates/rocks จะใช้ชุดนี้)
# ถ้ามีรูปใน templates/rocks/<dawn|day|night>/ บอทจะสร้างโปรไฟล์ใหม่จากรูปเองทุกครั้งที่เริ่ม
DEFAULT_PROFILES = {
    "dawn": {"s": [28, 158], "v": [40, 141], "hue": [[0, 9], [151, 179]]},
    "day": {"s": [0, 83], "v": [40, 172], "any_hue": True},
    "night": {"s": [4, 173], "v": [40, 143], "hue": [[99, 119]]},
}

DAY_MIN = 24 * 60
DEBUG = False


_real_sleep = time.sleep
STOP_FLAG = threading.Event()
STOP_FILE = os.path.join(BASE_DIR, "STOP")
STOP_WORDS = {"q", "quit", "stop", "s", "x", "หยุด"}


class Stop(BaseException):      # ไม่ใช่ Exception: กัน except Exception ที่ไหนก็ตามกลืนคำสั่งหยุด
    pass


QUIET = False                          # True ตอน run: log() ลงไฟล์อย่างเดียว หน้าจอแสดงเฉพาะ ui() (สั้น อ่านง่าย)
BOT_LOG = os.path.join(DEBUG_DIR, "bot.log")


def _dw(t):
    import unicodedata
    return sum(0 if unicodedata.category(c) in ("Mn", "Me", "Cf") else (2 if unicodedata.east_asian_width(c) in ("W", "F") else 1)
               for c in t)


def _fit(t, cols):
    out, w = "", 0
    for c in t:
        cw = _dw(c)
        if w + cw > cols - 1:
            return out.rstrip() + "…"
        out, w = out + c, w + cw
    return out


def _to_file(prefix, text):
    try:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        with open(BOT_LOG, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {prefix}{text}\n")
    except Exception:
        pass


def log(msg):
    """รายละเอียดเต็ม: ลงไฟล์ debug/bot.log เสมอ  หน้าจอแสดงเมื่อไม่ได้อยู่โหมด run (คำสั่ง doctor/test-input ฯลฯ)"""
    text = str(msg).lstrip(" ")
    _to_file("", text)
    if not QUIET:
        sys.stdout.write(f"\r{time.strftime('%H:%M:%S')}  {text}\n")
        sys.stdout.flush()


def ui(msg):
    """บรรทัดสั้นๆ สำหรับหน้าจอ (ชิดซ้าย ตัดให้พอดีความกว้างจอ ไม่มีเวลา) + เก็บลงไฟล์ด้วย"""
    text = str(msg).strip()
    _to_file("▸ ", text)
    cols = shutil.get_terminal_size((40, 20)).columns
    sys.stdout.write("\r" + _fit(text, cols) + "\n")
    sys.stdout.flush()


def stop_requested():
    if STOP_FLAG.is_set():
        return True
    if os.path.exists(STOP_FILE):
        STOP_FLAG.set()
        try:
            os.remove(STOP_FILE)                 # ใช้แล้วลบทิ้ง ไม่ให้ค้างไปหยุดรอบหน้า
        except OSError:
            pass
        return True
    return False


def check_stop():
    if stop_requested():
        raise Stop("ผู้ใช้สั่งหยุด")


def nap(sec):
    """sleep ที่ถูกขัดจังหวะได้ทันทีเมื่อสั่งหยุด"""
    end = time.time() + sec
    while True:
        check_stop()
        left = end - time.time()
        if left <= 0:
            return
        _real_sleep(min(0.1, left))


def install_stop_handlers():
    """หยุดได้ตลอด: Ctrl+C / พิมพ์ q แล้ว Enter / สร้างไฟล์ STOP (คำสั่ง `stop`) / SIGTERM"""
    try:
        os.remove(STOP_FILE)
    except OSError:
        pass
    STOP_FLAG.clear()
    hits = {"n": 0}

    def on_sig(signum, frame):
        hits["n"] += 1
        STOP_FLAG.set()
        if hits["n"] >= 2:
            sys.stdout.write("\r\nบังคับออก\r\n")
            os._exit(130)
        ui("⏹ หยุดแล้ว (กด Ctrl+C อีกครั้ง=บังคับ)")

    for sg in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sg, on_sig)
        except Exception:
            pass

    def reader():
        while not STOP_FLAG.is_set():
            try:
                line = sys.stdin.readline()
            except Exception:
                return
            if not line:
                return
            if line.strip().lower() in STOP_WORDS:
                STOP_FLAG.set()
                ui("⏹ รับ q -> หยุด")
                return

    if sys.stdin is not None and sys.stdin.isatty():
        threading.Thread(target=reader, daemon=True).start()
    ui("⏹ หยุดได้: Ctrl+C / พิมพ์ q+Enter")
    ui("   หรือหน้าต่างอื่น: python rock_loop.py stop")


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
        check_stop()
        p = subprocess.Popen(SHELL_PREFIX + [cmd], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, start_new_session=True)
        out = {}
        th = threading.Thread(target=lambda: out.update(r=p.communicate()), daemon=True)
        th.start()
        t0 = time.time()
        while th.is_alive():
            if stop_requested() or time.time() - t0 > timeout:
                try:
                    p.kill()
                except Exception:
                    pass
                th.join(1.0)
                check_stop()
                raise subprocess.TimeoutExpired(cmd, timeout)
            th.join(0.05)
        stdout = out.get("r", (b"", b""))[0] or b""
        return stdout if binary else stdout.decode("utf-8", "ignore")

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


# =============================================================================
# OCR ที่ไม่ขึ้นกับพื้นหลัง: white top-hat (เก็บเฉพาะ "เส้นขาวบางๆ" ตัดท้องฟ้า/เมฆ/พื้นหลังสว่างทิ้ง)
# =============================================================================
_LAST_LAYOUT = [None]


def layouts_to_try():
    order = ["A", "B"] if LAYOUT == "auto" else [LAYOUT]
    if _LAST_LAYOUT[0] in order:
        order.remove(_LAST_LAYOUT[0])
        order.insert(0, _LAST_LAYOUT[0])
    return order


def tophat_mask(roi, sc=4, ks=17, blur=True):
    """-> ภาพขาวดำ (ตัวหนังสือ=ขาว) ขยาย sc เท่า"""
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gb = cv2.resize(g, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
    k = int(ks * sc / 6) | 1
    th = cv2.morphologyEx(gb, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if blur:
        th = cv2.GaussianBlur(th, (3, 3), 0)
    _, b = cv2.threshold(th, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b


def tess(img_black_on_white, whitelist, psm=7):
    if not HAS_TESS:
        return ""
    v = cv2.copyMakeBorder(img_black_on_white, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=255)
    try:
        return pytesseract.image_to_string(v, config=f"--psm {psm} -c tessedit_char_whitelist={whitelist}").strip()
    except Exception:
        return ""


def segment_glyphs(mask):
    """แยกตัวอักษรตามคอลัมน์ -> [[x0,x1,y0,y1],...] ตัดจุด/ลูกน้ำ (เตี้ย) ทิ้ง"""
    H, W = mask.shape
    col = (mask > 0).sum(axis=0)
    runs, x = [], 0
    while x < W:
        if col[x] > 0:
            x0 = x
            while x < W and col[x] > 0:
                x += 1
            runs.append([x0, x])
        else:
            x += 1
    out = []
    for x0, x1 in runs:
        ys = np.where(mask[:, x0:x1].any(axis=1))[0]
        if len(ys):
            out.append([x0, x1, int(ys.min()), int(ys.max()) + 1])
    if not out:
        return []
    hmax = max(o[3] - o[2] for o in out)
    keep = [o for o in out if (o[3] - o[2]) >= 0.62 * hmax and (o[1] - o[0]) >= 0.10 * hmax]
    res = []
    for x0, x1, y0, y1 in keep:
        w, h = x1 - x0, y1 - y0
        if w > 1.15 * h:                      # ตัวเลขสองตัวติดกัน -> ตัดตรงจุดที่บางสุดกลางๆ
            seg = (mask[:, x0:x1] > 0).sum(axis=0).astype(float)
            lo, hi = int(0.35 * w), int(0.65 * w)
            cut = x0 + lo + int(np.argmin(seg[lo:hi]))
            res += [[x0, cut, y0, y1], [cut, x1, y0, y1]]
        else:
            res.append([x0, x1, y0, y1])
    return res


def read_digits_by_glyph(roi, kernels=(15, 17, 21)):
    """อ่านทีละตัวเลข (tesseract psm 10) หลายขนาดเคอร์เนล แล้วโหวตรายหลัก -> (x, y) หรือ None
    ใช้เมื่อแบ่งได้ 9 ตัวพอดี (xxxx,yyyyy) ตำแหน่งลูกน้ำ = ช่องว่างกว้างสุด"""
    votes, boxes_ref = None, None
    for ks in kernels:
        m = tophat_mask(roi, 4, ks)
        gl = segment_glyphs(m)
        if len(gl) > 9:
            gl = gl[-9:]                  # เลย์เอาต์ B มีไอคอน/เลขเกาะนำหน้า เอา 9 ตัวท้าย
        if len(gl) != 9:
            continue
        if votes is None:
            votes = [Counter() for _ in gl]
            gaps = [gl[i + 1][0] - gl[i][1] for i in range(8)]
            boxes_ref = int(np.argmax(gaps)) + 1
        for i, (x0, x1, y0, y1) in enumerate(gl):
            g = 255 - m[max(0, y0 - 2):y1 + 2, max(0, x0 - 2):x1 + 2]
            t = tess(g, "0123456789", psm=10)
            if t[:1].isdigit():
                votes[i][t[0]] += 1
    if votes is None or any(not v for v in votes):
        return None
    d = "".join(v.most_common(1)[0][0] for v in votes)
    k = boxes_ref
    if not (3 <= k <= 5):
        k = 4
    xy = int(d[:k]), int(d[k:])
    return xy if (xy[0] >= 1000 and xy[1] >= 1000) else None      # พิกัดจริงของแผนที่นี้เป็นเลข 4-5 หลักทั้งคู่ (กันอ่านผิดโซน)


class ClockReader:
    """อ่านเวลาในเกม: OCR หลายแบบ+โหวต -> กรองความสมเหตุสมผล -> เดินนาฬิกาเองด้วยอัตราที่วัดได้ -> สำรอง: ดูแสงฉาก"""

    def __init__(self):
        self.last_min = self.last_real = None
        self.pairs, self.rate, self.unwrapped = [], None, 0.0
        self.source, self._t = "none", 0.0
        self.override = None

    @staticmethod
    def _variants(roi):
        """สร้างภาพให้ tesseract ทีละแบบ (top-hat ก่อน ทนพื้นหลังท้องฟ้า แล้วค่อยแบบเดิมสำหรับพื้นหลังเข้ม)"""
        for ks in (17, 13, 21):
            for bl in (True, False):
                yield 255 - tophat_mask(roi, 5, ks, bl)
        g = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
        _, o = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        a = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, -8)
        yield 255 - o
        yield o
        yield 255 - a

    def _ocr(self, fr):
        if not HAS_TESS:
            return []
        out = []
        for lay in layouts_to_try():
            roi = fr.crop(LAYOUT_ROI[lay]["clock"])
            if roi.size == 0:
                continue
            got = []
            for v in self._variants(roi):
                d = "".join(c for c in tess(v, "0123456789:") if c.isdigit())
                if len(d) == 4 and int(d[:2]) < 24 and int(d[2:]) < 60:
                    got.append(int(d[:2]) * 60 + int(d[2:]))
                    if Counter(got).most_common(1)[0][1] >= 3:
                        break
            if got and Counter(got).most_common(1)[0][1] >= 2:
                _LAST_LAYOUT[0] = lay
                return got
            out += got
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


def _runs_longest(m):
    """ช่วง True ต่อเนื่องยาวสุดของเวกเตอร์ -> (ความยาว, เริ่ม, จบ)"""
    best, i, n = (0, 0, 0), 0, len(m)
    while i < n:
        if m[i]:
            j = i
            while j + 1 < n and m[j + 1]:
                j += 1
            if j - i + 1 > best[0]:
                best = (j - i + 1, i, j)
            i = j + 1
        else:
            i += 1
    return best


def _white_line_bar(fr, strip, x1, y1, out):
    """แถบเป้าหมายแบบเส้นขาว (เครื่องจริง): ชื่อ -> เส้นขาวบางยาว ~310px -> แถบเลือดแดงใต้เส้น"""
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    S, V, H = hsv[..., 1], hsv[..., 2], hsv[..., 0]
    white = (S < 60) & (V > 200)
    red = (((H <= 9) | (H >= 171)) & (S > 110) & (V > 130))
    rows = [(_runs_longest(white[y]), y) for y in range(white.shape[0])]
    lo, hi = 230 * fr.sx, 380 * fr.sx
    good = [(r, y) for r, y in rows if lo <= r[0] <= hi]
    if not good:
        return out
    (n, xa, xb), ly = max(good, key=lambda t: t[0][0])
    thick = sum(1 for r, y in good if abs(y - ly) <= 8)
    if thick > 8:                                       # หนาเกิน = เมฆ/ขอบอย่างอื่น ไม่ใช่เส้นแถบ
        return out
    ya, yb = ly + int(3 * fr.sy), min(strip.shape[0], ly + int(22 * fr.sy))
    if yb <= ya:
        return out
    below_red = red[ya:yb, xa:xb + 1]
    cols = np.where(below_red.sum(axis=0) >= max(2, int(3 * fr.sy)))[0]
    dark = float((V[ya:yb, xa:xb + 1].mean(axis=0) < 140).mean())
    if cols.size == 0 and dark < 0.7:
        return out
    hp = (float(cols.max()) + 1.0) / float(xb - xa + 1) if cols.size else 0.0
    out.update(engaged=True, hp=0.0 if hp < 0.03 else min(1.0, hp), x0=x1 + xa, x1=x1 + xb, y=y1 + ly)
    return out


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
        return _white_line_bar(fr, strip, x1, y1, out)
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


NAME_TPL_B64 = {"iron mine": [
    "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",
]}   # ต้นแบบตัวหนังสือชื่อ (ฝังในไฟล์ ไม่ต้องมีโฟลเดอร์ templates)


_NAME_TPL = {}


def _dog(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.GaussianBlur(g, (0, 0), 0.8) - cv2.GaussianBlur(g, (0, 0), 3.0)


def name_templates():
    """ภาพต้นแบบตัวหนังสือชื่อเป้าหมาย templates/ui/name_<ชื่อ>_<n>.png (ชื่อขึ้นต้น bad- = ห้ามตี) -> {ชื่อ: [dog]}"""
    if not _NAME_TPL:
        import base64
        for label, lst in NAME_TPL_B64.items():                  # ต้นแบบที่ฝังในไฟล์
            for b64 in lst:
                im = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
                t = _dog(im)
                _NAME_TPL.setdefault(label, []).append((t - t.mean()) / (t.std() + 1e-6))
        files = glob.glob(os.path.join(TEMPLATE_DIR, "ui", "name_*.png")) + glob.glob(os.path.join(ICONS_DIR, "name_*.png"))
        for f in sorted(files):                                  # เพิ่มเองได้: name_<ชื่อ>_<n>.png (ชื่อขึ้นต้น bad- = ห้ามตี)
            im = cv2.imread(f)
            if im is None:
                continue
            label = os.path.basename(f)[5:-4].rsplit("_", 1)[0].replace("_", " ").replace("bad-", "bad ")
            t = _dog(im)
            _NAME_TPL.setdefault(label, []).append((t - t.mean()) / (t.std() + 1e-6))
    return _NAME_TPL


def name_by_template(fr, bar, thr=0.5):
    """จับคู่ภาพตัวหนังสือชื่อกับต้นแบบ (ทนพื้นหลังท้องฟ้า/หญ้า) -> (ชื่อ, คะแนน) หรือ ('', 0)"""
    tpls = name_templates()
    if not tpls or not bar["engaged"]:
        return "", 0.0
    ya, yb = max(0, bar["y"] - int(40 * fr.sy)), min(fr.h, bar["y"] + int(4 * fr.sy))
    xa, xb = max(0, bar["x0"] - int(10 * fr.sx)), min(fr.w, bar["x1"] + int(10 * fr.sx))
    band = _dog(fr.img[ya:yb, xa:xb])
    best = ("", 0.0)
    for label, ts in tpls.items():
        for t in ts:
            for sc in (0.9, 1.0, 1.1):
                tt = cv2.resize(t, None, fx=sc * fr.sx, fy=sc * fr.sy, interpolation=cv2.INTER_AREA)
                if tt.shape[0] > band.shape[0] or tt.shape[1] > band.shape[1]:
                    continue
                v = float(cv2.matchTemplate(band, tt.astype(np.float32), cv2.TM_CCOEFF_NORMED).max())
                if v > best[1]:
                    best = (label, v)
    return (best[0], best[1]) if best[1] >= thr else ("", best[1])


def read_target_name(fr, bar):
    """อ่านชื่อเป้าหมายบนแถบ (ตัวพิมพ์เล็ก): จับคู่ภาพต้นแบบก่อน (แม่นกว่า) แล้ว OCR เสริม"""
    if not bar["engaged"]:
        return ""
    texts = []
    lab, sc = name_by_template(fr, bar)
    if lab:
        texts.append(lab)
    if HAS_TESS:
        ya, yb = max(0, bar["y"] - int(40 * fr.sy)), max(1, bar["y"] - int(1 * fr.sy))
        w = bar["x1"] - bar["x0"]
        roi = fr.img[ya:yb, bar["x0"] + int(w * 0.18):bar["x1"] - int(w * 0.18)]
        if roi.size:
            g = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            imgs = []
            _, o = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            imgs += [o, 255 - o]
            bg = cv2.GaussianBlur(g, (0, 0), 25)
            d = cv2.normalize(cv2.subtract(g, bg), None, 0, 255, cv2.NORM_MINMAX)
            imgs.append(255 - cv2.threshold(d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
            imgs.append(255 - cv2.createCLAHE(3.0, (2, 2)).apply(g))
            for ks in (13, 17):
                imgs.append(255 - tophat_mask(roi, 4, ks, True))
            for t in imgs:
                t = cv2.copyMakeBorder(t, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
                try:
                    texts.append(pytesseract.image_to_string(t, config="--psm 7").strip().lower())
                except Exception:
                    pass
    return " | ".join(x for x in texts if x)


def name_verdict(text, period, strict=None):
    """-> 'ore' | 'bad' | 'unknown' ตาม ORE_NAME_KEYWORDS / NAME_BLOCKLIST (เทียบแบบเผื่อ OCR เพี้ยน)"""
    if not text:
        return "unknown"
    words = [w for w in text.replace("|", " ").split() if w]
    close = lambda k, thr: k in text or any(difflib.SequenceMatcher(None, k, w).ratio() >= thr for w in words)
    if any(close(b, 0.75) for b in NAME_BLOCKLIST):
        return "bad"
    if any(close(k, 0.7) for k in ORE_NAME_KEYWORDS.get(period, [])):
        return "ore"
    if (NAME_STRICT if strict is None else strict) and any(sum(ch.isalpha() for ch in seg) >= 4 for seg in text.split("|")):
        return "bad"                                    # อ่านชื่อออกเป็นตัวหนังสือ แต่ไม่ใช่แร่ = ไม่ใช่หิน
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
ROI["coords"] = LAYOUT_ROI["A"]["coords"]
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
    for ks in (17, 21, 13):
        yield 255 - tophat_mask(roi, 4, ks, True)
    g = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    _, o = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield 255 - o
    yield o


def _parse_coord(text):
    import re
    m = re.findall(r"\d+", text)
    if len(m) >= 3:
        m = m[-2:]                      # เลย์เอาต์ B มีเลขเกาะ/โซนนำหน้า เอาสองกลุ่มท้าย
    if len(m) >= 2 and 4 <= len(m[0]) <= 5 and 4 <= len(m[1]) <= 6:
        return int(m[0]), int(m[1])
    d = "".join(m)
    if len(m) == 1 and len(d) == 9:
        return int(d[:4]), int(d[4:])
    return None


def read_coords_once(fr):
    """อ่านพิกัด 'x, y': (1) อ่านทีละตัวเลขแล้วโหวต (2) ทั้งบรรทัดหลายแบบแล้วโหวต -> (x, y) หรือ None"""
    if not HAS_TESS:
        return None
    for lay in layouts_to_try():
        roi = fr.crop(LAYOUT_ROI[lay]["coords"])
        if roi.size == 0:
            continue
        c = read_digits_by_glyph(roi)
        if c:
            _LAST_LAYOUT[0] = lay
            return c
        votes = []
        for v in _coord_variants(roi):
            c = _parse_coord(tess(v, "0123456789,. "))
            if c:
                votes.append(c)
        if votes:
            best, n = Counter(votes).most_common(1)[0]
            if n >= 2 or len(set(votes)) == 1:
                _LAST_LAYOUT[0] = lay
                return best
    return None


# =============================================================================
# กล้อง: วัดการหมุนจากการขยับของฉาก (phase correlation) แล้วควบคุมแบบวงปิด
# =============================================================================
def scene_gray(fr):
    x1, y1, x2, y2 = fr.box(CAM_SCENE_ROI)
    g = cv2.cvtColor(fr.img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(g, (5, 5), 0).astype(np.float32)


def scene_shift(a, b):
    """ฉากใน b ขยับเทียบกับ a -> (dx, dy, ความมั่นใจ)  dx บวก = เนื้อฉากเลื่อนขวา = กล้องหันซ้าย / dy = เงย-ก้ม"""
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
    return float(dx), float(dy), float(resp)


class Camera:
    """ratio = ฉากขยับแนวนอนกี่ px ต่อการลากนิ้ว 1 px (มีเครื่องหมาย วัดจริง)
    axis = ลากแนวไหนของจอแนวนอนแล้วกล้อง 'หมุนซ้าย-ขวา' ("x" ลากซ้าย-ขวา / "y" ลากขึ้น-ลง) วัดเอง
    (บางเครื่องลากแนวนอนแล้วกล้องเงย-ก้มแทน จึงต้องลองทั้งสองแกน)"""

    def __init__(self, dev):
        self.dev = dev
        self.ratio = None
        self.axis = "x"
        self.ok = False

    def _snap(self):
        fr = self.dev.grab()
        if fr is None:
            raise Stop("จับภาพหน้าจอไม่ได้ (เช็คสิทธิ์รูท / su)")
        return fr

    def _drag_axis(self, d, axis):
        if axis == "x":
            self.dev.camera_drag(d, 0, CAM_DRAG_MS)
        else:
            self.dev.camera_drag(0, d, CAM_DRAG_MS)

    def drag(self, d, axis=None):
        """ลากกล้อง d px ตามแกน -> (ฉากขยับแนวนอน px, แนวตั้ง px)"""
        a = scene_gray(self._snap())
        self._drag_axis(d, axis or self.axis)
        nap(0.35)
        dx, dy, _ = scene_shift(a, scene_gray(self._snap()))
        return dx, dy

    def try_calibrate(self, verbose=True):
        """ลองลากสองแกน (ซ้าย-ขวา แล้วขึ้น-ลง) หาแกนที่ทำให้ฉากเลื่อนแนวนอน (= หมุนกล้อง)"""
        for axis in ("x", "y"):
            rs, pitch = [], 0.0
            for d in (-140, 140, -140, 140):
                dx, dy = self.drag(d, axis)
                pitch = max(pitch, abs(dy))
                if verbose:
                    log(f"[กล้อง] ลาก{'ซ้าย-ขวา' if axis == 'x' else 'ขึ้น-ลง'} {d:+d}px -> ฉากขยับ แนวนอน {dx:+.0f} แนวตั้ง {dy:+.0f}")
                if abs(dx) >= CAM_MIN_SHIFT and abs(dx) >= 1.5 * abs(dy):
                    rs.append(dx / d)
            pos, neg = [r for r in rs if r > 0], [r for r in rs if r < 0]
            if len(rs) >= 3 and (not pos or not neg):
                self.ratio, self.axis, self.ok = float(np.median(rs)), axis, True
                return True
            if verbose and pitch >= CAM_MIN_SHIFT:
                log(f"[กล้อง] แกน{'ซ้าย-ขวา' if axis == 'x' else 'ขึ้น-ลง'} = เงย/ก้ม (ฉากขยับแนวตั้ง {pitch:.0f}px) ไม่ใช่หมุนซ้ายขวา")
        return False

    def calibrate(self, backends=None):
        """ลองหลายวิธีสั่งลาก (input/sendevent, ช้า/เร็ว) จนกว่ากล้องจะหมุนซ้ายขวา แล้วจำค่าที่ใช้ได้"""
        global CAM_DRAG_MS
        combos = backends or [("input", 380), ("input", 700), ("sendevent", 600), ("input", 200)]
        for be, ms in combos:
            TOUCH_BACKEND["camera"] = be
            CAM_DRAG_MS = ms
            log(f"[กล้อง] ลองวิธี {be} ลาก {ms}ms")
            if self.try_calibrate():
                log(f"[กล้อง] ✅ หมุนซ้ายขวาได้ ({be}, {ms}ms, ลากแนว{'นอน' if self.axis == 'x' else 'ตั้ง'}) ลาก 100px -> ฉากขยับ {self.ratio * 100:+.0f}px")
                self.save()
                return True
        log("[กล้อง] ❌ กล้องไม่หมุนซ้ายขวาทุกวิธี -> บอทจะหามองหินด้วยการเดินแทน")
        TOUCH_BACKEND["camera"] = "none"
        self.ok = False
        return False

    def save(self):
        os.makedirs(DEBUG_DIR, exist_ok=True)
        with open(CAM_FILE, "w") as f:
            json.dump({"v": 2, "backend": TOUCH_BACKEND["camera"], "ms": CAM_DRAG_MS, "ratio": self.ratio, "axis": self.axis}, f)

    def load(self):
        global CAM_DRAG_MS
        try:
            d = json.load(open(CAM_FILE))
            if d.get("v") != 2:
                return False                        # ไฟล์เก่า (ยังไม่รู้เรื่องแกน) -> วัดใหม่
            TOUCH_BACKEND["camera"], CAM_DRAG_MS, self.ratio, self.axis = d["backend"], d["ms"], d["ratio"], d["axis"]
            self.ok = abs(self.ratio) > 1e-3 and TOUCH_BACKEND["camera"] != "none"
            return self.ok
        except Exception:
            return False

    def ensure(self):
        if getattr(self.dev, "dry", False):
            return False
        if self.load():
            log(f"[กล้อง] ใช้ค่าที่วัดไว้ ({TOUCH_BACKEND['camera']}, {CAM_DRAG_MS}ms, แกน {self.axis}, อัตรา {self.ratio:+.2f})")
            return True
        return self.calibrate()

    def sweep_left(self):
        """หมุนกล้องไปทางซ้าย 1 ก้าว (ฉากเลื่อนไปทางขวา) -> ฉากขยับจริงกี่ px"""
        d = math.copysign(min(CAM_DRAG_PX, CAM_MAX_SHIFT / abs(self.ratio)), self.ratio)
        sh, _ = self.drag(d)
        if abs(sh) < CAM_MIN_SHIFT:
            log("[กล้อง] ลากแล้วฉากไม่หมุน")
        return sh

    def turn_by(self, shift_px):
        """หมุนให้ฉากขยับ shift_px (บวก = กล้องหันซ้าย / ลบ = หันขวา) -> ฉากขยับจริงกี่ px"""
        shift_px = max(-CAM_MAX_SHIFT, min(CAM_MAX_SHIFT, shift_px))
        d = shift_px / self.ratio
        d = max(-CAM_DRAG_PX, min(CAM_DRAG_PX, d))
        if abs(d) < 12:
            d = math.copysign(12, d)
        sh, _ = self.drag(d)
        return sh


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
            nap(0.4)
        return None

    def read_stable(self, timeout=5.0):
        """พิกัดบนจอดีเล: อ่านซ้ำจนสองครั้งติดกันเท่ากัน"""
        t0, prev = time.time(), None
        nap(COORD_SETTLE)
        while time.time() - t0 < timeout:
            c = self.read(tries=2)
            if c and c == prev:
                return c
            prev = c or prev
            nap(0.5)
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


# =============================================================================
# ตัวบอทวน
# =============================================================================
class RockLoop:
    def __init__(self, dev, profiles, period_override=None):
        self.dev, self.profiles = dev, profiles
        self.clock = ClockReader()
        self.clock.override = period_override
        self.nav = Nav(dev)
        self.cam = Camera(dev)
        self.fr = None
        self.names, self.period, self._tstr, self._src = ["day"], "day", "--:--", "none"
        self.black = []
        self.search_pt = SEARCH_COORD
        self.box_pt = BOX_COORD
        self.wander_ang, self.wander_left = 0.0, 0
        self._tkey, self._tui = None, 0.0
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
        """บันทึกเวลาเกมที่อ่านได้/โปรไฟล์ที่ใช้/หินที่แต่ละโปรไฟล์เห็น ลงไฟล์ + หน้าจอแสดงบรรทัดเดียวเมื่อเปลี่ยน"""
        fr = self.fr
        counts = {p: len(find_rocks(fr, [p], self.profiles)[0]) for p in PERIODS}
        vis = visual_period(fr.img)
        best = max(counts, key=counts.get) if any(counts.values()) else "-"
        msg = (f"เวลาเกม {self._tstr} ช่วง={self.period} (แหล่ง {self._src}) ใช้โปรไฟล์ {self.names} | แสงฉาก={vis} | "
               f"หินที่เห็น dawn={counts['dawn']} day={counts['day']} night={counts['night']}")
        log("[เวลา] " + msg)
        key = (self._tstr, self.period, self._src)
        if key != self._tkey and (time.time() - self._tui > 40 or self._tkey is None or key[1:] != self._tkey[1:]):
            self._tkey, self._tui = key, time.time()
            if self._tstr == "--:--":
                ui(f"⚠️ อ่านเวลาไม่ได้ ใช้แสงฉาก={PERIOD_TH.get(vis, vis)}")
            else:
                ui(f"🕐 {self._tstr} {PERIOD_TH[self.period]}")
        if best != "-" and best not in self.names and counts[best] >= 2 and counts[best] > 2 * max(
                [counts[n] for n in self.names if n in counts] + [0]):
            ui(f"⚠️ รูปหินไม่ตรงเวลา ({PERIOD_TH[best]}เจอมากกว่า)")
        log_time_row([time.strftime("%Y-%m-%d %H:%M:%S"), self._tstr, self.period, self._src, "+".join(self.names), vis,
                      f"{counts['dawn']}/{counts['day']}/{counts['night']}", self.nav.last or "", event])

    # ---------------------------------------------------------------- ไปจุดต่างๆ
    def go_search(self):
        if not self.search_pt:
            return True
        ui("📍 ไปจุดมองหา")
        ok = self.nav.goto(self.search_pt, GOTO_TOL, "จุดมองหา")
        if not ok:
            ui("⚠️ ไปจุดมองหาไม่ถึง (อ่านพิกัดไม่ได้?)")
        return ok

    def go_box(self):
        if not self.box_pt:
            log("[กล่อง] ยังไม่ได้ตั้ง BOX_COORD -> ข้ามขั้นไปกล่อง")
            ui("⏭ ข้ามกล่อง (ยังไม่ตั้ง BOX_COORD)")
            return None
        ui("🏠 ไปกล่อง")
        ok = self.nav.goto(self.box_pt, BOX_TOL, "กล่อง")
        for k in range(6):                     # ใกล้กล่องแล้วรูปมือจะขึ้น ถ้ายังไม่ขึ้นขยับเข้าใกล้อีกนิด
            fr = self.frame()
            seen, score = hand_present(fr)
            log(f"[กล่อง] ตรวจรูปมือ คะแนน {score:.2f} (เกณฑ์ {HAND_THRESHOLD}) {'เห็น' if seen else 'ยังไม่เห็น'}")
            if seen:
                self.stats["กลับบ้านสำเร็จ"] += 1
                ui("🏠 ✅ กลับบ้านสำเร็จ")
                self.report_time("กลับบ้านสำเร็จ")
                return True
            self.dev.joystick(random.uniform(-1, 1), random.uniform(-1, 1), 450)
            nap(0.8)
        ui("⚠️ ถึงกล่องแล้วไม่เห็นรูปมือ")
        self.report_time("กลับบ้านไม่สำเร็จ")
        return False

    # ---------------------------------------------------------------- มองหา / เดินเข้าหา / ตี
    def explore(self):
        if self.wander_left <= 0:
            self.wander_ang = random.uniform(-math.pi * 0.8, math.pi * 0.8)
            self.wander_left = random.randint(2, 3)
        self.wander_left -= 1
        self.dev.joystick(math.sin(self.wander_ang), -math.cos(self.wander_ang), 800)
        nap(0.3)

    def find_target(self):
        """ยืนมองจากจุดนี้: หมุนกล้องกวาดไปทางซ้ายหาหิน เจอแล้วคืนก้อนที่ใกล้สุด ไม่เจอเดินสุ่มสั้นๆ แล้วกวาดใหม่"""
        for rnd in range(ROCKS_PER_VISIT):
            steps = CAM_SWEEP_STEPS if self.cam.ok else 1
            ui("🔎 กวาดหาหิน ←ซ้าย" if self.cam.ok else "🔎 มองหาหิน")
            for k in range(steps):
                fr = self.frame()
                rocks, _ = find_rocks(fr, self.names, self.profiles)
                rocks = [r for r in rocks if not self.blacklisted(r)]
                if k == 0:
                    self.report_time(f"มองหาหิน รอบ{rnd + 1} เจอ{len(rocks)}")
                if rocks:
                    ui(f"🪨 เจอหิน {len(rocks)} ก้อน")
                    log(f"[หา] ใกล้สุดฐาน={rocks[0]['base']}")
                    return rocks[0]
                if self.cam.ok:
                    log(f"[กล้อง] ไม่เจอหิน -> หมุนกล้องไปทางซ้าย ({k + 1}/{steps})")
                    self.cam.sweep_left()
            ui(f"🔎 ไม่เจอ เดินสุ่ม ({rnd + 1}/{ROCKS_PER_VISIT})")
            self.explore()
            if self.search_pt and self.nav.last:      # กันเดินไกลจากจุดมองหาเกินไป
                c = read_coords_once(self.dev.grab())
                if c and math.hypot(c[0] - self.search_pt[0], c[1] - self.search_pt[1]) > LEASH:
                    ui("📍 ไกลเกิน กลับจุดมองหา")
                    self.go_search()
        return None

    def near_pixels(self, fr):
        """พื้นที่สีหินรอบตัวละคร (px อ้างอิง) ใช้ดูว่าหินใหญ่จนชิด/บังตัวละครแล้วไหม"""
        m = rock_mask(fr.img, self.names, self.profiles)
        x1, y1, x2, y2 = fr.box(NEAR_BOX)
        sub = cv2.morphologyEx(m[y1:y2, x1:x2], cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return float((sub > 0).sum()) / (fr.sx * fr.sy)

    def _arrived(self, fr):
        feet = fr.A("char_feet")
        return {"base": feet, "bbox": (feet[0] - 40, feet[1] - 80, 80, 80), "area": 0, "dist": 0.0, "near": True}

    def unstick(self, n, off):
        """ติดต้นไม้/หิน/พุ่มไม้: เบี่ยงข้างสลับซ้ายขวา เดินอ้อม ถ้ายังติดถอยหลังก่อน ติดเกิน UNSTICK_MAX = ยอมแพ้ก้อนนี้"""
        if n >= UNSTICK_MAX:
            ui("🧱 ติดไม่หลุด ข้ามก้อนนี้")
            return False
        sgn = (1 if off >= 0 else -1) * (1 if n % 2 == 1 else -1)
        ui(f"🧱 ติดสิ่งกีดขวาง เบี่ยง{'ขวา' if sgn > 0 else 'ซ้าย'} {n}/{UNSTICK_MAX}")
        if n >= 3:
            self.dev.joystick(0, 1, 500)                       # ถอยออกมาก่อน
            nap(0.15)
        self.dev.joystick(sgn, 0, 600 + 250 * n)
        nap(0.15)
        self.dev.joystick(0, -1, 700)                          # เดินหน้าผ่านสิ่งกีดขวาง
        nap(0.15)
        return True

    def approach(self, rock):
        """เดินเข้าหาหิน: หินซ้าย/ขวา -> หมุนกล้อง, อยู่กลาง -> เดินตรง, ติดของขวาง -> เบี่ยง, ชิดแล้ว (หินใหญ่บังตัว/หายตอนชิด) -> ตี"""
        d, lost, last = self.dev, 0, rock["base"]
        blocked = no_prog = 0
        last_dist = None
        for step in range(APPROACH_MAX_STEPS):
            fr = self.frame()
            feet = fr.A("char_feet")
            if step >= 1 and self.near_pixels(fr) >= NEAR_PIX:
                ui("📍 ชิดหินแล้ว")
                return self._arrived(fr)
            rocks, _ = find_rocks(fr, self.names, self.profiles)
            rocks = [r for r in rocks if not self.blacklisted(r)]
            same = [r for r in rocks if math.hypot(r["base"][0] - last[0], r["base"][1] - last[1]) < 170 * fr.sx]
            tgt = (same or rocks or [None])[0]
            if tgt is None:
                lost += 1
                if last_dist is not None and last_dist <= LOST_NEAR_DIST * fr.sy:
                    ui("📍 ชิดหินแล้ว (หินบังกล้อง)")
                    return self._arrived(fr)
                if lost >= 3:
                    ui("⚠️ หินหลุดจอ")
                    return None
                d.joystick(0, -1, 400)
                continue
            lost, last = 0, tgt["base"]
            dx, dy = tgt["base"][0] - feet[0], tgt["base"][1] - feet[1]
            dist = math.hypot(dx, dy)
            if dist <= NEAR_DIST * fr.sy:
                ui("📍 ชิดหินแล้ว")
                return tgt
            ms = int(min(STEP_MS_MAX, max(STEP_MS_MIN, dist * 1.6)))
            cx = tgt["bbox"][0] + tgt["bbox"][2] / 2.0
            off = cx - feet[0]                                   # + = หินอยู่ขวาของกลางจอ
            band = CAM_CENTER_BAND * fr.sx
            side = "ซ้าย" if off < -band else ("ขวา" if off > band else "กลาง")
            if self.cam.ok:
                log(f"[เดิน] ก้าว {step + 1}: หินอยู่ด้าน{side} (เยื้อง {off:+.0f}px) ห่าง {dist:.0f}px")
                if side != "กลาง":
                    ui(f"🎥 หินอยู่{side} → หมุนกล้อง{side}")
                    for _ in range(CAM_ALIGN_MAX):
                        self.cam.turn_by(-off)
                        fr = self.frame()
                        rr, _ = find_rocks(fr, self.names, self.profiles)
                        near = [r for r in rr if math.hypot(r["base"][0] - last[0], r["base"][1] - last[1]) < 260 * fr.sx]
                        if not near:
                            break
                        last = near[0]["base"]
                        off = near[0]["bbox"][0] + near[0]["bbox"][2] / 2.0 - fr.A("char_feet")[0]
                        if abs(off) <= band:
                            break
                    last_dist = None
                    continue
                ui(f"🚶 เดินตรง ห่าง {dist:.0f}")
                dx, dy = 0.0, -1.0
            else:
                ui(f"🚶 เดิน ห่าง {dist:.0f}")
                log(f"[เดิน] dx={dx:+.0f} dy={dy:+.0f} นาน {ms}ms (กล้องไม่หมุน ใช้จอยทิศตรง)")
            before = fr.img
            d.joystick(dx, dy, ms)
            nap(0.15)
            fr2 = d.grab()
            moved = fr2 is not None and frame_diff(before, fr2.img) >= STUCK_FRAME_DIFF
            if last_dist is not None and dist > last_dist - PROGRESS_MIN * fr.sy:
                no_prog += 1
            else:
                no_prog = 0
            last_dist = dist
            if not moved or no_prog >= 2:
                blocked += 1
                no_prog = 0
                if not self.unstick(blocked, off):
                    return None
                last_dist = None
        return None

    def hit_once(self, rock):
        """ตีแล้วดูแถบชื่อ -> 'ore' (ตีสำเร็จ) | 'unknown' (ตีโดนแต่ชื่อไม่ชัด) | 'bad' (ไม่ใช่แร่) | 'miss'"""
        d = self.dev
        for t in range(ENGAGE_TRIES):
            fr = self.frame()
            d.hold(*fr.A("sword"), 1000)
            nap(0.4)
            fr = self.frame()
            bar = read_target_bar(fr)
            if bar["engaged"]:
                name = read_target_name(fr, bar)
                v = name_verdict(name, self.period)
                short = (name.split("|")[0].strip() or "?")[:14]
                ui(f"⚔️ ตี: {short.title()} เลือด {bar['hp'] * 100:.0f}%")
                log(f"[ตี] ชื่อที่อ่านได้ '{name}' -> {v}")
                return v
            log(f"[ตี] ยังไม่โดนเป้า ({t + 1}/{ENGAGE_TRIES}) -> ขยับเข้าใกล้อีกนิด")
            rocks, _ = find_rocks(fr, self.names, self.profiles)
            if rocks and not rock.get("near"):
                feet = fr.A("char_feet")
                d.joystick(rocks[0]["base"][0] - feet[0], rocks[0]["base"][1] - feet[1], 300)
            else:
                d.joystick(0, -1, 250)                       # หินชิด/บังตัว: ดันไปข้างหน้าอีกนิดแล้วตีใหม่
            nap(0.2)
        ui("⚠️ ตีไม่โดนเป้า")
        return "miss"

    def leave(self, rock, why):
        """ไม่ใช่หิน/ตีไม่โดน: ถอยออกจากเป้า เบี่ยงข้าง และจำตำแหน่งไว้ ไม่กลับมาตีอีก"""
        ui(f"🚪 {why} ออกจากเป้า")
        self.dev.joystick(0, 1, 1000)
        nap(0.2)
        self.dev.joystick(random.choice([-1, 1]), 0, 700)
        nap(0.2)
        self.black.append((rock["base"], time.time() + BLACKLIST_SEC))

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
                ui("✅ ตีสำเร็จ (ชื่อตรงแร่)")
                self.report_time("ตีสำเร็จ")
                return True
            if res == "unknown":
                self.stats["ตีสำเร็จ"] += 1
                ui("✅ ตีสำเร็จ (อ่านชื่อไม่ชัด)")
                self.report_time("ตีสำเร็จ(ชื่อไม่ยืนยัน)")
                return True
            self.leave(rock, "ไม่ใช่หิน" if res == "bad" else "ตีไม่โดน")
            self.report_time(f"ตีไม่สำเร็จ:{res}")
        return False

    # ---------------------------------------------------------------- ลูปหลัก
    def setup(self):
        self.frame()
        ui("🎥 ตรวจกล้อง...")
        self.cam.ensure()
        ui("🎥 กล้อง: " + (f"ใช้ได้ (ลากแนว{'นอน' if self.cam.axis == 'x' else 'ตั้ง'})" if self.cam.ok else "หมุนไม่ได้ → เดินหาแทน"))
        ui(f"🕐 {self._tstr} {PERIOD_TH[self.period]} ({self._src})")
        if not self.search_pt:
            c = self.nav.read_stable()
            if c:
                self.search_pt = c
                ui(f"📍 จุดมองหา = ตำแหน่งนี้ {c[0]},{c[1]}")
            else:
                ui("⚠️ อ่านพิกัดไม่ได้ → ไม่เดินตามพิกัด")
        if not self.box_pt:
            log("ไม่ได้ตั้ง BOX_COORD -> จะข้ามขั้นตอนไปกล่อง")

    def run(self, max_rounds=0):
        self.setup()
        while True:
            self.stats["รอบ"] += 1
            r = self.stats["รอบ"]
            ui(f"━━ รอบ {r} │ ตี {self.stats['ตีสำเร็จ']} │ บ้าน {self.stats['กลับบ้านสำเร็จ']} ━━")
            self.go_search()
            ok = self.mine_one()
            if ok:
                self.go_search()
                self.go_box()
            else:
                ui("↩ รอบนี้ยังไม่สำเร็จ เริ่มใหม่")
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

    def camera_available(self):
        return False


# =============================================================================
# คำสั่งเสริม
# =============================================================================
def countdown(sec, msg):
    log(msg)
    for s in range(sec, 0, -1):
        sys.stdout.write(f"\r{time.strftime('%H:%M:%S')}  เริ่มใน {s} วินาที...   ")
        sys.stdout.flush()
        nap(1)
    sys.stdout.write("\r\n")


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
    install_stop_handlers()
    countdown(5, "สลับเข้าเกมแล้วรอ (ตัวละครอย่าอยู่ใกล้มอน)")
    a = dev.grab()
    if a is None:
        sys.exit("จับภาพไม่ได้")
    os.makedirs(DEBUG_DIR, exist_ok=True)
    if args.what == "camera":
        global CAM_DRAG_MS
        TOUCH_BACKEND["camera"] = args.backend
        CAM_DRAG_MS = args.ms
        cam = Camera(dev)
        log(f"ทดสอบกล้อง: วิธี {args.backend} ลาก {args.ms}ms (ลากซ้าย-ขวา-ซ้าย แล้ววัดว่าฉากขยับกี่ px)")
        cv2.imwrite(os.path.join(DEBUG_DIR, "camera_before.png"), a.img)
        good = cam.try_calibrate()
        b = dev.grab()
        cv2.imwrite(os.path.join(DEBUG_DIR, "camera_after.png"), b.img)
        if good:
            cam.save()
            log(f"✅ กล้องหมุนได้ (ลาก 100px -> ฉากขยับ {cam.ratio * 100:+.0f}px) บันทึกค่าแล้ว ใช้ `run` ได้เลย")
        else:
            log("❌ กล้องไม่หมุน ลอง: python rock_loop.py camera-cal   (ลองทุกวิธีให้อัตโนมัติ)")
        return
    if args.what == "joystick":
        dev.joystick(0, -1, 1500)
    else:
        dev.hold(*a.A("sword"), 1500)
    nap(1.0)
    b = dev.grab()
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{args.what}_before.png"), a.img)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{args.what}_after.png"), b.img)
    log(f"ภาพต่างกัน {frame_diff(a.img, b.img):.1f} (เกิน ~6 = ฉากขยับจริง) เซฟภาพก่อน-หลังใน debug/")


def cmd_camera_cal(args):
    dev = Device(dry=args.dry)
    ok, who = dev.root_ok()
    log(f"รูท: {'OK' if ok else 'ไม่ใช่ -> ' + who}")
    install_stop_handlers()
    countdown(5, "สลับเข้าเกมแล้วรอ (ยืนในที่โล่ง ให้เห็นฉากไกล/ท้องฟ้า)")
    cam = Camera(dev)
    if dev.grab() is None:
        sys.exit("จับภาพไม่ได้")
    cam.calibrate()


def cmd_stop(args):
    open(STOP_FILE, "w").close()
    log(f"ส่งคำสั่งหยุดแล้ว ({STOP_FILE}) บอทที่กำลังรันจะหยุดภายในไม่กี่วินาที")


def cmd_doctor(args):
    """ตรวจทุกอย่างทีเดียว: root / tesseract / จับภาพ / อ่านเวลา-พิกัด-ชื่อ -> บอกว่าอะไรพัง"""
    log("--- ตรวจระบบ ---")
    log(f"pytesseract: {'มี' if pytesseract else 'ไม่มี  -> pip install pytesseract'}")
    tb = shutil.which("tesseract")
    log(f"tesseract โปรแกรม: {tb or 'ไม่พบ  -> pkg install tesseract (Termux) / apt install tesseract-ocr'}")
    if tb:
        try:
            v = subprocess.run([tb, "--version"], capture_output=True, timeout=10, stdin=subprocess.DEVNULL)
            log("tesseract รุ่น: " + (v.stdout or v.stderr).decode("utf-8", "ignore").splitlines()[0])
        except Exception as e:
            log(f"tesseract รันไม่ได้: {e}")
    if args.image:
        raw = cv2.imread(args.image)
        if raw is None:
            sys.exit(f"เปิดไฟล์ภาพไม่ได้: {args.image}")
        fr = Frame(cv2.resize(raw, (REF_W, REF_H)), raw.shape[:2], True) if args.landscape else make_frame(raw)
    else:
        dev = Device(dry=args.dry)
        ok, who = dev.root_ok()
        log(f"รูท (su): {'OK' if ok else 'ไม่ใช่ -> ' + who}")
        fr = dev.grab()
        if fr is None:
            sys.exit("จับภาพไม่ได้ (screencap ผ่าน su ใช้ไม่ได้)")
        log(f"จับภาพได้ {fr.raw_shape[1]}x{fr.raw_shape[0]} -> แนวนอน {fr.w}x{fr.h}")
    os.makedirs(DEBUG_DIR, exist_ok=True)
    clk = ClockReader()
    got = clk._ocr(fr)
    log(f"นาฬิกา: อ่านได้ {[fmt_min(m) for m in got][:6]} เลย์เอาต์ที่ใช้ได้={_LAST_LAYOUT[0]}")
    log(f"พิกัด: {read_coords_once(fr)}")
    bar = read_target_bar(fr)
    log(f"แถบเป้าหมาย: ขึ้น={bar['engaged']} ชื่อ='{read_target_name(fr, bar)}'")
    for lay in ("A", "B"):
        for k in ("clock", "coords"):
            roi = fr.crop(LAYOUT_ROI[lay][k])
            if roi.size:
                cv2.imwrite(os.path.join(DEBUG_DIR, f"doctor_{k}_{lay}.png"), cv2.resize(roi, None, fx=4, fy=4))
    log(f"เซฟภาพโซนอ่านเวลา/พิกัดไว้ที่ {DEBUG_DIR}/doctor_*.png (ถ้ากรอบไม่ตรงตัวเลข ใช้ --layout A|B หรือแก้ LAYOUT_ROI)")


def main():
    ap = argparse.ArgumentParser(description="บอทวนทดสอบ: จุดมองหา -> หาหิน -> ตี -> กล่อง -> วน")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--max-rounds", type=int, default=0, help="วนกี่รอบแล้วหยุด (0 = ไม่หยุด)")
    r.add_argument("--period", choices=list(PERIODS), help="บังคับช่วงเวลา (ทดสอบ)")
    r.add_argument("--no-countdown", action="store_true")
    r.add_argument("--verbose", action="store_true", help="แสดงรายละเอียดเต็มบนจอ (ปกติลงไฟล์ debug/bot.log)")
    r.add_argument("--dry", action="store_true", help="ไม่ส่งคำสั่งจริง")
    r.add_argument("--image", help="ทดสอบกับภาพนิ่ง (ภาพแนวนอนที่หมุนกลับแล้ว) ไม่ต่อเครื่อง")
    c = sub.add_parser("calibrate")
    c.add_argument("--image")
    c.add_argument("--landscape", action="store_true")
    c.add_argument("--dry", action="store_true")
    t = sub.add_parser("test-input")
    t.add_argument("what", choices=["joystick", "hold", "camera"])
    t.add_argument("--backend", choices=["input", "sendevent"], default="input")
    t.add_argument("--ms", type=int, default=380, help="ลากนานกี่ ms (กล้อง)")
    t.add_argument("--dry", action="store_true")
    cc = sub.add_parser("camera-cal")
    cc.add_argument("--dry", action="store_true")
    sub.add_parser("stop")
    dr = sub.add_parser("doctor")
    dr.add_argument("--image")
    dr.add_argument("--landscape", action="store_true")
    dr.add_argument("--dry", action="store_true")
    for sp in (r, c, t, cc, dr):
        sp.add_argument("--layout", choices=["auto", "A", "B"], default=None, help="ตำแหน่ง HUD: A=ซ้ายบน B=ขวาบน/ใต้มินิแมพ (ค่าเริ่มต้น auto)")
    args = ap.parse_args()
    global LAYOUT, QUIET
    if args.cmd != "stop":                       # ไฟล์ STOP เก่าที่ค้างอยู่ ต้องไม่ทำให้รันรอบใหม่หยุดทันที
        try:
            os.remove(STOP_FILE)
        except OSError:
            pass
    if getattr(args, "layout", None):
        LAYOUT = args.layout
    if args.cmd == "run":
        QUIET = not args.verbose
    if args.cmd == "stop":
        cmd_stop(args)
        return
    if args.cmd == "doctor":
        cmd_doctor(args)
        return
    if args.cmd == "camera-cal":
        try:
            cmd_camera_cal(args)
        except Stop as e:
            log(f"[หยุด] {e}")
        return
    if args.cmd == "calibrate":
        cmd_calibrate(args)
        return
    if args.cmd == "test-input":
        try:
            cmd_test_input(args)
        except Stop as e:
            log(f"[หยุด] {e}")
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
    _to_file("", "=" * 20 + " เริ่มรัน " + "=" * 20)
    install_stop_handlers()
    try:
        if not (args.no_countdown or args.image):
            countdown(COUNTDOWN_SEC, "สลับเข้าเกม Utopia: Origin ได้เลย!")
        bot.run(args.max_rounds)
    except Stop as e:
        ui(f"⏹ หยุด: {e}")
    except KeyboardInterrupt:
        ui("⏹ หยุด (Ctrl+C)")
    QUIET = False
    ui(f"📊 ตี {bot.stats['ตีสำเร็จ']} │ บ้าน {bot.stats['กลับบ้านสำเร็จ']} │ รอบ {bot.stats['รอบ']}")
    ui("📄 รายละเอียด: debug/bot.log")


if __name__ == "__main__":
    main()
