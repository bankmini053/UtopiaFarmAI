#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mining.py - UtopiaFarmAI : บอทฟามแร่ (Termux + su บนมือถือ / มือถือคลาวด์)

วนลูป:  หาแร่ -> เดินเข้าไป -> ตี (ปุ่มดาบ) -> เช็กชื่อ -> ตีจนเลือดหมด -> มือ -> Pick All
        -> รอแร่หาย -> หาแร่ต่อ
ก่อนขุดทุกก้อนเช็กพิก (ช่องแรกแถบล่าง): เขียว = ขุดต่อ
  แดง = กดที่พิกแดง -> fix.png -> equip_no.png -> ขุดต่อ
เช็กกระเป๋า (ปุ่มวงแหวนมุมขวาล่าง): เขียว = ขุดต่อ, ส้ม = เดินไปหน้าบ้าน -> เข้าบ้าน -> หยุด
แสดงสถานะสดๆ ตลอดเวลา (กำลังทำอะไร / จำนวนขุด / พิก / กระเป๋า / พิกัด / เหตุการณ์ล่าสุด)

รองรับจอ 2 แบบ (ตรวจเองอัตโนมัติ):
  - จอแนวนอนปกติ (มือถือจริง)
  - มือถือคลาวด์ที่ screencap ออกมาเป็นแนวตั้ง แล้วเกมหมุนข้าง 90° มีแถบดำบน/ล่าง
    -> บอทตัดแถบดำ + หมุนภาพกลับ + แปลงพิกัดแตะให้เอง

คำสั่งช่วยตรวจ (ลองก่อนรันจริง):
  python mining.py debug    เซฟ debug.jpg ให้ดูว่าบอทเห็นแร่/ปุ่มตรงไหน (+พิมพ์สถานะ)
  python mining.py tap      แตะปุ่มดาบ 1 ครั้ง (เช็กว่าแตะตรงปุ่มจริง)
  python mining.py          รันบอท

ต้องมี:  pip install opencv-python-headless numpy   (+ pytesseract/tesseract ถ้าจะใช้เดินตามพิกัด)
ไอคอนใน icons/ (ถ้าไม่มีในเครื่องจะดึงจาก GitHub อัตโนมัติ):
  sword.png  hand.png  pickall.png  iron_mine.png  pickaxe_red.png  bag_icon.png
  fix.png  equip_no.png  (ของเดิมจากบอทตกปลา)   กระเป๋าส้ม.png (ไม่บังคับ ใช้เสริมการเช็กสีส้ม)
"""
import math
import os
import random
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request

import cv2
import numpy as np

try:
    import pytesseract
except ImportError:
    pytesseract = None

# ----------------------------- ตั้งค่า ---------------------------------
HOME = (6374, 14207)          # หน้าบ้าน
STORAGE = (6381, 14224)       # จุดที่เดินขึ้นไปแล้วเสร็จ
FARM = (6360, 14170)          # จุดฟามแร่
FARM_RADIUS = 70
HOME_EVERY = 0                # ขุดกี่ก้อนแล้วกลับบ้าน (0 = ไม่กลับ ขุดอย่างเดียว)
RETURN_TO_FARM = True
COUNTDOWN = 10

ROTATE = "auto"               # auto | none | ccw   (auto: ภาพแนวตั้ง -> หมุน ccw)

ICON_DIR = "icons"
GITHUB_RAW = "https://raw.githubusercontent.com/bankmini053/UtopiaFarmAI/main/icons/"
TARGET_NAME = "iron_mine.png"  # รูปชื่อแร่ที่ต้องการ ถ้าไม่มีไฟล์จะไม่เช็กชื่อ
SWORD_NAME = "sword.png"       # ปุ่มดาบ (ใช้หาตำแหน่งปุ่ม ถ้าไม่เจอใช้ตำแหน่งที่ตั้งไว้)
PICK_RED_NAME = "pickaxe_red.png"
FIX_NAME = "fix.png"
EQUIP_NO_NAME = "equip_no.png"
BAG_ICON_NAME = "bag_icon.png"
BAG_FULL_NAME = "กระเป๋าส้ม.png"   # ไม่บังคับ
TITLE_THR = 0.6
MIN_BLOB = 500                # ก้อนสีเล็กกว่านี้ (พิกเซลบนภาพย่อ) ไม่นับเป็นแร่
SCALE = 0.4                   # ย่อภาพก่อนจับเพื่อให้เร็ว
NEAR_PX = 150                 # แร่ห่างตัวละครไม่เกินนี้ = ถึงแล้ว
MINE_TIMEOUT = 45
POS_TOL = 3

# ---- ตำแหน่ง UI: พิกัดอ้างอิงจากมือถือ 2576x1189 + จุดยึด (l=ซ้าย c=กลาง r=ขวา) ----
REF_W, REF_H = 2576, 1189
UI_REF = {
    "joy": ((630, 890), "l"),
    "attack": ((2050, 890), "r"),      # ปุ่มดาบ
    "hand": ((1720, 795), "r"),
    "pickall": ((1470, 788), "c"),
    "char": ((1290, 700), "c"),
    "cam": ((1500, 330), "c"),
    "slot1": ((718, 1112), "l"),       # ช่องพิก (ช่องแรกแถบล่าง)
    "bag": ((1989, 1114), "r"),        # ปุ่มกระเป๋า (วงแหวน)
}
BOX_REF = {
    "bar": ((1070, 50, 1625, 88), "c"),      # แถบเลือดเป้าหมายด้านบน
    "title": ((1000, 0, 1700, 60), "c"),     # ชื่อเป้าหมาย
    "dura": ((755, 1105, 767, 1170), "l"),   # แถบความทนทานพิก
    "pos": ((280, 425, 600, 482), "l"),      # ตัวเลขพิกัด
}
JOY_R = 95
# ระยะเลื่อนของ UI บนมือถือคลาวด์ (วัดจากภาพจริง 1999x1189)
SHIFT_CLOUD = {"l": -260, "c": -288, "r": -330}
PLAY_ROI = (0.12, 0.20, 0.78, 0.80)       # พื้นที่หาแร่ (สัดส่วนของจอ)
EXCLUDE = [(0.0, 0.0, 0.28, 0.30)]        # โซน HUD มุมซ้ายบน ไม่หาแร่

# ----------------------------- พื้นฐาน ---------------------------------
W = H = 0                    # ขนาดภาพหลังหมุนเป็นแนวนอน (logical)
RAW_W = RAW_H = 0
CROP_Y0 = 0
ROTATED = False
_crop = None
stats = {"mined": 0, "since_home": 0, "trips": 0, "fails": 0, "sweeps": 0, "repairs": 0}
t_start = time.time()
HAND_TPL = PICKALL_TPL = TITLE_TPL = None
SWORD_TPL = PICK_RED_TPL = FIX_TPL = EQUIP_NO_TPL = None
BAG_ICON = BAG_FULL = None          # (เทมเพลตสีเทาขนาดเต็ม)
ATTACK_POS = None


def sh(cmd):
    return subprocess.run(["su", "-c", cmd], stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL).stdout


def _game_rows(img):
    """หาช่วงแถวที่เป็นภาพเกม (ตัดแถบดำบน/ล่าง/แถบนำทาง)"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok = (g > 20).mean(axis=1) > 0.5
    best, start, bs, be = 0, None, 0, 0
    for i, v in enumerate(list(ok) + [False]):
        if v and start is None:
            start = i
        if not v and start is not None:
            if i - start > best:
                best, bs, be = i - start, start, i
            start = None
    return (bs, be) if best > 200 else (0, img.shape[0])


def process(raw):
    """ภาพดิบจาก screencap -> ภาพแนวนอนพร้อมใช้ (และจำค่าสำหรับแปลงพิกัดแตะ)"""
    global W, H, RAW_W, RAW_H, CROP_Y0, ROTATED, _crop
    RAW_H, RAW_W = raw.shape[:2]
    rot = ROTATE == "ccw" or (ROTATE == "auto" and RAW_H > RAW_W)
    if not rot:
        ROTATED, CROP_Y0 = False, 0
        H, W = raw.shape[:2]
        return raw
    if _crop is None:
        _crop = _game_rows(raw)
    y0, y1 = _crop
    CROP_Y0, ROTATED = y0, True
    img = cv2.rotate(raw[y0:y1], cv2.ROTATE_90_COUNTERCLOCKWISE)
    H, W = img.shape[:2]
    return img


def shot():
    for _ in range(5):
        data = sh("screencap -p")
        raw = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if raw is not None:
            return process(raw)
        time.sleep(0.3)
    raise RuntimeError("screencap ไม่ได้ (เช็กสิทธิ์ su)")


def lg(x, y, a="c"):
    """พิกัดอ้างอิง (มือถือ 2576x1189) -> พิกัดในภาพแนวนอนจริง"""
    s = H / REF_H
    if ROTATED:
        dx = SHIFT_CLOUD[a]
    else:
        dx = {"l": 0, "c": (W - REF_W * s) / 2, "r": W - REF_W * s}[a]
    return x * s + dx, y * s


def P(name):
    (x, y), a = UI_REF[name]
    return lg(x, y, a)


def B(name):
    (x1, y1, x2, y2), a = BOX_REF[name]
    a1, b1 = lg(x1, y1, a)
    a2, b2 = lg(x2, y2, a)
    return int(max(0, a1)), int(max(0, b1)), int(min(W, a2)), int(min(H, b2))


def to_raw(lx, ly):
    """พิกัดบนภาพแนวนอน -> พิกัดจริงของจอ (ที่ใช้กับ input tap)"""
    if not ROTATED:
        return int(lx), int(ly)
    return int(RAW_W - 1 - ly), int(lx + CROP_Y0)


def tap(pt):
    x, y = to_raw(*pt)
    sh(f"input tap {x} {y}")


def swipe(p1, p2, ms):
    a, b = to_raw(*p1)
    c, d = to_raw(*p2)
    sh(f"input swipe {a} {b} {c} {d} {int(ms)}")


STATE = {"action": "เริ่มระบบ", "pick": "?", "bag": "?", "pos": "-", "events": []}
_lines = 0
_COL = {"green": "\033[32m", "red": "\033[31m", "orange": "\033[33m", "none": "\033[90m", "?": "\033[90m"}
_TXT = {"green": "เขียว", "red": "แดง", "orange": "ส้ม", "none": "ไม่เห็น", "?": "?"}


def _dot(k):
    return f"{_COL.get(k, '')}●\033[0m {_TXT.get(k, k)}"


def render():
    """วาดหน้าจอสถานะ (วาดทับที่เดิม)"""
    global _lines
    up = int(time.time() - t_start)
    lines = [
        f"━━ Mining Bot ━━━━ {up//3600:02d}:{up%3600//60:02d}:{up%60:02d}",
        f"ทำอยู่ : {STATE['action']}",
        f"ขุดได้ : {stats['mined']}  พลาด : {stats['fails']}  ซ่อมพิก : {stats['repairs']}",
        f"พิก    : {_dot(STATE['pick'])}",
        f"กระเป๋า: {_dot(STATE['bag'])}",
        f"พิกัด  : {STATE['pos']}",
        "──────────────────────",
    ] + (STATE["events"][-4:] or ["(ยังไม่มีเหตุการณ์)"])
    if not sys.stdout.isatty():
        print(lines[1], flush=True)
        return
    out = (f"\033[{_lines}F" if _lines else "") + "".join(f"{ln}\033[K\n" for ln in lines)
    sys.stdout.write(out)
    sys.stdout.flush()
    _lines = len(lines)


def say(msg):
    """บอกสถานะปัจจุบัน"""
    STATE["action"] = msg
    render()


def event(msg):
    """บันทึกเหตุการณ์สำคัญ (โชว์ 4 รายการล่าสุด)"""
    STATE["events"].append(time.strftime("%H:%M ") + msg)
    STATE["events"] = STATE["events"][-12:]
    render()


log = say


# ----------------------------- ไอคอน -----------------------------------
def load_icon(name):
    path = os.path.join(ICON_DIR, name)
    if not os.path.exists(path):
        try:
            os.makedirs(ICON_DIR, exist_ok=True)
            urllib.request.urlretrieve(GITHUB_RAW + urllib.parse.quote(name), path)
        except Exception:
            return None
    return cv2.imread(path, cv2.IMREAD_COLOR)


def prep(gray):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def to_gray_small(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    return prep(g)


def load_tpl(name):
    im = load_icon(name)
    return None if im is None else to_gray_small(im)


def load_gray_full(name):
    im = load_icon(name)
    return None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


def find_peaks(gray, tpl, thr, scales, limit=5):
    out = []
    for s in scales:
        t = cv2.resize(tpl, None, fx=s, fy=s)
        if t.shape[0] >= gray.shape[0] or t.shape[1] >= gray.shape[1]:
            continue
        res = cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED)
        for _ in range(limit):
            _, mx, _, loc = cv2.minMaxLoc(res)
            if mx < thr:
                break
            out.append((mx, loc[0] + t.shape[1] // 2, loc[1] + t.shape[0] // 2))
            x0, y0 = max(0, loc[0] - t.shape[1] // 2), max(0, loc[1] - t.shape[0] // 2)
            res[y0:loc[1] + t.shape[0] // 2, x0:loc[0] + t.shape[1] // 2] = 0
    return out


def find_ui(img, tpl, thr=0.7, scales=(1.0,)):
    """หาไอคอน -> พิกัดกลางบนภาพแนวนอน หรือ None (เลือกที่คะแนนสูงสุด)"""
    if tpl is None:
        return None
    p = find_peaks(to_gray_small(img), tpl, thr, list(scales), 1)
    if not p:
        return None
    _, cx, cy = max(p)
    return cx / SCALE, cy / SCALE


def find_ores(img):
    """จับหินด้วยสี: เทาม่วงเข้ม = กลางวัน/เย็น, น้ำเงิน = กลางคืน
    คืน [(ระยะจากตัวละคร, x, y)] เรียงใกล้ -> ไกล"""
    small = cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(int)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    notgreen = (h < 30) | (h > 95) | (s < 45)
    day = (v < 135) & (s < 105) & notgreen
    night = (h >= 98) & (h <= 130) & (s > 90) & (v > 55)
    m = (day | night).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(m)
    x1, y1, x2, y2 = PLAY_ROI
    cx0, cy0 = P("char")
    found = []
    for i in range(1, n):
        x, y, w, hh, area = st[i]
        if area < MIN_BLOB or area > 0.12 * m.shape[0] * m.shape[1]:
            continue
        if w / hh > 2.2 or hh / w > 3 or area / (w * hh) < 0.35:
            continue
        rx, ry = cen[i][0] / SCALE, cen[i][1] / SCALE
        if not (x1 * W < rx < x2 * W and y1 * H < ry < y2 * H):
            continue
        if any(a * W < rx < c * W and b * H < ry < d * H for a, b, c, d in EXCLUDE):
            continue
        if abs(rx - cx0) < 110 and cy0 - 220 < ry < cy0 + 200:   # ตัวละครเอง
            continue
        found.append((math.hypot(rx - cx0, ry - cy0), rx, ry))
    found.sort()
    return found


def pick_state(img):
    """แถบความทนทานของพิก: 'green' ขุดต่อ / 'red' ต้องซ่อม / 'none' ไม่เห็นแถบ"""
    x1, y1, x2, y2 = B("dura")
    crop = img[y1:y2, x1:x2].astype(int)
    if crop.size == 0:
        return "none"
    Bc, G, R = crop[..., 0], crop[..., 1], crop[..., 2]
    green = ((G > 190) & (R < 175) & (Bc < 175)).sum()
    red = ((R > 170) & (G < R - 60) & (Bc < R - 60)).sum()
    if red >= 6 and red > green:
        return "red"
    if green >= 6:
        return "green"
    return "none"


def title_ok(img):
    """ชื่อเป้าหมายตรงที่ต้องการไหม: True/False (None = ไม่มีรูปชื่อ ข้ามการเช็ก)"""
    if TITLE_TPL is None:
        return None
    x1, y1, x2, y2 = B("title")
    reg = to_gray_small(img[y1:y2, x1:x2])
    if reg.shape[0] < TITLE_TPL.shape[0] or reg.shape[1] < TITLE_TPL.shape[1]:
        return None
    return float(cv2.matchTemplate(reg, TITLE_TPL, cv2.TM_CCOEFF_NORMED).max()) >= TITLE_THR


def attack_pt():
    """ตำแหน่งปุ่มดาบ: หาด้วย sword.png (จำไว้) ถ้าไม่เจอใช้ตำแหน่งที่ตั้งไว้"""
    global ATTACK_POS
    if ATTACK_POS is None:
        exp = P("attack")
        pos = find_ui(shot(), SWORD_TPL, 0.6, (0.9, 1.0, 1.1))
        if pos and math.hypot(pos[0] - exp[0], pos[1] - exp[1]) < 150:
            ATTACK_POS = pos
        else:
            ATTACK_POS = exp
    return ATTACK_POS


def pick_pt(img):
    """ตำแหน่งพิกที่ช่องแรก (หา pickaxe_red.png ใกล้ๆ ตำแหน่งช่อง ถ้าไม่เจอใช้ตำแหน่งช่อง)"""
    exp = P("slot1")
    pos = find_ui(img, PICK_RED_TPL, 0.6, (0.9, 1.0, 1.1))
    if pos and math.hypot(pos[0] - exp[0], pos[1] - exp[1]) < 120:
        return pos
    return exp


def locate_bag(img):
    ex, ey = P("bag")
    if BAG_ICON is None:
        return ex, ey
    x0, y0 = int(max(0, ex - 90)), int(max(0, ey - 90))
    win = cv2.cvtColor(img[y0:int(ey + 90), x0:int(ex + 90)], cv2.COLOR_BGR2GRAY)
    if win.shape[0] <= BAG_ICON.shape[0] or win.shape[1] <= BAG_ICON.shape[1]:
        return ex, ey
    res = cv2.matchTemplate(win, BAG_ICON, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(res)
    if mx < 0.55:
        return ex, ey
    return x0 + loc[0] + BAG_ICON.shape[1] / 2, y0 + loc[1] + BAG_ICON.shape[0] / 2


def bag_state(img):
    """สถานะกระเป๋าจากสีวงแหวนรอบปุ่ม: 'green' = ขุดต่อ, 'orange' = เต็ม, 'none' = ไม่เห็น"""
    cx, cy = locate_bag(img)
    s = H / REF_H
    x0, y0 = int(max(0, cx - 70 * s)), int(max(0, cy - 70 * s))
    patch = img[y0:int(cy + 70 * s), x0:int(cx + 70 * s)]
    if patch.size == 0:
        return "none"
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).astype(int)
    yy, xx = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
    d = np.hypot(xx - (cx - x0), yy - (cy - y0))
    ring = (d >= 45 * s) & (d <= 51 * s)
    if ring.sum() == 0:
        return "none"
    h, sa, v = hsv[..., 0][ring], hsv[..., 1][ring], hsv[..., 2][ring]
    orange = ((h >= 8) & (h <= 24) & (sa > 130) & (v > 170)).mean()
    green = ((h >= 40) & (h <= 75) & (sa > 90) & (v > 150)).mean()
    if orange > 0.10:
        return "orange"
    if BAG_FULL is not None:                       # เสริม: เทมเพลต กระเป๋าส้ม.png
        x1, y1 = int(max(0, cx - 90)), int(max(0, cy - 90))
        win = cv2.cvtColor(img[y1:int(cy + 90), x1:int(cx + 90)], cv2.COLOR_BGR2GRAY)
        if win.shape[0] > BAG_FULL.shape[0] and win.shape[1] > BAG_FULL.shape[1]:
            if cv2.matchTemplate(win, BAG_FULL, cv2.TM_CCOEFF_NORMED).max() > 0.85:
                return "orange"
    return "green" if green > 0.08 else "none"


def repair_pickaxe():
    """พิกแดง: กดที่พิก -> fix.png -> equip_no.png -> เช็กว่ากลับเป็นเขียว"""
    if FIX_TPL is None or EQUIP_NO_TPL is None:
        event("ไม่มี fix.png / equip_no.png ซ่อมไม่ได้")
        return False
    for attempt in range(1, 4):
        say(f"พิกแดง -> กดที่พิก (ครั้งที่ {attempt})")
        tap(pick_pt(shot()))
        fix, t0 = None, time.time()
        while time.time() - t0 < 4 and not fix:
            time.sleep(0.4)
            fix = find_ui(shot(), FIX_TPL, 0.7, (0.85, 1.0, 1.15))
        if not fix:
            continue
        say("กด Fix")
        tap(fix)
        eq, t0 = None, time.time()
        while time.time() - t0 < 4 and not eq:
            time.sleep(0.4)
            eq = find_ui(shot(), EQUIP_NO_TPL, 0.7, (0.85, 1.0, 1.15))
        if eq:
            say("กด Equip No")
            tap(eq)
        time.sleep(1.0)
        if pick_state(shot()) != "red":
            stats["repairs"] += 1
            event("ซ่อมพิกสำเร็จ")
            return True
    event("ซ่อมพิกไม่สำเร็จ")
    return False


def bar_state(img):
    """(มีแถบเป้าหมายด้านบน, เลือดยังเหลือ)"""
    x1, y1, x2, y2 = B("bar")
    crop = img[y1:y2, x1:x2].astype(int)
    if crop.size == 0:
        return False, False
    Bc, G, R = crop[..., 0], crop[..., 1], crop[..., 2]
    red = ((R > 190) & (G < 110) & (Bc < 110)).mean()
    cyan = ((Bc > 190) & (G > 190) & (R < 150)).mean()
    return cyan > 0.15, red > 0.03


# ----------------------------- พิกัด / เดิน ----------------------------
def read_pos(img=None):
    if pytesseract is None:
        return None
    img = img if img is not None else shot()
    x1, y1, x2, y2 = B("pos")
    g = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    cfg = "--psm 7 -c tessedit_char_whitelist=0123456789,"
    for variant in (cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                    cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]):
        m = re.search(r"(\d{4,5})\D+(\d{4,5})", pytesseract.image_to_string(variant, config=cfg))
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


def stable_pos():
    """อ่านพิกัด 2 ครั้งติดกันที่ตรงกัน (กัน OCR ผิด และรอให้พิกัดที่หน่วงนิ่ง)"""
    last = None
    for _ in range(8):
        p = read_pos()
        if p and p == last:
            return p
        last = p
        time.sleep(0.2)
    return last


def joystick(theta, secs):
    """theta: 0 = ขึ้น, +90° = ขวา (หน่วยเรเดียน)"""
    cx, cy = P("joy")
    swipe((cx, cy), (cx + JOY_R * math.sin(theta), cy - JOY_R * math.cos(theta)), secs * 1000)


def calibrate():
    p0 = stable_pos()
    if not p0:
        return None
    joystick(0, 0.6)
    time.sleep(0.5)
    p1 = stable_pos()
    joystick(math.pi / 2, 0.6)
    time.sleep(0.5)
    p2 = stable_pos()
    if not p1 or not p2:
        return None
    f = (p1[0] - p0[0], p1[1] - p0[1])
    r = (p2[0] - p1[0], p2[1] - p1[1])
    if math.hypot(*f) < 1:
        return None
    hand = 1 if (f[0] * r[1] - f[1] * r[0]) >= 0 else -1
    return math.atan2(f[1], f[0]), hand


def goto(target, tol=POS_TOL, max_steps=140):
    cal = calibrate()
    if cal is None:
        log("calibrate ไม่ผ่าน (อ่านพิกัดไม่ได้/ตัวละครไม่ขยับ)")
        return False
    phi_f, hand = cal
    stuck, prev = 0, None
    for step in range(max_steps):
        if step and step % 12 == 0:
            cal = calibrate()
            if cal:
                phi_f, hand = cal
        p = stable_pos()
        if not p:
            continue
        dx, dy = target[0] - p[0], target[1] - p[1]
        dist = math.hypot(dx, dy)
        log(f"เดินไป {target} ตอนนี้ {p} เหลือ {dist:.0f}")
        if dist <= tol:
            return True
        stuck = stuck + 1 if prev == p else 0
        prev = p
        theta = hand * (math.atan2(dy, dx) - phi_f)
        theta = (theta + math.pi) % (2 * math.pi) - math.pi
        if stuck >= 3:
            theta += random.choice([-1, 1]) * math.radians(70)
            stuck = 0
        secs = 0.8 if dist > 20 else 0.5 if dist > 8 else 0.3
        joystick(theta, secs)
        time.sleep(secs + 0.35)
    return False


# ----------------------------- ฟามแร่ ----------------------------------
def sweep_camera():
    """หมุนกล้องไปทางขวาทีละนิดเพื่อหาแร่"""
    cx, cy = P("cam")
    swipe((cx, cy), (cx + 380, cy), 300)
    stats["sweeps"] += 1
    time.sleep(0.5)


def try_lock():
    """ตีดาบ 1 ครั้ง แล้วดูว่ามีแถบเป้าหมาย (ชื่อ/เลือด) ขึ้นไหม -> แร่อยู่ในระยะแล้ว"""
    say("ลองตีดาบ (ดูว่ามีแร่ในระยะไหม)")
    tap(attack_pt())
    time.sleep(0.6)
    return bar_state(shot())[0]


def approach():
    for _ in range(30):
        img = shot()
        ores = find_ores(img)
        if not ores:
            return False
        d = ores[0][0]
        if d < NEAR_PX * 2 and try_lock():     # ใกล้พอ ลองตีดู ถ้าติดเป้าก็พอ
            return True
        if d < NEAR_PX:
            return True
        _, rx, ry = ores[0]
        say(f"เดินเข้าหาแร่ (ห่าง {d:.0f}px)")
        cx, cy = P("char")
        joystick(math.atan2(rx - cx, -(ry - cy)), 0.5 if d > 400 else 0.3)
        time.sleep(0.6)
    return False


def wait_gone(timeout=8):
    """หลังกด Pick All: รอจนแร่หาย (แถบ/มือ/หน้าต่าง Pick All หายหมดต่อเนื่อง 2 ครั้ง)"""
    t0, clear = time.time(), 0
    while time.time() - t0 < timeout:
        img = shot()
        busy = bar_state(img)[0] or find_ui(img, HAND_TPL) or find_ui(img, PICKALL_TPL)
        clear = 0 if busy else clear + 1
        if clear >= 2:
            return True
        time.sleep(0.4)
    return False


def mine_current():
    seen_red, empty, hand_tapped, named = False, 0, 0, False
    t0 = time.time()
    while time.time() - t0 < MINE_TIMEOUT:
        img = shot()
        bar, hp = bar_state(img)
        pk = find_ui(img, PICKALL_TPL)
        if pk:
            say("กด Pick All")
            tap(pk)
            time.sleep(0.6)
            say("รอแร่หาย")
            wait_gone()
            return True
        hand = find_ui(img, HAND_TPL)
        if hp:
            seen_red, empty = True, 0
        elif seen_red and bar:
            empty += 1
        if bar and not named:                      # ตีครั้งแรกแล้วชื่อขึ้น -> เช็กว่าใช่แร่ที่ต้องการ
            if title_ok(img) is False:
                event("ชื่อเป้าหมายไม่ใช่ที่ต้องการ -> เปลี่ยนเป้าหมาย")
                return False
            named = True
        if hand or empty >= 2:
            say("เลือดหมด -> กดมือ")
            tap(hand or P("hand"))
            hand_tapped += 1
            time.sleep(0.8)
            if PICKALL_TPL is None:                # ไม่มีรูป Pick All -> กดตำแหน่งตายตัว
                tap(P("pickall"))
                time.sleep(0.6)
                wait_gone()
                return True
            continue
        if not bar and not seen_red and time.time() - t0 > 6:
            return False                           # ไม่ได้ล็อกเป้าแร่
        say("ตีแร่ (เลือดเหลือ)" if hp else "ตีแร่...")
        tap(attack_pt())
        time.sleep(0.7)
    return False


def go_home():
    stats["trips"] += 1
    log("กลับบ้าน")
    goto(HOME)
    goto(STORAGE)
    log("ถึงจุดเก็บของแล้ว")
    stats["since_home"] = 0
    if RETURN_TO_FARM:
        goto(FARM, tol=6)
    else:
        summary()
        sys.exit(0)


def bag_full_stop():
    """กระเป๋าส้ม: เดินไปหน้าบ้าน -> เข้าบ้าน -> หยุด"""
    event("กระเป๋าเต็ม (ส้ม) -> กลับบ้าน")
    if pytesseract is None:
        event("ไม่มี pytesseract เดินตามพิกัดไม่ได้ -> หยุดทันที")
    else:
        say(f"เดินไปหน้าบ้าน {HOME}")
        ok1 = goto(HOME)
        say(f"เข้าบ้าน {STORAGE}")
        ok2 = goto(STORAGE)
        event("ถึงในบ้านแล้ว หยุดบอท" if ok1 and ok2 else "เดินไม่ถึงบ้านครบ หยุดบอท")
    summary()
    sys.exit(0)


def summary():
    up = int(time.time() - t_start)
    print("\n========= สรุป =========")
    print(f"ขุดได้ {stats['mined']} ก้อน | ซ่อมพิก {stats['repairs']} | รอบกลับบ้าน {stats['trips']} | พลาด {stats['fails']}")
    print(f"หมุนกล้อง {stats['sweeps']} ครั้ง | เวลา {up//3600}h {up%3600//60}m")


# ----------------------------- เริ่มทำงาน ------------------------------
def load_all():
    global HAND_TPL, PICKALL_TPL, TITLE_TPL, SWORD_TPL, PICK_RED_TPL, FIX_TPL, EQUIP_NO_TPL
    global BAG_ICON, BAG_FULL
    HAND_TPL, PICKALL_TPL = load_tpl("hand.png"), load_tpl("pickall.png")
    TITLE_TPL = load_tpl(TARGET_NAME)
    SWORD_TPL, PICK_RED_TPL = load_tpl(SWORD_NAME), load_tpl(PICK_RED_NAME)
    FIX_TPL, EQUIP_NO_TPL = load_tpl(FIX_NAME), load_tpl(EQUIP_NO_NAME)
    BAG_ICON, BAG_FULL = load_gray_full(BAG_ICON_NAME), load_gray_full(BAG_FULL_NAME)


def debug():
    load_all()
    img = shot()
    print(f"จอดิบ {RAW_W}x{RAW_H} | หมุน={'ใช่' if ROTATED else 'ไม่'} | ตัดแถวเกม y0={CROP_Y0} | "
          f"ภาพแนวนอน {W}x{H}")
    print("พิก:", pick_state(img), "| กระเป๋า:", bag_state(img), "| แถบเป้าหมาย/เลือด:",
          bar_state(img), "| ชื่อตรง:", title_ok(img))
    ores = find_ores(img)
    print(f"เจอแร่ {len(ores)} ก้อน (ใกล้สุด {ores[0][0]:.0f}px)" if ores else "ไม่เจอแร่")
    vis = img.copy()
    for k, (d, x, y) in enumerate(ores):
        cv2.circle(vis, (int(x), int(y)), 40, (0, 0, 255) if k == 0 else (0, 255, 255), 5)
    for name, col in (("joy", (255, 0, 255)), ("attack", (0, 255, 0)), ("hand", (255, 255, 0)),
                      ("pickall", (255, 128, 0)), ("char", (255, 255, 255)),
                      ("slot1", (0, 128, 255)), ("bag", (0, 255, 255))):
        x, y = P(name)
        cv2.drawMarker(vis, (int(x), int(y)), col, cv2.MARKER_CROSS, 50, 5)
        cv2.putText(vis, name, (int(x) + 25, int(y) - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 3)
    for name in ("dura", "bar", "title"):
        x1, y1, x2, y2 = B(name)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 165, 255), 3)
    cv2.imwrite("debug.jpg", vis)
    print("เซฟ debug.jpg แล้ว: ดูว่า + ของ attack ทับปุ่มดาบ, joy ทับจอย, วงแดง = แร่ที่จะเดินไปหา")


def main():
    load_all()
    if HOME_EVERY and pytesseract is None:
        sys.exit("HOME_EVERY > 0 ต้องติดตั้ง pytesseract + tesseract เพื่ออ่านพิกัด")
    have = lambda t: "มี" if t is not None else "ไม่มี"
    print(f"ไอคอน: ดาบ={have(SWORD_TPL)} มือ={have(HAND_TPL)} pickall={have(PICKALL_TPL)} "
          f"ชื่อแร่={have(TITLE_TPL)} fix={have(FIX_TPL)} equip_no={have(EQUIP_NO_TPL)}")
    if pytesseract is None:
        print("เตือน: ไม่มี pytesseract -> ถ้ากระเป๋าเต็มจะเดินกลับบ้านไม่ได้ (บอทจะหยุดเฉยๆ)")
    for i in range(COUNTDOWN, 0, -1):
        print(f"เริ่มใน {i} วิ (สลับไปหน้าเกม)", end="\r", flush=True)
        time.sleep(1)

    empty_sweeps, warned_none = 0, False
    while True:
        if HOME_EVERY and stats["since_home"] >= HOME_EVERY:
            go_home()
            continue
        if HOME_EVERY:
            p = read_pos()
            if p and math.hypot(p[0] - FARM[0], p[1] - FARM[1]) > FARM_RADIUS:
                log("ออกนอกพื้นที่ฟาม -> กลับจุดฟาม")
                goto(FARM, tol=6)
                continue

        img = shot()
        ps, bs = pick_state(img), bag_state(img)
        STATE["pick"], STATE["bag"] = ps, bs
        try:
            p = read_pos(img)
        except Exception:
            p = None
        if p:
            STATE["pos"] = f"{p[0]}, {p[1]}"
        render()

        if bs == "orange":                         # กระเป๋าเต็ม -> เดินไปหน้าบ้าน เข้าบ้าน แล้วหยุด
            bag_full_stop()
        if ps == "red":                            # พิกแดง -> ซ่อมแล้วขุดต่อ
            if not repair_pickaxe():
                event("หยุดบอท: ซ่อมพิกไม่ได้")
                summary()
                sys.exit(0)
            continue
        if ps == "none" and not warned_none:
            warned_none = True
            event("ไม่เห็นแถบพิก (ถือพิกช่องแรกไหม? ลอง debug)")

        # แร่อยู่ติดตัวอยู่แล้ว? (ตัวละครบังหิน เลยหาด้วยสีไม่เจอ) -> ตีเลย
        if try_lock():
            if mine_current():
                stats["mined"] += 1
                stats["since_home"] += 1
                event(f"ขุดสำเร็จ #{stats['mined']}")
                continue
            stats["fails"] += 1
            joystick(random.uniform(-math.pi, math.pi), 0.6)
            time.sleep(0.8)
            continue

        if not find_ores(img):
            empty_sweeps += 1
            sweep_camera()
            if empty_sweeps >= 10:
                joystick(random.uniform(-math.pi, math.pi), 1.0)
                time.sleep(1.3)
                empty_sweeps = 0
            log("หาแร่...")
            continue
        empty_sweeps = 0

        if approach() and mine_current():
            stats["mined"] += 1
            stats["since_home"] += 1
            event(f"ขุดสำเร็จ #{stats['mined']}")
        else:
            stats["fails"] += 1
            event("ขุดไม่สำเร็จ ลองใหม่")
            joystick(random.uniform(-math.pi, math.pi), 0.6)   # ขยับหนีก้อนเดิม
            time.sleep(0.8)
            sweep_camera()


if __name__ == "__main__":
    try:
        cmd = sys.argv[1] if len(sys.argv) > 1 else ""
        if cmd == "debug":
            debug()
        elif cmd == "tap":
            shot()
            load_all()
            tap(attack_pt())
            print("แตะปุ่มดาบแล้ว 1 ครั้ง")
        else:
            main()
    except KeyboardInterrupt:
        summary()
