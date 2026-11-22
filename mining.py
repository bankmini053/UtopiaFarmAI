#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mining.py - UtopiaFarmAI : บอทฟามแร่ (Termux + su บนมือถือคลาวด์)

วนลูป:  หาแร่ -> เดินเข้าไป -> ขุด -> มือ -> Pick All -> หาแร่ต่อ
ขุดครบ HOME_EVERY ก้อน -> เดินกลับบ้าน (6374,14207) -> ขึ้นไป (6381,14224) -> กลับมาฟามต่อ

ต้องมี:  pip install opencv-python-headless numpy pytesseract   (+ pkg install tesseract)
ไอคอนในโฟลเดอร์ icons/ (หรือดึงจาก GitHub ให้อัตโนมัติถ้าเน็ตได้):
    ore_day.png, ore_dusk.png, ore_night.png   รูปหิน (ครอปจากจอจริง เช้า/เย็น/มืด)
    hand.png                                   ไอคอนมือ (ขึ้นตอนเลือดแร่หมด)
    pickall.png                                ปุ่ม Pick All
รันก่อนใช้งาน: เปิดเกมค้างไว้ ถือพิกแล้ว  ->  python mining.py
"""
import math
import os
import random
import re
import subprocess
import sys
import time
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
FARM_RADIUS = 70              # ออกห่างจาก FARM เกินนี้ให้เดินกลับ
HOME_EVERY = 15               # ขุดกี่ก้อนแล้วกลับบ้าน
RETURN_TO_FARM = True         # เสร็จที่บ้านแล้วกลับมาฟามต่อไหม
COUNTDOWN = 10                # หน่วงเวลาก่อนเริ่ม (วินาที)

ICON_DIR = "icons"
GITHUB_RAW = "https://raw.githubusercontent.com/bankmini053/UtopiaFarmAI/main/icons/"
ORE_ICONS = ["ore_day.png", "ore_dusk.png", "ore_night.png"]

MATCH_THR = 0.62              # ความเหมือน template (ต่ำ=จับง่ายแต่พลาดบ่อย)
SCALE = 0.5                   # ย่อภาพก่อนจับเพื่อให้เร็ว
TPL_SCALES = [0.7, 0.85, 1.0, 1.2, 1.4]
NEAR_PX = 150                 # แร่ห่างตัวละครไม่เกินนี้ (หน่วยพิกัดอ้างอิง) = ถึงแล้ว
MINE_TIMEOUT = 45
POS_TOL = 3                   # คลาดพิกัดได้เท่านี้

# พิกัดอ้างอิงจากภาพหน้าจอ 2576x1189 (สเกลอัตโนมัติตามจอจริง)
REF_W, REF_H = 2576, 1189
JOY_CENTER, JOY_R = (630, 890), 95
CHAR = (1290, 700)
ATTACK_BTN = (2050, 890)
HAND_BTN = (1720, 795)
PICKALL_BTN = (1470, 788)
CAM_SWIPE_Y = 330
POS_BOX = (280, 425, 600, 482)        # x1,y1,x2,y2 ตัวเลขพิกัด
BAR_BOX = (1070, 50, 1625, 88)        # แถบเลือดชื่อแร่ด้านบน
PLAY_ROI = (0.12, 0.17, 0.78, 0.80)   # พื้นที่หาแร่ (ตัด UI ออก)

# ----------------------------- พื้นฐาน ---------------------------------
W = H = 0
stats = {"mined": 0, "since_home": 0, "trips": 0, "fails": 0, "sweeps": 0}
t_start = time.time()


def sh(cmd):
    return subprocess.run(["su", "-c", cmd], stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL).stdout


def shot():
    global W, H
    for _ in range(5):
        data = sh("screencap -p")
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            H, W = img.shape[:2]
            return img
        time.sleep(0.3)
    raise RuntimeError("screencap ไม่ได้ (เช็กสิทธิ์ su)")


def px(x, y):
    return int(x * W / REF_W), int(y * H / REF_H)


def tap(x, y):
    a, b = px(x, y)
    sh(f"input tap {a} {b}")


def swipe(x1, y1, x2, y2, ms):
    a, b = px(x1, y1)
    c, d = px(x2, y2)
    sh(f"input swipe {a} {b} {c} {d} {int(ms)}")


def log(msg):
    up = int(time.time() - t_start)
    print(f"[{up//60:02d}:{up%60:02d}] ขุด {stats['mined']} | รอบบ้าน {stats['trips']} | "
          f"พลาด {stats['fails']} | {msg}        ", flush=True)


# ----------------------------- ไอคอน -----------------------------------
def load_icon(name):
    path = os.path.join(ICON_DIR, name)
    if not os.path.exists(path):
        try:
            os.makedirs(ICON_DIR, exist_ok=True)
            urllib.request.urlretrieve(GITHUB_RAW + name, path)
        except Exception:
            return None
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    return im


def prep(gray):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def to_gray_small(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    return prep(g)


def load_tpl(name):
    im = load_icon(name)
    return None if im is None else to_gray_small(im)


ORE_TPLS, HAND_TPL, PICKALL_TPL = [], None, None


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
            cx = loc[0] + t.shape[1] // 2
            cy = loc[1] + t.shape[0] // 2
            out.append((mx, cx, cy))
            x0, y0 = max(0, loc[0] - t.shape[1] // 2), max(0, loc[1] - t.shape[0] // 2)
            res[y0:loc[1] + t.shape[0] // 2, x0:loc[0] + t.shape[1] // 2] = 0
    return out


def to_ref(cx, cy):
    """พิกัดบนภาพย่อ -> พิกัดอ้างอิง 2576x1189"""
    return cx / SCALE * REF_W / W, cy / SCALE * REF_H / H


def find_ui(img, tpl, thr=0.7):
    if tpl is None:
        return None
    g = to_gray_small(img)
    p = find_peaks(g, tpl, thr, [1.0], 1)
    if not p:
        return None
    return to_ref(p[0][1], p[0][2])


def find_ores(img):
    g = to_gray_small(img)
    x1, y1, x2, y2 = PLAY_ROI
    found = []
    for tpl in ORE_TPLS:
        for sc, cx, cy in find_peaks(g, tpl, MATCH_THR, TPL_SCALES):
            rx, ry = to_ref(cx, cy)
            if x1 * REF_W < rx < x2 * REF_W and y1 * REF_H < ry < y2 * REF_H:
                found.append((math.hypot(rx - CHAR[0], ry - CHAR[1]), rx, ry))
    found.sort()
    return found


def bar_state(img):
    x1, y1, x2, y2 = BAR_BOX
    a, b = px(x1, y1)
    c, d = px(x2, y2)
    crop = img[b:d, a:c].astype(int)
    B, G, R = crop[..., 0], crop[..., 1], crop[..., 2]
    red = ((R > 190) & (G < 110) & (B < 110)).mean()
    cyan = ((B > 190) & (G > 190) & (R < 150)).mean()
    return cyan > 0.15, red > 0.03      # (มีแถบเป้าหมาย, เลือดยังเหลือ)


# ----------------------------- พิกัด / เดิน ----------------------------
def read_pos(img=None):
    if pytesseract is None:
        return None
    img = img if img is not None else shot()
    x1, y1, x2, y2 = POS_BOX
    a, b = px(x1, y1)
    c, d = px(x2, y2)
    g = cv2.cvtColor(img[b:d, a:c], cv2.COLOR_BGR2GRAY)
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
    """theta: 0=ขึ้น, +90=ขวา (องศา เป็นเรเดียน)"""
    cx, cy = JOY_CENTER
    swipe(cx, cy, cx + JOY_R * math.sin(theta), cy - JOY_R * math.cos(theta), secs * 1000)


def calibrate():
    """ดันขึ้น/ขวา แล้ววัดพิกัด -> รู้ว่าทิศ 'ขึ้น' ของจอ = ทิศไหนในพิกัดโลก"""
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
    stuck = 0
    prev = None
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
        if stuck >= 3:                       # ติดสิ่งกีดขวาง: เบี่ยงออก
            theta += random.choice([-1, 1]) * math.radians(70)
            stuck = 0
        secs = 0.8 if dist > 20 else 0.5 if dist > 8 else 0.3
        joystick(theta, secs)
        time.sleep(secs + 0.35)              # รอให้พิกัดที่หน่วงอัปเดต
    return False


# ----------------------------- ฟามแร่ ----------------------------------
def sweep_camera():
    """หมุนกล้องไปทางขวาทีละนิดเพื่อหาแร่"""
    swipe(1500, CAM_SWIPE_Y, 1500 + 380, CAM_SWIPE_Y, 300)
    stats["sweeps"] += 1
    time.sleep(0.5)


def approach():
    for _ in range(30):
        img = shot()
        ores = find_ores(img)
        if not ores:
            return False
        d, rx, ry = ores[0]
        if d < NEAR_PX:
            return True
        theta = math.atan2(rx - CHAR[0], -(ry - CHAR[1]))
        joystick(theta, 0.5 if d > 400 else 0.3)
        time.sleep(0.6)
    return False


def mine_current():
    seen_red, empty, t0, hand_tapped = False, 0, time.time(), 0
    while time.time() - t0 < MINE_TIMEOUT:
        img = shot()
        bar, hp = bar_state(img)
        pk = find_ui(img, PICKALL_TPL)
        if pk:
            tap(*pk)
            time.sleep(1.0)
            return True
        hand = find_ui(img, HAND_TPL)
        if hp:
            seen_red, empty = True, 0
        elif seen_red and bar:
            empty += 1
        if hand or empty >= 2:
            tap(*(hand or HAND_BTN))
            hand_tapped += 1
            time.sleep(0.8)
            if PICKALL_TPL is None and hand_tapped >= 1:   # ไม่มีรูป Pick All -> กดตำแหน่งตายตัว
                tap(*PICKALL_BTN)
                time.sleep(1.0)
                return True
            continue
        if not bar and not seen_red and time.time() - t0 > 6:
            return False                                   # ไม่ได้ล็อกเป้าแร่
        tap(*ATTACK_BTN)
        time.sleep(0.7)
    return False


def go_home():
    stats["trips"] += 1
    log("กลับบ้าน")
    goto(HOME)
    goto(STORAGE)
    log("ถึงจุดเก็บของแล้ว")
    # ถ้าต้องกดฝากของ/เปิดตู้ ใส่ tap(...) ตรงนี้
    stats["since_home"] = 0
    if RETURN_TO_FARM:
        goto(FARM, tol=6)
    else:
        summary()
        sys.exit(0)


def summary():
    up = int(time.time() - t_start)
    print("\n========= สรุป =========")
    print(f"ขุดได้ {stats['mined']} ก้อน | รอบกลับบ้าน {stats['trips']} | พลาด {stats['fails']}")
    print(f"หมุนกล้อง {stats['sweeps']} ครั้ง | เวลา {up//3600}h {up%3600//60}m")


def main():
    global ORE_TPLS, HAND_TPL, PICKALL_TPL
    ORE_TPLS = [t for t in (load_tpl(n) for n in ORE_ICONS) if t is not None]
    HAND_TPL, PICKALL_TPL = load_tpl("hand.png"), load_tpl("pickall.png")
    if not ORE_TPLS:
        sys.exit("ไม่พบรูปแร่ใน icons/ (ore_day.png / ore_dusk.png / ore_night.png)")
    if pytesseract is None:
        sys.exit("ต้องติดตั้ง pytesseract + tesseract เพื่ออ่านพิกัด")
    print(f"แร่ {len(ORE_TPLS)} แบบ | hand={'มี' if HAND_TPL is not None else 'ไม่มี'} | "
          f"pickall={'มี' if PICKALL_TPL is not None else 'ไม่มี'}")
    for i in range(COUNTDOWN, 0, -1):
        print(f"เริ่มใน {i} วิ (สลับไปหน้าเกม)", end="\r", flush=True)
        time.sleep(1)

    empty_sweeps = 0
    while True:
        if stats["since_home"] >= HOME_EVERY:
            go_home()
            continue
        p = read_pos()
        if p and math.hypot(p[0] - FARM[0], p[1] - FARM[1]) > FARM_RADIUS:
            log("ออกนอกพื้นที่ฟาม -> กลับจุดฟาม")
            goto(FARM, tol=6)
            continue

        img = shot()
        if not find_ores(img):
            empty_sweeps += 1
            sweep_camera()
            if empty_sweeps >= 10:             # หมุนรอบแล้วไม่เจอ: เดินสุ่มไปที่ใหม่
                joystick(random.uniform(-math.pi, math.pi), 1.0)
                time.sleep(1.3)
                empty_sweeps = 0
            log("หาแร่...")
            continue
        empty_sweeps = 0

        if approach() and mine_current():
            stats["mined"] += 1
            stats["since_home"] += 1
            log("ขุดสำเร็จ")
        else:
            stats["fails"] += 1
            log("ขุดไม่สำเร็จ ลองใหม่")
            sweep_camera()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        summary()
