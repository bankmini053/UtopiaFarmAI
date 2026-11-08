import os
import re
import time
import math
import statistics
import subprocess
import cv2
import numpy as np
import pytesseract

ROOT_SU = "/system/xbin/su"
TMP_PATH = "/data/local/tmp/screen.png"
TARGET_CHEST = (17527, 15336)

CAL_MS = 900        # ระยะเวลาก้าวสอบเทียบ
STOP_DIST = 2.2     # ระยะที่ถือว่าถึงกล่อง (ปรับตามจริง)
USE_SCAN = False    # ปิดการกวาดหากล่องด้วยสี (จับผิดบ่อย) ใช้พิกัดอย่างเดียว

JOY_CENTER_X = 250
JOY_CENTER_Y = 530
JOY_RADIUS = 145


# ---------------------------------------------------------------- พื้นฐาน
def run_cmd(cmd):
    subprocess.run(f"{ROOT_SU} -c '{cmd}'", shell=True)


def tap(x, y):
    run_cmd(f"input tap {x} {y}")


def swipe_screen(x1, y1, x2, y2, duration=350):
    run_cmd(f"input swipe {x1} {y1} {x2} {y2} {duration}")


def get_screen():
    run_cmd(f"screencap -p {TMP_PATH}")
    if os.path.exists(TMP_PATH):
        img = cv2.imread(TMP_PATH)
        if img is not None:
            h, w = img.shape[:2]
            if h > w:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            return img
    return None


# ---------------------------------------------------------------- อ่านพิกัด
def extract_coords(screen):
    if screen is None:
        return None
    roi = screen[180:300, 40:270]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    for th in [175, 195, 215]:
        _, thresh = cv2.threshold(resized, th, 255, cv2.THRESH_BINARY)
        matches = re.findall(
            r"\d{5}",
            pytesseract.image_to_string(
                thresh, config=r"--psm 6 -c tessedit_char_whitelist=0123456789,."
            ),
        )
        if len(matches) >= 2:
            try:
                x, y = int(matches[0]), int(matches[1])
                if 17300 <= x <= 17700 and 15200 <= y <= 15500:
                    return (x, y)
            except Exception:
                pass
    return None


def read_pos(n=3):
    """อ่านพิกัดหลายรอบแล้วเอามัธยฐาน กัน OCR อ่านผิด"""
    xs, ys = [], []
    for _ in range(n):
        p = extract_coords(get_screen())
        if p:
            xs.append(p[0])
            ys.append(p[1])
        time.sleep(0.1)
    if len(xs) < 2:
        return None
    return (statistics.median(xs), statistics.median(ys))


# ---------------------------------------------------------------- ควบคุมตัวละคร
def walk_direction(deg, duration_ms=450):
    """0°=หน้า, 90°=ขวา, 180°=หลัง, 270°=ซ้าย"""
    rad = math.radians(deg)
    tx = int(JOY_CENTER_X + JOY_RADIUS * math.sin(rad))
    ty = int(JOY_CENTER_Y - JOY_RADIUS * math.cos(rad))
    swipe_screen(JOY_CENTER_X, JOY_CENTER_Y, tx, ty, duration=int(duration_ms))
    time.sleep(0.15)


def scan_rotate_right(dist=160):
    cx, cy = 850, 260
    swipe_screen(cx, cy, cx - dist, cy, duration=220)
    time.sleep(0.4)


# ---------------------------------------------------------------- ตรวจจับ
def detect_chest(screen):
    if screen is None:
        return False, 0, 0
    roi = screen[60:550, 100:1180]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([5, 60, 60]), np.array([25, 255, 255])),
        cv2.inRange(hsv, np.array([4, 20, 20]), np.array([28, 180, 140])),
    )
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_x, max_a = 0, 0
    for cnt in contours:
        a = cv2.contourArea(cnt)
        if a > 700:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if 0.5 <= float(cw) / ch <= 2.2 and a > max_a:
                max_a = a
                best_x = x + 100 + (cw // 2)
    return (True, best_x, max_a) if max_a > 0 else (False, 0, 0)


def check_hand_button(screen):
    if screen is None:
        return False
    hand_roi = screen[350:440, 1000:1090]
    gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
    return np.sum(gray > 200) > 220


# ---------------------------------------------------------------- เดินตามพิกัด
def measure_step(deg):
    """เดิน 1 ก้าวสอบเทียบ แล้วคืนค่าพิกัดที่เปลี่ยน (dx, dy)"""
    for ms in (CAL_MS, 1500, 2200):
        p1 = read_pos()
        walk_direction(deg, ms)
        time.sleep(0.5)
        p2 = read_pos()
        print(f"[cal] deg={deg} ms={ms} p1={p1} p2={p2}")
        if p1 and p2:
            v = (p2[0] - p1[0], p2[1] - p1[1])
            if math.hypot(*v) > 1:
                k = CAL_MS / ms
                return (v[0] * k, v[1] * k)
    return None


def calibrate():
    f = measure_step(0)    # เดินหน้า
    r = measure_step(90)   # เดินขวา
    if not f or not r:
        return None
    return np.array([[f[0], r[0]], [f[1], r[1]]], dtype=float)


def press_hand():
    tap(1045, 390)
    time.sleep(0.3)
    tap(1045, 390)


def go_to_target(max_steps=80):
    M = calibrate()
    if M is None or abs(np.linalg.det(M)) < 1e-3:
        print("[!] สอบเทียบไม่ได้ ตรวจ ROI ของ OCR / เช็กว่าตัวละครเดินจริง")
        return False
    print("[✓] สอบเทียบแล้ว", M.tolist())

    stuck = 0
    last = None
    for i in range(max_steps):
        scr = get_screen()
        if check_hand_button(scr):
            print("[🎯] เจอปุ่มรูปมือ")
            press_hand()
            return True

        pos = read_pos()
        if pos is None:
            print("[?] อ่านพิกัดไม่ได้ ลองใหม่")
            time.sleep(0.3)
            continue

        t = np.array(
            [TARGET_CHEST[0] - pos[0], TARGET_CHEST[1] - pos[1]], dtype=float
        )
        dist = float(np.hypot(*t))
        print(f"[{i}] pos={pos} dist={dist:.1f}")
        if dist <= STOP_DIST:
            print("[🎯] ถึงระยะประชิดกล่อง")
            press_hand()
            return True

        # แก้สมการ t = a*เดินหน้า + b*เดินขวา
        a, b = np.linalg.solve(M, t)
        deg = math.degrees(math.atan2(b, a)) % 360
        ms = int(np.clip(math.hypot(a, b) * CAL_MS * 0.7, 120, 1500))
        walk_direction(deg, ms)
        time.sleep(0.25)

        # ตรวจติดสิ่งกีดขวาง
        if last and math.hypot(pos[0] - last[0], pos[1] - last[1]) < 0.5:
            stuck += 1
            if stuck >= 3:
                print("[!] ติด ลองเดินหลบข้าง")
                walk_direction((deg + 90) % 360, 700)
                stuck = 0
        else:
            stuck = 0
        last = pos
    return False


# ---------------------------------------------------------------- main
def main():
    print("=" * 60)
    print("[*] UtopiaFarmAI: เจอกล่อง -> พัก 10 วิ -> เดินคุมพิกัด")
    print(f"[*] พิกัดเป้าหมายกล่อง: {TARGET_CHEST}")
    print("=" * 60)
    time.sleep(2)

    # 1. กวาดสายตา 360 องศาตรวจหากล่อง
    found_chest = not USE_SCAN
    for step in range(8 if USE_SCAN else 0):
        scr = get_screen()
        found, chest_x, area = detect_chest(scr)
        if found:
            print(f"\n[👁] ตรวจพบกล่องที่สเต็ป {step + 1}/8 (X={chest_x}, ขนาด: {area:.1f})")
            found_chest = True
            break
        print(f"[🔄] กวาดสายตามุมที่ {step + 1}/8 -> หมุนต่อ...")
        scan_rotate_right(dist=160)

    if not found_chest:
        print("\n[!] ไม่สำเร็จ: กวาดครบ 360 องศาแล้วไม่พบกล่อง")
        return

    # 2. หยุดนิ่ง 10 วินาที
    for sec in range(10 if USE_SCAN else 0, 0, -1):
        print(f"[*] หยุดรอภาพนิ่ง... เหลืออีก {sec} วินาที")
        time.sleep(1.0)

    # 3. สอบเทียบแกน + เดินเข้าหากล่องตามพิกัด
    ok = go_to_target()
    if ok:
        print("\n[✓] ภารกิจเปิดกล่องเสร็จสมบูรณ์!")
    else:
        final = read_pos()
        print(f"\n[!] ไม่สำเร็จ: เดินเข้าหากล่องไม่ถึง (พิกัดล่าสุด: {final})")


if __name__ == "__main__":
    main()
