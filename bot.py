import os
import re
import time
import math
import subprocess
import cv2
import numpy as np
import pytesseract

ROOT_SU = "/system/xbin/su"
TMP_PATH = "/data/local/tmp/screen.png"
TARGET_CHEST = (17527, 15336)

def run_cmd(cmd):
    subprocess.run(f"{ROOT_SU} -c '{cmd}'", shell=True)

def tap(x, y):
    run_cmd(f"input tap {x} {y}")

def swipe_screen(x1, y1, x2, y2, duration=280):
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
        matches = re.findall(r"\d{5}", pytesseract.image_to_string(thresh, config=r"--psm 6 -c tessedit_char_whitelist=0123456789,."))
        if len(matches) >= 2:
            try:
                x, y = int(matches[0]), int(matches[1])
                if 17300 <= x <= 17700 and 15200 <= y <= 15500:
                    return (x, y)
            except:
                pass
    return None

def rotate_camera_towards(diff):
    """
    คำนวณการหมุนกล้องเข้าหากึ่งกลางจอโดยตรง:
    diff = chest_x - 640
    - ถ้า diff > 0 (กล่องอยู่ขวา) -> ต้องหันกล้องไปทางขวา (ลากนิ้วจากขวาไปซ้าย)
    - ถ้า diff < 0 (กล่องอยู่ซ้าย) -> ต้องหันกล้องไปทางซ้าย (ลากนิ้วจากซ้ายไปขวา)
    """
    cx, cy = 850, 260
    # คำนวณระยะรูดตามความห่างจริง เพื่อไม่ให้แกว่งหลุดเฟรม
    dist = int(abs(diff) * 0.55)
    dist = min(max(dist, 50), 260)

    if diff > 0:
        # กล่องอยู่ขวา -> รูด 850 ไปทางซ้าย
        swipe_screen(cx, cy, cx - dist, cy, duration=240)
    else:
        # กล่องอยู่ซ้าย -> รูดจากซ้ายมาขวา
        swipe_screen(cx - dist, cy, cx, cy, duration=240)
    time.sleep(0.4)

def scan_rotate_right(dist=220):
    """กวาดสายตาหมุนรอบตัวปกติ"""
    cx, cy = 850, 260
    swipe_screen(cx, cy, cx - dist, cy, duration=280)
    time.sleep(0.4)

JOY_CENTER_X, JOY_CENTER_Y, JOY_RADIUS = 250, 530, 145

def walk_straight(duration_ms=500):
    swipe_screen(JOY_CENTER_X, JOY_CENTER_Y, JOY_CENTER_X, JOY_CENTER_Y - JOY_RADIUS, duration=duration_ms)
    time.sleep(0.15)

def detect_chest(screen):
    if screen is None:
        return False, 0, 0
    roi = screen[60:550, 100:1180]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([5, 60, 60]), np.array([25, 255, 255])),
        cv2.inRange(hsv, np.array([4, 20, 20]), np.array([28, 180, 140]))
    )
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_x, max_a = 0, 0
    for cnt in contours:
        a = cv2.contourArea(cnt)
        if a > 650:
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

def main():
    print("=" * 60)
    print("[*] UtopiaFarmAI: กวาด 360° -> คำนวณหันจอจัดกลาง -> เดินชนเปิด")
    print("=" * 60)
    time.sleep(2)

    CENTER_X = 640
    TOLERANCE = 85

    # 1. กวาดสายตา 360 องศา (8 มุม)
    found_chest = False
    for step in range(8):
        scr = get_screen()
        found, chest_x, area = detect_chest(scr)
        if found:
            print(f"\n[👁] ตรวจพบกล่องแล้วที่มุมมองสเต็ป {step + 1}/8! (X={chest_x}, ขนาด: {area:.1f})")
            found_chest = True
            break
        else:
            print(f"[🔄] กวาดสายตามุมที่ {step + 1}/8 -> กำลังหมุนต่อ...")
            scan_rotate_right(dist=220)

    if not found_chest:
        print("\n[!] ไม่สำเร็จ: กวาดสายตาครบ 360 องศาแล้วไม่พบกล่อง")
        return

    # 2. หยุดรอภาพนิ่ง 2 วินาที
    print("[⏸] หยุดรอภาพนิ่งสนิท 2 วินาที...")
    time.sleep(2.0)

    # 3. คำนวณปรับมุมกล้องดึงกล่องเข้ากึ่งกลางจอ
    aligned = False
    for attempt in range(8):
        still_scr = get_screen()
        still_found, still_x, _ = detect_chest(still_scr)

        if not still_found:
            print("[?] ยังไม่พบกล่องชัดเจน สะกิดมุมมอง...")
            scan_rotate_right(dist=100)
            continue

        diff = still_x - CENTER_X
        if abs(diff) <= TOLERANCE:
            print(f"\n[🎯] ล็อกกล่องอยู่ตรงกลางจอสำเร็จแล้ว! (X={still_x}, ห่าง: {diff:+d})")
            aligned = True
            break

        print(f"[*] คำนวณปรับจอ: กล่องอยู่ที่ X={still_x} (ห่างกลางจอ: {diff:+d})")
        rotate_camera_towards(diff)

    if not aligned:
        print("\n[!] ไม่สำเร็จ: ไม่สามารถล็อกกล่องให้อยู่ตรงกลางจอได้")
        return

    # 4. เดินตรงเข้าหากล่อง (0 องศา) พร้อมรอปุ่มรูปมือขึ้น
    print("\n[↑] กำลังเดินตรงไปข้างหน้า (0°) เข้าชนกล่อง...")
    hand_opened = False

    for step in range(7):
        walk_straight(duration_ms=500)
        time.sleep(0.15)
        chk_scr = get_screen()

        if check_hand_button(chk_scr):
            print(f"\n[🎯] ตรวจพบปุ่มรูปมือขึ้นมาแล้วที่ก้าวที่ {step + 1}! แตะเปิดกล่อง...")
            time.sleep(0.2)
            tap(1045, 390)
            time.sleep(0.3)
            tap(1045, 390)
            hand_opened = True
            break
        else:
            print(f"[*] เดินชนก้าวที่ {step + 1}/7... ยังไม่ขึ้นปุ่มมือ")

    time.sleep(0.5)
    final_pos = extract_coords(get_screen())

    if hand_opened:
        print(f"\n[✓] ภารกิจเปิดกล่องเสร็จสมบูรณ์! (พิกัดเปิด: {final_pos})")
    else:
        print(f"\n[!] ไม่สำเร็จ: เดินชนแล้วปุ่มรูปมือไม่ขึ้น (พิกัดปัจจุบัน: {final_pos})")

if __name__ == "__main__":
    main()
