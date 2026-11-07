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

def swipe_screen(x1, y1, x2, y2, duration=260):
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

def get_current_distance():
    """อ่านพิกัดปัจจุบันและคำนวณระยะห่างถึงกล่อง"""
    scr = get_screen()
    pos = extract_coords(scr)
    if pos is not None:
        cur_x, cur_y = pos
        dx = TARGET_CHEST[0] - cur_x
        dy = TARGET_CHEST[1] - cur_y
        dist = math.sqrt(dx**2 + dy**2)
        return pos, dist
    return None, 999.0

def rotate_camera_towards(diff):
    """
    แก้ทิศทางการรูดให้ถูกต้อง:
    - ถ้า diff < 0 (กล่องอยู่ซ้าย X < 640) -> ต้องดึงกล่องมาทางขวา ลากจากขวาไปซ้าย
    - ถ้า diff > 0 (กล่องอยู่ขวา X > 640) -> ต้องดึงกล่องมาทางซ้าย ลากจากซ้ายไปขวา
    """
    cx, cy = 850, 260
    dist = int(abs(diff) * 0.50)
    dist = min(max(dist, 60), 220)

    if diff < 0:
        # กล่องอยู่ซ้าย -> รูดไปซ้ายเพื่อหมุนกล้องรับ
        swipe_screen(cx, cy, cx - dist, cy, duration=240)
    else:
        # กล่องอยู่ขวา -> รูดไปขวา
        swipe_screen(cx - dist, cy, cx, cy, duration=240)
    time.sleep(0.4)

def scan_rotate_right(dist=220):
    cx, cy = 850, 260
    swipe_screen(cx, cy, cx - dist, cy, duration=280)
    time.sleep(0.4)

JOY_CENTER_X, JOY_CENTER_Y, JOY_RADIUS = 250, 530, 145

def walk_straight(duration_ms=450):
    """เดินตรงไปข้างหน้า (0 องศา)"""
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
    print("[*] UtopiaFarmAI: กวาดหา -> จัดกึ่งกลางเป๊ะ -> เดินจนเข้าใกล้กล่อง/พบรูปมือ")
    print(f"[*] พิกัดเป้าหมาย: {TARGET_CHEST}")
    print("=" * 60)
    time.sleep(2)

    CENTER_X = 640
    TOLERANCE = 45  # บีบให้แคบลง (595 - 685) เพื่อให้กล่องอยู่ตรงกลางเป๊ะจริงๆ ก่อนเริ่มเดิน

    # 1. กวาดสายตา 360 องศา (8 สเต็ป)
    found_chest = False
    for step in range(8):
        scr = get_screen()
        found, chest_x, area = detect_chest(scr)
        if found:
            print(f"\n[👁] ตรวจพบกล่องแล้วที่มุมมองสเต็ป {step + 1}/8! (X={chest_x}, ขนาด: {area:.1f})")
            found_chest = True
            break
        else:
            print(f"[🔄] กวาดสายตามุมที่ {step + 1}/8 -> หมุนต่อ...")
            scan_rotate_right(dist=220)

    if not found_chest:
        print("\n[!] ไม่สำเร็จ: กวาดสายตาครบ 360 องศาแล้วไม่พบกล่อง")
        return

    # 2. หยุดรอภาพนิ่ง 2 วินาที
    print("[⏸] หยุดรอภาพนิ่งสนิท 2 วินาที...")
    time.sleep(2.0)

    # 3. หันหน้าจอจัดให้กล่องอยู่ตรงกลางจอเป๊ะๆ
    aligned = False
    for attempt in range(10):
        still_scr = get_screen()
        still_found, still_x, _ = detect_chest(still_scr)

        if not still_found:
            print("[?] กำลังค้นหาตำแหน่งกล่องซ้ำ...")
            scan_rotate_right(dist=90)
            continue

        diff = still_x - CENTER_X
        if abs(diff) <= TOLERANCE:
            print(f"\n[🎯] ล็อกกล่องอยู่ตรงกลางจอเป๊ะแล้ว! (X={still_x}, คลาดเคลื่อน: {diff:+d}px)")
            aligned = True
            break

        print(f"[*] กำลังจัดกึ่งกลาง: กล่องอยู่ที่ X={still_x} (ห่างกลางจอ: {diff:+d})")
        rotate_camera_towards(diff)

    if not aligned:
        print("\n[!] ไม่สำเร็จ: ไม่สามารถล็อกกล่องให้อยู่ตรงกลางจอได้")
        return

    # 4. เดินตรงเข้าหากล่องแบบตรวจสอบสด (ไม่จำกัดรอบก้าวตายตัว)
    print("\n" + "=" * 50)
    print("[↑] เริ่มเดินตรงเข้าชนกล่อง (ตรวจพิกัดและปุ่มรูปมือแบบ Real-time)...")
    print("=" * 50)

    opened = False
    walk_attempts = 0

    while walk_attempts < 15:
        walk_attempts += 1
        walk_straight(duration_ms=450)
        time.sleep(0.15)

        curr_scr = get_screen()
        pos, dist = get_current_distance()

        # ตรวจสอบว่าปุ่มรูปมือขึ้นมาหรือยัง
        if check_hand_button(curr_scr):
            print(f"\n[🎯] ตรวจพบปุ่มรูปมือแล้ว! (พิกัดคน: {pos}, ห่างกล่อง: {dist:.2f} หน่วย)")
            print("[✓] แตะเปิดกล่องทันที...")
            tap(1045, 390)
            time.sleep(0.3)
            tap(1045, 390)
            opened = True
            break

        # ตรวจสอบว่าเดินเข้าใกล้ระยะประชิดกล่องแล้วหรือไม่ (ห่าง <= 2.2 หน่วย)
        if dist <= 2.2:
            print(f"\n[🎯] เดินถึงระยะประชิดกล่องแล้ว! (พิกัดคน: {pos}, ห่าง: {dist:.2f} หน่วย)")
            time.sleep(0.3)
            # เผื่อปุ่มมือเพิ่งโผล่หลังจากตัวละครหยุดเดิน
            chk = get_screen()
            tap(1045, 390)
            time.sleep(0.3)
            tap(1045, 390)
            opened = True
            break

        print(f"[*] กำลังเดินเข้าหากล่อง (ก้าวที่ {walk_attempts}) | พิกัด: {pos} | ระยะห่าง: {dist:.2f}")

    if opened:
        print("\n[✓] ภารกิจเปิดกล่องเสร็จสมบูรณ์!")
    else:
        final_pos, final_dist = get_current_distance()
        print(f"\n[!] ไม่สำเร็จ: เดินเข้าหากล่องไม่สำเร็จ (พิกัด: {final_pos}, ห่าง: {final_dist:.2f})")

if __name__ == "__main__":
    main()
