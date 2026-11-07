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

def swipe_screen(x1, y1, x2, y2, duration=320):
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
    """อ่านพิกัดตำแหน่งคนจากหน้าจอ"""
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

def rotate_view(direction="right", dist=240):
    """รูดหมุนหน้าจอให้หันจริง (ระยะรูดยาว 240px ในโซนว่างด้านขวา)"""
    cx, cy = 920, 300
    if direction == "right":
        # รูดจากขวาไปซ้าย -> จอหันขวา
        swipe_screen(cx, cy, cx - dist, cy, duration=280)
    else:
        # รูดจากซ้ายไปขวา -> จอหันซ้าย
        swipe_screen(cx - dist, cy, cx, cy, duration=280)
    time.sleep(0.4)

JOY_CENTER_X, JOY_CENTER_Y, JOY_RADIUS = 250, 530, 145

def walk_straight(duration_ms=500):
    """ดันจอยตรงไปข้างหน้า (0 องศา) เข้าชนกล่อง"""
    swipe_screen(JOY_CENTER_X, JOY_CENTER_Y, JOY_CENTER_X, JOY_CENTER_Y - JOY_RADIUS, duration=duration_ms)
    time.sleep(0.2)

def detect_chest(screen):
    """ตรวจหากล่องทั้งเช้าและเย็น"""
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
        if a > 800:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if 0.55 <= float(cw) / ch <= 1.9 and a > max_a:
                max_a = a
                best_x = x + 100 + (cw // 2)
    return (True, best_x, max_a) if max_a > 0 else (False, 0, 0)

def check_hand_button(screen):
    """ตรวจจับปุ่มรูปมือสีขาวเหนือปุ่มดาบ (X: 1000-1090, Y: 350-440)"""
    if screen is None:
        return False
    hand_roi = screen[350:440, 1000:1090]
    gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
    return np.sum(gray > 200) > 250

def is_at_chest_coords():
    """ตรวจสอบว่าคนยืนอยู่ที่พิกัดกล่องเป้าหมายจริงหรือไม่"""
    for _ in range(3):
        scr = get_screen()
        pos = extract_coords(scr)
        if pos is not None:
            cur_x, cur_y = pos
            dx = TARGET_CHEST[0] - cur_x
            dy = TARGET_CHEST[1] - cur_y
            dist = math.sqrt(dx**2 + dy**2)
            print(f"[*] เช็กพิกัดปัจจุบัน: ({cur_x}, {cur_y}) | ห่างเป้าหมายกล่อง: {dist:.2f} หน่วย")
            if dist <= 2.5:
                return True, pos, dist
            else:
                return False, pos, dist
        time.sleep(0.3)
    return False, None, 999.0

def main():
    print("=" * 60)
    print(f"[*] UtopiaFarmAI: เจอกล่อง -> หันจอ -> เดินชน -> เช็กพิกัดกล่อง {TARGET_CHEST}")
    print("=" * 60)
    time.sleep(2)

    CENTER_X = 640
    TOLERANCE = 75
    scan_start_time = time.time()

    while True:
        scr = get_screen()
        found, chest_x, area = detect_chest(scr)

        if found:
            scan_start_time = time.time()
            print(f"\n[👁] ตรวจพบกล่องแล้ว! (X={chest_x}, ขนาด: {area:.1f})")
            print("[⏸] หยุดรอภาพนิ่ง 2 วินาที...")
            time.sleep(2.0)

            aligned = False
            for step in range(8):
                still_scr = get_screen()
                still_found, still_x, _ = detect_chest(still_scr)
                if not still_found:
                    rotate_view("right", dist=180)
                    continue

                diff = still_x - CENTER_X
                if still_x < (CENTER_X - TOLERANCE):
                    print(f"[<] หมุนซ้ายจัดกลางจอ (ห่าง {diff:+d})...")
                    rotate_view("left", dist=200)
                elif still_x > (CENTER_X + TOLERANCE):
                    print(f"[>] หมุนขวาจัดกลางจอ (ห่าง {diff:+d})...")
                    rotate_view("right", dist=200)
                else:
                    print(f"\n[🎯] ล็อกกล่องอยู่ตรงกลางจอสำเร็จ! (X={still_x})")
                    aligned = True
                    break

            if aligned:
                print("[↑] กำลังเดินตรงเข้าชนกล่อง...")
                for i in range(5):
                    walk_straight(duration_ms=500)
                    time.sleep(0.15)

                time.sleep(0.5)
                # ตรวจสอบพิกัดจริงหลังเดินเข้าชน
                print("\n" + "=" * 50)
                print("[🔍] ตรวจสอบพิกัดตำแหน่งหลังเดินชนกล่อง...")
                is_correct, pos, dist = is_at_chest_coords()

                if not is_correct:
                    print(f"\n[!] ไม่สำเร็จ: ไม่ใช่พิกัดกล่อง! (ตำแหน่งปัจจุบัน: {pos}, ห่างเป้าหมาย {dist:.2f} หน่วย)")
                    print("=" * 50)
                    break

                # ตรวจสอบปุ่มรูปมือ
                chk_scr = get_screen()
                if check_hand_button(chk_scr):
                    print(f"[🎯] พิกัดถูกต้อง ({pos}) และพบปุ่มรูปมือ! แตะเปิดกล่อง...")
                    tap(1045, 390)
                    time.sleep(0.3)
                    tap(1045, 390)
                    print("[✓] ภารกิจเปิดกล่องเสร็จสมบูรณ์!")
                else:
                    print(f"[🎯] ถึงพิกัดกล่องถูกต้อง ({pos})! แตะเปิดกล่อง...")
                    tap(1045, 390)
                    time.sleep(0.3)
                    tap(1045, 390)
                    print("[✓] ภารกิจเปิดกล่องเสร็จสมบูรณ์!")
                break
        else:
            elapsed = time.time() - scan_start_time
            print(f"[🔄] กวาดสายตาหากล่อง... ({elapsed:.1f}/5.0 วิ)")
            if elapsed >= 5.0:
                print("\n[!] ไม่สำเร็จ: กวาดสายตาเกิน 5 วิแล้วไม่พบกล่อง")
                break
            rotate_view("right", dist=240)

if __name__ == "__main__":
    main()
