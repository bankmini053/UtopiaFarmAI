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

def get_live_distance(screen):
    pos = extract_coords(screen)
    if pos is not None:
        dx = TARGET_CHEST[0] - pos[0]
        dy = TARGET_CHEST[1] - pos[1]
        dist = math.sqrt(dx**2 + dy**2)
        return pos, dist, dx, dy
    return None, 999.0, 0, 0

JOY_CENTER_X = 250
JOY_CENTER_Y = 530
JOY_RADIUS   = 145

def walk_direction(deg, duration_ms=450):
    """
    ดันจอยสติ๊กตามองศา (เดินตรง, สี่เหลี่ยม หรือเดินเฉียง):
    0° = หน้า, 90° = ขวา, 180° = หลัง, 270° = ซ้าย
    45° = เฉียงหน้าขวา, 135° = เฉียงหลังขวา, 225° = เฉียงหลังซ้าย, 315° = เฉียงหน้าซ้าย
    """
    rad = math.radians(deg)
    tx = int(JOY_CENTER_X + JOY_RADIUS * math.sin(rad))
    ty = int(JOY_CENTER_Y - JOY_RADIUS * math.cos(rad))
    swipe_screen(JOY_CENTER_X, JOY_CENTER_Y, tx, ty, duration=duration_ms)
    time.sleep(0.15)

def scan_rotate_right(dist=160):
    cx, cy = 850, 260
    swipe_screen(cx, cy, cx - dist, cy, duration=220)
    time.sleep(0.4)

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

def main():
    print("=" * 60)
    print("[*] UtopiaFarmAI: เจอกล่อง -> พัก 10 วิ -> เดินคุมพิกัด (ตรง/เฉียง)")
    print(f"[*] พิกัดเป้าหมายกล่อง: {TARGET_CHEST}")
    print("=" * 60)
    time.sleep(2)

    # 1. กวาดสายตา 360 องศาตรวจหากล่อง
    found_chest = False
    for step in range(8):
        scr = get_screen()
        found, chest_x, area = detect_chest(scr)
        if found:
            print(f"\n[👁] ตรวจพบกล่องแล้วที่มุมมองสเต็ป {step + 1}/8! (X={chest_x}, ขนาด: {area:.1f})")
            print("[🛑] สั่งหยุดนิ่งสนิท 10 วินาทีตามที่สั่ง...")
            found_chest = True
            break
        else:
            print(f"[🔄] กวาดสายตามุมที่ {step + 1}/8 -> หมุนต่อ...")
            scan_rotate_right(dist=160)

    if not found_chest:
        print("\n[!] ไม่สำเร็จ: กวาดสายตาครบ 360 องศาแล้วไม่พบกล่อง")
        return

    # 2. หยุดนิ่งสนิท 10 วินาที
    for sec in range(10, 0, -1):
        print(f"[*] หยุดรอภาพนิ่ง... เหลืออีก {sec} วินาที")
        time.sleep(1.0)

    # 3. ทดสอบก้าว 1 ครั้งเพื่อเทียบแกนแผนที่กับหน้าจอ (0 องศา)
    print("\n[*] ทดสอบเดินหน้า 1 ก้าวสั้นเพื่อเทียบมุมพิกัด...")
    scr1 = get_screen()
    pos1 = extract_coords(scr1)
    walk_direction(0, duration_ms=450)
    time.sleep(0.3)
    scr2 = get_screen()
    pos2 = extract_coords(scr2)

    facing_angle = 0
    if pos1 and pos2 and pos1 != pos2:
        step_dx = pos2[0] - pos1[0]
        step_dy = pos2[1] - pos1[1]
        facing_angle = math.degrees(math.atan2(step_dx, step_dy))
        print(f"[✓] ทิศหน้าจอเทียบพิกัด: {facing_angle:.1f}°")

    # 4. เดินคุมด้วยพิกัดจริง (ตรง / สี่เหลี่ยม / เฉียง) มุ่งเข้าหากล่อง
    print("\n" + "=" * 50)
    print("[↑] เริ่มต้นเดินเข้าหากล่องตามเวกเตอร์พิกัด...")
    print("=" * 50)

    opened = False
    for step in range(20):
        scr = get_screen()
        pos, dist, dx, dy = get_live_distance(scr)

        # ก. ตรวจสอบปุ่มรูปมือ
        if check_hand_button(scr):
            print(f"\n[🎯] ตรวจพบปุ่มรูปมือแล้ว! (พิกัด: {pos}, ห่าง: {dist:.2f} หน่วย)")
            tap(1045, 390)
            time.sleep(0.3)
            tap(1045, 390)
            opened = True
            break

        # ข. ตรวจสอบระยะประชิด
        if dist <= 2.2:
            print(f"\n[🎯] ถึงระยะประชิดกล่องแล้ว! (พิกัด: {pos}, ห่าง: {dist:.2f} หน่วย)")
            time.sleep(0.3)
            tap(1045, 390)
            time.sleep(0.3)
            tap(1045, 390)
            opened = True
            break

        # ค. คำนวณทิศทางการเดิน (ตรง, ข้าง, หรือเฉียง)
        target_world_angle = math.degrees(math.atan2(dx, dy))
        walk_deg = (target_world_angle - facing_angle + 360) % 360

        print(f"[*] ก้าวที่ {step + 1} | พิกัด: {pos} | ห่าง: {dist:.2f} หน่วย | เดินทิศ: {walk_deg:.1f}°")
        walk_direction(walk_deg, duration_ms=450)
        time.sleep(0.15)

    if opened:
        print("\n[✓] ภารกิจเปิดกล่องเสร็จสมบูรณ์!")
    else:
        final_pos, final_dist, _, _ = get_live_distance(get_screen())
        print(f"\n[!] ไม่สำเร็จ: เดินเข้าหากล่องไม่สำเร็จ (พิกัด: {final_pos}, ห่าง: {final_dist:.2f})")

if __name__ == "__main__":
    main()
