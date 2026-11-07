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

def swipe_screen(x1, y1, x2, y2, duration=220):
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
        return pos, dist
    return None, 999.0

def rotate_camera_towards(diff):
    """
    ปรับมุมกล้องแบบนุ่มนวล ไม่สะบัดแรง:
    diff < 0 (กล่องอยู่ซ้าย) -> ลากซ้ายไปขวา
    diff > 0 (กล่องอยู่ขวา) -> ลากขวาไปซ้าย
    """
    dist = int(abs(diff) * 0.28)  # ลดอัตราทดเพื่อกันไม่ให้สะบัดข้ามหัว
    dist = min(max(dist, 35), 140)

    cy = 260
    if diff < 0:
        swipe_screen(750, cy, 750 + dist, cy, duration=200)
    else:
        swipe_screen(850, cy, 850 - dist, cy, duration=200)
    time.sleep(0.35)

def scan_rotate_right(dist=160):
    cx, cy = 850, 260
    swipe_screen(cx, cy, cx - dist, cy, duration=220)
    time.sleep(0.4)

JOY_CENTER_X, JOY_CENTER_Y, JOY_RADIUS = 250, 530, 145

def walk_straight(duration_ms=400):
    swipe_screen(JOY_CENTER_X, JOY_CENTER_Y, JOY_CENTER_X, JOY_CENTER_Y - JOY_RADIUS, duration=duration_ms)
    time.sleep(0.1)

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
    print("[*] UtopiaFarmAI: เจอกล่อง -> พัก 5 วิ -> คำนวณภาพ+พิกัดสดตลอดเวลา")
    print(f"[*] พิกัดเป้าหมาย: {TARGET_CHEST}")
    print("=" * 60)
    time.sleep(2)

    CENTER_X = 640
    TOLERANCE = 50

    # 1. กวาดสายตา 360 องศา
    found_chest = False
    for step in range(8):
        scr = get_screen()
        found, chest_x, area = detect_chest(scr)
        if found:
            print(f"\n[👁] ตรวจพบกล่องแล้วที่มุมสเต็ป {step + 1}/8! (X={chest_x}, ขนาด: {area:.1f})")
            print("[🛑] สั่งหยุดนิ่งสนิททันที 5 วินาที เพื่อให้ภาพนิ่ง...")
            found_chest = True
            break
        else:
            print(f"[🔄] กวาดสายตามุมที่ {step + 1}/8 -> กำลังหมุนต่อ...")
            scan_rotate_right(dist=160)

    if not found_chest:
        print("\n[!] ไม่สำเร็จ: กวาดสายตาครบ 360 องศาแล้วไม่พบกล่อง")
        return

    # 2. หยุดนิ่งสนิท 5 วินาทีตามที่สั่ง
    time.sleep(5.0)

    # 3. จัดกึ่งกลางรอบแรกให้กล่องเข้ามาอยู่ตรงหน้า
    aligned = False
    for attempt in range(8):
        scr = get_screen()
        found, chest_x, _ = detect_chest(scr)

        if not found:
            print("[?] กำลังค้นหากล่องใหม่...")
            time.sleep(0.3)
            continue

        diff = chest_x - CENTER_X
        if abs(diff) <= TOLERANCE:
            print(f"\n[🎯] ล็อกกล่องกึ่งกลางจอเป๊ะแล้ว! (X={chest_x}, ห่าง: {diff:+d}px)")
            aligned = True
            break

        print(f"[*] จัดกึ่งกลาง: กล่องอยู่ที่ X={chest_x} (ห่าง: {diff:+d})")
        rotate_camera_towards(diff)

    if not aligned:
        print("\n[!] ไม่สำเร็จ: ไม่สามารถล็อกกล่องให้อยู่ตรงกลางจอได้")
        return

    # 4. ลูปเดินเข้าชน: คำนวณทั้งภาพจัดกลางจอ + พิกัดระยะห่าง ตลอดเวลาแบบ Real-time
    print("\n" + "=" * 50)
    print("[↑] เริ่มเดินเข้าหากล่อง (คำนวณภาพ + พิกัด สดทุกก้าว)...")
    print("=" * 50)

    opened = False
    for step in range(15):
        scr = get_screen()
        pos, dist = get_live_distance(scr)
        found, chest_x, _ = detect_chest(scr)

        # ก. ตรวจสอบปุ่มรูปมือ
        if check_hand_button(scr):
            print(f"\n[🎯] ตรวจพบปุ่มรูปมือแล้ว! (พิกัด: {pos}, ห่างกล่อง: {dist:.2f} หน่วย)")
            tap(1045, 390)
            time.sleep(0.3)
            tap(1045, 390)
            opened = True
            break

        # ข. ตรวจสอบระยะพิกัดจริง
        if dist <= 2.2:
            print(f"\n[🎯] ถึงระยะประชิดกล่องแล้ว! (พิกัด: {pos}, ห่าง: {dist:.2f} หน่วย)")
            time.sleep(0.3)
            tap(1045, 390)
            time.sleep(0.3)
            tap(1045, 390)
            opened = True
            break

        # ค. คำนวณปรับหน้าจอสดระหว่างเดิน (ถ้าเดินแล้วกล่องเอียงออกจากกึ่งกลางจอ)
        if found:
            diff = chest_x - CENTER_X
            if abs(diff) > TOLERANCE:
                print(f"[🔄] ปรับมุมมองระหว่างเดิน: กล่องเอียง X={chest_x} ({diff:+d})")
                rotate_camera_towards(diff)

        # ง. เดินตรงไปข้างหน้า 1 จังหวะ
        print(f"[*] ก้าวที่ {step + 1} | พิกัด: {pos} | ระยะห่าง: {dist:.2f} หน่วย | กล่อง X={chest_x if found else '?'}")
        walk_straight(duration_ms=450)

    if opened:
        print("\n[✓] ภารกิจเปิดกล่องเสร็จสมบูรณ์!")
    else:
        final_scr = get_screen()
        final_pos, final_dist = get_live_distance(final_scr)
        print(f"\n[!] ไม่สำเร็จ: เดินไม่ถึงกล่อง (พิกัด: {final_pos}, ห่าง: {final_dist:.2f})")

if __name__ == "__main__":
    main()
