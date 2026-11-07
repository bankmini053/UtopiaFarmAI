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

def tap(x, y):
    subprocess.run(f"{ROOT_SU} -c 'input tap {x} {y}'", shell=True)

def swipe(x1, y1, x2, y2, duration=250):
    subprocess.run(f"{ROOT_SU} -c 'input swipe {x1} {y1} {x2} {y2} {duration}'", shell=True)

def get_screen():
    subprocess.run(f"{ROOT_SU} -c 'screencap -p {TMP_PATH}'", shell=True)
    if os.path.exists(TMP_PATH):
        return cv2.imread(TMP_PATH)
    return None

def extract_coords(screen):
    if screen is None:
        return None

    h, w = screen.shape[:2]
    if w > h:
        roi = screen[180:300, 40:270]
    else:
        roi = screen[40:270, 180:300]
        roi = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)

    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    custom_cfg = r"--psm 6 -c tessedit_char_whitelist=0123456789,."
    for th_val in [175, 195, 215]:
        _, thresh = cv2.threshold(resized, th_val, 255, cv2.THRESH_BINARY)
        raw_text = pytesseract.image_to_string(thresh, config=custom_cfg)

        matches = re.findall(r"\d{5}", raw_text)
        if len(matches) >= 2:
            try:
                x, y = int(matches[0]), int(matches[1])
                if 17300 <= x <= 17700 and 15200 <= y <= 15500:
                    return (x, y)
            except ValueError:
                pass

        nums = re.findall(r"\d{4,6}", raw_text)
        if len(nums) >= 2:
            try:
                x, y = int(nums[0]), int(nums[1])
                if 17300 <= x <= 17700 and 15200 <= y <= 15500:
                    return (x, y)
            except ValueError:
                pass

    return None

JOY_CENTER_X = 250
JOY_CENTER_Y = 530
JOY_RADIUS   = 145

def walk_forward(duration_ms=600):
    """ดันจอยสติ๊กตรงไปข้างหน้า (0 องศา) เพื่อเดินตรงเข้าหากล่อง"""
    target_x = JOY_CENTER_X
    target_y = JOY_CENTER_Y - JOY_RADIUS
    swipe(JOY_CENTER_X, JOY_CENTER_Y, target_x, target_y, duration=duration_ms)
    time.sleep(0.2)

def rotate_camera(direction="right", dist=200):
    """รูดจอฝั่งขวาเพื่อหมุนมุมมองหน้าจอ"""
    cx, cy = 900, 360
    if direction == "right":
        swipe(cx, cy, cx - dist, cy, duration=220)
    else:
        swipe(cx, cy, cx + dist, cy, duration=220)
    time.sleep(0.35)

def detect_chest_position(screen):
    """
    สแกนหากล่องสมบัติบนหน้าจอทั้งช่วงเช้าและเย็น
    คืนค่า: (เจอกล่องไหม, พิกัดกึ่งกลาง X ของกล่อง, พื้นที่กล่อง)
    """
    if screen is None:
        return False, 0, 0

    h, w = screen.shape[:2]
    if h > w:
        screen = cv2.rotate(screen, cv2.ROTATE_90_CLOCKWISE)

    # ตัดเฉพาะโซนมองเห็น (ไม่รวมปุ่มควบคุมด้านล่าง)
    roi = screen[50:560, 150:1130]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # สีกล่องตอนเช้า (ส้มไม้สว่าง)
    mask_day = cv2.inRange(hsv, np.array([8, 100, 110]), np.array([25, 255, 255]))
    # สีกล่องตอนเย็น/ค่ำ (น้ำตาลเข้มหม่น)
    mask_night = cv2.inRange(hsv, np.array([5, 30, 30]), np.array([28, 170, 130]))

    mask = cv2.bitwise_or(mask_day, mask_night)
    kernel = np.ones((5, 5), np.uint8)
    clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            x, y, cw, ch = cv2.boundingRect(cnt)
            ratio = float(cw) / ch
            if 0.6 <= ratio <= 1.5:  # ทรงสี่เหลี่ยมกล่อง
                center_x = x + 150 + (cw // 2)
                return True, center_x, area

    return False, 0, 0

def wait_coords(max_wait=3.0):
    st = time.time()
    last = None
    while time.time() - st < max_wait:
        scr = get_screen()
        pos = extract_coords(scr)
        if pos is not None:
            if pos == last:
                return pos
            last = pos
        time.sleep(0.25)
    return last

TARGET_CHEST = (17527, 15336)

def main():
    print("=" * 60)
    print(f"[*] UtopiaFarmAI: หมุนหาจอกล่อง -> จัดกึ่งกลาง -> เดินตรงเข้าพิกัด")
    print(f"[*] พิกัดเป้าหมายกล่อง: {TARGET_CHEST}")
    print("=" * 60)
    time.sleep(2)

    while True:
        # 1. เช็กพิกัดปัจจุบันก่อน
        scr = get_screen()
        pos = wait_coords(max_wait=2.5)

        if pos is not None:
            cur_x, cur_y = pos
            dx = TARGET_CHEST[0] - cur_x
            dy = TARGET_CHEST[1] - cur_y
            dist = math.sqrt(dx**2 + dy**2)
            print(f"[✓] พิกัด: ({cur_x}, {cur_y}) | ห่างจากกล่อง: {dist:.1f} หน่วย (dx={dx:+d}, dy={dy:+d})")

            # ถ้าถึงจุดหน้ากล่องแล้ว (ระยะห่าง <= 1.5 หรือ +-1 แต้ม)
            if dist <= 1.5 or (abs(dx) <= 1 and abs(dy) <= 1):
                print("\n" + "=" * 50)
                print(f"[🎯] ถึงพิกัดกล่องเรียบร้อยแล้ว! ({cur_x}, {cur_y})")
                print("[🎯] แตะปุ่มรูปมือเปิดกล่อง...")
                print("=" * 50)
                time.sleep(0.3)
                tap(820, 430)
                time.sleep(0.3)
                tap(820, 430)
                break
        else:
            print("[?] กำลังค้นหาตำแหน่งพิกัด...")

        # 2. ตรวจสอบว่าเห็นกล่องบนหน้าจอหรือไม่
        found, chest_x, area = detect_chest_position(scr)

        if found:
            print(f"[👁] เห็นกล่องบนหน้าจอ! (ตำแหน่งแกน X: {chest_x}, ขนาด: {area})")

            # 3. จัดกล่องให้อยู่กลางจอ (จอแนวนอน 1280 กลางจอคือ 640)
            if chest_x < 530:
                print("[<] กล่องอยู่เอียงซ้าย -> หมุนกล้องซ้ายนิดหน่อย")
                rotate_camera("left", dist=90)
                continue
            elif chest_x > 750:
                print("[>] กล่องอยู่เอียงขวา -> หมุนกล้องขวานิดหน่อย")
                rotate_camera("right", dist=90)
                continue
            else:
                print("[↑] กล่องอยู่ตรงกลางจอแล้ว! เดินตรงไปข้างหน้า (0°)...")
                # เดินตรงเข้าหากล่อง
                walk_forward(duration_ms=650)
        else:
            # 4. ถ้ายังไม่เห็นกล่องบนหน้าจอ ให้หมุนกล้องค้นหารอบตัว
            print("[🔄] ยังไม่เห็นกล่องบนหน้าจอ -> หมุนหน้าจอค้นหา...")
            rotate_camera("right", dist=200)

if __name__ == "__main__":
    main()
    
