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

    # ตัดกรอบโซนพิกัดกว้างพิเศษ
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

# พิกัดกึ่งกลางปุ่มจอยสติ๊กสำหรับจอ 1280x720
JOY_CENTER_X = 250
JOY_CENTER_Y = 530
JOY_RADIUS   = 145

def walk_in_direction(angle_deg, duration_ms=350):
    angle_rad = math.radians(angle_deg)
    target_x = int(JOY_CENTER_X + JOY_RADIUS * math.sin(angle_rad))
    target_y = int(JOY_CENTER_Y - JOY_RADIUS * math.cos(angle_rad))
    swipe(JOY_CENTER_X, JOY_CENTER_Y, target_x, target_y, duration=duration_ms)
    time.sleep(0.2)

def walk_square_step(dx, dy, duration_sec):
    if abs(dx) >= abs(dy):
        if dx > 0:
            angle = 90
            axis_name = f"ขวา (X: +{dx})"
        else:
            angle = 270
            axis_name = f"ซ้าย (X: {dx})"
    else:
        if dy > 0:
            angle = 0
            axis_name = f"ตรงไปข้างหน้า (Y: +{dy})"
        else:
            angle = 180
            axis_name = f"ถอยหลัง (Y: {dy})"

    print(f"[->] เดินตรงสี่เหลี่ยม: มุ่งหน้าทาง {axis_name} เป็นเวลา {duration_sec} วิ")
    walk_in_direction(angle, duration_ms=int(duration_sec * 1000))

# พิกัดตำแหน่งกล่องเป้าหมาย (X, Y)
TARGET_CHEST = (17527, 15331)

TIMEOUT_LIMIT = 10
DIRECTIONS = [0, 45, 90, 135, 180, 225, 270, 315]

def main():
    print("=" * 60)
    print(f"[*] UtopiaFarmAI: ระบบล็อกพิกัดกล่อง {TARGET_CHEST}")
    print(f"[*] เงื่อนไขหยุด: พิกัดคลาดเคลื่อนได้ +-1 หรือ +-2 แต้ม")
    print("=" * 60)
    time.sleep(3)

    not_found_start = None
    dir_index = 0

    while True:
        scr = get_screen()
        pos = extract_coords(scr)

        # 1. หากหาพิกัดไม่เจอ
        if pos is None:
            if not_found_start is None:
                not_found_start = time.time()

            elapsed = int(time.time() - not_found_start)
            current_dir = DIRECTIONS[dir_index % len(DIRECTIONS)]
            dir_index += 1
            print(f"[?] สำรวจหาพิกัด... ({elapsed}/{TIMEOUT_LIMIT} วิ) -> ก้าวเปลี่ยนมุม {current_dir}°")

            if elapsed >= TIMEOUT_LIMIT:
                print(f"\n[!] หาพิกัดไม่เจอเกิน {TIMEOUT_LIMIT} วิ -> สั่งหยุดการทำงานทันที")
                break

            walk_in_direction(current_dir, duration_ms=350)
            continue

        # 2. อ่านพิกัดสำเร็จ
        not_found_start = None
        cur_x, cur_y = pos
        
        diff_x = TARGET_CHEST[0] - cur_x
        diff_y = TARGET_CHEST[1] - cur_y
        dist = math.sqrt(diff_x**2 + diff_y**2)

        print("-" * 55)
        print(f"[✓] พิกัดปัจจุบัน: ({cur_x}, {cur_y}) | เป้าหมาย: {TARGET_CHEST}")
        print(f"[✓] ผลต่างพิกัด: diff_X = {diff_x:+d}, diff_Y = {diff_y:+d} (ห่าง {dist:.1f} หน่วย)")

        # 3. เงื่อนไขระยะ +-1 หรือ +-2 แต้ม
        # ตัวอย่าง: ถ้า X ต่างกันไม่เกิน 2 (-2, -1, 0, +1, +2) และ Y ต่างกันไม่เกิน 2
        if abs(diff_x) <= 2 and abs(diff_y) <= 2:
            print("\n" + "=" * 50)
            print(f"[✓] พิกัดอยู่ในระยะ +-1 หรือ +-2 ของกล่องแล้ว!")
            print(f"[✓] หยุดเดินทันที และกดปุ่มรูปมือเปิดกล่อง (X:{cur_x}, Y:{cur_y})")
            print("=" * 50)
            time.sleep(0.5)
            tap(820, 430)  # ปุ่มรูปมือเปิดกล่อง
            break

        # 4. กำหนดเวลาเดินและหยุดพักตามระยะห่าง
        if dist > 100.0:
            step_duration = 4.0   # ห่างเกิน 100 เดินยาว 4 วิ
            pause_time = 1.5      # พักสั้น 1.5 วิ
            print(f"[>>] ระยะไกลมาก (>100) -> เดินยาว {step_duration} วิ (พักคำนวณ {pause_time} วิ)")
        elif dist > 20.0:
            step_duration = 2.0
            pause_time = 2.0
            print(f"[->] ระยะปานกลาง -> เดิน {step_duration} วิ (พัก {pause_time} วิ)")
        elif dist > 6.0:
            step_duration = 1.0
            pause_time = 3.0
            print(f"[->] เริ่มเข้าใกล้ -> เดิน {step_duration} วิ (พัก {pause_time} วิ)")
        else:
            step_duration = 0.45  # ระยะประชิด ก้าวสั้นป้องกันเดินเลย
            pause_time = 3.5
            print(f"[->] ระยะประชิด -> ก้าวสั้น {step_duration} วิ (พักนิ่ง {pause_time} วิ)")

        print(f"[⏸] หยุดนิ่ง {pause_time} วินาที เพื่อคำนวณเส้นทาง...")
        time.sleep(pause_time)

        # 5. สั่งเดินสี่เหลี่ยม
        walk_square_step(diff_x, diff_y, step_duration)

if __name__ == "__main__":
    main()
