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
    """
    ขยายจุดรับพิกัดให้ใหญ่ขึ้น ไม่ตกขอบ รองรับทั้งแนวนอนและแนวตั้ง
    """
    if screen is None:
        return None

    h, w = screen.shape[:2]

    if w > h:
        # แนวนอน 1280x720: Y: 180-300, X: 40-270
        roi = screen[180:300, 40:270]
    else:
        # แนวตั้ง 720x1280: หมุนภาพตามเข็มนาฬิกา
        roi = screen[40:270, 180:300]
        roi = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)

    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    custom_cfg = r"--psm 6 -c tessedit_char_whitelist=0123456789,."
    
    # ทดสอบฟิลเตอร์แสง 3 ระดับ
    for th_val in [175, 195, 215]:
        _, thresh = cv2.threshold(resized, th_val, 255, cv2.THRESH_BINARY)
        raw_text = pytesseract.image_to_string(thresh, config=custom_cfg)

        matches = re.findall(r"\d{5}", raw_text)
        if len(matches) >= 2:
            try:
                x, y = int(matches[0]), int(matches[1])
                if 17400 <= x <= 17600 and 15250 <= y <= 15400:
                    return (x, y)
            except ValueError:
                pass

        nums = re.findall(r"\d{4,6}", raw_text)
        if len(nums) >= 2:
            try:
                x, y = int(nums[0]), int(nums[1])
                if 17400 <= x <= 17600 and 15250 <= y <= 15400:
                    return (x, y)
            except ValueError:
                pass

    return None

# พิกัดกึ่งกลางปุ่มจอยสติ๊กสำหรับจอ 1280x720
JOY_CENTER_X = 250
JOY_CENTER_Y = 530
JOY_RADIUS   = 145

def walk_in_direction(angle_deg, duration_ms=350):
    """เดินสั้นสำรวจเปลี่ยนมุมแสงเวลาหาพิกัดไม่เจอ"""
    angle_rad = math.radians(angle_deg)
    target_x = int(JOY_CENTER_X + JOY_RADIUS * math.sin(angle_rad))
    target_y = int(JOY_CENTER_Y - JOY_RADIUS * math.cos(angle_rad))
    swipe(JOY_CENTER_X, JOY_CENTER_Y, target_x, target_y, duration=duration_ms)
    time.sleep(0.2)

def walk_square_step(dx, dy, dist):
    """
    ระบบเดินสี่เหลี่ยมปรับสั้นกระชับ:
    ระยะไกลใช้ 1.1 วิ | ระยะใกล้ใช้ 0.45 วิ ป้องกันเดินเลยกล่อง
    """
    duration = 1.1 if dist > 6.0 else 0.45

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

    print(f"[->] เดินเป็นเส้นตรงสี่เหลี่ยม: มุ่งหน้าทาง {axis_name} เป็นเวลา {duration} วิ")
    walk_in_direction(angle, duration_ms=int(duration * 1000))

TARGET_CHEST = (17529, 15339)
TIMEOUT_LIMIT = 10     # ลดเวลาหาพิกัดไม่เจอเหลือ 10 วินาที แล้วสั่งหยุดทันที
STOP_DURATION = 3.5    # ปรับเวลาหยุดนิ่งเป็น 3.5 วินาที เพื่อความแม่นยำ

DIRECTIONS = [0, 45, 90, 135, 180, 225, 270, 315]

def main():
    print("=" * 60)
    print(f"[*] UtopiaFarmAI: ก้าวสั้นกระชับ + หยุดคำนวณ {STOP_DURATION} วิ")
    print(f"[*] พิกัดเป้าหมายกล่อง: {TARGET_CHEST}")
    print(f"[*] ระบบความปลอดภัย: หาพิกัดไม่เจอเกิน {TIMEOUT_LIMIT} วิ จะหยุดทันที")
    print("=" * 60)
    time.sleep(3)

    not_found_start = None
    dir_index = 0

    while True:
        scr = get_screen()
        pos = extract_coords(scr)

        # 1. หากยังไม่พบพิกัด
        if pos is None:
            if not_found_start is None:
                not_found_start = time.time()

            elapsed = int(time.time() - not_found_start)
            current_dir = DIRECTIONS[dir_index % len(DIRECTIONS)]
            dir_index += 1
            print(f"[?] สำรวจหาพิกัด... ({elapsed}/{TIMEOUT_LIMIT} วิ) -> ก้าวเปลี่ยนมุม {current_dir}°")

            # เช็กเงื่อนไข Timeout 10 วินาที
            if elapsed >= TIMEOUT_LIMIT:
                print(f"\n[!] ไม่พบพิกัดต่อเนื่องครบ {TIMEOUT_LIMIT} วิ -> สั่งหยุดการทำงานทันที")
                break

            walk_in_direction(current_dir, duration_ms=350)
            continue

        # 2. อ่านพิกัดสำเร็จ
        not_found_start = None
        cur_x, cur_y = pos
        
        dx = TARGET_CHEST[0] - cur_x
        dy = TARGET_CHEST[1] - cur_y
        dist = math.sqrt(dx**2 + dy**2)

        print("-" * 55)
        print(f"[✓] พิกัดปัจจุบัน: ({cur_x}, {cur_y}) | ห่างเป้าหมาย: {dist:.1f} หน่วย (dx={dx}, dy={dy})")
        print(f"[⏸] สั่งหยุดนิ่ง {STOP_DURATION} วินาที เพื่อให้ตัวละครหยุดสนิทและคำนวณ...")

        # หยุดนิ่ง 3.5 วินาที
        time.sleep(STOP_DURATION)

        # 3. ถึงเป้าหมายประชิด (ระยะห่างไม่เกิน 1.5 หน่วย จะอยู่หน้ากล่องพอดี)
        if dist <= 1.5:
            print("\n" + "=" * 50)
            print("[✓] ประชิดหน้ากล่องเรียบร้อยแล้ว!")
            print("=" * 50)
            time.sleep(0.5)
            tap(820, 430)  # แตะเปิดกล่อง
            break

        # 4. เดินสี่เหลี่ยมแบบก้าวสั้น
        walk_square_step(dx, dy, dist)

if __name__ == "__main__":
    main()
            
