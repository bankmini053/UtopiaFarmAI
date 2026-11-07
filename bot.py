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

    # ขยายพื้นที่ Crop ให้กว้างขึ้นทั้งความสูงและความกว้าง
    if w > h:
        # แนวนอน 1280x720: ขยายขอบเขตเป็น Y: 180-300, X: 40-270
        roi = screen[180:300, 40:270]
    else:
        # แนวตั้ง 720x1280: ขยายขอบเขตและหมุนภาพให้ตรง
        roi = screen[40:270, 180:300]
        roi = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)

    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    custom_cfg = r"--psm 6 -c tessedit_char_whitelist=0123456789,."
    
    # ทดสอบฟิลเตอร์แสง 3 ระดับ เพื่อความติดง่ายในทุกสภาพสีพื้นหลัง
    for th_val in [175, 195, 215]:
        _, thresh = cv2.threshold(resized, th_val, 255, cv2.THRESH_BINARY)
        raw_text = pytesseract.image_to_string(thresh, config=custom_cfg)

        matches = re.findall(r"\d{5}", raw_text)
        if len(matches) >= 2:
            try:
                x, y = int(matches[0]), int(matches[1])
                # กรองค่าต้องอยู่ในเกาะ ไม่ให้หลุดไป 155xx ที่เพี้ยน
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

def walk_in_direction(angle_deg, duration_ms=450):
    """เดินสั้น 360° สำรวจหาพิกัดตั้งต้น"""
    angle_rad = math.radians(angle_deg)
    target_x = int(JOY_CENTER_X + JOY_RADIUS * math.sin(angle_rad))
    target_y = int(JOY_CENTER_Y - JOY_RADIUS * math.cos(angle_rad))
    swipe(JOY_CENTER_X, JOY_CENTER_Y, target_x, target_y, duration=duration_ms)
    time.sleep(0.2)

def walk_square_step(dx, dy, dist):
    """
    ระบบเดินเป็นสี่เหลี่ยม (Grid Movement):
    เลือกเดินแกนที่มีระยะห่างมากที่สุดตรงๆ ก่อน (0°, 90°, 180°, 270°)
    """
    duration = 1.8 if dist > 8.0 else 0.8

    # เลือกเดินแกนที่ห่างมากกว่าก่อน
    if abs(dx) >= abs(dy):
        # เดินแกน X (ซ้าย หรือ ขวา)
        if dx > 0:
            angle = 90   # ขวา
            axis_name = f"ขวา (X: +{dx})"
        else:
            angle = 270  # ซ้าย
            axis_name = f"ซ้าย (X: {dx})"
    else:
        # เดินแกน Y (หน้า หรือ หลัง)
        if dy > 0:
            angle = 0    # ตรงไปข้างหน้า
            axis_name = f"ตรงไปข้างหน้า (Y: +{dy})"
        else:
            angle = 180  # ถอยหลัง
            axis_name = f"ถอยหลัง (Y: {dy})"

    print(f"[->] เดินเป็นเส้นตรงสี่เหลี่ยม: มุ่งหน้าทาง {axis_name} เป็นเวลา {duration} วิ")
    walk_in_direction(angle, duration_ms=int(duration * 1000))

TARGET_CHEST = (17529, 15339)
TIMEOUT_LIMIT = 60
STOP_DURATION = 2  # หยุด 2 วินาทีเพื่อรอภาพนิ่งและคำนวณ

DIRECTIONS = [0, 45, 90, 135, 180, 225, 270, 315]

def main():
    print("=" * 60)
    print(f"[*] UtopiaFarmAI: ขยายกรอบพิกัด + เดินสี่เหลี่ยมฉาก 90°")
    print(f"[*] พิกัดเป้าหมาย: {TARGET_CHEST}")
    print("=" * 60)
    time.sleep(3)

    not_found_start = None
    dir_index = 0

    while True:
        scr = get_screen()
        pos = extract_coords(scr)

        # 1. หากยังไม่พบพิกัด ใช้การเดินสั้น 360° เพื่อค้นหา
        if pos is None:
            if not_found_start is None:
                not_found_start = time.time()

            elapsed = int(time.time() - not_found_start)
            current_dir = DIRECTIONS[dir_index % len(DIRECTIONS)]
            dir_index += 1
            print(f"[?] สำรวจหาพิกัด... ({elapsed}/{TIMEOUT_LIMIT} วิ) -> ก้าวทดสอบมุม {current_dir}°")

            if elapsed >= TIMEOUT_LIMIT:
                print(f"\n[!] ไม่พบพิกัดต่อเนื่องเกิน {TIMEOUT_LIMIT} วิ -> หยุดอัตโนมัติ")
                break

            walk_in_direction(current_dir, duration_ms=400)
            continue

        # 2. อ่านพิกัดสำเร็จ
        not_found_start = None
        cur_x, cur_y = pos
        
        dx = TARGET_CHEST[0] - cur_x
        dy = TARGET_CHEST[1] - cur_y
        dist = math.sqrt(dx**2 + dy**2)

        print("-" * 55)
        print(f"[✓] พิกัดปัจจุบัน: ({cur_x}, {cur_y}) | ห่างเป้าหมาย: {dist:.1f} หน่วย (dx={dx}, dy={dy})")
        print(f"[⏸] หยุดนิ่ง {STOP_DURATION} วินาที รอคำนวณแกนสี่เหลี่ยม...")

        time.sleep(STOP_DURATION)

        # 3. ถึงเป้าหมาย
        if dist <= 3.0:
            print("\n" + "=" * 50)
            print("[✓] ถึงตำแหน่งหน้ากล่องเรียบร้อยแล้ว!")
            print("=" * 50)
            time.sleep(0.5)
            tap(820, 430)
            break

        # 4. สั่งเดินเป็นมุมฉากสี่เหลี่ยมตามแกนหลัก
        walk_square_step(dx, dy, dist)

if __name__ == "__main__":
    main()
