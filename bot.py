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
    """ตัดกรอบพิกัดและถอดรหัสตัวเลข (ตรวจจับหลายระดับแสงเพื่อความแม่นยำ)"""
    if screen is None:
        return None

    h, w = screen.shape[:2]
    if w > h:
        roi = screen[200:280, 50:240]
    else:
        roi = screen[50:240, 200:280]
        roi = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)

    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    # ทดสอบ 2 ระดับแสง (180 และ 200) เพื่อไม่ให้หลุดเวลาแสงเปลี่ยน
    custom_cfg = r"--psm 6 -c tessedit_char_whitelist=0123456789,."
    for th_val in [185, 205]:
        _, thresh = cv2.threshold(resized, th_val, 255, cv2.THRESH_BINARY)
        raw_text = pytesseract.image_to_string(thresh, config=custom_cfg)

        matches = re.findall(r"\d{5}", raw_text)
        if len(matches) >= 2:
            try:
                x, y = int(matches[0]), int(matches[1])
                if 17000 <= x <= 18000 and 15000 <= y <= 16000:
                    return (x, y)
            except ValueError:
                pass

        nums = re.findall(r"\d{4,6}", raw_text)
        if len(nums) >= 2:
            try:
                x, y = int(nums[0]), int(nums[1])
                if 17000 <= x <= 18000 and 15000 <= y <= 16000:
                    return (x, y)
            except ValueError:
                pass

    return None

# พิกัดกึ่งกลางปุ่มจอยสติ๊กสำหรับจอ 1280x720
JOY_CENTER_X = 250
JOY_CENTER_Y = 530
JOY_RADIUS   = 145

def walk_in_direction(angle_deg, duration_ms=450):
    """เดินสั้น 360° สำรวจหาพิกัดตั้งต้นหรือเปลี่ยนมุมแสง"""
    angle_rad = math.radians(angle_deg)
    target_x = int(JOY_CENTER_X + JOY_RADIUS * math.sin(angle_rad))
    target_y = int(JOY_CENTER_Y - JOY_RADIUS * math.cos(angle_rad))
    swipe(JOY_CENTER_X, JOY_CENTER_Y, target_x, target_y, duration=duration_ms)
    time.sleep(0.2)

def walk_vector(dx, dy, duration_sec):
    """เดินตามเวกเตอร์มุ่งหน้าสู่กล่อง"""
    angle_rad = math.atan2(dx, dy)
    target_x = int(JOY_CENTER_X + JOY_RADIUS * math.sin(angle_rad))
    target_y = int(JOY_CENTER_Y - JOY_RADIUS * math.cos(angle_rad))
    duration_ms = int(duration_sec * 1000)
    swipe(JOY_CENTER_X, JOY_CENTER_Y, target_x, target_y, duration=duration_ms)
    time.sleep(0.2)

TARGET_CHEST = (17529, 15339)
TIMEOUT_LIMIT = 60
STOP_DURATION = 2  # หยุด 2 วินาทีเพื่อรอภาพนิ่งและคำนวณ

DIRECTIONS = [0, 45, 90, 135, 180, 225, 270, 315]

def main():
    print("=" * 60)
    print(f"[*] UtopiaFarmAI: เดินสำรวจ 360° -> คำนวณเดินยาว -> เป้าหมาย {TARGET_CHEST}")
    print(f"[*] ตรวจจับพิกัดคลาดเคลื่อนอัตโนมัติ | พัก {STOP_DURATION} วิ ทุกช่วงก้าว")
    print("=" * 60)
    time.sleep(3)

    not_found_start = None
    dir_index = 0
    last_distance = None

    while True:
        scr = get_screen()
        pos = extract_coords(scr)

        # 1. หากยังหาพิกัดไม่เจอ ใช้การเดิน 360° ก้าวสั้นเพื่อค้นหา
        if pos is None:
            if not_found_start is None:
                not_found_start = time.time()

            elapsed = int(time.time() - not_found_start)
            current_dir = DIRECTIONS[dir_index % len(DIRECTIONS)]
            dir_index += 1
            print(f"[?] กำลังหาทิศทาง 360°... ({elapsed}/{TIMEOUT_LIMIT} วิ) -> ก้าวทดสอบ {current_dir}°")

            if elapsed >= TIMEOUT_LIMIT:
                print(f"\n[!] ไม่พบพิกัดต่อเนื่องเกิน {TIMEOUT_LIMIT} วิ -> หยุดการทำงานอัตโนมัติ")
                break

            walk_in_direction(current_dir, duration_ms=400)
            continue

        # 2. อ่านพิกัดสำเร็จ -> รีเซ็ตตัวนับ Timeout
        not_found_start = None
        cur_x, cur_y = pos
        
        dx = TARGET_CHEST[0] - cur_x
        dy = TARGET_CHEST[1] - cur_y
        dist = math.sqrt(dx**2 + dy**2)

        # ตรวจสอบว่าพิกัดคลาดเคลื่อนหรือเดินออกห่างหรือไม่
        drift_status = ""
        if last_distance is not None:
            if dist < last_distance:
                drift_status = "[✓ เข้าใกล้ขึ้น]"
            elif dist > last_distance:
                drift_status = "[! คลาดเคลื่อน (ระยะเพิ่มขึ้น) -> คำนวณเวกเตอร์ใหม่]"
            else:
                drift_status = "[= อยู่จุดเดิม]"

        print("-" * 55)
        print(f"[✓] พิกัด: ({cur_x}, {cur_y}) | ห่างกล่อง: {dist:.1f} หน่วย (dx={dx}, dy={dy}) {drift_status}")
        print(f"[⏸] หยุดนิ่ง {STOP_DURATION} วินาที เพื่อประมวลผล...")

        # หยุดนิ่ง 2 วินาทีเพื่อให้ตัวละครหยุดสนิท
        time.sleep(STOP_DURATION)

        # 3. ตรวจสอบเงื่อนไขถึงเป้าหมาย
        if dist <= 3.0:
            print("\n" + "=" * 50)
            print("[✓] ถึงตำแหน่งกล่องเรียบร้อยแล้ว!")
            print("=" * 50)
            time.sleep(0.5)
            tap(820, 430)  # แตะเปิดกล่อง
            break

        # 4. คำนวณระยะเดิน: ไกลใช้เดินยาว ใกล้ปรับเป็นก้าวสั้นป้องกันเดินเลย
        if dist > 8.0:
            step_time = 2.0  # เดินยาว
            print(f"[->] คำนวณเสร็จสิ้น -> เดินยาว {step_time} วินาที...")
        else:
            step_time = 0.7  # ระยะประชิด ปรับเป็นก้าวสั้น
            print(f"[->] ใกล้ถึงจุดหมาย -> เดินสั้น {step_time} วินาที ป้องกันเดินเลย...")

        walk_vector(dx, dy, duration_sec=step_time)
        last_distance = dist

if __name__ == "__main__":
    main()
    
