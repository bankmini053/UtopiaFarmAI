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
    ตัดเฉพาะแถวตัวเลขพิกัดใต้แผนที่มินิแมป (จอแนวนอน 1280x720)
    Y: 228-265, X: 65-215 (ตรงตำแหน่งตัวเลข 17xxx, 15xxx พอดี)
    """
    if screen is None:
        return None

    # ตัดเฉพาะบรรทัดตัวเลขพิกัด
    coord_roi = screen[228:265, 65:215]

    # บันทึกภาพล่าสุดไว้ตรวจสอบ
    cv2.imwrite("/data/local/tmp/current_coord_box.png", coord_roi)

    gray = cv2.cvtColor(coord_roi, cv2.COLOR_BGR2GRAY)
    
    # ดึงเฉพาะเนื้อสีขาวของตัวเลข (Value >= 200)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    resized = cv2.resize(thresh, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_NEAREST)

    custom_cfg = r"--psm 7 -c tessedit_char_whitelist=0123456789,"
    raw_text = pytesseract.image_to_string(resized, config=custom_cfg)

    # สกัดตัวเลข 5 หลัก
    matches = re.findall(r"\d{5}", raw_text)
    if len(matches) >= 2:
        try:
            x, y = int(matches[0]), int(matches[1])
            if 17000 <= x <= 18000 and 15000 <= y <= 16000:
                return (x, y)
        except ValueError:
            pass

    # กรณีมีจุลภาคคั่น
    parts = re.split(r"[, \n]+", raw_text.strip())
    valid_nums = [int(p) for p in parts if p.isdigit() and len(p) >= 4]
    if len(valid_nums) >= 2:
        x, y = valid_nums[0], valid_nums[1]
        if 17000 <= x <= 18000 and 15000 <= y <= 16000:
            return (x, y)

    return None

# พิกัดกึ่งกลางปุ่มจอยสติ๊กสำหรับจอ 1280x720
JOY_CENTER_X = 250
JOY_CENTER_Y = 530
JOY_RADIUS   = 140

def walk_by_direction(dx, dy, duration_sec=1.5):
    """เดินตามทิศทางเวกเตอร์ที่คำนวณได้"""
    angle_rad = math.atan2(dx, dy)
    target_x = int(JOY_CENTER_X + JOY_RADIUS * math.sin(angle_rad))
    target_y = int(JOY_CENTER_Y - JOY_RADIUS * math.cos(angle_rad))
    duration_ms = int(duration_sec * 1000)
    
    swipe(JOY_CENTER_X, JOY_CENTER_Y, target_x, target_y, duration=duration_ms)
    time.sleep(0.3)

def walk_forward_step(duration_ms=500):
    swipe(JOY_CENTER_X, JOY_CENTER_Y, JOY_CENTER_X, JOY_CENTER_Y - JOY_RADIUS, duration=duration_ms)
    time.sleep(0.3)

TARGET_CHEST = (17529, 15339)
TIMEOUT_LIMIT = 60
STOP_DURATION = 15  # หยุด 15 วินาทีเมื่ออ่านพิกัดได้

def main():
    print("=" * 60)
    print(f"[*] UtopiaFarmAI: แก้ไขตำแหน่งพิกัดตรงจุด -> เป้าหมาย {TARGET_CHEST}")
    print(f"[*] ตั้งค่าหยุดนิ่ง {STOP_DURATION} วินาทีเมื่อจับพิกัดได้")
    print("=" * 60)
    time.sleep(3)

    not_found_start = None

    while True:
        scr = get_screen()
        pos = extract_coords(scr)

        # กรณีหาพิกัดไม่เจอ
        if pos is None:
            if not_found_start is None:
                not_found_start = time.time()

            elapsed = int(time.time() - not_found_start)
            print(f"[?] กำลังจับพิกัด... ({elapsed}/{TIMEOUT_LIMIT} วิ) ก้าวสั้นๆ เพื่อเปลี่ยนมุมแสง")

            if elapsed >= TIMEOUT_LIMIT:
                print(f"\n[!] ไม่พบพิกัดครบ {TIMEOUT_LIMIT} วิ -> หยุดทำงานอัตโนมัติ")
                break

            walk_forward_step(400)
            continue

        # อ่านพิกัดสำเร็จ
        not_found_start = None
        cur_x, cur_y = pos
        
        dx = TARGET_CHEST[0] - cur_x
        dy = TARGET_CHEST[1] - cur_y
        dist = math.sqrt(dx**2 + dy**2)

        print("\n" + "=" * 50)
        print(f"[✓] อ่านพิกัดสำเร็จ: ({cur_x}, {cur_y})")
        print(f"[✓] ระยะห่างถึงกล่อง: {dist:.1f} หน่วย (dx={dx}, dy={dy})")
        print(f"[⏸] หยุดนิ่ง {STOP_DURATION} วินาที เพื่อคำนวณทิศทาง...")
        print("=" * 50)

        # สั่งหยุดรอ 15 วินาที
        time.sleep(STOP_DURATION)

        # ถ้าระยะห่างน้อยกว่า 3 หน่วย ถึงหน้ากล่องแล้ว
        if dist <= 3.0:
            print("[✓] ถึงตำแหน่งหน้ากล่องเรียบร้อยแล้ว!")
            time.sleep(0.5)
            tap(820, 430)
            break

        print(f"[->] กำลังเดินเข้าหากล่อง...")
        walk_by_direction(dx, dy, duration_sec=1.6)

if __name__ == "__main__":
    main()
