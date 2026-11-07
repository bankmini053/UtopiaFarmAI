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

# พิกัดกึ่งกลางปุ่มจอยสติ๊กสำหรับจอ 1280x720
JOY_CENTER_X = 250
JOY_CENTER_Y = 530
JOY_RADIUS   = 145

def walk_in_direction(angle_deg, duration_ms=300):
    angle_rad = math.radians(angle_deg)
    target_x = int(JOY_CENTER_X + JOY_RADIUS * math.sin(angle_rad))
    target_y = int(JOY_CENTER_Y - JOY_RADIUS * math.cos(angle_rad))
    swipe(JOY_CENTER_X, JOY_CENTER_Y, target_x, target_y, duration=duration_ms)
    time.sleep(0.15)

def move_fixed_grid(diff_x, diff_y, duration_sec):
    """
    แกนทิศทางตามมุมกล้องเกม:
    - เพิ่ม X (+X) -> ดัน ซ้าย (270°)
    - ลด X (-X)   -> ดัน ขวา (90°)
    - เพิ่ม Y (+Y) -> ดัน ถอยหลัง (180°)
    - ลด Y (-Y)   -> ดัน ไปข้างหน้า (0°)
    """
    # ล็อกเดินแกน X ก่อนจนกว่าจะตรงเป้าหมาย แล้วจึงเดินแกน Y
    if abs(diff_x) > 0:
        if diff_x > 0:
            angle = 270
            name = f"ซ้ายเพื่อเพิ่ม X ({diff_x:+d})"
        else:
            angle = 90
            name = f"ขวาเพื่อลด X ({diff_x:+d})"
    else:
        if diff_y > 0:
            angle = 180
            name = f"ถอยหลังเพื่อเพิ่ม Y ({diff_y:+d})"
        else:
            angle = 0
            name = f"ตรงไปข้างหน้าเพื่อลด Y ({diff_y:+d})"

    print(f"[->] ล็อกแกนเดิน: {name} เป็นเวลา {duration_sec:.2f} วิ")
    walk_in_direction(angle, duration_ms=int(duration_sec * 1000))

def wait_and_settle_coords(max_wait=4.0):
    """
    วนอ่านค่าคำนวณตลอดเวลาช่วงหยุดนิ่ง:
    เช็กจนกว่าพิกัดจะนิ่งสนิทตรงกัน 2 ครั้งติด แล้วคืนค่าทันที ไม่ต้องรอครบเวลา
    """
    start_t = time.time()
    last_val = None

    while time.time() - start_t < max_wait:
        scr = get_screen()
        curr_val = extract_coords(scr)
        
        if curr_val is not None:
            # ถ้าพิกัดนิ่งตรงกัน 2 รอบติด แสดงว่าตัวละครหยุดสนิทแล้ว
            if curr_val == last_val:
                return curr_val
            last_val = curr_val
            
        time.sleep(0.3)

    return last_val

TARGET_CHEST = (17527, 15331)
TIMEOUT_LIMIT = 10
DIRECTIONS = [0, 45, 90, 135, 180, 225, 270, 315]

def main():
    print("=" * 60)
    print(f"[*] UtopiaFarmAI: คำนวณตลอดเวลาช่วงหยุดนิ่ง + ชิดพิกัดที่สุด")
    print(f"[*] พิกัดเป้าหมายกล่อง: {TARGET_CHEST}")
    print("=" * 60)
    time.sleep(2)

    not_found_start = None
    dir_index = 0

    while True:
        # วนอ่านพิกัดอย่างต่อเนื่องช่วงหยุดนิ่ง (Active Polling)
        pos = wait_and_settle_coords(max_wait=3.5)

        # 1. กรณีหาพิกัดไม่เจอ
        if pos is None:
            if not_found_start is None:
                not_found_start = time.time()

            elapsed = int(time.time() - not_found_start)
            current_dir = DIRECTIONS[dir_index % len(DIRECTIONS)]
            dir_index += 1
            print(f"[?] สำรวจหาพิกัด... ({elapsed}/{TIMEOUT_LIMIT} วิ) -> ขยับมุม {current_dir}°")

            if elapsed >= TIMEOUT_LIMIT:
                print(f"\n[!] ไม่พบพิกัดต่อเนื่องครบ {TIMEOUT_LIMIT} วิ -> สั่งหยุดการทำงาน")
                break

            walk_in_direction(current_dir, duration_ms=250)
            continue

        # 2. อ่านพิกัดสำเร็จ
        not_found_start = None
        cur_x, cur_y = pos
        
        diff_x = TARGET_CHEST[0] - cur_x
        diff_y = TARGET_CHEST[1] - cur_y
        dist = math.sqrt(diff_x**2 + diff_y**2)

        print("-" * 55)
        print(f"[✓] พิกัดนิ่ง: ({cur_x}, {cur_y}) | เป้าหมาย: {TARGET_CHEST}")
        print(f"[✓] ส่วนต่าง: diff_X = {diff_x:+d}, diff_Y = {diff_y:+d} (ห่าง {dist:.2f} หน่วย)")

        # 3. เงื่อนไขถึงจุดหมาย: เข้าใกล้ที่สุด (ห่างไม่เกิน 1.0 หน่วย หรือตรงเป้าหมาย)
        if dist <= 1.0 or (diff_x == 0 and diff_y == 0):
            print("\n" + "=" * 50)
            print(f"[✓] ถึงจุดหมายใกล้ที่สุดแล้ว! ({cur_x}, {cur_y})")
            print("[✓] แตะปุ่มรูปมือเปิดกล่องทันที...")
            print("=" * 50)
            time.sleep(0.4)
            tap(820, 430)  # ปุ่มเปิดกล่อง
            break

        # 4. คำนวณระยะการเดิน
        target_diff = abs(diff_x) if abs(diff_x) > 0 else abs(diff_y)
        
        if target_diff > 80:
            step_duration = 3.5  # ระยะไกลมาก เดินยาว
        elif target_diff > 20:
            step_duration = 1.6  # ระยะปานกลาง
        elif target_diff > 5:
            step_duration = 0.6  # เริ่มใกล้
        elif target_diff > 2:
            step_duration = 0.35 # ใกล้มาก
        else:
            step_duration = 0.18 # ระยะประชิด 1-2 แต้ม สะกิดเบาๆ เพื่อให้เข้าใกล้ที่สุด

        # 5. เดินตามแกน
        move_fixed_grid(diff_x, diff_y, step_duration)

if __name__ == "__main__":
    main()
