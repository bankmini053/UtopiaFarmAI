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
    """โฟกัสตัดเฉพาะกรอบพิกัด รองรับทั้งแนวนอนและแนวตั้ง"""
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

def walk_by_angle(angle_deg, duration_ms=400):
    """ดันจอยสติ๊กตามมุมองศาบนหน้าจอ (0=ดันขึ้น, 90=ดันขวา, 180=ดันลง, 270=ดันซ้าย)"""
    angle_rad = math.radians(angle_deg)
    target_x = int(JOY_CENTER_X + JOY_RADIUS * math.sin(angle_rad))
    target_y = int(JOY_CENTER_Y - JOY_RADIUS * math.cos(angle_rad))
    swipe(JOY_CENTER_X, JOY_CENTER_Y, target_x, target_y, duration=duration_ms)
    time.sleep(0.15)

def wait_and_settle_coords(max_wait=3.5):
    """วนอ่านพิกัดจนกว่าจะนิ่งตรงกัน 2 รอบติด ป้องกันภาพเบลอระหว่างขยับ"""
    start_t = time.time()
    last_val = None

    while time.time() - start_t < max_wait:
        scr = get_screen()
        curr_val = extract_coords(scr)
        
        if curr_val is not None:
            if curr_val == last_val:
                return curr_val
            last_val = curr_val
            
        time.sleep(0.25)

    return last_val

TARGET_CHEST = (17527, 15336)
TIMEOUT_LIMIT = 10
DIRECTIONS = [0, 45, 90, 135, 180, 225, 270, 315]

def main():
    print("=" * 60)
    print(f"[*] UtopiaFarmAI: โฟกัสพิกัด -> คำนวณมุมกล้องจริง -> เดินเข้าหากล่อง")
    print(f"[*] พิกัดเป้าหมายกล่อง: {TARGET_CHEST}")
    print("=" * 60)
    time.sleep(2)

    not_found_start = None
    dir_index = 0
    camera_angle_offset = None

    while True:
        # 1. โฟกัสอ่านตำแหน่งพิกัดปัจจุบันให้นิ่งสนิท
        pos1 = wait_and_settle_coords(max_wait=3.5)

        if pos1 is None:
            if not_found_start is None:
                not_found_start = time.time()

            elapsed = int(time.time() - not_found_start)
            current_dir = DIRECTIONS[dir_index % len(DIRECTIONS)]
            dir_index += 1
            print(f"[?] กำลังจับโฟกัสพิกัด... ({elapsed}/{TIMEOUT_LIMIT} วิ) -> ก้าวสั้นมุม {current_dir}°")

            if elapsed >= TIMEOUT_LIMIT:
                print(f"\n[!] ไม่พบพิกัดต่อเนื่องครบ {TIMEOUT_LIMIT} วิ -> หยุดอัตโนมัติ")
                break

            walk_by_angle(current_dir, duration_ms=250)
            continue

        not_found_start = None
        cur_x, cur_y = pos1

        dx = TARGET_CHEST[0] - cur_x
        dy = TARGET_CHEST[1] - cur_y
        dist = math.sqrt(dx**2 + dy**2)

        print("-" * 55)
        print(f"[✓] โฟกัสพิกัดนิ่ง: ({cur_x}, {cur_y}) | เป้าหมาย: {TARGET_CHEST}")
        print(f"[✓] ระยะห่างถึงกล่อง: {dist:.2f} หน่วย (dx={dx:+d}, dy={dy:+d})")

        # ตรวจสอบว่าถึงหน้ากล่องแล้วหรือยัง
        if dist <= 1.5 or (abs(dx) <= 1 and abs(dy) <= 1):
            print("\n" + "=" * 50)
            print(f"[✓] ถึงตำแหน่งหน้ากล่องแล้ว! ({cur_x}, {cur_y})")
            print("[✓] แตะปุ่มรูปมือเปิดกล่อง...")
            print("=" * 50)
            time.sleep(0.4)
            tap(820, 430)
            break

        # 2. สอบเทียบมุมกล้องจริง
        if camera_angle_offset is None:
            print("[*] กำลังทดสอบทิศทางหน้าจอเพื่อวัดมุมกล้องจริง...")
            walk_by_angle(0, duration_ms=450)
            pos2 = wait_and_settle_coords(max_wait=3.0)
            
            if pos2 is not None and pos2 != pos1:
                step_dx = pos2[0] - pos1[0]
                step_dy = pos2[1] - pos1[1]
                camera_angle_offset = math.atan2(step_dx, step_dy)
                print(f"[✓] สอบเทียบมุมกล้องสำเร็จ! (ก้าวทดสอบ: dx={step_dx:+d}, dy={step_dy:+d})")
                cur_x, cur_y = pos2
                dx = TARGET_CHEST[0] - cur_x
                dy = TARGET_CHEST[1] - cur_y
                dist = math.sqrt(dx**2 + dy**2)
            else:
                print("[!] ก้าวทดสอบไม่เห็นผล ขยับมุมรอบตัวเพื่อหาพื้นที่ว่าง...")
                walk_by_angle(90, duration_ms=350)
                continue

        # 3. คำนวณมุมจอยสติ๊กโดยหักลบมุมกล้อง
        target_world_angle = math.atan2(dx, dy)
        joystick_angle_rad = target_world_angle - camera_angle_offset
        joystick_angle_deg = (math.degrees(joystick_angle_rad) + 360) % 360

        # 4. คำนวณระยะก้าวเดิน
        if dist > 80:
            step_duration = 3.5
        elif dist > 20:
            step_duration = 1.6
        elif dist > 6:
            step_duration = 0.8
        elif dist > 2.5:
            step_duration = 0.45
        else:
            step_duration = 0.22

        print(f"[->] ดันจอยมุม {joystick_angle_deg:.0f}° เป็นเวลา {step_duration:.2f} วิ...")
        walk_by_angle(joystick_angle_deg, duration_ms=int(step_duration * 1000))

        # 5. ตรวจสอบผลลัพธ์หลังเดิน
        after_pos = wait_and_settle_coords(max_wait=3.5)
        if after_pos is not None:
            new_dx = TARGET_CHEST[0] - after_pos[0]
            new_dy = TARGET_CHEST[1] - after_pos[1]
            new_dist = math.sqrt(new_dx**2 + new_dy**2)
            if new_dist > dist + 1.0:
                print("[!] มุมกล้องหน้าจอเปลี่ยนไปจากเดิม -> ทำการรีเซ็ตเพื่อสอบเทียบมุมกล้องใหม่")
                camera_angle_offset = None

if __name__ == "__main__":
    main()
