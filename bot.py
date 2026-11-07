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
ICONS_DIR = os.path.expanduser("~/UtopiaFarmAI/icons")

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
    angle_rad = math.radians(angle_deg)
    target_x = int(JOY_CENTER_X + JOY_RADIUS * math.sin(angle_rad))
    target_y = int(JOY_CENTER_Y - JOY_RADIUS * math.cos(angle_rad))
    swipe(JOY_CENTER_X, JOY_CENTER_Y, target_x, target_y, duration=duration_ms)
    time.sleep(0.15)

def rotate_camera(direction="right", swipe_dist=280):
    """รูดนิ้วฝั่งขวาของหน้าจอเพื่อหมุนมุมกล้องรอบตัวละคร"""
    center_x, center_y = 950, 420
    if direction == "right":
        swipe(center_x, center_y, center_x - swipe_dist, center_y, duration=220)
    else:
        swipe(center_x, center_y, center_x + swipe_dist, center_y, duration=220)
    time.sleep(0.35)

def match_chest_template(screen):
    """
    ค้นหากล่องสมบัติบนหน้าจอโดยใช้รูปภาพต้นแบบในโฟลเดอร์ icons/ (Template Matching)
    รองรับทั้งรูปกล่องตอนเช้า (chest_day.png) และตอนค่ำ (chest_night.png)
    """
    if screen is None:
        return False, 0, 0.0

    h, w = screen.shape[:2]
    if h > w:
        screen = cv2.rotate(screen, cv2.ROTATE_90_CLOCKWISE)

    # โซนค้นหาตัดแถบปุ่มจอยสติ๊กด้านล่างออก
    search_roi = screen[60:600, 150:1150]
    gray_roi = cv2.cvtColor(search_roi, cv2.COLOR_BGR2GRAY)

    best_val = 0.0
    best_center_x = 0

    templates = [
        os.path.join(ICONS_DIR, "chest_day.png"),
        os.path.join(ICONS_DIR, "chest_night.png")
    ]

    for t_path in templates:
        if not os.path.exists(t_path):
            continue
        template = cv2.imread(t_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            continue

        # ค้นหาแบบหลายขนาด (Multi-scale Matching: 60% ถึง 140%) เพื่อรองรับระยะใกล้-ไกล
        for scale in [0.6, 0.8, 1.0, 1.2, 1.4]:
            tw = int(template.shape[1] * scale)
            th = int(template.shape[0] * scale)
            if tw >= gray_roi.shape[1] or th >= gray_roi.shape[0]:
                continue

            resized_t = cv2.resize(template, (tw, th), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(gray_roi, resized_t, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val > best_val:
                best_val = max_val
                best_center_x = max_loc[0] + 150 + (tw // 2)

    # ค่าความเหมือน (Confidence) ตั้งเกณฑ์ไว้ที่ >= 0.55
    if best_val >= 0.55:
        return True, best_center_x, best_val

    return False, 0, best_val

def wait_and_settle_coords(max_wait=3.5):
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
    print(f"[*] UtopiaFarmAI: ค้นหากล่องด้วย Template Matching ใน icons/")
    print(f"[*] พิกัดเป้าหมายกล่อง: {TARGET_CHEST}")
    print("=" * 60)
    time.sleep(2)

    not_found_start = None
    dir_index = 0
    camera_angle_offset = None

    while True:
        pos = wait_and_settle_coords(max_wait=3.5)

        if pos is None:
            if not_found_start is None:
                not_found_start = time.time()

            elapsed = int(time.time() - not_found_start)
            current_dir = DIRECTIONS[dir_index % len(DIRECTIONS)]
            dir_index += 1
            print(f"[?] สำรวจหาพิกัด... ({elapsed}/{TIMEOUT_LIMIT} วิ) -> ก้าวสั้น {current_dir}°")

            if elapsed >= TIMEOUT_LIMIT:
                print(f"\n[!] ไม่พบพิกัดต่อเนื่องครบ {TIMEOUT_LIMIT} วิ -> หยุดการทำงาน")
                break

            walk_by_angle(current_dir, duration_ms=250)
            continue

        not_found_start = None
        cur_x, cur_y = pos
        dx = TARGET_CHEST[0] - cur_x
        dy = TARGET_CHEST[1] - cur_y
        dist = math.sqrt(dx**2 + dy**2)

        print("-" * 55)
        print(f"[✓] พิกัดนิ่ง: ({cur_x}, {cur_y}) | ห่างเป้าหมาย: {dist:.2f} หน่วย (dx={dx:+d}, dy={dy:+d})")

        # -------------------------------------------------------------
        # โหมดเข้าประชิดหน้ากล่อง: เมื่อเข้าใกล้ระยะ <= 6 หน่วย
        # สแกนหากล่องจริงด้วย Template Matching ใน icons/
        # -------------------------------------------------------------
        if dist <= 6.0:
            print("\n" + "=" * 50)
            print("[🔍] เข้าสู่ระยะประชิด (<= 6 หน่วย) -> สแกนหากล่องจากไฟล์ใน icons/...")
            print("=" * 50)

            found_chest = False
            for step in range(8):  # หมุนสแกน 8 มุมรอบตัว 360 องศา
                scr = get_screen()
                is_seen, chest_cx, conf = match_chest_template(scr)

                if is_seen:
                    print(f"[🎯] พบกล่องสมบัติบนหน้าจอ! (กึ่งกลาง X: {chest_cx}, ความแม่นยำ: {conf:.2f})")
                    # ปรับกล้องให้ตัวกล่องอยู่กึ่งกลางจอ
                    if chest_cx < 540:
                        rotate_camera("left", swipe_dist=100)
                    elif chest_cx > 740:
                        rotate_camera("right", swipe_dist=100)

                    print("[->] ดันจอยสติ๊กตรงไปข้างหน้า (0°) มุ่งเข้าชนกล่อง 0.75 วิ...")
                    walk_by_angle(0, duration_ms=750)
                    time.sleep(0.5)

                    print("[✓] แตะปุ่มรูปมือเปิดกล่อง...")
                    tap(820, 430)
                    time.sleep(0.3)
                    tap(820, 430)
                    found_chest = True
                    break

                print(f"[*] หมุนกล้องฉากที่ {step + 1}/8 เพื่อเทียบหาภาพกล่อง...")
                rotate_camera("right", swipe_dist=240)

            if found_chest:
                print("\n[✓] ทำภารกิจเปิดกล่องเสร็จสิ้น!")
                break
            else:
                print("[!] ยังไม่พบภาพกล่องที่ตรงกัน -> สะกิดก้าวเข้าหาพิกัดเพิ่ม 1 สเต็ป")

        # -------------------------------------------------------------
        # โหมดเดินทางไกล: ยังอยู่ห่างเกิน 6 หน่วย นำทางด้วยพิกัด GPS
        # -------------------------------------------------------------
        if camera_angle_offset is None:
            print("[*] ทดสอบวัดมุมกล้องจริงสำหรับการเดินทางไกล...")
            walk_by_angle(0, duration_ms=450)
            pos2 = wait_and_settle_coords(max_wait=3.0)
            
            if pos2 is not None and pos2 != pos:
                step_dx = pos2[0] - pos[0]
                step_dy = pos2[1] - pos[1]
                camera_angle_offset = math.atan2(step_dx, step_dy)
                print(f"[✓] สอบเทียบมุมกล้องสำเร็จ: dx={step_dx:+d}, dy={step_dy:+d}")
            else:
                walk_by_angle(90, duration_ms=350)
                continue

        target_world_angle = math.atan2(dx, dy)
        joystick_angle_rad = target_world_angle - camera_angle_offset
        joystick_angle_deg = (math.degrees(joystick_angle_rad) + 360) % 360

        step_duration = 3.5 if dist > 80 else (1.6 if dist > 20 else 0.8)
        print(f"[->] เดินทางไกล: ดันจอยมุม {joystick_angle_deg:.0f}° นาน {step_duration} วิ...")
        walk_by_angle(joystick_angle_deg, duration_ms=int(step_duration * 1000))

if __name__ == "__main__":
    main()
