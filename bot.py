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
    โฟกัสตัดเฉพาะกรอบพิกัด รองรับทั้งแนวนอนและแนวตั้ง
    """
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
    camera_angle_
