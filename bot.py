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
    ตัดเฉพาะแถบพิกัดใต้ชื่อเกาะ No.6Archipelago ให้ตรงกับตำแหน่งจริง
    """
    if screen is None:
        return None

    h, w = screen.shape[:2]
    
    # กรณีรูปแคปมาในแนวตั้ง 720x1280
    if h > w:
        roi = screen[110:180, 420:530]
        roi = cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        roi = screen[260:340, 110:270]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY)
    resized = cv2.resize(thresh, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_LINEAR)

    custom_cfg = r"--psm 6 -c tessedit_char_whitelist=0123456789,"
    raw_text = pytesseract.image_to_string(resized, config=custom_cfg)

    matches = re.findall(r"\d{5}", raw_text)
    if len(matches) >= 2:
        try:
            x, y = int(matches[0]), int(matches[1])
            if 17400 <= x <= 17600 and 15200 <= y <= 15400:
                return (x, y)
        except ValueError:
            return None

    parts = re.split(r"[, \n]+", raw_text.strip())
    valid_nums = [int(p) for p in parts if p.isdigit() and len(p) >= 4]
    if len(valid_nums) >= 2:
        x, y = valid_nums[0], valid_nums[1]
        if 17400 <= x <= 17600 and 15200 <= y <= 15400:
            return (x, y)

    return None

# พิกัดจอยสติ๊กสำหรับจอ 1280x720
JOY_CENTER_X = 250
JOY_CENTER_Y = 530
JOY_RADIUS   = 140

def walk_by_direction(dx, dy, duration_sec=1.5):
    """คำนวณเวกเตอร์และเดินตามทิศทาง"""
    angle_rad = math.atan2(dx, dy)
    target_x = int(JOY_CENTER_X + JOY_RADIUS * math.sin(angle_rad))
    target_y = int(JOY_CENTER_Y - JOY_RADIUS * math.cos(
