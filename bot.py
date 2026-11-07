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

def preprocess_for_ocr(crop_img):
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    contrast = clahe.apply(resized)
    _, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

def read_coordinates(screen):
    """
    ตัดเฉพาะแถบพิกัดใต้แผนที่มินิแมป (Y: 280-335, X: 110-275 บนจอ 1280x720)
    """
    if screen is None:
        return None

    coord_roi = screen[280:335, 110:275]
    processed_img = preprocess_for_ocr(coord_roi)

    custom_cfg = r"--psm 7 -c tessedit_char_whitelist=0123456789.,"
    text = pytesseract.image_to_string(processed_img, config=custom_cfg)

    nums = re.findall(r"\d{4,6}", text)
    if len(nums) >= 2:
        try:
            x, y = int(nums[0]), int(nums[1])
            # กรองค่าขยะ: พิกัดบริเวณเกาะนี้ X ต้องอยู่ช่วง 17000-18000 และ Y อยู่ช่วง 15000-16000
            if 17000 <= x <= 18000 and 15000 <= y <= 16000:
                return (x, y)
        except ValueError:
            return None
    return None

# พิกัดคันโยกเดิน
JOY_CENTER_X = 250
JOY_CENTER_Y = 530
JOY_RADIUS   = 140

def walk_by_direction(dx, dy, duration_sec=1.5):
    """เดินตามทิศทางเวกเตอร์ (dx, dy) ที่คำนวณได้"""
    angle_rad = math.atan2(dx, dy)
    target_x = int(JOY_CENTER_X + JOY_RADIUS * math.sin(angle_rad))
    target_y = int(JOY_CENTER_Y - JOY_RADIUS * math.cos(angle_rad))
    duration_ms = int(duration_sec * 1000)
    
    swipe(JOY_CENTER_X, JOY_CENTER_Y, target_x, target_y, duration=duration_ms)
    time.sleep(0.3)

def walk_forward_step(duration_ms=600):
    swipe(JOY_CENTER_X, JOY_CENTER_Y, JOY_CENTER_X, JOY_CENTER_Y - JOY_RADIUS, duration=duration_ms)
    time.sleep(0.3)

TARGET_CHEST = (17529, 15339)
TIMEOUT_LIMIT = 60

def main():
    print("=" * 60)
    print(f"[*] UtopiaFarmAI: ระบบอ่านพิกัด -> หยุด 5 วิ -> คำนวณทิศทางหากล่อง {TARGET_CHEST}")
    print("=" * 60)
    time.sleep(3)

    not_found_start = None

    while True:
        scr = get_screen()
        pos = read_coordinates(scr)

        # กรณีหาพิกัดไม่เจอ
        if pos is None:
            if not_found_start is None:
                not_found_start = time.time()

            elapsed = int(time.time() - not_found_start)
            print(f"[?] ไม่พบพิกัด... ({elapsed}/{TIMEOUT_LIMIT} วิ) ขยับก้าวสั้นๆ เพื่อเปลี่ยนแสง")

            if elapsed >= TIMEOUT_LIMIT:
                print(f"[!] ไม่พบพิกัดครบ {TIMEOUT_LIMIT} วินาที -> หยุดการทำงานเพื่อความปลอดภัย")
                break

            walk_forward_step(500)
            continue

        # รีเซ็ตตัวจับเวลาเมื่ออ่านพิกัดเจอ
        not_found_start = None
        cur_x, cur_y = pos
        
        dx = TARGET_CHEST[0] - cur_x
        dy = TARGET_CHEST[1] - cur_y
        dist = math.sqrt(dx**2 + dy**2)

        print("\n" + "=" * 50)
        print(f"[✓] ยืนยันพิกัดปัจจุบัน: ({cur_x}, {cur_y})")
        print(f"[✓] ระยะห่างถึงกล่อง: {dist:.1f} หน่วย")
        print("[⏸] สั่งหยุดนิ่ง 5 วินาทีเพื่อคำนวณเส้นทาง...")
        print("=" * 50)
        
        # สั่งหยุดนิ่ง 5 วินาที
        time.sleep(5)

        # ถ้าระยะห่างเหลือไม่เกิน 3 หน่วย ถือว่าถึงหน้ากล่องแล้ว
        if dist <= 3.0:
            print("[✓] ถึงตำแหน่งหน้ากล่องเรียบร้อยแล้ว!")
            time.sleep(0.5)
            tap(820, 430)  # ปุ่มเปิดกล่อง
            break

        # คำนวณทิศทางแล้วเดินตามเวกเตอร์ไปยังกล่อง
        print(f"[->] คำนวณเสร็จสิ้น: dx={dx}, dy={dy} -> เริ่มก้าวเดินเข้าหากล่อง...")
        walk_by_direction(dx, dy, duration_sec=1.8)

if __name__ == "__main__":
    main()
