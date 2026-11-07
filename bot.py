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

def swipe(x1, y1, x2, y2, duration=250):
    subprocess.run(f"{ROOT_SU} -c 'input swipe {x1} {y1} {x2} {y2} {duration}'", shell=True)

def get_screen():
    subprocess.run(f"{ROOT_SU} -c 'screencap -p {TMP_PATH}'", shell=True)
    if os.path.exists(TMP_PATH):
        return cv2.imread(TMP_PATH)
    return None

def get_current_coordinates(screen):
    """
    ตัดเฉพาะกรอบตัวหนังสือใต้แผนที่มินิแมป แล้วใช้ OCR อ่านค่าตัวเลขพิกัด (X, Y)
    """
    if screen is None:
        return None

    # ตัดกรอบใต้แผนที่ (Y: 265-335, X: 110-275) สำหรับจอ 1280x720
    coord_roi = screen[265:335, 110:275]

    # แปลงเป็นภาพขาว-ดำ และดึงเฉพาะสีขาวของตัวเลขให้อ่านง่าย
    gray = cv2.cvtColor(coord_roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    # ขยายขนาดภาพ 2 เท่าเพื่อให้ OCR อ่านตัวเลขแม่นยำขึ้น
    resized = cv2.resize(thresh, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # อ่านข้อความด้วย Tesseract
    config = "--psm 6"
    text = pytesseract.image_to_string(resized, config=config)

    # ค้นหาตัวเลขพิกัด เช่น 17514,15323 หรือ 17514.15323
    nums = re.findall(r"\d{3,6}", text)
    if len(nums) >= 2:
        try:
            cur_x = int(nums[0])
            cur_y = int(nums[1])
            return (cur_x, cur_y)
        except ValueError:
            return None
    return None

# พิกัดกึ่งกลางปุ่มจอยสติ๊กสำหรับจอ 1280x720
JOY_CENTER_X = 195
JOY_CENTER_Y = 520
JOY_RADIUS   = 160

# กำหนดพิกัดเป้าหมาย (ตำแหน่งกล่อง)
TARGET_POS = (17529, 15339)

def walk_step(dx, dy, duration_sec=1.5):
    """ขยับเดินตามทิศทางเวกเตอร์ช่วงสั้นๆ"""
    angle_rad = math.atan2(dx, dy)
    target_x = int(JOY_CENTER_X + JOY_RADIUS * math.sin(angle_rad))
    target_y = int(JOY_CENTER_Y - JOY_RADIUS * math.cos(angle_rad))
    duration_ms = int(duration_sec * 1000)
    
    swipe(JOY_CENTER_X, JOY_CENTER_Y, target_x, target_y, duration=duration_ms)
    time.sleep(0.3)

def main():
    print("=" * 55)
    print(f"[*] UtopiaFarmAI: ระบบอ่านพิกัดสดและเดินเข้าหาเป้าหมาย {TARGET_POS}")
    print("=" * 55)
    time.sleep(2)

    last_dist = None

    while True:
        screen = get_screen()
        if screen is None:
            time.sleep(0.3)
            continue

        # 1. อ่านพิกัดปัจจุบันจากใต้แผนที่
        pos = get_current_coordinates(screen)
        if pos is None:
            print("[?] ไม่สามารถอ่านตัวเลขพิกัดได้ -> ขยับเดินตรงไปข้างหน้าเบาๆ เพื่ออัปเดตหน้าจอ...")
            walk_step(0, 1, duration_sec=0.8)
            continue

        cur_x, cur_y = pos
        dx = TARGET_POS[0] - cur_x
        dy = TARGET_POS[1] - cur_y
        dist = math.sqrt(dx**2 + dy**2)

        # 2. ตรวจสอบระยะห่างว่าเข้าใกล้เป้าหมายหรือไม่
        status = ""
        if last_dist is not None:
            if dist < last_dist:
                status = "[✓ กำลังไปถูกทาง]"
            elif dist > last_dist:
                status = "[! กำลังเดินออกห่าง]"
            else:
                status = "[= อยู่ตำแหน่งเดิม]"

        print(f"[*] พิกัดปัจจุบัน: ({cur_x}, {cur_y}) | ระยะห่าง: {dist:.1f} หน่วย {status}")
        last_dist = dist

        # 3. ถ้าระยะห่างน้อยกว่า 3 หน่วย ถือว่าถึงจุดหมายแล้ว
        if dist <= 3.0:
            print("[✓] ถึงตำแหน่งเป้าหมายเรียบร้อยแล้ว!")
            break

        # 4. บังคับเดินเข้าหาพิกัดเป้าหมายทีละช่วง
        walk_step(dx, dy, duration_sec=1.2)

if __name__ == "__main__":
    main()
    
