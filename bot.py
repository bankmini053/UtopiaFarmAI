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
    เจาะจงตัดเฉพาะแถวตัวเลขพิกัดใต้ชื่อเกาะ (Y: 285-330, X: 110-260)
    และกรองเฉพาะตัวอักษรสีขาวสว่าง
    """
    if screen is None:
        return None

    # ตัดเฉพาะบรรทัดตัวเลขพิกัด (เลี่ยงชื่อ 1st Archipelago ด้านบน)
    coord_roi = screen[285:330, 110:260]

    # กรองเฉพาะเนื้อสีขาวของตัวหนังสือด้วย HSV
    hsv = cv2.cvtColor(coord_roi, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 190])
    upper_white = np.array([180, 50, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # ขยายภาพ 3 เท่าเพื่อให้เส้นตัวเลขหนาชัดเจนขึ้น
    enlarged = cv2.resize(mask, (0, 0), fx=3, fy=3, interpolation=cv2.INTER_NEAREST)

    # ตั้งค่า Tesseract ให้อ่านเฉพาะตัวเลขและจุด/คอมม่า
    custom_config = r"--psm 7 -c tessedit_char_whitelist=0123456789.,"
    text = pytesseract.image_to_string(enlarged, config=custom_config)

    # สกัดตัวเลขชุดละ 4-6 หลัก (เช่น 17514 และ 15323 หรือ 1000)
    nums = re.findall(r"\d{3,6}", text)
    if len(nums) >= 2:
        try:
            return (int(nums[0]), int(nums[1]))
        except ValueError:
            return None
    return None

# พิกัดกึ่งกลางปุ่มจอยสติ๊กสำหรับจอ 1280x720
JOY_CENTER_X = 195
JOY_CENTER_Y = 520
JOY_RADIUS   = 160

# พิกัดเป้าหมาย (ตำแหน่งกล่อง)
TARGET_POS = (17529, 15339)

def walk_step(dx, dy, duration_sec=1.2):
    """ขยับเดินตามเวกเตอร์สั้นๆ"""
    angle_rad = math.atan2(dx, dy)
    target_x = int(JOY_CENTER_X + JOY_RADIUS * math.sin(angle_rad))
    target_y = int(JOY_CENTER_Y - JOY_RADIUS * math.cos(angle_rad))
    duration_ms = int(duration_sec * 1000)
    
    swipe(JOY_CENTER_X, JOY_CENTER_Y, target_x, target_y, duration=duration_ms)
    time.sleep(0.3)

def main():
    print("=" * 55)
    print(f"[*] UtopiaFarmAI: ระบบอ่านพิกัด + เดินเข้าหากล่อง {TARGET_POS}")
    print("=" * 55)
    time.sleep(2)

    last_dist = None

    while True:
        screen = get_screen()
        if screen is None:
            time.sleep(0.3)
            continue

        pos = get_current_coordinates(screen)
        if pos is None:
            print("[?] กำลังประมวลผลตัวเลขพิกัด...")
            time.sleep(0.4)
            continue

        cur_x, cur_y = pos
        dx = TARGET_POS[0] - cur_x
        dy = TARGET_POS[1] - cur_y
        dist = math.sqrt(dx**2 + dy**2)

        direction_status = ""
        if last_dist is not None:
            if dist < last_dist:
                direction_status = "[✓ ถูกทาง]"
            elif dist > last_dist:
                direction_status = "[! กำลังเดินออกห่าง]"
            else:
                direction_status = "[= อยู่จุดเดิม]"

        print(f"[*] พิกัด: ({cur_x}, {cur_y}) | เหลือระยะ: {dist:.1f} {direction_status}")
        last_dist = dist

        # ถ้าระยะห่างน้อยกว่า 3 หน่วย ถึงจุดหมาย
        if dist <= 3.0:
            print("[✓] ถึงตำแหน่งกล่องเรียบร้อยแล้ว!")
            break

        # เดินเข้าหาพิกัด
        walk_step(dx, dy, duration_sec=1.2)

if __name__ == "__main__":
    main()
