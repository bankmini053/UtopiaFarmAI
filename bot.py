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

def read_coordinates(screen):
    """
    อ่านพิกัด (X, Y) ใต้แผนที่มินิแมป
    กรอบตัวเลขบรรทัดล่าง (Y: 285-330, X: 110-260)
    """
    if screen is None:
        return None
    roi = screen[285:330, 110:260]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 50, 255]))
    resized = cv2.resize(mask, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    
    text = pytesseract.image_to_string(resized, config="--psm 7 -c tessedit_char_whitelist=0123456789.,")
    nums = re.findall(r"\d{4,6}", text)
    if len(nums) >= 2:
        try:
            return (int(nums[0]), int(nums[1]))
        except ValueError:
            return None
    return None

# พิกัดคันโยกเดินตรงไปข้างหน้า (ดันขึ้นตรง)
JOY_CENTER_X = 250
JOY_CENTER_Y = 530
JOY_UP_Y     = 380

def walk_forward_step(duration_ms=800):
    """เดินตรงไปข้างหน้า 1 ก้าวสั้นๆ"""
    swipe(JOY_CENTER_X, JOY_CENTER_Y, JOY_CENTER_X, JOY_UP_Y, duration=duration_ms)
    time.sleep(0.4)

def turn_camera_right(pixels=150):
    """ปัดหน้าจอบริเวณขวามือเพื่อหมุนมุมกล้องไปทางขวาเล็กน้อย"""
    print("[↻] ทิศทางไม่ตรง -> หมุนมุมกล้องไปทางขวาเล็กน้อย...")
    swipe(950, 400, 950 - pixels, 400, duration=200)
    time.sleep(0.4)

# พิกัดกล่องเป้าหมาย
TARGET_CHEST = (17529, 15339)

def main():
    print("=" * 60)
    print(f"[*] UtopiaFarmAI: ระบบเดินลองทิศทาง + หมุนกล้องอัตโนมัติหากล่อง {TARGET_CHEST}")
    print("=" * 60)
    time.sleep(3)

    last_dist = None

    while True:
        scr = get_screen()
        pos = read_coordinates(scr)

        # ถ้าอ่านพิกัดยังไม่ได้ ให้เดินขยับไปข้างหน้า 1 ก้าวเพื่อเปลี่ยนแสงฉากหลัง
        if pos is None:
            print("[?] กำลังค้นหาพิกัด... ขยับตัวเล็กน้อย")
            walk_forward_step(600)
            continue

        cur_x, cur_y = pos
        dist = math.sqrt((TARGET_CHEST[0] - cur_x)**2 + (TARGET_CHEST[1] - cur_y)**2)
        print(f"[*] พิกัดปัจจุบัน: ({cur_x}, {cur_y}) | ระยะห่างถึงกล่อง: {dist:.1f} หน่วย")

        # ถ้าระยะห่างเหลือไม่เกิน 3 หน่วย ถือว่าถึงหน้ากล่องแล้ว
        if dist <= 3.0:
            print("[✓] ถึงตำแหน่งหน้ากล่องเรียบร้อยแล้ว!")
            time.sleep(0.5)
            tap(820, 430)  # ปุ่มเปิดกล่อง
            break

        # ตรวจสอบทิศทางเทียบกับระยะเดิม
        if last_dist is not None:
            # ถ้าระยะห่างลดลง แสดงว่าหันหน้าถูกทางแล้ว
            if dist < last_dist:
                print("[✓] ถูกทางแล้ว! เดินหน้าต่อ...")
                walk_forward_step(1000)
            else:
                # ถ้าระยะห่างเพิ่มขึ้นหรือเท่าเดิม แสดงว่าเดินผิดทิศ
                print("[!] ผิดทาง (ระยะห่างเพิ่มขึ้น) -> ปรับหมุนมุมกล้อง")
                turn_camera_right(180)
                walk_forward_step(600)
        else:
            # ก้าวแรก: เดินทดสอบทิศทาง
            walk_forward_step(800)

        # บันทึกระยะห่างล่าสุดไว้เปรียบเทียบในรอบถัดไป
        last_dist = dist

if __name__ == "__main__":
    main()
        
