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
            return (int(nums[0]), int(nums[1]))
        except ValueError:
            return None
    return None

# พิกัดคันโยกเดินตรง
JOY_CENTER_X = 250
JOY_CENTER_Y = 530
JOY_UP_Y     = 380

def walk_forward_step(duration_ms=800):
    swipe(JOY_CENTER_X, JOY_CENTER_Y, JOY_CENTER_X, JOY_UP_Y, duration=duration_ms)
    time.sleep(0.3)

def turn_camera_right(pixels=140):
    print("[↻] หมุนมุมกล้องไปทางขวาเล็กน้อย...")
    swipe(950, 400, 950 - pixels, 400, duration=200)
    time.sleep(0.3)

TARGET_CHEST = (17529, 15339)
TIMEOUT_LIMIT = 60  # จำกัดเวลาไม่เจอพิกัด 60 วินาที (1 นาที)

def main():
    print("=" * 60)
    print(f"[*] UtopiaFarmAI: นำทางไปยัง {TARGET_CHEST}")
    print(f"[*] ระบบความปลอดภัย: หากหาพิกัดไม่เจอใน {TIMEOUT_LIMIT} วิ จะหยุดทำงานทันที")
    print("=" * 60)
    time.sleep(3)

    last_dist = None
    not_found_start_time = None  # ตัวแปรจับเวลาเริ่มต้นที่ไม่เจอพิกัด

    while True:
        scr = get_screen()
        pos = read_coordinates(scr)

        # กรณีอ่านพิกัดไม่เจอ
        if pos is None:
            # เริ่มจับเวลาตั้งแต่ครั้งแรกที่อ่านไม่เจอ
            if not_found_start_time is None:
                not_found_start_time = time.time()

            elapsed = int(time.time() - not_found_start_time)
            print(f"[?] ไม่พบพิกัด... ({elapsed}/{TIMEOUT_LIMIT} วินาที) ขยับตัวเพื่อหาแสงใหม่")

            # เช็กเงื่อนไข: ครบ 60 วินาทีแล้วหรือยัง
            if elapsed >= TIMEOUT_LIMIT:
                print("\n" + "!" * 55)
                print(f"[!] แจ้งเตือน: หาพิกัดไม่เจอติดต่อกันเกิน {TIMEOUT_LIMIT} วินาที")
                print("[!] ทำการหยุดการทำงานของบอททันทีเพื่อความปลอดภัย")
                print("!" * 55)
                break

            walk_forward_step(500)
            continue

        # หากอ่านพิกัดเจอ ให้รีเซ็ตตัวจับเวลา Timeout กลับเป็น None
        not_found_start_time = None

        cur_x, cur_y = pos
        dist = math.sqrt((TARGET_CHEST[0] - cur_x)**2 + (TARGET_CHEST[1] - cur_y)**2)
        print(f"[*] อ่านพิกัดสำเร็จ: ({cur_x}, {cur_y}) | เหลือระยะห่าง: {dist:.1f} หน่วย")

        if dist <= 3.0:
            print("[✓] ถึงตำแหน่งกล่องเรียบร้อยแล้ว!")
            time.sleep(0.5)
            tap(820, 430)
            break

        if last_dist is not None:
            if dist < last_dist:
                print("[✓ ถูกทาง] กำลังเข้าใกล้เป้าหมาย -> เดินหน้าต่อ")
                walk_forward_step(1000)
            else:
                print("[! ผิดทาง] ระยะห่างเพิ่มขึ้น -> หมุนกล้องแก้ทิศ")
                turn_camera_right(160)
                walk_forward_step(600)
        else:
            walk_forward_step(800)

        last_dist = dist

if __name__ == "__main__":
    main()
