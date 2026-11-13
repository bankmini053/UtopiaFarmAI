import os
import re
import time
import subprocess
import cv2
import numpy as np
import pytesseract

ROOT_SU = "/system/xbin/su"
TMP_PATH = "/data/local/tmp/screen.png"
ICONS_DIR = os.path.expanduser("~/UtopiaFarmAI/icons")

# พิกัดปุ่มตกปลา (ไอคอนคันเบ็ดขวาล่าง)
FISH_BTN = (1075, 530)

def run_cmd(cmd):
    subprocess.run(f"{ROOT_SU} -c '{cmd}'", shell=True)

def tap(x, y):
    run_cmd(f"input tap {x} {y}")

def get_screen():
    run_cmd(f"screencap -p {TMP_PATH}")
    if os.path.exists(TMP_PATH):
        for _ in range(6):
            if os.path.getsize(TMP_PATH) > 50000:
                break
            time.sleep(0.03)

        img = cv2.imread(TMP_PATH)
        if img is not None:
            h, w = img.shape[:2]
            if h > w:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            return img
    return None

def check_minigame_active(screen):
    """
    ตรวจสอบว่าหลอดมินิเกมตกปลาด้านบนจอแสดงขึ้นมาหรือยัง
    โซนแถบเส้น: X=400 ถึง 880, Y=45 ถึง 95
    """
    if screen is None:
        return False, None
    bar_roi = screen[45:95, 400:880]
    gray = cv2.cvtColor(bar_roi, cv2.COLOR_BGR2GRAY)
    
    # ตรวจสอบเส้นสีขาวหรือวงกลมปลา
    white_pixels = np.sum(gray > 210)
    if white_pixels > 120:
        return True, bar_roi
    return False, None

def analyze_fish_and_bar(bar_roi):
    """
    วิเคราะห์ตำแหน่งของตัวปลา (Fish Icon) และเส้นขาว (Cursor/Bar)
    คืนค่า: (ควรดึงเบ็ดหรือไม่, ระยะห่าง diff_px)
    """
    if bar_roi is None:
        return False, 999

    hsv = cv2.cvtColor(bar_roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bar_roi, cv2.COLOR_BGR2GRAY)

    # 1. ค้นหาไอคอนวงกลมปลาสีทอง/เหลือง
    # HSV สีเหลือง/ทองของปลา
    lower_gold = np.array([15, 100, 150])
    upper_gold = np.array([35, 255, 255])
    mask_fish = cv2.inRange(hsv, lower_gold, upper_gold)
    
    fish_x = None
    fish_contours, _ = cv2.findContours(mask_fish, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if fish_contours:
        largest_cnt = max(fish_contours, key=cv2.contourArea)
        if cv2.contourArea(largest_cnt) > 25:
            M = cv2.moments(largest_cnt)
            if M["m00"] != 0:
                fish_x = int(M["m10"] / M["m00"])

    # 2. ค้นหาตำแหน่งเส้นสีขาวตัวชี้ (Cursor)
    lower_white = np.array([0, 0, 220])
    upper_white = np.array([180, 40, 255])
    mask_cursor = cv2.inRange(hsv, lower_white, upper_white)
    
    cursor_x = None
    cursor_contours, _ = cv2.findContours(mask_cursor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cursor_contours:
        # หาวัตถุสีขาวที่มีความแคบแนวตั้ง (เส้นชี้)
        for cnt in cursor_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h >= 12 and w <= 25:
                cursor_x = x + (w // 2)
                break

    # 3. คำนวณระยะห่างระหว่างเส้นชี้กับปลา
    if fish_x is not None and cursor_x is not None:
        diff = abs(fish_x - cursor_x)
        # ถ้าระยะห่างน้อยกว่า 35px แสดงว่าเส้นอยู่ใกล้/ทับตัวปลาพอดี -> กดดึงได้
        if diff <= 38:
            return True, diff
        return False, diff

    # ถ้าตรวจไม่เจอลวดลายสี ให้ใช้ตำแหน่งพิกเซลสว่างสุดแทน
    return False, 999

def check_fish_reward(screen):
    """
    ตรวจสอบชื่อปลาที่ได้รับบริเวณกลางจอขวา (ข้อความชื่อปลา +1)
    """
    if screen is None:
        return ""

    # โซนข้อความชื่อปลาเด้งขึ้นมา: X=620 ถึง 980, Y=450 ถึง 620
    roi = screen[450:620, 620:980]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    text = pytesseract.image_to_string(thresh, lang="eng", config=r"--psm 6").strip()
    return text

def main():
    print("=" * 60)
    print("[🎣] เริ่มต้นระบบบอทตกปลาอัตโนมัติ (Utopia: Origin)")
    print(f"[*] พิกัดปุ่มตกปลา: {FISH_BTN}")
    print("=" * 60)
    time.sleep(2)

    elfish_count = 0
    total_fish_count = 0
    cast_round = 0

    while True:
        cast_round += 1
        print(f"\n{'='*15} เริ่มต้นเหวี่ยงเบ็ดรอบที่ {cast_round} {'='*15}")
        
        # 1. กดปุ่มเหวี่ยงเบ็ด
        print("[*] กดปุ่มเริ่มตกปลา...")
        tap(FISH_BTN[0], FISH_BTN[1])
        time.sleep(1.5)

        # 2. รอปลากินเบ็ด (ตรวจจับหลอดมินิเกมสีขาวด้านบน)
        print("[⏳] กำลังรอปลากินเบ็ด...")
        minigame_started = False
        wait_start = time.time()

        while time.time() - wait_start < 20.0:  # รอปลากินเบ็ดไม่เกิน 20 วินาที
            scr = get_screen()
            is_active, bar_roi = check_minigame_active(scr)
            if is_active:
                minigame_started = True
                print("[🐟] ปลากินเบ็ดแล้ว! หลอดมินิเกมปรากฏ -> เริ่มเล่นจังหวะดึง...")
                break
            time.sleep(0.3)

        if not minigame_started:
            print("[⚠️] ปลารอบนี้ไม่กินเบ็ด หรือเหยื่อหมด -> ลองกดเหวี่ยงใหม่")
            continue

        # 3. ลูปมินิเกม: ตรวจจับเส้นสีขาวเข้าใกล้ตัวปลาแล้วกดย้ำ
        minigame_timer = time.time()
        while time.time() - minigame_timer < 12.0:  # มินิเกมนานประมาณ 5-8 วินาที
            scr = get_screen()
            is_active, bar_roi = check_minigame_active(scr)
            if not is_active:
                print("[✓] จบมินิเกมตกปลาแล้ว!")
                break

            should_pull, diff = analyze_fish_and_bar(bar_roi)
            if should_pull:
                print(f"[🎯] เส้นทับตัวปลาพอดี (ต่างกัน {diff}px) -> กดดึงเบ็ด!")
                tap(FISH_BTN[0], FISH_BTN[1])
                time.sleep(0.12)  # หน่วงสั้นๆ กันกดย้ำรัวเกินไป
            else:
                time.sleep(0.04)

        # 4. หยุดรอแอนิเมชันปลาขึ้นมา และตรวจเช็กชื่อปลา
        time.sleep(1.2)
        end_scr = get_screen()
        reward_text = check_fish_reward(end_scr)
        
        if reward_text:
            print(f"[📦] ได้รับไอเทม: '{reward_text}'")
            total_fish_count += 1
            
            # ตรวจสอบว่าใช่ Elfish หรือไม่
            if any(k in reward_text.lower() for k in ["elfish", "eife", "el"]):
                elfish_count += 1
                print(f"\n[🌟] ตกได้ Elfish สำเร็จ! (สะสมรวม: {elfish_count} ตัว)")
        else:
            print("[*] ตกปลาสำเร็จรอบนี้")

        print(f"[📊] สถิติ: ตกได้ทั้งหมด {total_fish_count} ตัว | Elfish: {elfish_count} ตัว")
        
        # รอนิ่ง 2 วินาทีก่อนเริ่มเหวี่ยงเบ็ดรอบถัดไป
        time.sleep(2.0)

if __name__ == "__main__":
    main()
