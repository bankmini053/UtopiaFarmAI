import os
import time
import subprocess
import cv2
import numpy as np

ROOT_SU = "/system/xbin/su"
TMP_PATH = "/data/local/tmp/screen.png"

def run_cmd(cmd):
    subprocess.run(f"{ROOT_SU} -c '{cmd}'", shell=True)

def tap(x, y):
    run_cmd(f"input tap {x} {y}")

def swipe_screen(x1, y1, x2, y2, duration=240):
    run_cmd(f"input swipe {x1} {y1} {x2} {y2} {duration}")

def get_screen():
    run_cmd(f"screencap -p {TMP_PATH}")
    if os.path.exists(TMP_PATH):
        img = cv2.imread(TMP_PATH)
        if img is not None:
            h, w = img.shape[:2]
            if h > w:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            return img
    return None

def rotate_view_right(dist=180):
    """กวาดสายตาไปทางขวาในโซนว่าง (X: 850, Y: 240)"""
    cx, cy = 850, 240
    swipe_screen(cx, cy, cx - dist, cy, duration=260)
    time.sleep(0.4)

JOY_CENTER_X, JOY_CENTER_Y, JOY_RADIUS = 250, 530, 145

def walk_forward(duration_ms=500):
    """ดันจอยตรงไปข้างหน้า (0 องศา) มุ่งเข้าหาต้นไม้"""
    swipe_screen(JOY_CENTER_X, JOY_CENTER_Y, JOY_CENTER_X, JOY_CENTER_Y - JOY_RADIUS, duration=duration_ms)
    time.sleep(0.15)

def detect_tree(screen):
    """
    กรองเฉดสีต้นไม้ (เฉดสีน้ำตาลลำต้นไม้ หรือทรงพุ่มไม้)
    HSV Range สำหรับเนื้อไม้สีน้ำตาล: H=10-25, S=50-200, V=40-160
    """
    if screen is None:
        return False, 0, 0
    
    # สนใจเฉพาะพื้นที่ครึ่งจอบนถึงกลางจอ ไม่รวมปุ่มควบคุม
    roi = screen[80:500, 150:1130]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # มาสก์สีน้ำตาลลำต้นไม้
    lower_brown = np.array([10, 50, 40])
    upper_brown = np.array([25, 200, 160])
    mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_x, max_area = 0, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1200:  # กรองเฉพาะวัตถุขนาดใหญ่พอเป็นต้นไม้
            x, y, w, h = cv2.boundingRect(cnt)
            # ลำต้นไม้จะมีลักษณะเป็นทรงแนวตั้ง (h > w หรืออัตราส่วนแนวตั้ง)
            if h > w * 0.8 and area > max_area:
                max_area = area
                best_x = x + 150 + (w // 2)
                
    return (True, best_x, max_area) if max_area > 0 else (False, 0, 0)

def check_interact_or_drop(screen):
    """ตรวจจับปุ่มรูปมือเก็บของ หรือไอเทมดรอปบนพื้น (X: 1000-1090, Y: 350-440)"""
    if screen is None:
        return False
    roi = screen[350:440, 1000:1090]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return np.sum(gray > 200) > 220

def main():
    print("=" * 60)
    print("[*] UtopiaFarmAI: กวาดหาต้นไม้ 360° -> เจอลำต้น -> เดินเข้าฟัน")
    print("=" * 60)
    time.sleep(2)

    while True:
        tree_found = False
        target_x = 0

        # 1. กวาดสายตาค้นหาต้นไม้รอบตัว 360 องศา (8 สเต็ป)
        print("\n[*] กวาดสายตาค้นหาต้นไม้รอบตัว...")
        for step in range(8):
            scr = get_screen()
            found, x_pos, area = detect_tree(scr)
            if found:
                print(f"[🌲] ตรวจพบต้นไม้ตรงกับเฉดสี! (X={x_pos}, ขนาด: {area:.1f})")
                tree_found = True
                target_x = x_pos
                break
            else:
                print(f"[🔄] สแกนมุม {step + 1}/8 -> หมุนขวาต่อ...")
                rotate_view_right(dist=190)

        if not tree_found:
            print("[!] ไม่พบต้นไม้รอบตัวในระยะสายตา -> ก้าวเดินสุ่มไปข้างหน้า...")
            walk_forward(duration_ms=1200)
            continue

        # 2. หยุดนิ่ง 2 วินาทีเพื่อให้ตำแหน่งนิ่ง
        print("[⏸] หยุดรอภาพนิ่ง 2 วินาที...")
        time.sleep(2.0)

        # 3. เดินตรงเข้าหาต้นไม้
        print("[↑] เดินตรงเข้าหาเป้าหมายต้นไม้...")
        for i in range(6):
            walk_forward(duration_ms=500)
            time.sleep(0.15)

        # 4. ฟันต้นไม้ (กดปุ่มโจมตี X=1100, Y=530 รัวๆ)
        print("[🪓] ถึงจุดหมายแล้ว -> เริ่มต้นฟันต้นไม้...")
        for hit in range(8):
            tap(1100, 530)
            time.sleep(0.4)

        # 5. เก็บของ (กดปุ่มรูปมือเปิด/เก็บ X=1045, Y=390)
        time.sleep(0.5)
        chk_scr = get_screen()
        if check_interact_or_drop(chk_scr):
            print("[✓] เก็บผลผลิตไม้/ไอเทมดรอป...")
            tap(1045, 390)
            time.sleep(0.3)
            tap(1045, 390)

        print("[✓] เสร็จสิ้น 1 รอบ -> เริ่มต้นสแกนหาต้นต่อไป...")
        time.sleep(1.0)

if __name__ == "__main__":
    main()
