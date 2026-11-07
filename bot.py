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

def run_cmd(cmd):
    subprocess.run(f"{ROOT_SU} -c '{cmd}'", shell=True)

def tap(x, y):
    run_cmd(f"input tap {x} {y}")

def swipe(x1, y1, x2, y2, duration=280):
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

def extract_coords(screen):
    if screen is None:
        return None

    h, w = screen.shape[:2]
    roi = screen[180:300, 40:270]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    custom_cfg = r"--psm 6 -c tessedit_char_whitelist=0123456789,."
    for th in [175, 195, 215]:
        _, thresh = cv2.threshold(resized, th, 255, cv2.THRESH_BINARY)
        matches = re.findall(r"\d{5}", pytesseract.image_to_string(thresh, config=custom_cfg))
        if len(matches) >= 2:
            try:
                x, y = int(matches[0]), int(matches[1])
                if 17300 <= x <= 17700 and 15200 <= y <= 15500:
                    return (x, y)
            except:
                pass
    return None

def wait_coords(max_wait=3.5):
    st = time.time()
    last = None
    while time.time() - st < max_wait:
        c = extract_coords(get_screen())
        if c is not None:
            if c == last:
                return c
            last = c
        time.sleep(0.25)
    return last

JOY_CENTER_X, JOY_CENTER_Y, JOY_RADIUS = 250, 530, 140

def walk_straight(duration_ms=500):
    """เดินตรงไปข้างหน้า (0 องศา) เข้าชนกล่อง"""
    swipe(JOY_CENTER_X, JOY_CENTER_Y, JOY_CENTER_X, JOY_CENTER_Y - JOY_RADIUS, duration=duration_ms)
    time.sleep(0.2)

def walk_by_angle(deg, duration_ms=400):
    rad = math.radians(deg)
    tx = int(JOY_CENTER_X + JOY_RADIUS * math.sin(rad))
    ty = int(JOY_CENTER_Y - JOY_RADIUS * math.cos(rad))
    swipe(JOY_CENTER_X, JOY_CENTER_Y, tx, ty, duration=duration_ms)
    time.sleep(0.15)

def rotate_camera_by_degree(deg):
    """
    รูดหมุนหน้าจอตามองศาที่คำนวณได้:
    บนจอ 1280x720: รูดประมาณ 8.5 pixels = หมุนกล้อง 1 องศา
    """
    cx, cy = 880, 280
    px = int(abs(deg) * 8.5)
    px = min(max(px, 50), 380)

    if deg > 0:
        # หันขวา: ลากจากขวาไปซ้าย
        swipe(cx, cy, cx - px, cy, duration=240)
    else:
        # หันซ้าย: ลากจากซ้ายไปขวา
        swipe(cx - px, cy, cx, cy, duration=240)
    time.sleep(0.35)

TARGET_CHEST = (17527, 15336)

def main():
    print("=" * 60)
    print("[*] UtopiaFarmAI: คำนวณพิกัดคน -> หมุนหน้าจอเข้าหากล่องตรงๆ")
    print(f"[*] พิกัดกล่องคงที่: {TARGET_CHEST}")
    print("=" * 60)
    time.sleep(2)

    # ขั้นที่ 1: วัดมุมกล้องปัจจุบันของคนเทียบกับทิศเหนือในแผนที่
    print("[*] ก้าวสั้น 1 จังหวะเพื่อตรวจมุมหันหน้าปัจจุบันของคน...")
    pos1 = wait_coords(max_wait=3.5)
    walk_straight(duration_ms=450)
    pos2 = wait_coords(max_wait=3.5)

    if pos1 is None or pos2 is None or pos1 == pos2:
        print("[!] อ่านพิกัดก้าวทดสอบไม่ติด สะกิดเดินใหม่...")
        walk_by_angle(90, duration_ms=350)
        pos2 = wait_coords(max_wait=3.0)

    step_dx = pos2[0] - pos1[0]
    step_dy = pos2[1] - pos1[1]
    # มุมที่คนกำลังหันหน้าไป
    current_facing_angle = math.atan2(step_dx, step_dy)
    print(f"[✓] รู้ทิศหันหน้าของคนแล้ว (เวกเตอร์ก้าว: dx={step_dx:+d}, dy={step_dy:+d})")

    while True:
        # อ่านพิกัดคนปัจจุบัน
        cur_pos = wait_coords(max_wait=3.0)
        if cur_pos is None:
            continue

        cur_x, cur_y = cur_pos
        dx = TARGET_CHEST[0] - cur_x
        dy = TARGET_CHEST[1] - cur_y
        dist = math.sqrt(dx**2 + dy**2)

        print("-" * 55)
        print(f"[✓] คนอยู่ที่: ({cur_x}, {cur_y}) | ห่างจากกล่อง: {dist:.2f} หน่วย (dx={dx:+d}, dy={dy:+d})")

        # -------------------------------------------------------------
        # คำนวณองศาที่ต้องหมุนหน้าจอให้หันหน้าตรงเข้าหากล่อง
        # -------------------------------------------------------------
        target_angle = math.atan2(dx, dy)
        turn_angle_rad = target_angle - current_facing_angle
        turn_angle_deg = (math.degrees(turn_angle_rad) + 180) % 360 - 180

        # ถ้าถึงระยะใกล้กล่องแล้ว (<= 3.0 หน่วย)
        if dist <= 3.0:
            print("\n" + "=" * 50)
            print(f"[🎯] อยู่ในระยะประชิดกล่องแล้ว! (ห่าง {dist:.2f} หน่วย)")
            
            # ถ้าหน้าจอยังไม่ตรงกล่อง ให้หมุนจอเข้าหากล่อง
            if abs(turn_angle_deg) > 15:
                print(f"[🔄] หมุนหน้าจอ {turn_angle_deg:+.1f}° เพื่อหันหน้าตรงเข้ากล่อง...")
                rotate_camera_by_degree(turn_angle_deg)
                current_facing_angle = target_angle
                time.sleep(0.3)

            # เดินตรงหน้า (0°) ชนกล่อง
            print("[↑] เดินตรงไปข้างหน้า (0°) เข้าชนกล่อง...")
            walk_straight(duration_ms=500)
            time.sleep(0.4)

            # แตะปุ่มรูปมือเปิดกล่อง (พิกัดเหนือปุ่มดาบ)
            print("[✓] แตะปุ่มรูปมือเปิดกล่อง (1050, 385)...")
            tap(1050, 385)
            time.sleep(0.3)
            tap(1050, 385)
            print("[✓] ภารกิจเปิดกล่องเสร็จสมบูรณ์!")
            break

        # ถ้ายังอยู่ห่างจากกล่อง (> 3.0 หน่วย): หันจอเข้าหากล่องแล้วเดินตรงหน้าเข้าไป
        if abs(turn_angle_deg) > 20:
            print(f"[🔄] ปรับหน้าจอให้หันไปทางกล่อง ({turn_angle_deg:+.1f}°)...")
            rotate_camera_by_degree(turn_angle_deg)
            current_facing_angle = target_angle

        # เดินตรงไปหากล่อง
        step_time = 2.0 if dist > 30 else (1.0 if dist > 10 else 0.5)
        print(f"[->] เดินตรงเข้าหากล่อง {step_time} วินาที...")
        walk_straight(duration_ms=int(step_time * 1000))

if __name__ == "__main__":
    main()
