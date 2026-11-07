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

def swipe_screen(x1, y1, x2, y2, duration=350):
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

def rotate_view(direction="right", dist=200):
    cx, cy = 900, 300
    if direction == "right":
        swipe_screen(cx, cy, cx - dist, cy, duration=280)
    else:
        swipe_screen(cx - dist, cy, cx, cy, duration=280)
    time.sleep(0.35)

# พิกัดจอยสติ๊กเดิน
JOY_CENTER_X = 250
JOY_CENTER_Y = 530
JOY_RADIUS   = 145

def walk_straight(duration_ms=650):
    """ดันจอยสติ๊กตรงไปข้างหน้า (0 องศา) เดินเข้าชนกล่อง"""
    target_x = JOY_CENTER_X
    target_y = JOY_CENTER_Y - JOY_RADIUS
    swipe_screen(JOY_CENTER_X, JOY_CENTER_Y, target_x, target_y, duration=duration_ms)
    time.sleep(0.2)

def detect_chest(screen):
    if screen is None:
        return False, 0, 0
    roi = screen[60:550, 100:1180]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([5, 60, 60]), np.array([25, 255, 255])),
        cv2.inRange(hsv, np.array([4, 20, 20]), np.array([28, 180, 140]))
    )
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_x, max_a = 0, 0
    for cnt in contours:
        a = cv2.contourArea(cnt)
        if a > 600:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if 0.5 <= float(cw)/ch <= 2.0 and a > max_a:
                max_a = a
                best_x = x + 100 + (cw // 2)
    return (True, best_x, max_a) if max_a > 0 else (False, 0, 0)

def main():
    print("=" * 60)
    print("[*] UtopiaFarmAI: เจอกล่อง -> หันจอกลาง -> เดินตรงชนกล่อง -> แตะเปิด")
    print("=" * 60)
    time.sleep(2)

    CENTER_X = 640
    TOLERANCE = 80  # ขอบเขตตรงกลางจอ (560 ถึง 720)

    while True:
        scr = get_screen()
        found, chest_x, area = detect_chest(scr)

        if found:
            diff = chest_x - CENTER_X
            print(f"[👁] ตรวจพบกล่องที่ X={chest_x} (ขนาด: {area:.1f}, ห่างกลางจอ: {diff:+d})")

            # หากกล่องยังไม่ตรงกลางจอ ให้หันมุมกล้องจัดกึ่งกลาง
            if chest_x < (CENTER_X - TOLERANCE):
                print("[<] กล่องอยู่ซ้าย -> หมุนกล้องซ้าย...")
                rotate_view("left", dist=110)
            elif chest_x > (CENTER_X + TOLERANCE):
                print("[>] กล่องอยู่ขวา -> หมุนกล้องขวา...")
                rotate_view("right", dist=110)
            else:
                # กล่องอยู่ตรงกลางจอพอดี
                print("\n" + "=" * 50)
                print(f"[🎯] กล่องอยู่ตรงหน้าแล้ว! (X={chest_x})")
                print("[↑] เดินตรงไปข้างหน้า (0°) มุ่งเข้าชนกล่อง...")
                print("=" * 50)

                # เดินตรงเข้าไปชนกล่อง 3-4 ก้าวสั้นๆ
                for i in range(3):
                    print(f"[->] ก้าวตรงเข้าชนกล่อง ({i+1}/3)...")
                    walk_straight(duration_ms=600)
                    time.sleep(0.15)

                # แตะปุ่มรูปมือเปิดกล่อง (พิกัดปุ่มมือเหนือปุ่มดาบ X: 1045, Y: 390)
                print("[✓] ถึงตัวกล่องแล้ว! กำลังแตะปุ่มรูปมือเปิดกล่อง...")
                time.sleep(0.4)
                tap(1045, 390)
                time.sleep(0.3)
                tap(1045, 390)
                print("[✓] ภารกิจเปิดกล่องเสร็จสมบูรณ์!")
                break
        else:
            print("[🔄] ยังไม่เจอกล่อง -> กวาดหน้าจอหา...")
            rotate_view("right", dist=200)

if __name__ == "__main__":
    main()
