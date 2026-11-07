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

def swipe(x1, y1, x2, y2, duration=350):
    # ปรับ duration 350ms เพื่อให้ระบบเกมรับรู้ว่าเป็น gesture ลากหมุนกล้องจริง
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

# โซนรูดหมุนกล้อง (ตามรอยทัชในคลิป X: 750 ถึง 1000, Y: 280)
CAM_Y = 280

def rotate_camera(direction="right", dist=240):
    """รูดหมุนมุมกล้องรอบตัวแบบเดียวกับนิ้วผู้เล่นในวิดีโอ"""
    if direction == "right":
        # รูดจากขวาไปซ้าย -> กล้องหันไปทางขวา
        swipe(920, CAM_Y, 920 - dist, CAM_Y, duration=320)
    else:
        # รูดจากซ้ายไปขวา -> กล้องหันไปทางซ้าย
        swipe(720, CAM_Y, 720 + dist, CAM_Y, duration=320)
    time.sleep(0.4)

# พิกัดจอยสติ๊กเดินแนวนอน 1280x720
JOY_CENTER_X = 250
JOY_CENTER_Y = 530
JOY_RADIUS   = 140

def walk_straight(duration_ms=650):
    """ดันจอยสติ๊กตรงไปข้างหน้า (0 องศา) เดินเข้าชนกล่องตรงๆ"""
    target_x = JOY_CENTER_X
    target_y = JOY_CENTER_Y - JOY_RADIUS
    swipe(JOY_CENTER_X, JOY_CENTER_Y, target_x, target_y, duration=duration_ms)
    time.sleep(0.2)

def detect_chest(screen):
    """
    ตรวจจับกล่องสมบัติทั้งตอนเช้าและตอนค่ำ
    คืนค่า: (พบกล่องไหม, กึ่งกลาง X ของกล่อง, ขนาดพื้นที่กล่อง)
    """
    if screen is None:
        return False, 0, 0

    # ตัดขอบจอส่วนล่างที่มีจอยสติ๊กและปุ่มสกิลออก
    roi = screen[60:550, 100:1180]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 1. ช่วงสีส้มไม้ตอนเช้า
    mask_day = cv2.inRange(hsv, np.array([6, 70, 70]), np.array([24, 255, 255]))
    # 2. ช่วงสีน้ำตาลเข้มตอนค่ำ (เหมือนในคลิป)
    mask_night = cv2.inRange(hsv, np.array([4, 25, 25]), np.array([28, 180, 140]))

    mask = cv2.bitwise_or(mask_day, mask_night)
    kernel = np.ones((5, 5), np.uint8)
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_x = 0
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 800:
            x, y, cw, ch = cv2.boundingRect(cnt)
            ratio = float(cw) / ch
            if 0.55 <= ratio <= 1.8:
                if area > max_area:
                    max_area = area
                    best_x = x + 100 + (cw // 2)

    if max_area > 0:
        return True, best_x, max_area

    return False, 0, 0

def check_interact_button(screen):
    """ตรวจว่าปุ่มรูปมือเก็บของเด้งขึ้นมาหรือยัง (พิกัด X: 1010-1080, Y: 360-430)"""
    # โซนปุ่มรูปมือเหนือปุ่มดาบ
    hand_roi = screen[350:440, 1000:1090]
    gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
    # ปุ่มมือมีสีขาวสว่างชัดเจน
    white_pixels = np.sum(gray > 200)
    return white_pixels > 250

def main():
    print("=" * 60)
    print("[*] UtopiaFarmAI: ค้นหากล่อง -> หมุนมาตรงหน้า -> เดินตรงเข้าชน")
    print("=" * 60)
    time.sleep(2)

    SCREEN_CENTER = 640  # กึ่งกลางหน้าจอ 1280
    TOLERANCE = 80       # ขอบเขตตรงกลาง (560 ถึง 720)

    while True:
        scr = get_screen()
        if scr is None:
            continue

        # ตรวจสอบก่อนว่าเดินชนกล่องจนปุ่มรูปมือเด้งขึ้นมาหรือยัง
        if check_interact_button(scr):
            print("\n" + "=" * 50)
            print("[🎯] ตรวจพบปุ่มรูปมือเปิดกล่องแล้ว! แตะเปิดกล่อง...")
            print("=" * 50)
            tap(1040, 390)
            time.sleep(0.4)
            tap(1040, 390)
            break

        found, chest_x, area = detect_chest(scr)

        if found:
            diff = chest_x - SCREEN_CENTER
            print(f"[👁] เจอกล่องในจอ: พิกัด X={chest_x} (ขนาดวัตถุ: {area})")

            # ถ้ากล่องยังไม่ตรงกลางจอ ให้รูดหมุนหน้าจอจัดกึ่งกลาง
            if chest_x < (SCREEN_CENTER - TOLERANCE):
                print("[<] กล่องอยู่ทางซ้าย -> หมุนกล้องซ้ายเพื่อจัดให้อยู่ตรงหน้า...")
                rotate_camera("left", dist=130)
            elif chest_x > (SCREEN_CENTER + TOLERANCE):
                print("[>] กล่องอยู่ทางขวา -> หมุนกล้องขวาเพื่อจัดให้อยู่ตรงหน้า...")
                rotate_camera("right", dist=130)
            else:
                # กล่องอยู่ตรงหน้าแล้ว -> ดันจอยสติ๊กตรงไปข้างหน้า (0 องศา) เดินเข้าหาเลย
                print("[↑] กล่องอยู่ตรงหน้าพอดีแล้ว! เดินตรงไปข้างหน้า (0°)...")
                walk_straight(duration_ms=550)
        else:
            # ถ้ายังไม่เจอกล่อง ให้หมุนกล้องกวาดค้นหา 360 องศา
            print("[🔄] ยังไม่เจอกล่องในมุมมอง -> หมุนกล้องค้นหารอบตัว...")
            rotate_camera("right", dist=220)

if __name__ == "__main__":
    main()
