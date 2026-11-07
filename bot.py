import os
import time
import subprocess
import cv2
import numpy as np

ROOT_SU = "/system/xbin/su"
TMP_PATH = "/data/local/tmp/screen.png"

def swipe(x1, y1, x2, y2, duration=220):
    subprocess.run(f"{ROOT_SU} -c 'input swipe {x1} {y1} {x2} {y2} {duration}'", shell=True)

def get_screen():
    subprocess.run(f"{ROOT_SU} -c 'screencap -p {TMP_PATH}'", shell=True)
    if os.path.exists(TMP_PATH):
        return cv2.imread(TMP_PATH)
    return None

def rotate_camera(direction="right", dist=160):
    """
    รูดหมุนหน้าจอแนวนอน 1280x720:
    ใช้ตำแหน่ง X: 900, Y: 320 ซึ่งเป็นพื้นที่ว่างฝั่งขวาสำหรับหมุนมุมมอง
    """
    cx, cy = 900, 320
    if direction == "right":
        # รูดจากขวาไปซ้าย -> หน้าจอหมุนไปทางขวา
        swipe(cx, cy, cx - dist, cy, duration=200)
    else:
        # รูดจากซ้ายไปขวา -> หน้าจอหมุนไปทางซ้าย
        swipe(cx - dist, cy, cx, cy, duration=200)
    time.sleep(0.35)

def detect_chest(screen):
    """
    ตรวจหากล่องสมบัติจากรูปทรงและเฉดสี (เช้า/เย็น)
    คืนค่า: (พบกล่องหรือไม่, พิกัด X ของกล่อง, ขนาดพื้นที่)
    """
    if screen is None:
        return False, 0, 0

    h, w = screen.shape[:2]
    # แปลงภาพเป็นแนวนอน 1280x720 หากได้มาเป็นแนวตั้ง
    if h > w:
        screen = cv2.rotate(screen, cv2.ROTATE_90_CLOCKWISE)

    # สนใจเฉพาะพื้นที่ครึ่งบนของหน้าจอ (ตัดแถบควบคุมล่างออก)
    roi = screen[60:550, 100:1180]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # สีกล่องตอนเช้า (ส้มไม้สว่าง)
    mask_day = cv2.inRange(hsv, np.array([5, 70, 70]), np.array([25, 255, 255]))
    # สีกล่องตอนเย็น/ค่ำ (น้ำตาลเข้มหม่น)
    mask_night = cv2.inRange(hsv, np.array([4, 20, 20]), np.array([30, 190, 140]))

    mask = cv2.bitwise_or(mask_day, mask_night)
    kernel = np.ones((5, 5), np.uint8)
    clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 700:  # ขนาดก้อนวัตถุ
            x, y, cw, ch = cv2.boundingRect(cnt)
            ratio = float(cw) / ch
            if 0.5 <= ratio <= 1.9:  # สัดส่วนรูปทรงกล่อง
                center_x = x + 100 + (cw // 2)
                return True, center_x, area

    return False, 0, 0

def main():
    print("=" * 60)
    print("[*] UtopiaFarmAI: โหมดหมุนหันจอหาและจัดกล่องกลางจอ (ไม่มีการเดิน)")
    print("=" * 60)
    time.sleep(2)

    while True:
        scr = get_screen()
        found, chest_x, area = detect_chest(scr)

        if found:
            print(f"[👁] ตรวจพบกล่อง! ตำแหน่งแกน X: {chest_x} (ขนาด: {area})")

            # กลางจอแนวนอน 1280 คือ 640 (ช่วงยอมรับ: 560 ถึง 720)
            if chest_x < 560:
                print("[<] กล่องอยู่ฝั่งซ้าย -> กำลังหันจอไปทางซ้าย...")
                rotate_camera("left", dist=90)
            elif chest_x > 720:
                print("[>] กล่องอยู่ฝั่งขวา -> กำลังหันจอไปทางขวา...")
                rotate_camera("right", dist=90)
            else:
                print(f"[🎯] กล่องอยู่ตรงกลางจอพอดีแล้ว! (X={chest_x}) -> ล็อกมุมมองนิ่ง")
                time.sleep(0.8)
        else:
            print("[🔄] ไม่เห็นกล่อง -> กำลังหมุนหน้าจอค้นหารอบตัว...")
            rotate_camera("right", dist=180)

if __name__ == "__main__":
    main()
