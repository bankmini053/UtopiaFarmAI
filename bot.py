import os
import time
import subprocess
import cv2
import numpy as np

ROOT_SU = "/system/xbin/su"
TMP_PATH = "/data/local/tmp/screen.png"

def swipe(x1, y1, x2, y2, duration=200):
    subprocess.run(f"{ROOT_SU} -c 'input swipe {x1} {y1} {x2} {y2} {duration}'", shell=True)

def get_screen():
    subprocess.run(f"{ROOT_SU} -c 'screencap -p {TMP_PATH}'", shell=True)
    if os.path.exists(TMP_PATH):
        return cv2.imread(TMP_PATH)
    return None

def rotate_camera(direction="right", dist=180):
    """
    รูดหมุนมุมมองหน้าจอแนวนอน 1280x720:
    จุดรูดอยู่ที่ X: 850, Y: 320 (โซนหมุนมุมมองเกม)
    """
    cx, cy = 850, 320
    if direction == "right":
        # รูดจากขวาไปซ้าย -> กล้องจะหันขวา
        swipe(cx, cy, cx - dist, cy, duration=220)
    else:
        # รูดจากซ้ายไปขวา -> กล้องจะหันซ้าย
        swipe(cx - dist, cy, cx, cy, duration=220)
    time.sleep(0.4)

def detect_chest(screen):
    """
    ตรวจหากล่องและสีที่คล้ายกล่อง (เช้า/เย็น)
    คืนค่า: (เจอกล่องหรือไม่, พิกัด X ของกล่อง, ขนาดพื้นที่วัตถุ)
    """
    if screen is None:
        return False, 0, 0

    h, w = screen.shape[:2]
    # ปรับภาพเป็นแนวนอน 1280x720 หากได้ภาพแนวตั้ง
    if h > w:
        screen = cv2.rotate(screen, cv2.ROTATE_90_CLOCKWISE)

    # ตัดเฉพาะพื้นที่สายตามองเห็น (ไม่เอาแถบปุ่มด้านล่าง)
    roi = screen[50:560, 100:1180]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 1. ช่วงสีกล่องตอนเช้า (ส้มไม้/น้ำตาลสว่าง)
    mask_day = cv2.inRange(hsv, np.array([5, 80, 80]), np.array([25, 255, 255]))
    # 2. ช่วงสีกล่องตอนเย็น/ค่ำ (น้ำตาลเข้มหม่น)
    mask_night = cv2.inRange(hsv, np.array([5, 25, 25]), np.array([30, 180, 140]))

    mask = cv2.bitwise_or(mask_day, mask_night)
    kernel = np.ones((5, 5), np.uint8)
    clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 800:  # ขนาดก้อนวัตถุ
            x, y, cw, ch = cv2.boundingRect(cnt)
            ratio = float(cw) / ch
            if 0.5 <= ratio <= 1.8:  # สัดส่วนก้อนกล่อง
                center_x = x + 100 + (cw // 2)
                return True, center_x, area

    return False, 0, 0

def main():
    print("=" * 60)
    print("[*] UtopiaFarmAI: โหมดหมุนจัดกล่องให้อยู่กลางจออย่างเดียว")
    print("=" * 60)
    time.sleep(2)

    while True:
        scr = get_screen()
        found, chest_x, area = detect_chest(scr)

        if found:
            print(f"[👁] เจอกล่อง! พิกัด X: {chest_x} (ขนาด: {area})")

            # กลางจอแนวนอน 1280 คือ 640 (ยอมรับช่วง 560 - 720)
            if chest_x < 560:
                print("[<] กล่องอยู่ทางซ้าย -> หมุนกล้องไปทางซ้ายนิดหน่อย...")
                rotate_camera("left", dist=90)
            elif chest_x > 720:
                print("[>] กล่องอยู่ทางขวา -> หมุนกล้องไปทางขวานิดหน่อย...")
                rotate_camera("right", dist=90)
            else:
                print("\n" + "=" * 50)
                print(f"[🎯] กล่องอยู่ตรงกลางจอพอดีแล้ว! (X={chest_x})")
                print("=" * 50)
                time.sleep(1.0)
        else:
            print("[🔄] ยังไม่เจอกล่อง -> หมุนหน้าจอค้นหารอบตัว...")
            rotate_camera("right", dist=180)

if __name__ == "__main__":
    main()
