import os
import time
import subprocess
import cv2
import numpy as np

ROOT_SU = "/system/xbin/su"
TMP_PATH = "/data/local/tmp/screen.png"

def swipe_screen(x1, y1, x2, y2, duration=150):
    """ส่งคำสั่งรูดหน้าจอผ่าน root shell โดยตรง"""
    cmd = f"{ROOT_SU} -c 'input swipe {x1} {y1} {x2} {y2} {duration}'"
    subprocess.run(cmd, shell=True)

def get_screen():
    """แคปภาพหน้าจอปัจจุบันและอ่านเข้า OpenCV"""
    subprocess.run(f"{ROOT_SU} -c 'screencap -p {TMP_PATH}'", shell=True)
    if os.path.exists(TMP_PATH):
        return cv2.imread(TMP_PATH)
    return None

def rotate_camera(direction="right", dist=120):
    """
    รูดปรับมุมมองหน้าจอแนวนอน (1280x720):
    ใช้พิกัดจุดรูดว่างฝั่งขวา (X: 920, Y: 360) ซึ่งเป็นพื้นที่หมุนมุมมองเกม
    """
    cx, cy = 920, 360
    if direction == "right":
        # รูดจากขวาไปซ้าย -> มุมกล้องหันขวา
        swipe_screen(cx, cy, cx - dist, cy, duration=140)
    else:
        # รูดจากซ้ายไปขวา -> มุมกล้องหันซ้าย
        swipe_screen(cx - dist, cy, cx, cy, duration=140)
    time.sleep(0.1)

def detect_chest_continuous(screen):
    """
    ประมวลผลภาพแบบ Real-time หาวัตถุทรงกล่อง (รองรับทั้งเช้าและค่ำ)
    คืนค่า: (เจอกล่องไหม, จุดกึ่งกลาง X ของกล่อง, ขนาดพื้นที่วัตถุ)
    """
    if screen is None:
        return False, 0, 0

    h, w = screen.shape[:2]
    # ปรับภาพให้อยู่ในระนาบแนวนอน 1280x720 เสมอ
    if h > w:
        screen = cv2.rotate(screen, cv2.ROTATE_90_CLOCKWISE)

    # ตัดเฉพาะพื้นที่สายตามองเห็นครึ่งบน-กลาง (ไม่เอาจอยสติ๊กและแถบสกิลล่าง)
    roi = screen[80:560, 100:1180]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 1. ช่วงสีกล่องตอนเช้า (โทนสีส้มไม้/เหลืองทอง)
    mask_day = cv2.inRange(hsv, np.array([6, 70, 70]), np.array([24, 255, 255]))
    # 2. ช่วงสีกล่องตอนเย็น/ค่ำ (โทนสีน้ำตาลเข้มหม่น)
    mask_night = cv2.inRange(hsv, np.array([4, 20, 20]), np.array([28, 180, 140]))

    combined_mask = cv2.bitwise_or(mask_day, mask_night)
    kernel = np.ones((5, 5), np.uint8)
    clean_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_box = None
    max_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 800:  # ขนาดพื้นที่ของก้อนกล่อง
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect_ratio = float(cw) / ch
            if 0.55 <= aspect_ratio <= 1.75:  # สัดส่วนก้อนสี่เหลี่ยม
                if area > max_area:
                    max_area = area
                    best_box = (x + 100, y + 80, cw, ch)

    if best_box is not None:
        center_x = best_box[0] + (best_box[2] // 2)
        return True, center_x, max_area

    return False, 0, 0

def main():
    print("=" * 60)
    print("[*] UtopiaFarmAI: โหมดคำนวณภาพต่อเนื่อง + หมุนหันจอจัดกึ่งกลาง (ไม่เดิน)")
    print("=" * 60)
    time.sleep(1.5)

    SCREEN_CENTER_X = 640  # กึ่งกลางหน้าจอแนวนอน 1280
    TOLERANCE = 75         # ช่วงยอมรับกึ่งกลาง (565 ถึง 715)

    while True:
        # ดึงภาพหน้าจอมาคำนวณตลอดเวลา
        frame = get_screen()
        found, chest_x, area = detect_chest_continuous(frame)

        if found:
            diff_from_center = chest_x - SCREEN_CENTER_X
            print(f"[👁] เจอกล่อง! พิกัด X={chest_x} (ห่างจากกลางจอ: {diff_from_center:+d}px, ขนาด: {area})")

            # ปรับระยะรูดจอตามระยะห่างจากกึ่งกลาง (ยิ่งไกลยิ่งรูดยาว ยิ่งใกล้ยิ่งสะกิดเบา)
            swipe_step = min(max(int(abs(diff_from_center) * 0.45), 60), 160)

            if chest_x < (SCREEN_CENTER_X - TOLERANCE):
                print(f"[<] กล่องอยู่ฝั่งซ้าย -> กำลังหันจอไปทางซ้าย (ระยะรูด {swipe_step}px)")
                rotate_camera("left", dist=swipe_step)
            elif chest_x > (SCREEN_CENTER_X + TOLERANCE):
                print(f"[>] กล่องอยู่ฝั่งขวา -> กำลังหันจอไปทางขวา (ระยะรูด {swipe_step}px)")
                rotate_camera("right", dist=swipe_step)
            else:
                print(f"[🎯] กล่องอยู่ตรงกลางจอพอดีแล้ว! (X={chest_x}) -> ล็อกเป้าหมายนิ่ง")
                time.sleep(0.3)
        else:
            # หากยังไม่พบกล่องในมุมมองปัจจุบัน ให้หมุนสำรวจรอบตัวทีละสเต็ป
            print("[🔄] ยังไม่พบกล่องในมุมมอง -> กำลังหมุนสำรวจรอบตัว...")
            rotate_camera("right", dist=180)

if __name__ == "__main__":
    main()
    
