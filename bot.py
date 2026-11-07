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

def rotate_view(direction="right", dist=180):
    cx, cy = 900, 300
    if direction == "right":
        swipe_screen(cx, cy, cx - dist, cy, duration=260)
    else:
        swipe_screen(cx - dist, cy, cx, cy, duration=260)
    time.sleep(0.35)

JOY_CENTER_X, JOY_CENTER_Y, JOY_RADIUS = 250, 530, 145

def walk_straight(duration_ms=500):
    """ดันจอยสติ๊กตรงไปข้างหน้า (0 องศา) เดินเข้าชนกล่อง"""
    swipe_screen(JOY_CENTER_X, JOY_CENTER_Y, JOY_CENTER_X, JOY_CENTER_Y - JOY_RADIUS, duration=duration_ms)
    time.sleep(0.2)

def detect_chest(screen):
    """ตรวจจับกล่องสมบัติทั้งตอนเช้าและตอนค่ำ"""
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
            if 0.5 <= float(cw) / ch <= 2.0 and a > max_a:
                max_a = a
                best_x = x + 100 + (cw // 2)
    return (True, best_x, max_a) if max_a > 0 else (False, 0, 0)

def check_hand_button(screen):
    """ตรวจจับว่าปุ่มรูปมือสีขาวเหนือปุ่มดาบโผล่ขึ้นมาหรือยัง (พิกัด X: 1000-1090, Y: 350-440)"""
    if screen is None:
        return False
    hand_roi = screen[350:440, 1000:1090]
    gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
    white_pixels = np.sum(gray > 200)
    # ถ้ามีพิกเซลสีขาวสว่างเพียงพอ แสดงว่าปุ่มรูปมือปรากฏขึ้นแล้ว
    return white_pixels > 200

def main():
    print("=" * 60)
    print("[*] UtopiaFarmAI: เจอกล่อง -> พัก 2 วิ -> ล็อกกลางจอ -> รอรูปมือขึ้น -> เปิด")
    print("=" * 60)
    time.sleep(2)

    CENTER_X, TOLERANCE = 640, 80
    scan_start_time = time.time()

    while True:
        scr = get_screen()

        # ตรวจสอบว่ามีปุ่มรูปมืออยู่แล้วหรือไม่
        if check_hand_button(scr):
            print("\n" + "=" * 50)
            print("[🎯] ตรวจพบปุ่มรูปมือเรียบร้อยแล้ว! แตะเปิดกล่อง...")
            print("=" * 50)
            time.sleep(0.3)
            tap(1045, 390)
            time.sleep(0.3)
            tap(1045, 390)
            print("[✓] ภารกิจเปิดกล่องเสร็จสมบูรณ์!")
            break

        found, chest_x, area = detect_chest(scr)

        if found:
            # รีเซ็ตตัวจับเวลาเพราะเจอกล่องแล้ว
            scan_start_time = time.time()
            print(f"\n[👁] ตรวจพบกล่องแล้ว! (X={chest_x}, ขนาด: {area:.1f})")
            print("[⏸] หยุดรอภาพนิ่ง 2 วินาที...")
            time.sleep(2.0)

            # โหมดปรับละเอียดขณะภาพนิ่ง
            aligned = False
            for _ in range(8):
                still_scr = get_screen()
                still_found, still_x, _ = detect_chest(still_scr)

                if not still_found:
                    rotate_view("right", dist=90)
                    continue

                diff = still_x - CENTER_X
                if still_x < (CENTER_X - TOLERANCE):
                    print(f"[<] ปรับละเอียดหมุนซ้าย (ห่าง {diff:+d})...")
                    rotate_view("left", dist=90)
                elif still_x > (CENTER_X + TOLERANCE):
                    print(f"[>] ปรับละเอียดหมุนขวา (ห่าง {diff:+d})...")
                    rotate_view("right", dist=90)
                else:
                    print(f"\n[🎯] กล่องอยู่ตรงกลางจอแล้ว! (X={still_x})")
                    aligned = True
                    break

            if aligned:
                print("[↑] เดินตรงเข้าหากล่องและรอปุ่มรูปมือขึ้น...")
                hand_found = False
                
                # ก้าวเดินเข้าชนกล่องต่อเนื่อง สูงสุด 6 ก้าวสั้น
                for step in range(6):
                    walk_straight(duration_ms=500)
                    time.sleep(0.2)
                    chk_scr = get_screen()
                    if check_hand_button(chk_scr):
                        print(f"[✓] ตรวจพบปุ่มรูปมือขึ้นมาแล้วที่ก้าวที่ {step + 1}!")
                        hand_found = True
                        break
                    else:
                        print(f"[*] กำลังก้าวชนกล่อง ({step + 1}/6)... ยังไม่พบรูปมือ")

                if hand_found:
                    print("[🎯] แตะปุ่มรูปมือเปิดกล่อง (1045, 390)...")
                    time.sleep(0.3)
                    tap(1045, 390)
                    time.sleep(0.3)
                    tap(1045, 390)
                    print("[✓] ภารกิจเปิดกล่องเสร็จสมบูรณ์!")
                    break
                else:
                    print("[!] เดินชนครบแล้วยังไม่ขึ้นรูปมือ -> ตรวจสอบมุมมองใหม่")
        else:
            elapsed = time.time() - scan_start_time
            print(f"[🔄] กวาดสายตาหากล่อง... ({elapsed:.1f}/5.0 วิ)")

            # เงื่อนไข: ถ้ากวาดสายตาเกิน 5 วินาทีแล้วยังไม่เจอ ให้หยุดคำสั่ง
            if elapsed >= 5.0:
                print("\n" + "=" * 50)
                print("[!] กวาดสายตาต่อเนื่องเกิน 5 วินาทีแล้วไม่พบกล่อง -> หยุดคำสั่งอัตโนมัติ")
                print("=" * 50)
                break

            rotate_view("right", dist=200)

if __name__ == "__main__":
    main()
