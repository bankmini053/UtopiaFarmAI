import os
import time
import math
import subprocess
import cv2

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

def match(screen, template, thr=0.50):
    if screen is None or template is None:
        return None
    res = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if max_val >= thr:
        h, w = template.shape[:2]
        return (max_loc[0] + w // 2, max_loc[1] + h // 2)
    return None

tpl_home = cv2.imread("icons/hearthstone.png") if os.path.exists("icons/hearthstone.png") else None

# พิกัดกึ่งกลางปุ่มจอยสติ๊กสำหรับจอ 1280x720
JOY_CENTER_X = 195
JOY_CENTER_Y = 520
JOY_RADIUS   = 160

def walk_by_vector(dx, dy, duration_sec=4.0):
    """คำนวณทิศทางจอยสติ๊กตามระยะ dx, dy แล้วเดิน"""
    angle_rad = math.atan2(dx, dy)
    target_x = int(JOY_CENTER_X + JOY_RADIUS * math.sin(angle_rad))
    target_y = int(JOY_CENTER_Y - JOY_RADIUS * math.cos(angle_rad))
    duration_ms = int(duration_sec * 1000)
    swipe(JOY_CENTER_X, JOY_CENTER_Y, target_x, target_y, duration=duration_ms)
    time.sleep(0.2)

def main():
    print("[*] ค้นหาไอคอน Hearthstone เพื่อกดวาร์ป...")
    warped = False
    for _ in range(5):
        scr = get_screen()
        pos_home = match(scr, tpl_home, thr=0.50)
        if pos_home:
            print(f"[✓] พบหินวาร์ปที่ ({pos_home[0]}, {pos_home[1]}) -> กดวาร์ปทันที")
            tap(pos_home[0], pos_home[1])
            warped = True
            break
        time.sleep(1.0)

    if not warped:
        print("[!] ไม่พบรูป hearthstone.png บนจอ")
        return

    # รอโหลดฉากวาร์ป 10 วินาที
    print("[*] กำลังรอดาวน์โหลดวาร์ป 10 วินาที...")
    time.sleep(10)

    # พิกัดเริ่มต้น A: 17514, 15323 -> พิกัดเป้าหมาย B: 17529, 15339
    # dx = 17529 - 17514 = +15, dy = 15339 - 15323 = +16
    dx = 17529 - 17514
    dy = 15339 - 15323
    print(f"[->] อยู่ที่จุดเกิด (17514, 15323) -> กำลังเดินไปยังเป้าหมาย (17529, 15339)...")
    
    walk_by_vector(dx, dy, duration_sec=4.0)
    print("[✓] ถึงพิกัดเป้าหมายเรียบร้อยแล้ว!")

if __name__ == "__main__":
    main()
    
