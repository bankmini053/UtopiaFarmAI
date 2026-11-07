import os
import time
import math
import subprocess
import cv2
import numpy as np

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

def match(screen, template, thr=0.48):
    if screen is None or template is None:
        return None
    res = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if max_val >= thr:
        h, w = template.shape[:2]
        return (max_loc[0] + w // 2, max_loc[1] + h // 2)
    return None

def load_img(path):
    return cv2.imread(path) if os.path.exists(path) else None

# โหลดไอคอน
tpl_hand  = load_img("icons/pickup_hand.png")
tpl_pick  = load_img("icons/pick_all.png")
tpl_home  = load_img("icons/hearthstone.png")

# โหลดรูปป้ายชื่อ
tree_templates = []
for name in ["tree_rubber.png", "tree_rubber2.png", "tree_target.png"]:
    img = load_img(f"icons/{name}")
    if img is not None:
        tree_templates.append(img)

# พิกัดหน้าจอสำหรับจอ 1280x720
attack_x, attack_y = 1100, 560
JOY_CENTER_X = 195
JOY_CENTER_Y = 520
JOY_RADIUS   = 160

def walk_by_vector(dx, dy, duration_sec=1.5):
    """
    บังคับจอยสติ๊กเดินตามทิศทางเวกเตอร์ (dx, dy)
    dx > 0 = ขวา, dx < 0 = ซ้าย
    dy > 0 = หน้า/เหนือ, dy < 0 = หลัง/ใต้
    """
    angle_rad = math.atan2(dx, dy)
    target_x = int(JOY_CENTER_X + JOY_RADIUS * math.sin(angle_rad))
    target_y = int(JOY_CENTER_Y - JOY_RADIUS * math.cos(angle_rad))
    
    duration_ms = int(duration_sec * 1000)
    swipe(JOY_CENTER_X, JOY_CENTER_Y, target_x, target_y, duration=duration_ms)
    time.sleep(0.2)

def walk_straight(duration_ms=800):
    swipe(JOY_CENTER_X, JOY_CENTER_Y, JOY_CENTER_X, JOY_CENTER_Y - JOY_RADIUS, duration=duration_ms)
    time.sleep(0.2)

def warp_and_move_to_spot():
    """ขั้นตอนกดวาร์ป รอ 10 วินาที แล้วเดินจาก A ไป B"""
    print("[*] เริ่มต้นภารกิจ: ค้นหาไอคอน Hearthstone เพื่อวาร์ป...")
    
    warped = False
    for _ in range(5):
        scr = get_screen()
        pos_home = match(scr, tpl_home, thr=0.50)
        if pos_home:
            print(f"[✓] พบหินวาร์ปที่พิกัด ({pos_home[0]}, {pos_home[1]}) -> กดวาร์ปทันที")
            tap(pos_home[0], pos_home[1])
            warped = True
            break
        time.sleep(1.0)

    if not warped:
        print("[!] ไม่พบรูป hearthstone.png บนจอ -> แตะพิกัดลัดกระเป๋าสำรอง")
        tap(1220, 60)

    print("[*] กำลังรอดาวน์โหลดฉากวาร์ป 10 วินาที...")
    time.sleep(10)

    # ตัวละครมาหยุดอยู่ที่พิกัด A: (17514, 15323)
    # ต้องการเดินไปยังพิกัด B: (17529, 15339)
    # dx = +15, dy = +16
    print("[->] ถึงจุดเกิด (17514, 15323) แล้ว! กำลังเดินนำทางไปยังพิกัดเป้าหมาย (17529, 15339)...")
    
    # เดินเฉียงไปตามเวกเตอร์ dx=15, dy=16 (ประมาณ 4 วินาที)
    walk_by_vector(15, 16, duration_sec=4.0)
    print("[✓] มาถึงบริเวณพิกัดเป้าหมายเรียบร้อยแล้ว!")
    time.sleep(1.0)

def has_target_name(screen):
    if screen is None:
        return False
    for tpl in tree_templates:
        if match(screen, tpl, thr=0.42) is not None:
            return True
    roi = screen[120:180, 480:800]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 45, 255]))
    return np.sum(mask > 0) > 100

def has_hp_bar(screen):
    if screen is None:
        return False
    bar_roi = screen[180:220, 490:790]
    hsv = cv2.cvtColor(bar_roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 70, 70]), np.array([180, 255, 255]))
    return np.sum(mask > 0) > 150

def collect_items():
    print("[+] ต้นไม้ล้มแล้ว! กดรูปมือเก็บของ...")
    for _ in range(6):
        scr = get_screen()
        p_hand = match(scr, tpl_hand, thr=0.48)
        p_pick = match(scr, tpl_pick, thr=0.52)
        if p_hand:
            tap(p_hand[0], p_hand[1])
        elif p_pick:
            tap(p_pick[0], p_pick[1])
        else:
            tap(820, 430)
        time.sleep(0.7)

def main():
    print("=" * 55)
    print("[*] UtopiaFarmAI: กดวาร์ป Hearthstone -> เดินจาก A ไป B -> ตัดต้นไม้")
    print("=" * 55)
    time.sleep(2)

    # 1. รันกระบวนการวาร์ปแล้วเดินไปยังพิกัดฟาร์ม
    warp_and_move_to_spot()

    # 2. เริ่มลูปฟาร์มตัดต้นไม้
    while True:
        screen = get_screen()
        if screen is None:
            time.sleep(0.3)
            continue

        pos_pick = match(screen, tpl_pick, thr=0.55)
        if pos_pick:
            tap(pos_pick[0], pos_pick[1])
            time.sleep(1.0)
            continue

        # ตีเช็กชื่อต้นไม้
        tap(attack_x, attack_y)
        time.sleep(0.4)

        scr_chk = get_screen()
        if has_target_name(scr_chk):
            print("[✓] ขึ้นชื่อต้นไม้แล้ว! ฟันต่อเนื่องจนกว่าหลอดสีจะหาย...")
            while True:
                tap(attack_x, attack_y)
                time.sleep(1.05)

                scr_chopping = get_screen()
                hp_alive = has_hp_bar(scr_chopping)
                p_hand = match(scr_chopping, tpl_hand, thr=0.48)
                p_pick = match(scr_chopping, tpl_pick, thr=0.52)

                if not hp_alive or p_hand or p_pick:
                    print("[+] หลอดสีข้างล่างหายไปแล้ว! หยุดตี")
                    time.sleep(0.5)
                    collect_items()
                    break

            print("[->] เก็บของเสร็จสิ้น เดินหน้าหาต้นต่อไป...")
            walk_straight(1800)
        else:
            walk_straight(700)

if __name__ == "__main__":
    main()
