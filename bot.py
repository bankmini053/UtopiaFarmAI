import os
import time
import subprocess
import cv2
import numpy as np

ROOT_SU = "/system/xbin/su"
TMP_PATH = "/data/local/tmp/screen.png"

def tap(x, y):
    subprocess.run(f"{ROOT_SU} -c 'input tap {x} {y}'", shell=True)

def swipe(x1, y1, x2, y2, duration=300):
    subprocess.run(f"{ROOT_SU} -c 'input swipe {x1} {y1} {x2} {y2} {duration}'", shell=True)

def get_screen():
    subprocess.run(f"{ROOT_SU} -c 'screencap -p {TMP_PATH}'", shell=True)
    if os.path.exists(TMP_PATH):
        return cv2.imread(TMP_PATH)
    return None

def match(screen, template, thr=0.55):
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
tpl_hand   = load_img("icons/pickup_hand.png")
tpl_rubber = load_img("icons/tree_rubber.png")
tpl_pick   = load_img("icons/pick_all.png")
tpl_tree   = load_img("icons/tree_name1.png")

# พิกัดหน้าจอ 1280x720
attack_x, attack_y = 1100, 560

# พิกัดกึ่งกลางปุ่มจอยสติ๊กเดินซ้ายล่างสำหรับจอ 1280x720
JOY_CENTER_X = 205
JOY_CENTER_Y = 485
JOY_UP_Y     = 360  # ดึงตรงขึ้นด้านบนเพื่อให้เดินตรง 100%

def walk_straight(duration_ms=1200):
    """เดินตรงไปข้างหน้าแบบไม่เลี้ยว"""
    swipe(JOY_CENTER_X, JOY_CENTER_Y, JOY_CENTER_X, JOY_UP_Y, duration=duration_ms)
    time.sleep(0.2)

def turn_around():
    """หมุนตัวเบาๆ เฉพาะตอนไม่เจอต้นไม้จริงๆ"""
    swipe(850, 360, 680, 360, duration=250)
    time.sleep(0.3)

def is_bag_orange(screen):
    if screen is None:
        return False
    hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([5, 150, 150])
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    return np.sum(mask > 0) > 300

def main():
    print("[*] UtopiaFarmAI: ระบบล็อกเป้าและเดินตรงเข้าหาต้นไม้ พร้อมทำงาน...")
    time.sleep(2)

    while True:
        screen = get_screen()
        if screen is None:
            time.sleep(0.3)
            continue

        # 1. เช็กว่าอยู่ติดต้นไม้จนมีรูปมือขึ้นหรือไม่
        pos_hand = match(screen, tpl_hand, thr=0.55)
        pos_pick = match(screen, tpl_pick, thr=0.60)

        if pos_pick:
            print("[+] พบ Pick All -> เก็บของ")
            tap(pos_pick[0], pos_pick[1])
            time.sleep(1.0)
            walk_straight(800)
            continue

        # 2. ลองตีกระตุ้น 1 ครั้งเพื่อเช็กชื่อ
        tap(attack_x, attack_y)
        time.sleep(0.6)

        screen_chk = get_screen()
        pos_rubber = match(screen_chk, tpl_rubber, thr=0.50)

        if pos_rubber:
            print("[✓] ล็อกเป้า Rubber Tree สำเร็จ! กำลังฟันจนกว่าจะล้ม...")
            chopping = True
            hit_count = 0
            while chopping and hit_count < 25:
                tap(attack_x, attack_y)
                hit_count += 1
                time.sleep(1.1)

                scr_now = get_screen()
                p_hand = match(scr_now, tpl_hand, 0.55)
                p_pick = match(scr_now, tpl_pick, 0.60)

                if p_hand or p_pick:
                    print("[+] ต้นไม้ล้มแล้ว! กดเก็บของ")
                    if p_hand:
                        tap(p_hand[0], p_hand[1])
                    if p_pick:
                        tap(p_pick[0], p_pick[1])
                    time.sleep(1.2)
                    chopping = False

            # ตรวจสอบสีกระเป๋า
            scr_after = get_screen()
            if is_bag_orange(scr_after):
                print("[!] กระเป๋าขึ้นสีส้ม -> หยุดพัก")
                return

            # ตัดเสร็จแล้วเดินตรงไปหาต้นถัดไป
            walk_straight(1500)
            continue

        # 3. ถ้ายังไม่ถึงต้นไม้ ให้สแกนหาต้นไม้ในจอ
        pos_tree = match(screen, tpl_tree, thr=0.40)
        if pos_tree:
            print(f"[+] มองเห็นต้นไม้ที่ ({pos_tree[0]}, {pos_tree[1]}) -> เดินตรงเข้าหา...")
            # ถ้าต้นไม้อยู่เยื้องซ้ายหรือขวา ให้ปรับหันหน้าจอนิดเดียว
            diff_x = pos_tree[0] - 640
            if abs(diff_x) > 150:
                swipe(640, 360, 640 - int(diff_x * 0.3), 360, duration=150)
                time.sleep(0.2)
            # เดินตรงเข้าหาต้นไม้
            walk_straight(1200)
        else:
            # ไม่พบต้นไม้ ให้เดินตรงไปข้างหน้าแล้วค่อยหันมุมมองหา
            walk_straight(1000)
            turn_around()

if __name__ == "__main__":
    main()
          
