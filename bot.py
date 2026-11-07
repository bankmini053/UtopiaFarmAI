import os
import time
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

def match(screen, template, thr=0.45):
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
tpl_hand     = load_img("icons/pickup_hand.png")
tpl_pick     = load_img("icons/pick_all.png")
tpl_home     = load_img("icons/hearthstone.png")
tpl_hp_empty = load_img("icons/tree_hp_empty.png")  # รูปเลือดหมด

# โหลดรูปป้ายชื่อ Rubber Tree
tree_templates = []
for name in ["tree_rubber.png", "tree_rubber2.png", "tree_target.png"]:
    img = load_img(f"icons/{name}")
    if img is not None:
        tree_templates.append(img)

def has_target_tree(screen):
    for tpl in tree_templates:
        if match(screen, tpl, thr=0.40) is not None:
            return True
    return False

# พิกัดสำหรับหน้าจอ 1280x720
attack_x, attack_y = 1100, 560

JOY_X = 195
JOY_Y_START = 520
JOY_Y_END   = 350

def walk_straight(duration_ms=1200):
    swipe(JOY_X, JOY_Y_START, JOY_X, JOY_Y_END, duration=duration_ms)
    time.sleep(0.2)

def is_bag_orange(screen):
    if screen is None:
        return False
    hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([5, 150, 150])
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    return np.sum(mask > 0) > 300

def main():
    print("=" * 50)
    print("[*] UtopiaFarmAI: ล็อกเป้าฟันจนเลือดหมด ห้ามเดินหนี...")
    print("=" * 50)
    time.sleep(2)

    while True:
        screen = get_screen()
        if screen is None:
            time.sleep(0.3)
            continue

        # 1. เช็กปุ่ม Pick All ก่อนเสมอถ้ามีของตก
        pos_pick = match(screen, tpl_pick, thr=0.55)
        if pos_pick:
            print("[+] พบ Pick All -> กดเก็บของ")
            tap(pos_pick[0], pos_pick[1])
            time.sleep(1.0)
            continue

        # 2. ตีกระตุ้น 1 ครั้งเพื่อเรียกชื่อ
        tap(attack_x, attack_y)
        time.sleep(0.5)

        scr_chk = get_screen()
        if has_target_tree(scr_chk):
            print("[✓] ยืนยัน Rubber Tree! กำลังฟันจนเลือดหมด (ล็อกตัวห้ามเดิน)...")
            
            # ลูปฟันต่อเนื่อง ห้ามหลุดไปเดินเด็ดขาดจนกว่าจะพบเงื่อนไขเลือดหมด/ล้ม
            chopping = True
            miss_count = 0  # ตัวนับป้องกันชื่อกะพริบหายชั่วคราว
            
            while chopping:
                # ฟันรัวๆ ชุดละ 3 ครั้งติดกันเพื่อความเร็วและไม่หลุดจังหวะ
                for _ in range(3):
                    tap(attack_x, attack_y)
                    time.sleep(1.05)

                scr_now = get_screen()
                p_hand  = match(scr_now, tpl_hand, thr=0.48)
                p_pick  = match(scr_now, tpl_pick, thr=0.55)
                p_empty = match(scr_now, tpl_hp_empty, thr=0.55) if tpl_hp_empty is not None else None

                # ตรวจพบ: เลือดหมด หรือ ต้นไม้ล้ม (รูปมือ / Pick All)
                if p_hand or p_pick or p_empty:
                    print("[+] เลือดหมด/ต้นไม้ล้มแล้ว! เริ่มกดเก็บของ...")
                    
                    # กดเก็บของจนกว่ารูปมือและ Pick All จะหายไป
                    for _ in range(6):
                        if p_hand:
                            tap(p_hand[0], p_hand[1])
                        if p_pick:
                            tap(p_pick[0], p_pick[1])
                        time.sleep(0.8)
                        
                        scr_collect = get_screen()
                        p_hand = match(scr_collect, tpl_hand, thr=0.48)
                        p_pick = match(scr_collect, tpl_pick, thr=0.55)
                        if not p_hand and not p_pick:
                            break

                    print("[✓] เก็บของเรียบร้อย")
                    chopping = False
                    break

                # เช็กชื่อซ้ำ: ถ้าไม่เห็นชื่อ ให้ยอมให้พลาดได้ 4 ครั้งติดกัน (กันชื่อกะพริบ)
                if not has_target_tree(scr_now):
                    miss_count += 1
                    if miss_count >= 4:
                        print("[!] ป้ายชื่อหายไปต่อเนื่อง -> จบการฟันต้นนี้")
                        chopping = False
                else:
                    miss_count = 0

            # 3. เช็กสีกระเป๋าหลังฟันเสร็จ
            time.sleep(0.5)
            scr_after = get_screen()
            if is_bag_orange(scr_after):
                print("[!] กระเป๋าขึ้นสีส้ม (เต็ม) -> จบการทำงาน")
                return
            else:
                print("[✓] กระเป๋ายังเขียว -> เดินตรงไปหาต้นถัดไป")
                walk_straight(2000)
        else:
            # ถ้าตีแล้วยังไม่เจอชื่อ เดินขยับเข้าหาอีกนิด
            walk_straight(700)

if __name__ == "__main__":
    main()
