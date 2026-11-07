import os
import time
import subprocess
import cv2
import numpy as np

ROOT_SU = "/system/xbin/su"
TMP_PATH = "/data/local/tmp/screen.png"

def tap(x, y):
    """ส่งคำสั่งแตะสัมผัสหน้าจอ"""
    subprocess.run(f"{ROOT_SU} -c 'input tap {x} {y}'", shell=True)

def swipe(x1, y1, x2, y2, duration=300):
    """ส่งคำสั่งลากนิ้ว"""
    subprocess.run(f"{ROOT_SU} -c 'input swipe {x1} {y1} {x2} {y2} {duration}'", shell=True)

def get_screen():
    """แคปภาพลง tmp file เพื่อป้องกันปัญหาข้อมูลภาพเสีย (None)"""
    subprocess.run(f"{ROOT_SU} -c 'screencap -p {TMP_PATH}'", shell=True)
    if os.path.exists(TMP_PATH):
        return cv2.imread(TMP_PATH)
    return None

def match(screen, template, thr=0.55):
    """ค้นหาตำแหน่งรูปภาพด้วย OpenCV Template Matching"""
    if screen is None or template is None:
        return None
    res = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if max_val >= thr:
        h, w = template.shape[:2]
        return (max_loc[0] + w // 2, max_loc[1] + h // 2)
    return None

def find_trees_near_to_far(screen, template, thr=0.45):
    """
    ค้นหาตำแหน่งต้นไม้ทั้งหมดในจอ จัดลำดับจากใกล้ไปไกล
    (แกน Y ยิ่งมาก = อยู่ด้านล่างจอ = ใกล้ตัวละครที่สุด)
    """
    if screen is None or template is None:
        return []
    res = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= thr)
    h, w = template.shape[:2]
    
    candidates = []
    for pt in zip(*loc[::-1]):
        cx, cy = pt[0] + w // 2, pt[1] + h // 2
        # กรองจุดที่ซ้ำซ้อนกันในระยะใกล้เคียง
        if not any(abs(cx - ex) < 50 and abs(cy - ey) < 50 for ex, ey in candidates):
            candidates.append((cx, cy))
            
    # เรียงลำดับจากค่า Y มากไปหาน้อย (ใกล้ -> ไกล)
    candidates.sort(key=lambda coord: coord[1], reverse=True)
    return candidates

def load_img(path):
    """โหลดไฟล์ภาพไอคอน"""
    return cv2.imread(path) if os.path.exists(path) else None

# โหลดไฟล์ภาพไอคอนทั้งหมดจากโฟลเดอร์ icons/
tpl_hand   = load_img("icons/pickup_hand.png")
tpl_rubber = load_img("icons/tree_rubber.png")
tpl_pick   = load_img("icons/pick_all.png")
tpl_tree   = load_img("icons/tree_name1.png")

# พิกัดหน้าจอสำหรับจอ 1280x720
attack_x, attack_y = 1100, 560
joy_x, joy_y = 210, 530
joy_forward_y = 410

def move_forward(seconds=1.5):
    """เดินตรงไปข้างหน้า"""
    swipe(joy_x, joy_y, joy_x, joy_forward_y, int(seconds * 1000))
    time.sleep(0.3)

def rotate_camera(direction="right"):
    """หมุนกล้องเพื่อค้นหาต้นไม้รอบตัว"""
    print("[->] กำลังหมุนมุมกล้องเพื่อหาต้นไม้รอบตัว...")
    if direction == "right":
        swipe(850, 360, 600, 360, 350)
    else:
        swipe(600, 360, 850, 360, 350)
    time.sleep(0.5)

def is_bag_orange(screen):
    """ตรวจจับแถบแจ้งเตือนสีส้มของกระเป๋า"""
    if screen is None:
        return False
    hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([5, 150, 150])
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    return np.sum(mask > 0) > 300

def main():
    print("[*] UtopiaFarmAI: ระบบตรวจจับต้นไม้และเดินจากใกล้ไปไกล พร้อมทำงาน...")
    time.sleep(2)

    while True:
        screen = get_screen()
        if screen is None:
            time.sleep(0.3)
            continue

        # ตรวจจับต้นไม้ในหน้าจอ (เรียงจากใกล้ไปไกล)
        trees = find_trees_near_to_far(screen, tpl_tree, thr=0.45)
        
        if not trees:
            print("[?] ไม่พบต้นไม้ในมุมมองนี้ -> หมุนหามุมใหม่...")
            rotate_camera("right")
            move_forward(1.0)
            continue

        # ลูปเข้าหาต้นไม้ทีละต้นจากใกล้ไปไกล
        target_confirmed = False
        for tree_x, tree_y in trees:
            print(f"[+] พบต้นไม้ที่พิกัด ({tree_x}, {tree_y}) -> กำลังเดินเข้าหา...")
            
            # ปรับมุมมองกล้องหันหาเป้าหมาย (หากเยื้องซ้ายหรือขวา)
            screen_center_x = 640
            diff_x = tree_x - screen_center_x
            if abs(diff_x) > 120:
                swipe(screen_center_x, 360, screen_center_x - int(diff_x * 0.4), 360, 200)
                time.sleep(0.3)

            # เดินเข้าหาตามระยะ
            walk_time = 1.2 if tree_y > 450 else 2.2
            move_forward(walk_time)

            # ตีกระตุ้น 1 ครั้งเพื่อเรียกป้ายชื่อ
            tap(attack_x, attack_y)
            time.sleep(0.7)

            # ตรวจสอบป้ายชื่อเป้าหมาย
            screen_check = get_screen()
            pos_rubber = match(screen_check, tpl_rubber, 0.50)

            if pos_rubber:
                print("[✓] ยืนยันเป้าหมาย Rubber Tree! กำลังฟันจนกว่าจะล้ม...")
                target_confirmed = True
                
                # ฟันต่อเนื่องจนกว่าจะล้ม
                chopping = True
                hit_count = 0
                while chopping and hit_count < 25:
                    tap(attack_x, attack_y)
                    hit_count += 1
                    time.sleep(1.1)

                    scr_now = get_screen()
                    pos_hand = match(scr_now, tpl_hand, 0.55)
                    pos_pick = match(scr_now, tpl_pick, 0.60)

                    if pos_hand or pos_pick:
                        print("[+] ต้นไม้ล้มแล้ว -> เก็บของ")
                        if pos_hand:
                            tap(pos_hand[0], pos_hand[1])
                        if pos_pick:
                            tap(pos_pick[0], pos_pick[1])
                        time.sleep(1.2)
                        chopping = False

                # ตรวจสอบสีกระเป๋าหลังเก็บของ
                screen_after = get_screen()
                if is_bag_orange(screen_after):
                    print("[!] กระเป๋าขึ้นสีส้ม -> หยุดพักการฟาร์ม")
                    return
                break
            else:
                print("[x] ชื่อไม่ตรง -> ข้ามไปต้นถัดไป")
                continue

        if not target_confirmed:
            # หากตรวจต้นไม้รอบนี้แล้วไม่มีต้นไหนชื่อตรง ให้หมุนมุมมองใหม่
            rotate_camera("right")
            move_forward(1.5)

if __name__ == "__main__":
    main()
          
