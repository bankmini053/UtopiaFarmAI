import os
import time
import subprocess
import cv2
import numpy as np

ROOT_SU = "/system/xbin/su"
TMP_PATH = "/data/local/tmp/screen.png"

def tap(x, y):
    """ส่งคำสั่งแตะหน้าจอ"""
    subprocess.run(f"{ROOT_SU} -c 'input tap {x} {y}'", shell=True)

def swipe(x1, y1, x2, y2, duration=250):
    """ส่งคำสั่งลากนิ้ว"""
    subprocess.run(f"{ROOT_SU} -c 'input swipe {x1} {y1} {x2} {y2} {duration}'", shell=True)

def get_screen():
    """แคปภาพหน้าจอเกม"""
    subprocess.run(f"{ROOT_SU} -c 'screencap -p {TMP_PATH}'", shell=True)
    if os.path.exists(TMP_PATH):
        return cv2.imread(TMP_PATH)
    return None

def match(screen, template, thr=0.45):
    """ค้นหาตำแหน่งรูปภาพ"""
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

# โหลดไอคอนระบบ
tpl_hand  = load_img("icons/pickup_hand.png")
tpl_pick  = load_img("icons/pick_all.png")
tpl_home  = load_img("icons/hearthstone.png")

# โหลดรูปป้ายชื่อ Rubber Tree ทุกเวอร์ชัน
tree_templates = []
for name in ["tree_rubber.png", "tree_rubber2.png", "tree_target.png"]:
    img = load_img(f"icons/{name}")
    if img is not None:
        tree_templates.append(img)

def has_target_tree(screen):
    """ตรวจดูว่ายังมีป้ายชื่อ Rubber Tree อยู่บนจอหรือไม่"""
    for tpl in tree_templates:
        if match(screen, tpl, thr=0.42) is not None:
            return True
    return False

# พิกัดหน้าจอ 1280x720
attack_x, attack_y = 1100, 560

# พิกัดคันโยกเดินตรง
JOY_X = 195
JOY_Y_START = 520
JOY_Y_END   = 350

def walk_straight(duration_ms=1200):
    """เดินตรงไปข้างหน้า"""
    swipe(JOY_X, JOY_Y_START, JOY_X, JOY_Y_END, duration=duration_ms)
    time.sleep(0.2)

def is_bag_orange(screen):
    """ตรวจจับแถบเตือนสีส้ม (กระเป๋าเต็ม/น้ำหนักเกิน)"""
    if screen is None:
        return False
    hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([5, 150, 150])
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    return np.sum(mask > 0) > 300

def use_hearthstone(screen):
    """กดวาร์ปกลับบ้านเมื่อกระเป๋าเต็ม"""
    print("[!] กำลังกดหินวาร์ปกลับบ้าน (Hearthstone)...")
    if tpl_home is not None:
        pos_home = match(screen, tpl_home, thr=0.60)
        if pos_home:
            tap(pos_home[0], pos_home[1])
            time.sleep(15)
            return
    # พิกัดลัดสำรองของไอคอนวาร์ป/กระเป๋า
    tap(1220, 60)
    time.sleep(10)

def main():
    print("=" * 50)
    print("[*] UtopiaFarmAI: รันระบบตัดจนเลือดหมด + กดเก็บของ + ตรวจกระเป๋า")
    print("=" * 50)
    time.sleep(2)

    while True:
        screen = get_screen()
        if screen is None:
            time.sleep(0.3)
            continue

        # 1. เช็กปุ่ม Pick All ก่อนเสมอหากมีไอเทมตกค้าง
        pos_pick = match(screen, tpl_pick, thr=0.55)
        if pos_pick:
            print("[+] พบ Pick All ตกค้าง -> กดเก็บของ")
            tap(pos_pick[0], pos_pick[1])
            time.sleep(1.0)
            continue

        # 2. แตะตี 1 ครั้งเพื่อเรียกป้ายชื่อเป้าหมาย
        tap(attack_x, attack_y)
        time.sleep(0.5)

        # 3. แคปหน้าจอเพื่อเช็กชื่อ
        scr_chk = get_screen()
        if has_target_tree(scr_chk):
            print("[✓] ยืนยันเป้าหมาย Rubber Tree! ฟันต่อเนื่องจนเลือดหมด...")
            
            # ฟันต่อเนื่องไปเรื่อยๆ จนกว่าต้นไม้จะล้ม (ตรวจพบปุ่มรูปมือ หรือ Pick All)
            chopping = True
            hit_count = 0
            while chopping and hit_count < 30:
                tap(attack_x, attack_y)
                hit_count += 1
                time.sleep(1.05)

                scr_now = get_screen()
                p_hand = match(scr_now, tpl_hand, thr=0.50)
                p_pick = match(scr_now, tpl_pick, thr=0.55)

                # เลือดหมด / ต้นไม้ล้ม -> เข้าสู่ขั้นตอนกดเก็บของ
                if p_hand or p_pick:
                    print("[+] เลือดหมด/ต้นไม้ล้มแล้ว! กำลังกดเก็บของ...")
                    
                    # วนลูปแตะเก็บจนกว่าปุ่มรูปมือและ Pick All จะหายไป
                    collecting = True
                    collect_try = 0
                    while collecting and collect_try < 5:
                        if p_hand:
                            tap(p_hand[0], p_hand[1])
                        if p_pick:
                            tap(p_pick[0], p_pick[1])
                        time.sleep(0.8)
                        
                        scr_collect = get_screen()
                        p_hand = match(scr_collect, tpl_hand, thr=0.50)
                        p_pick = match(scr_collect, tpl_pick, thr=0.55)
                        if not p_hand and not p_pick:
                            collecting = False
                        collect_try += 1

                    print("[✓] เก็บของเสร็จสิ้น ป้ายชื่อต้นไม้หายไปแล้ว")
                    chopping = False

            # 4. ตรวจสอบสถานะสีกระเป๋าหลังเก็บของเสร็จสิ้น
            time.sleep(0.5)
            scr_after_loot = get_screen()
            if is_bag_orange(scr_after_loot):
                print("[!] กระเป๋าขึ้นแถบสีส้ม (เต็มแล้ว) -> ทำการวาร์ปกลับบ้าน")
                use_hearthstone(scr_after_loot)
                print("[*] ภารกิจฟาร์มเสร็จสิ้น หยุดการทำงาน")
                return
            else:
                print("[✓] กระเป๋ายังเป็นสีเขียว (ยังมีที่ว่าง) -> เดินตรงไปหาต้นถัดไป")
                walk_straight(1800)
        else:
            # ถ้าตีแล้วไม่มีชื่อ Rubber Tree ให้เดินหน้าอีกนิดเพื่อขยับหาเป้าหมาย
            walk_straight(800)

if __name__ == "__main__":
    main()
