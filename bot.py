import os
import time
import subprocess
import cv2
import numpy as np

ROOT_SU = "/system/xbin/su"
TMP_PATH = "/data/local/tmp/screen.png"

def tap(x, y):
    """แตะหน้าจอ"""
    subprocess.run(f"{ROOT_SU} -c 'input tap {x} {y}'", shell=True)

def swipe(x1, y1, x2, y2, duration=250):
    """ปัดหน้าจอ"""
    subprocess.run(f"{ROOT_SU} -c 'input swipe {x1} {y1} {x2} {y2} {duration}'", shell=True)

def get_screen():
    """แคปภาพหน้าจอ"""
    subprocess.run(f"{ROOT_SU} -c 'screencap -p {TMP_PATH}'", shell=True)
    if os.path.exists(TMP_PATH):
        return cv2.imread(TMP_PATH)
    return None

def match(screen, template, thr=0.50):
    """จับคู่ภาพไอคอน"""
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
tpl_hand = load_img("icons/pickup_hand.png")
tpl_pick = load_img("icons/pick_all.png")

# โหลดรูปป้ายชื่อ
tree_names = []
for fname in ["tree_rubber.png", "tree_rubber2.png", "tree_target.png"]:
    img = load_img(f"icons/{fname}")
    if img is not None:
        tree_names.append(img)

def is_target_tree(screen):
    """ตรวจว่าป้ายชื่อตรงกับต้นยางพาราหรือไม่"""
    for tpl in tree_names:
        if match(screen, tpl, thr=0.45) is not None:
            return True
    return False

# พิกัดสำหรับหน้าจอ 1280x720
attack_x, attack_y = 1100, 560
screen_center_x = 640

# พิกัดจอยสติ๊กเดินตรง
JOY_X = 195
JOY_Y_START = 520
JOY_Y_END   = 350

def walk_straight(duration_ms=1000):
    """เดินตรงไปข้างหน้า"""
    swipe(JOY_X, JOY_Y_START, JOY_X, JOY_Y_END, duration=duration_ms)
    time.sleep(0.2)

def find_trunk_contour(screen):
    """
    ตรวจหาลำต้นไม้ด้วยเฉดสีน้ำตาล (HSV)
    คืนค่า (center_x, center_y, area) ของลำต้นที่ใหญ่และชัดที่สุด
    """
    if screen is None:
        return None

    # ตัดขอบจอที่เป็น UI (จอยสติ๊กซ้ายล่าง, ปุ่มตีขวาล่าง, แถบสถานะบน) เพื่อไม่ให้สีกวน
    h, w = screen.shape[:2]
    roi = screen.copy()
    roi[:150, :] = 0          # แถบด้านบน
    roi[500:, :350] = 0        # โซนจอยสติ๊ก
    roi[450:, 950:] = 0        # โซนปุ่มตี

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # ช่วงเฉดสีน้ำตาลของลำต้นไม้ใน Utopia: Origin
    lower_brown = np.array([8, 60, 40])
    upper_brown = np.array([24, 255, 180])

    mask = cv2.inRange(hsv, lower_brown, upper_brown)
    # ตัดสัญญาณรบกวนเล็กๆ ออก
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # เลือกลำต้นที่มีขนาดใหญ่ที่สุด
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < 800:  # เล็กเกินไปแสดงว่าอยู่ไกลมากหรือเป็นเศษหิน
        return None

    x, y, cw, ch = cv2.boundingRect(largest)
    return (x + cw // 2, y + ch // 2, area)

def align_to_tree(trunk_x):
    """หันหน้าจอให้ลำต้นไม้มาอยู่ตรงกึ่งกลางจอ (X = 640)"""
    diff_x = trunk_x - screen_center_x
    if abs(diff_x) > 60:
        # คำนวณระยะปัดหน้าจอตามความห่างจากกึ่งกลาง
        swipe_dist = int(diff_x * 0.5)
        swipe(screen_center_x, 360, screen_center_x - swipe_dist, 360, duration=180)
        time.sleep(0.3)
        return False
    return True

def main():
    print("=" * 45)
    print("[*] UtopiaFarmAI: ระบบล็อกลำต้นสีน้ำตาล + หันตรงกลาง พร้อมทำงาน...")
    print("=" * 45)
    time.sleep(2)

    while True:
        screen = get_screen()
        if screen is None:
            time.sleep(0.3)
            continue

        # 1. เช็กเก็บของก่อนเสมอ
        pos_pick = match(screen, tpl_pick, thr=0.60)
        if pos_pick:
            print("[+] พบ Pick All -> กดเก็บของ")
            tap(pos_pick[0], pos_pick[1])
            time.sleep(1.0)
            walk_straight(800)
            continue

        # 2. ค้นหาลำต้นไม้สีน้ำตาลในฉาก
        trunk_info = find_trunk_contour(screen)

        if trunk_info is None:
            # หากมองไม่เห็นลำต้นไม้ ให้หมุนกล้องมองรอบๆ เบาๆ
            print("[?] ไม่พบลำต้นสีน้ำตาล -> หมุนกล้องค้นหา...")
            swipe(800, 360, 650, 360, duration=250)
            time.sleep(0.3)
            continue

        trunk_x, trunk_y, area = trunk_info
        print(f"[+] พบลำต้นสีน้ำตาลที่ X={trunk_x}, ขนาด={int(area)}")

        # 3. หันกล้องให้ต้นไม้อยู่ตรงกลางจอ
        centered = align_to_tree(trunk_x)
        if not centered:
            continue

        # 4. เมื่ออยู่ตรงกลางแล้ว ถ้ายังไม่ชิดต้น (area ยังเล็ก หรือยังไม่มีปุ่มรูปมือ) ให้เดินตรงเข้าหา
        pos_hand = match(screen, tpl_hand, thr=0.50)
        if pos_hand is None and area < 8000:
            print("[->] ต้นไม้อยู่ตรงกลางแล้ว -> เดินตรงเข้าหา...")
            walk_straight(1200)
            continue

        # 5. เมื่อเดินเข้ามาชิดต้นแล้ว ให้ตีกระตุ้น 1 ครั้งเพื่อเช็กชื่อ
        print("[*] ชิดต้นไม้แล้ว -> ตี 1 ครั้งเพื่อเช็กชื่อ...")
        tap(attack_x, attack_y)
        time.sleep(0.7)

        # 6. ตรวจสอบป้ายชื่อเป้าหมาย
        screen_chk = get_screen()
        if is_target_tree(screen_chk):
            print("[✓] ป้ายชื่อตรง: Rubber Tree! ฟันต่อเนื่องจนล้ม...")
            chopping = True
            hit_count = 0
            while chopping and hit_count < 30:
                tap(attack_x, attack_y)
                hit_count += 1
                time.sleep(1.1)

                scr_now = get_screen()
                p_hand = match(scr_now, tpl_hand, thr=0.50)
                p_pick = match(scr_now, tpl_pick, thr=0.55)

                if p_hand or p_pick:
                    print("[+] ต้นไม้ล้มแล้ว! กดเก็บของ")
                    if p_hand:
                        tap(p_hand[0], p_hand[1])
                    if p_pick:
                        tap(p_pick[0], p_pick[1])
                    time.sleep(1.2)
                    chopping = False

            # ฟันเสร็จแล้ว เดินตรงผ่านจุดเดิมไปหาต้นใหม่
            walk_straight(2000)
        else:
            # ถ้าชื่อไม่ใช่ Rubber Tree ให้หมุนตัวหลบไปทิศอื่น
            print("[x] ชื่อไม่ถูกต้อง -> หมุนกล้องหลบไปหาต้นอื่น...")
            swipe(850, 360, 550, 360, duration=300)
            time.sleep(0.3)
            walk_straight(1500)

if __name__ == "__main__":
    main()
          
