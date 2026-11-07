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
tpl_hand = load_img("icons/pickup_hand.png")
tpl_pick = load_img("icons/pick_all.png")

# โหลดรูปป้ายชื่อ
tree_templates = []
for name in ["tree_rubber.png", "tree_rubber2.png", "tree_target.png"]:
    img = load_img(f"icons/{name}")
    if img is not None:
        tree_templates.append(img)

# พิกัดสำหรับหน้าจอ 1280x720
attack_x, attack_y = 1100, 560

JOY_X = 195
JOY_Y_START = 520
JOY_Y_END   = 350

def walk_straight(duration_ms=800):
    """เดินตรงไปข้างหน้า"""
    swipe(JOY_X, JOY_Y_START, JOY_X, JOY_Y_END, duration=duration_ms)
    time.sleep(0.2)

def has_target_name(screen):
    """ตรวจดูว่ามีป้ายชื่อต้นไม้ปรากฏขึ้นบนจอหรือไม่"""
    if screen is None:
        return False
    # 1. ตรวจจากไฟล์ภาพ Template
    for tpl in tree_templates:
        if match(screen, tpl, thr=0.42) is not None:
            return True
    # 2. ตรวจจากข้อความสีขาวบริเวณชื่อเป้าหมาย (กลางจอด้านบน)
    roi = screen[120:180, 480:800]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 45, 255]))
    return np.sum(mask > 0) > 100

def has_hp_bar(screen):
    """
    ตรวจจับหลอดสีข้างล่างชื่อต้นไม้ (โซน Y: 180-220, X: 490-790)
    วัดค่าพิกเซลที่มีสีสด (เขียว, เหลือง, แดง, ส้ม)
    """
    if screen is None:
        return False
    bar_roi = screen[180:220, 490:790]
    hsv = cv2.cvtColor(bar_roi, cv2.COLOR_BGR2HSV)
    # คัดกรองพิกเซลที่มีความสดของสี (ไม่ใช่พื้นหลังโปร่งหรือสีเทา)
    mask = cv2.inRange(hsv, np.array([0, 70, 70]), np.array([180, 255, 255]))
    return np.sum(mask > 0) > 150

def collect_items():
    """กดเก็บของด้วยรูปมือหรือ Pick All จนกว่าจะหมด"""
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
            # พิกัดปุ่มเก็บของกลางจอขวาเผื่อรูปไม่ติด
            tap(820, 430)
        time.sleep(0.7)

def main():
    print("=" * 50)
    print("[*] UtopiaFarmAI: ตีเช็กชื่อ -> ฟันจนหลอดสีหาย -> กดรูปมือ")
    print("=" * 50)
    time.sleep(2)

    while True:
        screen = get_screen()
        if screen is None:
            time.sleep(0.3)
            continue

        # 1. เช็กปุ่ม Pick All ก่อนเสมอ
        pos_pick = match(screen, tpl_pick, thr=0.55)
        if pos_pick:
            tap(pos_pick[0], pos_pick[1])
            time.sleep(1.0)
            continue

        # 2. ตีกระตุ้น 1 ครั้ง
        tap(attack_x, attack_y)
        time.sleep(0.4)

        # 3. แคปหน้าจอเพื่อตรวจชื่อ
        scr_chk = get_screen()

        # 4. เมื่อขึ้นชื่อเป้าหมาย
        if has_target_name(scr_chk):
            print("[✓] ขึ้นชื่อต้นไม้แล้ว! เริ่มต้นฟันต่อเนื่องจนกว่าหลอดสีจะหาย...")
            
            # ลูปฟันต่อเนื่องจนกว่าหลอดสีข้างล่างจะหายไป
            while True:
                tap(attack_x, attack_y)
                time.sleep(1.05)

                scr_chopping = get_screen()
                
                # เช็กว่าหลอดสีข้างล่างหายไปหรือยัง หรือมีรูปมือ/Pick All ขึ้นมา
                hp_alive = has_hp_bar(scr_chopping)
                p_hand = match(scr_chopping, tpl_hand, thr=0.48)
                p_pick = match(scr_chopping, tpl_pick, thr=0.52)

                # ถ้าหลอดสีหายไป หรือตรวจพบรูปมือ/Pick All แสดงว่าต้นไม้ล้มแล้ว
                if not hp_alive or p_hand or p_pick:
                    print("[+] หลอดสีข้างล่างหายไปแล้ว! สั่งหยุดตี")
                    time
