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

def match(screen, template, thr=0.50):
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

# โหลดไอคอนเก็บของ
tpl_hand = load_img("icons/pickup_hand.png")
tpl_pick = load_img("icons/pick_all.png")

# พิกัดสำหรับหน้าจอ 1280x720
attack_x, attack_y = 1100, 560

# พิกัดจอยสติ๊กเดินตรง
JOY_X = 195
JOY_Y_START = 520
JOY_Y_END   = 350

def walk_straight(duration_ms=800):
    """เดินตรงไปข้างหน้า"""
    swipe(JOY_X, JOY_Y_START, JOY_X, JOY_Y_END, duration=duration_ms)
    time.sleep(0.2)

def has_white_name(screen):
    """
    ตรวจจับตัวอักษรสีขาวของป้ายชื่อต้นไม้
    บริเวณกึ่งกลางจอด้านบน (X: 420-860, Y: 100-240)
    """
    if screen is None:
        return False
    
    # ตัดเฉพาะพื้นที่แสดงป้ายชื่อด้านบนกลางจอ
    name_roi = screen[100:240, 420:860]
    hsv = cv2.cvtColor(name_roi, cv2.COLOR_BGR2HSV)
    
    # กรองเฉพาะสีขาวสว่าง (S น้อยมาก, V สูงมาก)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 45, 255])
    
    mask = cv2.inRange(hsv, lower_white, upper_white)
    white_pixels = np.sum(mask > 0)
    
    # ถ้ามีพิกเซลสีขาวรวมกันเป็นตัวหนังสือ (มากกว่า 120 พิกเซล)
    return white_pixels > 120

def main():
    print("=" * 50)
    print("[*] UtopiaFarmAI: ระบบตรวจชื่อตัวอักษรสีขาว -> หยุดเดินแล้วตีรัวๆ")
    print("=" * 50)
    time.sleep(2)

    while True:
        screen = get_screen()
        if screen is None:
            time.sleep(0.3)
            continue

        # 1. เช็กถ้ามีปุ่มเก็บของ Pick All ให้กดทันที
        pos_pick = match(screen, tpl_pick, thr=0.55)
        if pos_pick:
            print("[+] พบปุ่ม Pick All -> กดเก็บของ")
            tap(pos_pick[0], pos_pick[1])
            time.sleep(1.0)
            continue

        # 2. ตีกระตุ้น 1 ครั้ง
        tap(attack_x, attack_y)
        time.sleep(0.4)

        # 3. แคปหน้าจอเพื่อตรวจตัวอักษรสีขาว
        scr_chk = get_screen()

        # 4. ถ้าเห็นตัวหนังสือสีขาวขึ้น -> สั่งหยุดเดิน 100% แล้วยืนตีเรื่อยๆ ทันที
        if has_white_name(scr_chk):
            print("[✓] พบชื่อตัวอักษรสีขาว! สั่งหยุดเดิน -> เริ่มต้นยืนตีเรื่อยๆ...")
            
            no_name_count = 0
            while True:
                # กดตีอย่างเดียว ไม่มีการขยับเดินใดๆ
                tap(attack_x, attack_y)
                time.sleep(1.05)

                scr_hit = get_screen()
                p_hand = match(scr_hit, tpl_hand, thr=0.48)
                p_pick = match(scr_hit, tpl_pick, thr=0.55)

                # ถ้าต้นไม้ล้ม: มีรูปมือ หรือ Pick All ขึ้นมา
                if p_hand or p_pick:
                    print("[+] ต้นไม้ล้มแล้ว! กดเก็บของ...")
                    for _ in range(5):
                        if p_hand:
                            tap(p_hand[0], p_hand[1])
                        if p_pick:
                            tap(p_pick[0], p_pick[1])
                        time.sleep(0.7)
                        
                        s_c = get_screen()
                        p_hand = match(s_c, tpl_hand, thr=0.48)
                        p_pick = match(s_c, tpl_pick, thr=0.55)
                        if not p_hand and not p_pick:
                            break
                    print("[✓] เก็บของเสร็จสิ้น")
                    break

                # ตรวจดูว่าชื่อสีขาวยังอยู่ไหม
                if not has_white_name(scr_hit):
                    no_name_count += 1
                    # ถ้าชื่อหายไปต่อเนื่อง 3 ครั้ง แสดงว่าต้นไม้แตกหรือล้มแล้ว
                    if no_name_count >= 3:
                        print("[!] ชื่อสีขาวหายไปแล้ว -> หยุดตี")
                        break
                else:
                    no_name_count = 0

            # หลังตีเสร็จ ให้เดินตรงไปหาต้นต่อไปข้างหน้า
            walk_straight(1500)
        else:
            # ถ้ายังไม่เห็นชื่อสีขาว ให้เดินขยับเข้าหาอีกนิด
            walk_straight(700)

if __name__ == "__main__":
    main()
