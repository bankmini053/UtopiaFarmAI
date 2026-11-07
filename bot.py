import os, time, subprocess, cv2
import numpy as np

ROOT_SU = "/system/xbin/su"
TMP_PATH = "/data/local/tmp/screen.png"

def tap(x, y):
    subprocess.run(f"{ROOT_SU} -c 'input tap {x} {y}'", shell=True)

def get_screen():
    # แคปเจอร์ลง tmp file โดยตรง แก้ปัญหาข้อมูลภาพเสีย (None)
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
    if os.path.exists(path):
        img = cv2.imread(path)
        if img is not None:
            return img
    print(f"[!] ไม่พบไฟล์: {path}")
    return None

tpl_pick   = load_img("icons/pick_all.png")
tpl_hand   = load_img("icons/pickup_hand.png")
tpl_rubber = load_img("icons/tree_rubber.png")
tpl_weight = load_img("icons/weight_full.png")
tpl_home   = load_img("icons/hearthstone.png")

print("[*] UtopiaFarmAI พร้อมทำงาน (จอ 1280x720)...")
time.sleep(2)

# พิกัดปุ่มดาบตัดไม้ขวาล่างสำหรับจอ 1280x720
attack_x = 1100
attack_y = 560

while True:
    screen = get_screen()
    if screen is None:
        time.sleep(0.3)
        continue

    # 1. เช็ก Pick All เก็บของ
    pos_pick = match(screen, tpl_pick, 0.60)
    if pos_pick:
        print("[+] พบ Pick All -> กดเก็บของ")
        tap(pos_pick[0], pos_pick[1])
        time.sleep(1.0)
        continue

    # 2. เช็กเป้าหมาย Rubber Tree
    pos_rubber = match(screen, tpl_rubber, 0.50)
    if pos_rubber:
        print("[+] ตัด Rubber Tree...")
        tap(attack_x, attack_y)
        time.sleep(1.2)
        continue

    # 3. เช็กปุ่มรูปมือ
    pos_hand = match(screen, tpl_hand, 0.50)
    if pos_hand:
        print("[+] แตะปุ่มรูปมือ")
        tap(pos_hand[0], pos_hand[1])
        time.sleep(0.8)
        continue

    # แตะปุ่มฟัน 1 ครั้งเพื่อกระตุ้นเป้าหมาย
    tap(attack_x, attack_y)
    time.sleep(0.8)
  
