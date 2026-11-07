import os
import time
import subprocess
import cv2
import numpy as np

ROOT_SU = "/system/xbin/su"
TMP_PATH = "/data/local/tmp/screen.png"

def tap(x, y):
    subprocess.run(f"{ROOT_SU} -c 'input tap {x} {y}'", shell=True)

def get_screen():
    subprocess.run(f"{ROOT_SU} -c 'screencap -p {TMP_PATH}'", shell=True)
    if os.path.exists(TMP_PATH):
        return cv2.imread(TMP_PATH)
    return None

def match(screen, template, thr=0.45):
    """จับคู่ภาพ template (ลด threshold เหลือ 0.45 ให้อ่านป้ายชื่อติดง่ายขึ้น)"""
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

# โหลดรูปป้ายชื่อ Rubber Tree ทุกเวอร์ชัน
tree_templates = []
for name in ["tree_rubber.png", "tree_rubber2.png", "tree_target.png"]:
    img = load_img(f"icons/{name}")
    if img is not None:
        tree_templates.append(img)

def has_target_tree_name(screen):
    """ตรวจดูว่ามีป้ายชื่อ Rubber Tree ขึ้นอยู่บนจอหรือไม่"""
    for tpl in tree_templates:
        if match(screen, tpl, thr=0.45) is not None:
            return True
    return False

# พิกัดสำหรับหน้าจอ 1280x720
attack_x, attack_y = 1100, 560

def main():
    print("=" * 45)
    print(f"[*] โหลดป้ายชื่ออ้างอิงสำเร็จ {len(tree_templates)} รูป")
    print("[*] UtopiaFarmAI: ระบบตรวจป้ายชื่อแล้วฟันรัวๆ พร้อมทำงาน...")
    print("=" * 45)
    time.sleep(2)

    while True:
        screen = get_screen()
        if screen is None:
            time.sleep(0.3)
            continue

        # 1. ถ้ามีของตกหรือขึ้น Pick All ให้กดเก็บของก่อน
        pos_pick = match(screen, tpl_pick, thr=0.55)
        if pos_pick:
            print("[+] พบ Pick All -> กดเก็บของ")
            tap(pos_pick[0], pos_pick[1])
            time.sleep(1.0)
            continue

        # 2. ตีกระตุ้น 1 ครั้งเพื่อเรียก/ต่ออายุแถบชื่อเป้าหมาย
        tap(attack_x, attack_y)
        time.sleep(0.5)

        # 3. แคปภาพหน้าจอตรวจป้ายชื่อ
        scr_check = get_screen()
        
        # 4. ถ้าเห็นป้ายชื่อ Rubber Tree ให้ฟันรัวๆ จนกว่าจะล้ม
        if has_target_tree_name(scr_check):
            print("[✓] เห็นชื่อ Rubber Tree แล้ว! เริ่มต้นฟันต่อเนื่องเรื่อยๆ...")
            
            # ลูปฟันต่อเนื่องแบบไม่หยุด
            while True:
                tap(attack_x, attack_y)
                time.sleep(1.05)

                scr_chopping = get_screen()
                p_hand = match(scr_chopping, tpl_hand, thr=0.50)
                p_pick = match(scr_chopping, tpl_pick, thr=0.55)

                # ถ้าต้นไม้ล้ม: มีรูปมือ หรือ Pick All ขึ้นมา
                if p_hand or p_pick:
                    print("[+] ต้นไม้ล้มแล้ว! กดเก็บของทันที")
                    if p_hand:
                        tap(p_hand[0], p_hand[1])
                    if p_pick:
                        tap(p_pick[0], p_pick[1])
                    time.sleep(1.2)
                    break

                # ถ้าป้ายชื่อหายไปแล้ว (ต้นไม้อาจจะพังหรือหลุดโฟกัส) ให้หลุดลูปเพื่อรอเช็กใหม่
                if not has_target_tree_name(scr_chopping):
                    print("[!] ป้ายชื่อหายไป -> หยุดฟันเพื่อสแกนใหม่")
                    break
        else:
            # ถ้ายังไม่เห็นชื่อ ให้เว้นจังหวะนิดเดียวแล้ววนลูปใหม่
            time.sleep(0.4)

if __name__ == "__main__":
    main()
  
