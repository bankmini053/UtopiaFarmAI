import os
import re
import time
import math
import random
import subprocess
import cv2
import numpy as np
import pytesseract

ROOT_SU = "/system/xbin/su"
TMP_PATH = "/data/local/tmp/screen.png"
ICONS_DIR = os.path.expanduser("~/UtopiaFarmAI/icons")

def run_cmd(cmd):
    subprocess.run(f"{ROOT_SU} -c '{cmd}'", shell=True)

def tap(x, y):
    run_cmd(f"input tap {x} {y}")

def swipe_screen(x1, y1, x2, y2, duration=240):
    run_cmd(f"input swipe {x1} {y1} {x2} {y2} {duration}")

def get_screen():
    run_cmd(f"screencap -p {TMP_PATH}")
    if os.path.exists(TMP_PATH):
        img = cv2.imread(TMP_PATH)
        if img is not None:
            h, w = img.shape[:2]
            if h > w:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            return img
    return None

def extract_coords(screen):
    if screen is None:
        return None
    roi = screen[80:260, 580:780]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (0, 0), fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    for th in [160, 185, 210]:
        _, thresh = cv2.threshold(resized, th, 255, cv2.THRESH_BINARY)
        matches = re.findall(r"\d{5}", pytesseract.image_to_string(thresh, config=r"--psm 6 -c tessedit_char_whitelist=0123456789,."))
        if len(matches) >= 2:
            try:
                x, y = int(matches[0]), int(matches[1])
                return (x, y)
            except:
                pass
    return None

def match_icon(screen, icon_filename, threshold=0.68, roi=None):
    """ค้นหาตำแหน่งภาพไอคอนต้นแบบบนหน้าจอ"""
    if screen is None:
        return False, 0, 0, 0, 0, 0.0

    icon_path = os.path.join(ICONS_DIR, icon_filename)
    if not os.path.exists(icon_path):
        return False, 0, 0, 0, 0, 0.0

    template = cv2.imread(icon_path)
    if template is None:
        return False, 0, 0, 0, 0, 0.0

    target_img = screen
    offset_x, offset_y = 0, 0
    if roi is not None:
        y1, y2, x1, x2 = roi
        target_img = screen[y1:y2, x1:x2]
        offset_x, offset_y = x1, y1

    th, tw = template.shape[:2]
    sh, sw = target_img.shape[:2]
    if th > sh or tw > sw:
        return False, 0, 0, 0, 0, 0.0

    res = cv2.matchTemplate(target_img, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    if max_val >= threshold:
        center_x = offset_x + max_loc[0] + (tw // 2)
        center_y = offset_y + max_loc[1] + (th // 2)
        return True, center_x, center_y, tw, th, max_val

    return False, 0, 0, 0, 0, max_val

def check_pickup_hand(screen):
    """ตรวจจับปุ่ม pickup_hand.png สำหรับเก็บของ"""
    # 1. เทียบภาพกับ pickup_hand.png โดยตรง
    found, cx, cy, _, _, val = match_icon(screen, "pickup_hand.png", threshold=0.65, roi=(300, 480, 950, 1160))
    if found:
        return True, cx, cy, f"pickup_hand.png ({val:.2f})"

    # เผื่อชื่อไฟล์ hand.png อื่นๆ
    if os.path.exists(ICONS_DIR):
        for fname in os.listdir(ICONS_DIR):
            if "hand" in fname.lower() and fname != "pickup_hand.png" and fname.endswith((".png", ".jpg")):
                f_ok, fx, fy, _, _, fval = match_icon(screen, fname, threshold=0.65, roi=(300, 480, 950, 1160))
                if f_ok:
                    return True, fx, fy, f"{fname} ({fval:.2f})"

    # ระบบสำรองวัดพิกเซลสีขาวเหนือปุ่มดาบ
    hand_roi = screen[350:440, 1000:1090]
    gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
    if np.sum(gray > 200) > 220:
        return True, 1045, 390, "วัดความสว่างปุ่มขาว"

    return False, 0, 0, ""

def detect_coconut_tree(screen):
    """
    ตรวจหาต้นมะพร้าว (รองรับ tree_coconut.png และ tree_coconut2.png ทั้ง 2 ลาย)
    """
    if screen is None:
        return False, "ไม่พบ", 0, 0, 0

    # 1. สแกนหาจากไฟล์ template ลาย 1 และ ลาย 2
    tree_templates = ["tree_coconut.png", "tree_coconut2.png"]
    for t_name in tree_templates:
        found, cx, cy, tw, th, val = match_icon(screen, t_name, threshold=0.62, roi=(50, 480, 100, 1180))
        if found:
            return True, f"ไอคอน {t_name} ({val:.2f})", cx, tw * th, th

    # 2. ระบบสำรอง: กรองเฉดสีลำต้นมะพร้าว (สีเนื้อไม้/ครีมเทาลายปล้อง)
    roi = screen[50:460, 100:1180]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_bark = np.array([8, 20, 70])
    upper_bark = np.array([32, 180, 230])
    mask = cv2.inRange(hsv, lower_bark, upper_bark)
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_x, max_area, best_h = 0, 0, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1200:
            x, y, w, h = cv2.boundingRect(cnt)
            if h >= 80 and area > max_area:
                max_area = area
                best_h = h
                best_x = x + 100 + (w // 2)

    if max_area > 0:
        return True, "เฉดสีลำต้นมะพร้าว", best_x, max_area, best_h

    return False, "ไม่พบ", 0, 0, 0

def check_target_name_match(screen):
    """ตรวจสอบชื่อเป้าหมายและแถบเลือดหลังฟัน 1 ครั้ง"""
    if screen is None:
        return False, "ไม่พบชื่อ"

    # เทียบจากไฟล์ภาพป้ายชื่อถ้ามีใน icons/
    if os.path.exists(ICONS_DIR):
        for fname in os.listdir(ICONS_DIR):
            if "name" in fname.lower() and fname.endswith((".png", ".jpg")):
                found, _, _, _, _, val = match_icon(screen, fname, threshold=0.68, roi=(70, 220, 420, 860))
                if found:
                    return True, f"รูปป้ายชื่อตรงกับ {fname} ({val:.2f})"

    # OCR ตัวหนังสือบนหัว
    name_roi = screen[70:220, 420:860]
    gray = cv2.cvtColor(name_roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh, lang="eng+tha", config=r"--psm 7").strip()

    keywords = ["tree", "palm", "coconut", "ต้น", "ไม้", "มะพร้าว"]
    if any(k in text.lower() for k in keywords):
        return True, f"ข้อความตรง: '{text}'"

    # เช็กแถบเลือด (Health Bar)
    bar_roi = screen[80:230, 450:830]
    hsv = cv2.cvtColor(bar_roi, cv2.COLOR_BGR2HSV)
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255])),
        cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
    )
    if np.sum(mask_red > 0) > 80:
        return True, f"พบแถบเลือด (ข้อความ: '{text}')"

    return False, text if text else "ไม่มีชื่อ"

JOY_CENTER_X = 250
JOY_CENTER_Y = 530
JOY_RADIUS   = 145

def walk_direction(deg=0, duration_ms=450):
    rad = math.radians(deg)
    tx = int(JOY_CENTER_X + JOY_RADIUS * math.sin(rad))
    ty = int(JOY_CENTER_Y - JOY_RADIUS * math.cos(rad))
    swipe_screen(JOY_CENTER_X, JOY_CENTER_Y, tx, ty, duration=duration_ms)
    time.sleep(0.12)

def rotate_view_right(dist=180):
    cx, cy = 850, 240
    swipe_screen(cx, cy, cx - dist, cy, duration=240)
    time.sleep(0.4)

def main():
    print("=" * 60)
    print("[*] UtopiaFarmAI: ตรวจจับ tree_coconut (ลาย 1 & 2) + pickup_hand.png")
    print(f"[*] โฟลเดอร์ต้นแบบ: {ICONS_DIR}")
    print("=" * 60)
    time.sleep(2)

    while True:
        tree_found = False
        detect_src = ""
        base_area = 0

        # 1. กวาดสายตา 360 องศา (8 ทิศทาง)
        print("\n[*] กวาดสายตา 360 องศาค้นหาต้นมะพร้าว...")
        for step in range(8):
            scr = get_screen()
            found, src, cx, area, h = detect_coconut_tree(scr)
            if found:
                tree_found = True
                detect_src = src
                base_area = area
                print(f"\n[🌴] ตรวจพบต้นมะพร้าวที่มุม {step + 1}/8! [{src}] (ขนาด: {area:.0f}, สูง: {h}px)")
                break
            else:
                print(f"[🔄] สแกนมุม {step + 1}/8 -> หมุนขวาต่อ...")
                rotate_view_right(dist=180)

        if not tree_found:
            print("[!] ไม่พบต้นมะพร้าวรอบตัว -> ก้าวสุ่มเปลี่ยนมุมมอง...")
            walk_direction(0, duration_ms=900)
            continue

        # 2. หยุดนิ่ง 5 วินาทีเพื่อวิเคราะห์ข้อมูล
        print("[🛑] หยุดนิ่งสนิท 5 วินาทีเพื่อวิเคราะห์...")
        for sec in range(5, 0, -1):
            print(f"[*] หยุดรอ... เหลือ {sec} วิ")
            time.sleep(1.0)

        analysis_scr = get_screen()
        coords = extract_coords(analysis_scr)
        _, _, _, start_area, start_h = detect_coconut_tree(analysis_scr)

        print("\n" + "=" * 50)
        print("[🔍] ผลการวิเคราะห์:")
        print(f"  - แหล่งตรวจจับ       : {detect_src}")
        print(f"  - พิกัดตำแหน่งปัจจุบัน : {coords if coords else 'ไม่สามารถระบุได้'}")
        print(f"  - ขนาดลำต้นเริ่มต้น   : {start_area:.0f} พิกเซล (สูง {start_h}px)")
        print("=" * 50)

        # 3. เดินเข้าหาต้นมะพร้าว (เช็กขนาดขยายใหญ่ขึ้น)
        current_deg = 0
        last_area = start_area if start_area > 0 else base_area
        reached_tree = False

        print("\n[↑] เริ่มต้นเดินเข้าหาต้นมะพร้าว...")
        for step in range(16):
            # ตรวจสอบว่าปุ่มรูปมือขึ้นแล้วหรือไม่
            hand_up, hx, hy, h_info = check_pickup_hand(get_screen())
            if hand_up:
                print(f"\n[🎯] พบปุ่มเก็บของขึ้นมาแล้ว! ({h_info}) ที่ ({hx}, {hy})")
                reached_tree = True
                break

            walk_direction(current_deg, duration_ms=450)
            time.sleep(0.12)

            step_scr = get_screen()
            found_step, _, _, new_area, new_h = detect_coconut_tree(step_scr)

            if found_step:
                print(f"[*] ก้าวที่ {step + 1} | ขนาด: {new_area:.0f} (เดิม: {last_area:.0f}) | สูง: {new_h}px")

                if new_area > last_area * 1.05:
                    print("[✓] ถูกทางแล้ว! ลำต้นมะพร้าวขยายใหญ่ขึ้น")
                    last_area = new_area
                elif new_area < last_area * 0.85:
                    current_deg = random.choice([330, 30, 0])
                    print(f"[⚠️] ลำต้นเล็กลง ปรับมุมเดินเป็น: {current_deg}°")
                else:
                    last_area = max(last_area, new_area)

                if new_area >= 9500 or new_h >= 240:
                    print(f"\n[🎯] ลำต้นขยายใหญ่เต็มหน้าจอ -> ถึงระยะประชิดแล้ว!")
                    reached_tree = True
                    break
            else:
                current_deg = random.choice([45, 315, 0])

        if not reached_tree:
            print("[!] ยังไม่ถึงระยะประชิด -> เริ่มสแกนรอบใหม่")
            continue

        # 4. ฟัน 1 ครั้งเพื่อเช็กชื่อเป้าหมาย
        print("\n[🪓] ฟัน 1 ครั้งเพื่อตรวจสอบชื่อเป้าหมาย...")
        tap(1100, 530)
        time.sleep(0.7)

        target_scr = get_screen()
        name_matched, name_detail = check_target_name_match(target_scr)
        print(f"[*] ผลตรวจชื่อเป้าหมาย: {name_detail}")

        if name_matched:
            print("[✓] ชื่อถูกต้อง! ฟันต่อเนื่องจนกว่าหลอดเลือดจะหมด...")
            hits = 0
            while hits < 30:
                hits += 1
                tap(1100, 530)
                time.sleep(0.38)

                if hits % 3 == 0:
                    chk_scr = get_screen()
                    hand_ready, _, _, _ = check_pickup_hand(chk_scr)
                    name_alive, _ = check_target_name_match(chk_scr)
                    if hand_ready or not name_alive:
                        print(f"[✓] ต้นไม้ล้มเรียบร้อยแล้ว (ตีไปทั้งหมด {hits} ครั้ง)")
                        break

            # 5. เก็บของด้วยปุ่ม pickup_hand.png
            time.sleep(0.5)
            for _ in range(4):
                final_scr = get_screen()
                hand_found, hx, hy, hinfo = check_pickup_hand(final_scr)
                if hand_found:
                    print(f"[✓] แตะปุ่มเก็บไอเทม ({hinfo}) ที่พิกัด ({hx}, {hy})...")
                    tap(hx, hy)
                    time.sleep(0.3)
                else:
                    break
        else:
            print(f"[!] ชื่อเป้าหมายไม่ตรง ({name_detail}) -> เปลี่ยนไปหาต้นอื่น")

        time.sleep(1.5)

if __name__ == "__main__":
    main()
