import os
import re
import time
import math
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

def match_icon(screen, icon_filename, threshold=0.68, roi=None):
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
    found, cx, cy, _, _, val = match_icon(screen, "pickup_hand.png", threshold=0.65, roi=(300, 480, 950, 1160))
    if found:
        return True, cx, cy, f"pickup_hand.png ({val:.2f})"

    hand_roi = screen[350:440, 1000:1090]
    gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
    if np.sum(gray > 200) > 220:
        return True, 1045, 390, "วัดความสว่างปุ่มขาว"

    return False, 0, 0, ""

def detect_coconut_tree(screen):
    """ตรวจหาต้นมะพร้าว (tree_coconut.png, tree_coconut2.png หรือ HSV เปลือกไม้)"""
    if screen is None:
        return False, "ไม่พบ", 0, 0, 0

    for t_name in ["tree_coconut.png", "tree_coconut2.png"]:
        found, cx, cy, tw, th, val = match_icon(screen, t_name, threshold=0.62, roi=(50, 480, 100, 1180))
        if found:
            return True, f"ไอคอน {t_name} ({val:.2f})", cx, tw * th, th

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
        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            if h >= 75 and area > max_area:
                max_area = area
                best_h = h
                best_x = x + 100 + (w // 2)

    if max_area > 0:
        return True, "เฉดสีลำต้นมะพร้าว", best_x, max_area, best_h

    return False, "ไม่พบ", 0, 0, 0

def check_target_name_match(screen):
    """ตรวจสอบชื่อเป้าหมายหรือแถบเลือดหลังฟัน"""
    if screen is None:
        return False, "ไม่พบชื่อ"

    if os.path.exists(ICONS_DIR):
        for fname in os.listdir(ICONS_DIR):
            if "name" in fname.lower() and fname.endswith((".png", ".jpg")):
                found, _, _, _, _, val = match_icon(screen, fname, threshold=0.68, roi=(70, 220, 420, 860))
                if found:
                    return True, f"รูปป้ายชื่อตรง {fname} ({val:.2f})"

    name_roi = screen[70:220, 420:860]
    gray = cv2.cvtColor(name_roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh, lang="eng+tha", config=r"--psm 7").strip()

    keywords = ["tree", "palm", "coconut", "ต้น", "ไม้", "มะพร้าว"]
    if any(k in text.lower() for k in keywords):
        return True, f"ข้อความตรง: '{text}'"

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

def walk_direction(deg=0, duration_ms=400):
    rad = math.radians(deg)
    tx = int(JOY_CENTER_X + JOY_RADIUS * math.sin(rad))
    ty = int(JOY_CENTER_Y - JOY_RADIUS * math.cos(rad))
    swipe_screen(JOY_CENTER_X, JOY_CENTER_Y, tx, ty, duration=duration_ms)
    time.sleep(0.1)

def rotate_view_right(dist=180):
    cx, cy = 850, 240
    swipe_screen(cx, cy, cx - dist, cy, duration=240)
    time.sleep(0.4)

def calculate_walk_angle(target_x):
    """
    คำนวณทิศทางการเดินเอียงซ้าย/ขวา/ตรง ตามตำแหน่งภาพ:
    กึ่งกลางจออยู่ที่ X=640
    - target_x < 580: อยู่ซ้าย -> เดินเฉียงซ้าย 330°
    - target_x > 700: อยู่ขวา -> เดินเฉียงขวา 30°
    - 580 <= target_x <= 700: อยู่ตรงหน้า -> เดินตรง 0°
    """
    if target_x < 580:
        return 330, "เอียงซ้าย"
    elif target_x > 700:
        return 30, "เอียงขวา"
    else:
        return 0, "ตรง"

def main():
    print("=" * 60)
    print("[*] UtopiaFarmAI: กวาด 360° (ไม่เจอ=หยุดล้มเหลว) -> เดินเอียงซ้าย/ขวา -> ชิดตัวกดตี")
    print(f"[*] โฟลเดอร์ต้นแบบ: {ICONS_DIR}")
    print("=" * 60)
    time.sleep(2)

    # 1. กวาดสายตา 360 องศา (8 มุม)
    print("\n[*] กวาดสายตา 360 องศาค้นหาเป้าหมายรอบตัว...")
    tree_found = False
    detect_src = ""
    target_x = 0

    for step in range(8):
        scr = get_screen()
        found, src, cx, area, h = detect_coconut_tree(scr)
        if found:
            tree_found = True
            detect_src = src
            target_x = cx
            print(f"\n[🌴] ตรวจพบเป้าหมายที่มุม {step + 1}/8! [{src}] (X={cx}, ขนาด={area:.0f}, สูง={h}px)")
            break
        else:
            print(f"[🔄] สแกนมุม {step + 1}/8 -> กำลังหมุนขวาต่อ...")
            rotate_view_right(dist=180)

    # เงื่อนไขที่ 1: ถ้ากวาดสายตาครบ 360 องศาแล้วไม่เจอ ให้หยุดและบอกล้มเหลวทันที
    if not tree_found:
        print("\n" + "=" * 55)
        print("[!] ล้มเหลว: มองหา 360 องศาแล้วไม่พบเป้าหมาย -> หยุดการทำงานทันที")
        print("=" * 55)
        return

    # 2. หยุดนิ่ง 3 วินาทีเพื่อเคลียร์การเคลื่อนไหว
    print("[🛑] หยุดนิ่ง 3 วินาทีเพื่อเตรียมเดินเข้าหา...")
    time.sleep(3.0)

    # 3. เดินเข้าหาแบบ Real-time: ปรับองศา ตรง / เอียงซ้าย / เอียงขวา และเดินจนชิดตัวละครที่สุด
    print("\n[↑] เริ่มเดินเข้าหาเป้าหมาย (ปรับทิศทาง ตรง/เอียงซ้าย/เอียงขวา อัตโนมัติ)...")
    reached_tree = False

    for step in range(16):
        curr_scr = get_screen()
        
        # ถ้าขึ้นปุ่มมือแสดงว่าชิดติดตัวละครแล้ว
        if check_pickup_hand(curr_scr)[0]:
            print(f"\n[🎯] ถึงระยะประชิดตัวละครแล้ว (พบปุ่มเก็บของ)!")
            reached_tree = True
            break

        found_step, _, cur_x, cur_area, cur_h = detect_coconut_tree(curr_scr)
        if found_step:
            walk_deg, direction_label = calculate_walk_angle(cur_x)
            print(f"[*] ก้าวที่ {step + 1} | เดินทิศ: {direction_label} ({walk_deg}°) | พิกัดวัตถุ X={cur_x} | สูง={cur_h}px")

            # ตรวจสอบว่าชิดตัวละครที่สุดแล้วหรือยัง (ความสูง >= 340px หรือพื้นที่ >= 11000px)
            if cur_h >= 340 or cur_area >= 11000:
                print(f"\n[🎯] วัตถุขยายใหญ่เต็มหน้าจอแนบชิดตัวละครแล้ว! (สูง: {cur_h}px)")
                reached_tree = True
                break

            walk_direction(walk_deg, duration_ms=450)
            time.sleep(0.12)
        else:
            # ถ้าหลุดเฟรมชั่วคราว เดินตรงก้าวสั้น
            walk_direction(0, duration_ms=300)
            time.sleep(0.12)

    # 4. ฟัน 1 ครั้งเพื่อตรวจสอบชื่อ/แถบเลือด ถ้ายังไม่ขึ้น ให้ขยับชิดเข้าไปอีกแล้วตรวจใหม่
    name_verified = False
    for verify_attempt in range(3):
        print(f"\n[🪓] ทดสอบกดตีครั้งที่ {verify_attempt + 1} เพื่อเช็กชื่อเป้าหมาย...")
        tap(1100, 530)
        time.sleep(0.7)

        target_scr = get_screen()
        name_matched, name_detail = check_target_name_match(target_scr)
        print(f"[*] ผลการตรวจชื่อ: {name_detail}")

        if name_matched:
            name_verified = True
            break
        else:
            print("[?] ยังไม่พบชื่อหรือแถบเลือด -> เดินขยับประชิดตัวละครเข้าไปอีก 1 ก้าว...")
            walk_direction(0, duration_ms=350)
            time.sleep(0.2)

    # 5. ฟันจนหมดหลอด และเก็บของ
    if name_verified:
        print("[✓] ยืนยันชื่อถูกต้อง! ฟันต่อเนื่องจนกว่าหลอดเลือดจะหมด...")
        hits = 0
        while hits < 35:
            hits += 1
            tap(1100, 530)
            time.sleep(0.38)

            if hits % 3 == 0:
                chk_scr = get_screen()
                hand_ready, _, _, _ = check_pickup_hand(chk_scr)
                name_alive, _ = check_target_name_match(chk_scr)
                if hand_ready or not name_alive:
                    print(f"[✓] ต้นไม้ล้มเรียบร้อยแล้ว (ฟันไปทั้งหมด {hits} ครั้ง)")
                    break

        # เก็บของด้วย pickup_hand.png
        time.sleep(0.6)
        for _ in range(4):
            final_scr = get_screen()
            hand_found, hx, hy, hinfo = check_pickup_hand(final_scr)
            if hand_found:
                print(f"[✓] แตะปุ่มเก็บไอเทม ({hinfo}) ที่พิกัด ({hx}, {hy})...")
                tap(hx, hy)
                time.sleep(0.3)
            else:
                break
        print("\n[✓] เสร็จสิ้นภารกิจ 1 รอบสมบูรณ์!")
    else:
        print("\n[!] ล้มเหลว: เดินเข้าไปตีแล้วไม่พบชื่อเป้าหมาย")

if __name__ == "__main__":
    main()
