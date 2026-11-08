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
    """อ่านพิกัดตำแหน่งตัวละครจากมุมซ้ายบน"""
    if screen is None:
        return None
    roi = screen[180:300, 40:270]
    if roi.size == 0:
        return None
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    for th in [175, 195, 215]:
        _, thresh = cv2.threshold(resized, th, 255, cv2.THRESH_BINARY)
        matches = re.findall(r"\d{5}", pytesseract.image_to_string(thresh, config=r"--psm 6 -c tessedit_char_whitelist=0123456789,."))
        if len(matches) >= 2:
            try:
                x, y = int(matches[0]), int(matches[1])
                return (x, y)
            except:
                pass
    return None

def match_icon(screen, icon_filename, threshold=0.72, roi=None):
    """ค้นหาไอคอนจากโฟลเดอร์ icons/"""
    if screen is None:
        return False, 0, 0, 0.0

    icon_path = os.path.join(ICONS_DIR, icon_filename)
    if not os.path.exists(icon_path):
        return False, 0, 0, 0.0

    template = cv2.imread(icon_path)
    if template is None:
        return False, 0, 0, 0.0

    target_img = screen
    offset_x, offset_y = 0, 0
    if roi is not None:
        y1, y2, x1, x2 = roi
        target_img = screen[y1:y2, x1:x2]
        offset_x, offset_y = x1, y1

    th, tw = template.shape[:2]
    sh, sw = target_img.shape[:2]
    if th > sh or tw > sw:
        return False, 0, 0, 0.0

    res = cv2.matchTemplate(target_img, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    if max_val >= threshold:
        center_x = offset_x + max_loc[0] + (tw // 2)
        center_y = offset_y + max_loc[1] + (th // 2)
        return True, center_x, center_y, max_val

    return False, 0, 0, max_val

def check_hand_icon(screen):
    """ตรวจจับปุ่มรูปมือจากไอคอนและค่าแสง"""
    if os.path.exists(ICONS_DIR):
        for name in os.listdir(ICONS_DIR):
            if "hand" in name.lower():
                found, x, y, conf = match_icon(screen, name, threshold=0.70, roi=(320, 460, 960, 1140))
                if found:
                    return True, x, y, f"ไอคอน {name} ({conf:.2f})"

    # ระบบสำรองวัดพิกเซลสีขาวเหนือปุ่มดาบ
    hand_roi = screen[350:440, 1000:1090]
    gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
    if np.sum(gray > 200) > 220:
        return True, 1045, 390, "ตรวจจับปุ่มขาว"

    return False, 0, 0, ""

def analyze_target(screen):
    """
    วิเคราะห์วัตถุในจอ:
    - ตรวจหาไอคอนใน icons/
    - หรือตรวจหากล่องสมบัติ / ต้นไม้จากเฉดสีและขนาด
    คืนค่า: (found, target_name, center_x, area, height)
    """
    if screen is None:
        return False, "", 0, 0, 0

    # 1. เช็กไอคอนในโฟลเดอร์ก่อน
    if os.path.exists(ICONS_DIR):
        for fname in os.listdir(ICONS_DIR):
            if not "hand" in fname.lower() and fname.endswith((".png", ".jpg")):
                found, ix, iy, conf = match_icon(screen, fname, threshold=0.68, roi=(60, 520, 100, 1180))
                if found:
                    return True, f"ไอคอน {fname}", ix, 5000, 150

    # 2. ตรวจหากล่องสมบัติ (สีส้ม/น้ำตาลทอง)
    roi = screen[60:550, 100:1180]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask_chest = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([5, 60, 60]), np.array([25, 255, 255])),
        cv2.inRange(hsv, np.array([4, 20, 20]), np.array([28, 180, 140]))
    )
    clean_chest = cv2.morphologyEx(mask_chest, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours_c, _ = cv2.findContours(clean_chest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_c:
        a = cv2.contourArea(cnt)
        if a > 750:
            x, y, w, h = cv2.boundingRect(cnt)
            if 0.5 <= float(cw:=w) / h <= 2.2:
                return True, "กล่องสมบัติ (Treasure Chest)", x + 100 + (w // 2), a, h

    # 3. ตรวจหาต้นไม้ (สีเนื้อไม้และใบ)
    mask_tree = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([8, 30, 30]), np.array([28, 220, 180])),
        cv2.inRange(hsv, np.array([32, 60, 40]), np.array([75, 255, 200]))
    )
    clean_tree = cv2.morphologyEx(mask_tree, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours_t, _ = cv2.findContours(clean_tree, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_t:
        a = cv2.contourArea(cnt)
        if a > 900:
            x, y, w, h = cv2.boundingRect(cnt)
            if h >= 75:
                return True, "ต้นไม้ (Tree)", x + 100 + (w // 2), a, h

    return False, "", 0, 0, 0

JOY_CENTER_X = 250
JOY_CENTER_Y = 530
JOY_RADIUS   = 145

def walk_direction(deg=0, duration_ms=450):
    """ดันจอยสติ๊กตามองศา (0=หน้า, 90=ขวา, 180=หลัง, 270=ซ้าย หรือเฉียง)"""
    rad = math.radians(deg)
    tx = int(JOY_CENTER_X + JOY_RADIUS * math.sin(rad))
    ty = int(JOY_CENTER_Y - JOY_RADIUS * math.cos(rad))
    swipe_screen(JOY_CENTER_X, JOY_CENTER_Y, tx, ty, duration=duration_ms)
    time.sleep(0.15)

def rotate_view_right(dist=180):
    cx, cy = 850, 240
    swipe_screen(cx, cy, cx - dist, cy, duration=240)
    time.sleep(0.4)

def main():
    print("=" * 60)
    print("[*] UtopiaFarmAI: เจอดักหยุด 5 วิ -> ระบุวัตถุ+พิกัด -> เดินเช็กขนาดใหญ่ขึ้น")
    print(f"[*] โฟลเดอร์ไอคอน: {ICONS_DIR}")
    print("=" * 60)
    time.sleep(2)

    while True:
        target_found = False
        target_name = ""
        initial_area = 0

        # 1. กวาดสายตา 360 องศา (8 ทิศทาง)
        print("\n[*] กวาดสายตา 360 องศาค้นหาเป้าหมาย...")
        for step in range(8):
            scr = get_screen()
            found, name, cx, area, h = analyze_target(scr)
            if found:
                target_found = True
                target_name = name
                initial_area = area
                print(f"\n[🎯] ตรวจพบเป้าหมายที่มุมมอง {step + 1}/8!")
                break
            else:
                print(f"[🔄] สแกนมุม {step + 1}/8 -> หมุนขวาต่อ...")
                rotate_view_right(dist=180)

        if not target_found:
            print("[!] ไม่พบเป้าหมายรอบตัว -> สุ่มก้าวเดินเพื่อเปลี่ยนมุม...")
            walk_direction(0, duration_ms=900)
            continue

        # 2. หยุดนิ่ง 5 วินาที พร้อมวิเคราะห์สิ่งที่เห็นและอ่านพิกัด
        print("[🛑] สั่งหยุดนิ่งสนิท 5 วินาทีเพื่อวิเคราะห์...")
        for sec in range(5, 0, -1):
            print(f"[*] หยุดรอ... เหลือ {sec} วิ")
            time.sleep(1.0)

        analysis_scr = get_screen()
        coords = extract_coords(analysis_scr)
        _, final_name, _, base_area, base_h = analyze_target(analysis_scr)

        print("\n" + "=" * 50)
        print(f"[🔍] ผลการวิเคราะห์หน้าจอ:")
        print(f"  - สิ่งที่เห็นตรงหน้า : {final_name if final_name else target_name}")
        print(f"  - พิกัดตำแหน่งปัจจุบัน : {coords if coords else 'ไม่สามารถระบุได้'}")
        print(f"  - ขนาดเริ่มต้นของวัตถุ : {base_area:.0f} พิกเซล (สูง {base_h}px)")
        print("=" * 50)

        # 3. เริ่มเดินเข้าหา: เช็กขนาดใหญ่ขึ้น = ถูกทาง / ถ้าไม่ใหญ่ = สุ่มองศาวงล้อ
        current_deg = 0
        last_area = base_area
        success = False

        print("\n[↑] เริ่มต้นเดินเข้าหาเป้าหมาย...")
        for step in range(16):
            # ตรวจสอบปุ่มรูปมือก่อนทุกครั้ง
            curr_scr = get_screen()
            hand_ok, hx, hy, hinfo = check_hand_icon(curr_scr)
            if hand_ok:
                print(f"\n[🎯] พบปุ่มรูปมือขึ้นมาแล้ว! ({hinfo}) ที่พิกัด ({hx}, {hy})")
                tap(hx, hy)
                time.sleep(0.3)
                tap(hx, hy)
                success = True
                break

            # เดินตามองศาปัจจุบัน
            walk_direction(current_deg, duration_ms=450)
            time.sleep(0.15)

            # ตรวจจับขนาดวัตถุหลังก้าวเดิน
            step_scr = get_screen()
            found_step, _, _, new_area, new_h = analyze_target(step_scr)

            if found_step:
                print(f"[*] ก้าวที่ {step + 1} (มุม {current_deg}°) | ขนาด: {new_area:.0f} (เดิม: {last_area:.0f})")

                # ถ้าขนาดใหญ่ขึ้นชัดเจน แสดงว่าเดินถูกทางแล้ว
                if new_area > last_area * 1.05:
                    print(f"[✓] ถูกทางแล้ว! เป้าหมายขยายใหญ่ขึ้น -> เดินหน้ามุมนี้ต่อ")
                    last_area = new_area
                # ถ้าวัตถุเริ่มเล็กลง แสดงว่าเดินเฉไฉ -> สุ่มปรับองศาวงล้อเดินใหม่
                elif new_area < last_area * 0.85:
                    current_deg = random.choice([330, 30, 300, 60, 0])
                    print(f"[⚠️] เป้าหมายเล็กลง สุ่มปรับมุมเดินใหม่เป็น: {current_deg}°")
                else:
                    last_area = max(last_area, new_area)

                # ถ้าวัตถุใหญ่เต็มจอ (ระยะประชิด)
                if new_area >= 12000 or new_h >= 240:
                    print(f"\n[🎯] เป้าหมายขยายใหญ่เต็มจอแล้ว (ถึงตัวเป้าหมาย)!")
                    time.sleep(0.3)
                    # ลองแตะปุ่มรูปมือหรือปุ่มโจมตี
                    final_hand, fx, fy, _ = check_hand_icon(get_screen())
                    if final_hand:
                        tap(fx, fy)
                    else:
                        tap(1100, 530)
                    success = True
                    break
            else:
                # ถ้าหลุดเฟรม สุ่มองศาวงล้อหมุนกลับมารับ
                current_deg = random.choice([45, 315, 90, 270])
                print(f"[?] วัตถุคลาดเคลื่อน สุ่มมุมค้นหาใหม่: {current_deg}°")

        if success:
            print("\n[✓] ทำภารกิจกับเป้าหมายสำเร็จ!")
        else:
            print("\n[!] ครบจำนวนก้าวแล้ว -> เริ่มรอบสแกนใหม่")

        time.sleep(1.5)

if __name__ == "__main__":
    main()
