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

SCREEN_W = 1280
HFOV_DEG = 60
DEADZONE_PX = 40
TURN_THRESHOLD_DEG = 22
TURN_GAIN = 0.8
CAMERA_PX_PER_DEG = 4.0


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


def match_texture(target_img, icon_filename, threshold=0.55):
    """ฟังก์ชันจับคู่ลวดลายกับภาพต้นแบบใน icons/"""
    icon_path = os.path.join(ICONS_DIR, icon_filename)
    if not os.path.exists(icon_path):
        return False, 0.0

    template = cv2.imread(icon_path)
    if template is None:
        return False, 0.0

    th, tw = template.shape[:2]
    sh, sw = target_img.shape[:2]

    if th >= sh or tw >= sw:
        scale = min(sh / th, sw / tw) * 0.8
        template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
        th, tw = template.shape[:2]

    if th < 10 or tw < 10 or th >= sh or tw >= sw:
        return False, 0.0

    res = cv2.matchTemplate(target_img, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return (True, max_val) if max_val >= threshold else (False, max_val)


def optical_zoom_analysis(screen, target_x, target_y=260):
    """
    ระบบซูมสายตาบอท (Digital Zoom In/Out) โดยหน้าจอเกมไม่ขยับ
    - ดึงเฉพาะโซนเป้าหมายมาซูม 3 ระดับ (1.0x, 1.4x, 1.8x)
    - ปรับแสงด้วย CLAHE เพื่อตัดแสงสะท้อนรบกวน
    """
    if screen is None:
        return False, "ไม่พบภาพ"

    h, w = screen.shape[:2]
    box_w, box_h = 320, 360
    x1 = max(0, target_x - (box_w // 2))
    x2 = min(w, target_x + (box_w // 2))
    y1 = max(40, target_y - (box_h // 2))
    y2 = min(h - 80, target_y + (box_h // 2))

    cropped_view = screen[y1:y2, x1:x2]
    if cropped_view.size == 0:
        return False, "พิกัดหลุดขอบจอ"

    # ปรับแสงตัดเงาและแสงสะท้อนด้วย CLAHE
    lab = cv2.cvtColor(cropped_view, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    zoom_scales = [1.0, 1.4, 1.8]
    tree_icons = ["tree_name1.png", "tree_coconut.png", "tree_coconut2.png", "tree_rubber.png"]

    for sc in zoom_scales:
        scaled_img = cv2.resize(enhanced, (0, 0), fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
        for icon in tree_icons:
            matched, conf = match_texture(scaled_img, icon, threshold=0.52)
            if matched:
                return True, f"ซูม {sc:.1f}x พบลวดลาย {icon} (ความแม่นยำ: {conf:.2f})"

    # ระบบวิเคราะห์สีลำต้นผ่านเลนส์ซูม (HSV Bark Masking)
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    lower_bark = np.array([5, 30, 25])
    upper_bark = np.array([28, 220, 160])
    mask = cv2.inRange(hsv, lower_bark, upper_bark)
    bark_pixels = np.sum(mask > 0)

    if bark_pixels > 2500:
        return True, f"ซูมส่องทะลุแสง พบลวดลายลำต้นไม้แท้ ({bark_pixels} px)"

    return False, "แสงสะท้อนหรือวัตถุไม่ตรงลวดลาย"


def check_pickup_hand(screen):
    """ตรวจจับปุ่ม pickup_hand.png สำหรับเก็บไอเทม"""
    icon_path = os.path.join(ICONS_DIR, "pickup_hand.png")
    if os.path.exists(icon_path):
        tpl = cv2.imread(icon_path)
        if tpl is not None:
            roi = screen[300:480, 950:1160]
            if roi.shape[0] > tpl.shape[0] and roi.shape[1] > tpl.shape[1]:
                res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
                _, val, _, loc = cv2.minMaxLoc(res)
                if val >= 0.60:
                    return True, 950 + loc[0] + (tpl.shape[1] // 2), 300 + loc[1] + (tpl.shape[0] // 2), f"pickup_hand ({val:.2f})"

    hand_roi = screen[350:440, 1000:1090]
    gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
    if np.sum(gray > 200) > 220:
        return True, 1045, 390, "วัดความสว่างปุ่มขาว"

    return False, 0, 0, ""


def detect_target_tree(screen):
    """สแกนค้นหาตำแหน่งคร่าวๆ รอบตัว 360 องศา"""
    if screen is None:
        return False, "ไม่พบ", 0, 0, 0

    roi = screen[40:540, 80:1200]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_bark = np.array([5, 35, 30])
    upper_bark = np.array([28, 220, 160])
    mask = cv2.inRange(hsv, lower_bark, upper_bark)
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_x, max_area, best_h = 0, 0, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            if h >= 70 and area > max_area:
                max_area = area
                best_h = h
                best_x = x + 80 + (w // 2)

    if max_area > 0:
        return True, "รูปทรงลำต้นไม้", best_x, max_area, best_h

    return False, "ไม่พบ", 0, 0, 0


def check_target_name_match(screen):
    """ตรวจจับป้ายชื่อและแถบเลือด"""
    if screen is None:
        return False, "ไม่พบชื่อ"

    for r_name in ["tree_rubber.png", "tree_rubber2.png", "tree_hp_empty.png"]:
        icon_path = os.path.join(ICONS_DIR, r_name)
        if os.path.exists(icon_path):
            tpl = cv2.imread(icon_path)
            if tpl is not None:
                roi = screen[60:240, 380:900]
                if roi.shape[0] > tpl.shape[0] and roi.shape[1] > tpl.shape[1]:
                    res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
                    _, val, _, _ = cv2.minMaxLoc(res)
                    if val >= 0.58:
                        return True, f"ป้ายชื่อตรง {r_name} ({val:.2f})"

    bar_roi = screen[70:220, 420:860]
    hsv = cv2.cvtColor(bar_roi, cv2.COLOR_BGR2HSV)
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 120, 100]), np.array([14, 255, 255])),
        cv2.inRange(hsv, np.array([168, 120, 100]), np.array([180, 255, 255]))
    )
    if np.sum(mask_red > 0) > 55:
        return True, "พบแถบเลือดเป้าหมาย"

    gray = cv2.cvtColor(bar_roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh, lang="eng+tha", config=r"--psm 7").strip()
    if any(k in text.lower() for k in ["rubber", "tree", "palm", "coconut", "ยาง", "ต้นไม้"]):
        return True, f"OCR พบชื่อ: '{text}'"

    return False, text if text else "ไม่มีชื่อ"


JOY_CENTER_X = 250
JOY_CENTER_Y = 530
JOY_RADIUS = 145


def walk_direction(deg=0, duration_ms=450):
    rad = math.radians(deg)
    tx = int(JOY_CENTER_X + JOY_RADIUS * math.sin(rad))
    ty = int(JOY_CENTER_Y - JOY_RADIUS * math.cos(rad))
    swipe_screen(JOY_CENTER_X, JOY_CENTER_Y, tx, ty, duration=duration_ms)
    time.sleep(0.1)


def rotate_view_right(dist=180):
    cx, cy = 850, 240
    swipe_screen(cx, cy, cx - dist, cy, duration=240)
    time.sleep(0.4)


def pixel_to_angle(target_x):
    cx = SCREEN_W / 2
    off = target_x - cx
    if abs(off) <= DEADZONE_PX:
        return 0.0
    f = cx / math.tan(math.radians(HFOV_DEG / 2))
    return math.degrees(math.atan(off / f))


def turn_camera(angle_deg):
    px = int(min(abs(angle_deg) * TURN_GAIN * CAMERA_PX_PER_DEG, 550))
    if px < 15:
        return
    cx, cy = 850, 240
    if angle_deg > 0:
        swipe_screen(cx, cy, cx - px, cy, duration=220)
    else:
        swipe_screen(cx, cy, cx + min(px, 360), cy, duration=220)
    time.sleep(0.35)


def calculate_walk_angle(target_x):
    ang = pixel_to_angle(target_x)
    if ang < -1.5:
        label = "เอียงซ้าย"
    elif ang > 1.5:
        label = "เอียงขวา"
    else:
        label = "ตรง"
    return ang % 360, f"{label} {ang:+.1f}°"


def main():
    print("=" * 60)
    print("[*] UtopiaFarmAI: เจอดักหยุด -> Digital Optical Zoom วิเคราะห์ลวดลาย -> เดินแนบชิด")
    print(f"[*] โฟลเดอร์ต้นแบบ: {ICONS_DIR}")
    print("=" * 60)
    time.sleep(2)

    while True:
        # 1. กวาดสายตา 360 องศา (8 มุม)
        print("\n[*] กวาดสายตา 360 องศาค้นหาเป้าหมายรอบตัว...")
        candidate_found = False
        target_x = 0

        for step in range(8):
            scr = get_screen()
            found, src, cx, area, h = detect_target_tree(scr)
            if found:
                candidate_found = True
                target_x = cx
                print(f"\n[🌲] พบจุดน่าสงสัยที่มุม {step + 1}/8! (X={cx}, สูง={h}px)")
                break
            else:
                print(f"[🔄] สแกนมุม {step + 1}/8 -> กำลังหมุนขวาต่อ...")
                rotate_view_right(dist=180)

        if not candidate_found:
            print("\n[!] ล้มเหลว: กวาดหา 360 องศาแล้วไม่พบต้นไม้ -> หยุดการทำงาน")
            return

        # 2. ล็อกจอภาพนิ่งสนิท แล้วเปิดโหมด "Digital Optical Zoom"
        print("[🛑] ล็อกหน้าจอนิ่งสนิท ไม่ขยับมุมมอง...")
        time.sleep(1.0)
        print("[🔬] เปิดโหมดซูมสายตา (Digital Zoom In/Out) ตัดแสงรบกวนและส่องลวดลาย...")

        zoom_scr = get_screen()
        is_real_tree, zoom_detail = optical_zoom_analysis(zoom_scr, target_x)
        print(f"[*] ผลการวิเคราะห์ผ่านเลนส์ซูม: {zoom_detail}")

        if not is_real_tree:
            print("[⚠️] ปฏิเสธเป้าหมาย: เป็นแสงสะท้อนหรือสิ่งกีดขวาง -> สแกนหาต่อ")
            rotate_view_right(dist=180)
            continue

        print("[✓] ยืนยันต้นไม้แท้ 100%! เตรียมเดินตรงเข้าหา...")
        time.sleep(1.5)

        # 3. เดินเข้าหาแบบ Real-time: เล็งทิศและเดินจนแนบชิดเนื้อไม้จริง
        reached_close = False
        for step in range(20):
            curr_scr = get_screen()

            if check_pickup_hand(curr_scr)[0]:
                print(f"\n[🎯] ถึงระยะประชิดตัวละครแล้ว (พบปุ่มเก็บของ)!")
                reached_close = True
                break

            found_step, _, cur_x, cur_area, cur_h = detect_target_tree(curr_scr)
            if found_step:
                ang = pixel_to_angle(cur_x)
                if abs(ang) > TURN_THRESHOLD_DEG:
                    print(f"[*] ก้าวที่ {step + 1} | เป้าเบี่ยง {ang:+.1f}° -> หมุนกล้องเข้าหา")
                    turn_camera(ang)
                    curr_scr = get_screen()
                    found_step, _, cur_x, cur_area, cur_h = detect_target_tree(curr_scr)
                    if not found_step:
                        continue

                walk_deg, direction_label = calculate_walk_angle(cur_x)
                print(f"[*] ก้าวที่ {step + 1} | เดิน: {direction_label} ({walk_deg:.1f}°) | สูง={cur_h}px | ขนาด={cur_area:.0f}")

                # แนบชิดตัวละครจริง
                if cur_h >= 360 or cur_area >= 28000:
                    print(f"\n[🎯] ลำต้นแนบชิดตัวละครแล้ว! (สูง: {cur_h}px, ขนาด: {cur_area:.0f})")
                    reached_close = True
                    break

                walk_direction(walk_deg, duration_ms=460)
                time.sleep(0.1)
            else:
                walk_direction(0, duration_ms=350)
                time.sleep(0.1)

        # 4. ทดสอบฟัน 1 ครั้งเพื่อเช็กชื่อ
        name_verified = False
        for verify_attempt in range(4):
            print(f"\n[🪓] ทดสอบกดตีครั้งที่ {verify_attempt + 1} เพื่อเช็กลวดลายป้ายชื่อ...")
            tap(1100, 530)
            time.sleep(0.65)

            target_scr = get_screen()
            name_matched, name_detail = check_target_name_match(target_scr)
            print(f"[*] ผลตรวจป้ายชื่อ: {name_detail}")

            if name_matched:
                name_verified = True
                break
            else:
                print("[?] ยังไม่พบลวดลายชื่อเป้าหมาย -> ขยับเดินชนเข้าไปอีก 1 ก้าว...")
                walk_direction(0, duration_ms=450)
                time.sleep(0.2)

        # 5. ฟันต่อเนื่องจนเลือดหมด และกดเก็บของ
        if name_verified:
            print("[✓] ยืนยันลวดลายชื่อถูกต้อง! ฟันต่อเนื่องจนกว่าหลอดเลือดจะหมด...")
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
            print("\n[!] ยกเลิก: ไม่พบลวดลายชื่อเป้าหมาย (ป้องกันการตีอากาศ/ดิน)")

        time.sleep(1.5)


if __name__ == "__main__":
    main()
