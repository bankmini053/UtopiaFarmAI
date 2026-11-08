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

# --- พารามิเตอร์มุมมองกล้องและการเล็งทิศทาง ---
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


def match_icon(screen, icon_filename, threshold=0.50, roi=None):
    """ค้นหา template icon โดยปรับ threshold ให้อ่อนลงเพื่อรองรับสเกลระยะห่าง"""
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
        # ถ้ารูป template ใหญ่กว่าจอ ให้ย่อลงมาครึ่งหนึ่งเพื่อเทียบ
        scale = min(sh / th, sw / tw) * 0.85
        template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
        th, tw = template.shape[:2]

    res = cv2.matchTemplate(target_img, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    if max_val >= threshold:
        center_x = offset_x + max_loc[0] + (tw // 2)
        center_y = offset_y + max_loc[1] + (th // 2)
        return True, center_x, center_y, tw, th, max_val

    return False, 0, 0, 0, 0, max_val


def check_pickup_hand(screen):
    """ตรวจจับปุ่มเก็บของ pickup_hand.png"""
    found, cx, cy, _, _, val = match_icon(screen, "pickup_hand.png", threshold=0.58, roi=(300, 480, 950, 1160))
    if found:
        return True, cx, cy, f"pickup_hand.png ({val:.2f})"

    hand_roi = screen[350:440, 1000:1090]
    gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)
    if np.sum(gray > 200) > 220:
        return True, 1045, 390, "วัดความสว่างปุ่มขาว"

    return False, 0, 0, ""


def detect_target_tree(screen):
    """
    ระบบสแกนหาต้นไม้ (ครอบคลุมทั้งไฟล์ไอคอน และเฉดสีฉากจริง)
    """
    if screen is None:
        return False, "ไม่พบ", 0, 0, 0

    # 1. เทียบภาพ template จากไฟล์ใน icons/ (ลด threshold เหลือ 0.50)
    for t_name in ["tree_name1.png", "tree_coconut.png", "tree_coconut2.png"]:
        found, cx, cy, tw, th, val = match_icon(screen, t_name, threshold=0.50, roi=(40, 520, 80, 1200))
        if found:
            return True, f"ไอคอน {t_name} ({val:.2f})", cx, tw * th, th

    # 2. ระบบสแกนเฉดสีต้นไม้ (ครอบคลุมทั้งเนื้อไม้สีน้ำตาลเข้ม + เทาเปลือกไม้)
    roi = screen[40:540, 80:1200]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # ช่วงสีกว้าง: น้ำตาลเข้ม, น้ำตาลเทา, เปลือกไม้ทึบ
    lower_bark = np.array([4, 25, 20])
    upper_bark = np.array([32, 230, 180])
    mask = cv2.inRange(hsv, lower_bark, upper_bark)
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_x, max_area, best_h = 0, 0, 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 800:  # ลดขนาดขั้นต่ำลงเพื่อให้มองเห็นต้นไม้ที่อยู่ไกลได้
            x, y, w, h = cv2.boundingRect(cnt)
            # ลำต้นต้องมีความสูงทรงตั้ง
            if h >= 65 and area > max_area:
                max_area = area
                best_h = h
                best_x = x + 80 + (w // 2)

    if max_area > 0:
        return True, "ตรวจพบรูปทรงลำต้นไม้", best_x, max_area, best_h

    return False, "ไม่พบ", 0, 0, 0


def check_target_name_match(screen):
    """ตรวจสอบชื่อเป้าหมาย Rubber Tree และแถบเลือด"""
    if screen is None:
        return False, "ไม่พบชื่อ"

    for r_name in ["tree_rubber.png", "tree_rubber2.png"]:
        found, _, _, _, _, val = match_icon(screen, r_name, threshold=0.55, roi=(60, 240, 380, 900))
        if found:
            return True, f"ป้ายชื่อตรงกับ {r_name} ({val:.2f})"

    bar_roi = screen[70:220, 420:860]
    hsv = cv2.cvtColor(bar_roi, cv2.COLOR_BGR2HSV)
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 120, 100]), np.array([14, 255, 255])),
        cv2.inRange(hsv, np.array([168, 120, 100]), np.array([180, 255, 255]))
    )
    if np.sum(mask_red > 0) > 50:
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
    print("[*] UtopiaFarmAI: กวาด 360° -> เล็งมุมต่อเนื่อง -> ชิดติดเนื้อไม้ -> ฟันเช็กชื่อ")
    print(f"[*] โฟลเดอร์ไอคอน: {ICONS_DIR}")
    print("=" * 60)
    time.sleep(2)

    # 1. กวาดสายตา 360 องศา (8 มุม)
    print("\n[*] กวาดสายตา 360 องศาค้นหาเป้าหมายรอบตัว...")
    tree_found = False
    detect_src = ""

    for step in range(8):
        scr = get_screen()
        found, src, cx, area, h = detect_target_tree(scr)
        if found:
            tree_found = True
            detect_src = src
            print(f"\n[🌲] ตรวจพบเป้าหมายที่มุม {step + 1}/8! [{src}] (X={cx}, ขนาด={area:.0f}, สูง={h}px)")
            break
        else:
            print(f"[🔄] สแกนมุม {step + 1}/8 -> กำลังหมุนขวาต่อ...")
            rotate_view_right(dist=180)

    # ไม่พบเป้าหมาย = หยุดและแจ้งเตือนทันที
    if not tree_found:
        print("\n" + "=" * 55)
        print("[!] ล้มเหลว: มองหา 360 องศาแล้วไม่พบเป้าหมาย -> หยุดการทำงานทันที")
        print("=" * 55)
        return

    # 2. หยุดนิ่ง 3 วินาที
    print("[🛑] หยุดนิ่ง 3 วินาทีเพื่อเตรียมเดินเข้าหา...")
    time.sleep(3.0)

    # 3. เดินเข้าหาแบบ Real-time: เล็งทิศตามมุมจริง หมุนปรับมุมมองเมื่อเบี่ยง และเดินจนแนบชิด
    print("\n[↑] เริ่มเดินเข้าหาเป้าหมาย (เล็งทิศต่อเนื่อง + หมุนกล้องเมื่อเบี่ยงมาก)...")
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

            # เกณฑ์แนบชิดตัวละครจริงแบบรูป 6175: ลำต้นสูงเกิน 360px หรือขนาดกินพื้นที่เกิน 28,000px[span_4](start_span)[span_4](end_span)
            if cur_h >= 360 or cur_area >= 28000:
                print(f"\n[🎯] วัตถุขยายใหญ่เต็มหน้าจอแนบชิดตัวละครแล้ว! (สูง: {cur_h}px, ขนาด: {cur_area:.0f})")
                reached_close = True
                break

            walk_direction(walk_deg, duration_ms=460)
            time.sleep(0.1)
        else:
            walk_direction(0, duration_ms=350)
            time.sleep(0.1)

    # 4. ทดสอบฟัน 1 ครั้งเพื่อเช็กชื่อเป้าหมาย (ถ้าไม่ขึ้น ให้ก้าวชนเข้าไปอีก 1 ก้าวแล้วตีใหม่)
    name_verified = False
    for verify_attempt in range(4):
        print(f"\n[🪓] ทดสอบกดตีครั้งที่ {verify_attempt + 1} เพื่อเช็กชื่อเป้าหมาย...")
        tap(1100, 530)
        time.sleep(0.65)

        target_scr = get_screen()
        name_matched, name_detail = check_target_name_match(target_scr)
        print(f"[*] ผลการตรวจชื่อ: {name_detail}")

        if name_matched:
            name_verified = True
            break
        else:
            print("[?] ยังไม่ขึ้นชื่อ (ขวานยังไม่ถึงเนื้อไม้) -> ขยับเดินชนเข้าไปอีก 1 ก้าว...")
            walk_direction(0, duration_ms=450)
            time.sleep(0.2)

    # 5. ฟันต่อเนื่องจนเลือดหมด และกดเก็บของ
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
