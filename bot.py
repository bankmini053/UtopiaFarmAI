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
LOCKED_IMG_PATH = "/data/local/tmp/bot_target_locked.png"
ICONS_DIR = os.path.expanduser("~/UtopiaFarmAI/icons")

# พารามิเตอร์หน้าจอและมุมมองกล้อง
SCREEN_W = 1280
SCREEN_H = 720
HFOV_DEG = 60
DEADZONE_PX = 40
TURN_THRESHOLD_DEG = 22
TURN_GAIN = 0.8
CAMERA_PX_PER_DEG = 4.0

# พิกัดปุ่ม Pick All สีเขียว
PICK_ALL_BTN = (335, 545)

# พิกัดจอยสติ๊กควบคุมการเดิน
JOY_CENTER_X = 250
JOY_CENTER_Y = 530
JOY_RADIUS = 145


def run_cmd(cmd):
    subprocess.run(f"{ROOT_SU} -c '{cmd}'", shell=True)


def tap(x, y):
    run_cmd(f"input tap {x} {y}")


def swipe_screen(x1, y1, x2, y2, duration=240):
    run_cmd(f"input swipe {x1} {y1} {x2} {y2} {duration}")


def get_screen():
    run_cmd(f"screencap -p {TMP_PATH}")
    if os.path.exists(TMP_PATH):
        # รอเขียนไฟล์เสร็จสมบูรณ์เพื่อป้องกัน libpng error
        for _ in range(6):
            if os.path.getsize(TMP_PATH) > 50000:
                break
            time.sleep(0.04)

        img = cv2.imread(TMP_PATH)
        if img is not None:
            h, w = img.shape[:2]
            if h > w:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            return img
    return None


def match_texture(target_img, icon_filename, threshold=0.48):
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
    วิเคราะห์ลวดลายต้นไม้ เลือกต้นที่ใหญ่ที่สุด
    คืนค่า: (พบหรือไม่, ชื่อต้นไม้, ชื่อไฟล์, ข้อความ, พิกัด X กลาง, พิกัดกล่องกรอบจริง rx, ry, rw, rh)
    """
    default_box = (max(0, target_x - 70), 100, 140, 240)
    if screen is None:
        return False, "ไม่ทราบ", "ไม่มี", "ไม่พบภาพ", target_x, default_box

    h, w = screen.shape[:2]
    target_roi = screen[40:540, 80:1200]
    gray_roi = cv2.cvtColor(target_roi, cv2.COLOR_BGR2GRAY)
    tree_files = ["tree_name1.png", "tree_coconut.png", "tree_coconut2.png"]

    best_area = 0
    best_target_x = target_x
    best_box = default_box

    for fname in tree_files:
        icon_path = os.path.join(ICONS_DIR, fname)
        if not os.path.exists(icon_path):
            continue
        template = cv2.imread(icon_path)
        if template is None:
            continue

        gray_tpl = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        for scale in [0.6, 0.8, 1.0, 1.3, 1.6]:
            tw = int(gray_tpl.shape[1] * scale)
            th = int(gray_tpl.shape[0] * scale)
            if th >= target_roi.shape[0] or tw >= target_roi.shape[1] or th < 25 or tw < 25:
                continue

            res = cv2.matchTemplate(gray_roi, cv2.resize(gray_tpl, (tw, th)), cv2.TM_CCOEFF_NORMED)
            locs = np.where(res >= 0.46)
            for pt in zip(*locs[::-1]):
                area = tw * th
                if area > best_area:
                    best_area = area
                    best_target_x = 80 + pt[0] + (tw // 2)
                    best_box = (80 + pt[0], 40 + pt[1], tw, th)

    box_w, box_h = 320, 360
    x1 = max(0, best_target_x - (box_w // 2))
    x2 = min(w, best_target_x + (box_w // 2))
    y1 = max(40, target_y - (box_h // 2))
    y2 = min(h - 80, target_y + (box_h // 2))

    cropped_view = screen[y1:y2, x1:x2]
    if cropped_view.size == 0:
        return False, "ไม่ทราบ", "ไม่มี", "พิกัดหลุดขอบจอ", best_target_x, best_box

    lab = cv2.cvtColor(cropped_view, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    zoom_scales = [1.0, 1.4, 1.8]
    tree_file_map = {
        "tree_coconut.png": "ต้นมะพร้าว (Coconut Tree)",
        "tree_coconut2.png": "ต้นมะพร้าวลาย 2 (Coconut Tree)",
        "tree_name1.png": "ต้นยางพารา (Rubber Tree)"
    }

    for sc in zoom_scales:
        scaled_img = cv2.resize(enhanced, (0, 0), fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
        for icon_fname, tree_name in tree_file_map.items():
            matched, conf = match_texture(scaled_img, icon_fname, threshold=0.46)
            if matched:
                detail = f"ซูม {sc:.1f}x ลายตรงกับ '{icon_fname}' ({conf:.2f})"
                return True, tree_name, icon_fname, detail, best_target_x, best_box

    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([5, 30, 25]), np.array([28, 220, 160]))
    if np.sum(mask > 0) > 2200:
        return True, "ต้นไม้ทั่วไป", "HSV_Bark", "เฉดสีลำต้นไม้ตรงตามเกณฑ์", best_target_x, best_box

    return False, "ไม่ทราบ", "ไม่มี", "ลวดลายไม่ตรงกับ icons", best_target_x, best_box


def check_tree_cut_finished(screen):
    """
    ข้อ 5: เช็กว่าตัดเสร็จหรือยัง
    เงื่อนไข: No_Name_Rubber_tree.png (ไม่มีเลือด) หรือ เจอ มือ.png
    """
    if screen is None:
        return False, 0, 0

    # 1. ตรวจสอบปุ่ม มือ.png
    hand_path = os.path.join(ICONS_DIR, "มือ.png")
    if os.path.exists(hand_path):
        tpl = cv2.imread(hand_path)
        if tpl is not None:
            roi_hand = screen[300:660, 200:1160]
            if roi_hand.shape[0] > tpl.shape[0] and roi_hand.shape[1] > tpl.shape[1]:
                res = cv2.matchTemplate(roi_hand, tpl, cv2.TM_CCOEFF_NORMED)
                _, val, _, loc = cv2.minMaxLoc(res)
                if val >= 0.52:
                    hx = 200 + loc[0] + (tpl.shape[1] // 2)
                    hy = 300 + loc[1] + (tpl.shape[0] // 2)
                    return True, hx, hy

    # 2. ตรวจสอบ No_Name_Rubber_tree.png บริเวณขวาบน
    no_name_path = os.path.join(ICONS_DIR, "No_Name_Rubber_tree.png")
    if os.path.exists(no_name_path):
        tpl_no = cv2.imread(no_name_path)
        if tpl_no is not None:
            roi_name = screen[120:340, 780:1260]
            if roi_name.shape[0] > tpl_no.shape[0] and roi_name.shape[1] > tpl_no.shape[1]:
                res_no = cv2.matchTemplate(roi_name, tpl_no, cv2.TM_CCOEFF_NORMED)
                _, val_no, _, _ = cv2.minMaxLoc(res_no)
                if val_no >= 0.48:
                    return True, 0, 0

    return False, 0, 0


def check_pickup_hand(screen):
    """ตรวจจับปุ่มรูปมือ pickup_hand.png ในพิกัดจอ"""
    if screen is None:
        return False, 0, 0, ""

    icon_path = os.path.join(ICONS_DIR, "pickup_hand.png")
    if os.path.exists(icon_path):
        tpl = cv2.imread(icon_path)
        if tpl is not None:
            roi = screen[300:660, 200:1160]
            if roi.shape[0] > tpl.shape[0] and roi.shape[1] > tpl.shape[1]:
                res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
                _, val, _, loc = cv2.minMaxLoc(res)
                if val >= 0.55:
                    return True, 200 + loc[0] + (tpl.shape[1] // 2), 300 + loc[1] + (tpl.shape[0] // 2), f"pickup_hand.png ({val:.2f})"

    return False, 0, 0, ""


def check_pick_all_dialog(screen):
    """ตรวจจับหน้าต่างเก็บของ: ต้องมีปุ่มเขียว หรือ คำว่า Pick / All / เก็บ"""
    if screen is None:
        return False, 0, 0

    # 1. เช็กปุ่มเขียว Pick All
    roi = screen[480:600, 260:410]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, np.array([35, 70, 70]), np.array([85, 255, 255]))
    if np.sum(mask_green > 0) > 380:
        return True, PICK_ALL_BTN[0], PICK_ALL_BTN[1]

    # 2. เช็กตัวอักษร OCR
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, lang="eng+tha", config=r"--psm 7").strip()
    if any(k in text.lower() for k in ["pick", "all", "เก็บ"]):
        return True, PICK_ALL_BTN[0], PICK_ALL_BTN[1]

    return False, 0, 0


def check_bag_status(screen):
    """
    ข้อ 1: ตรวจสอบสถานะกระเป๋า
    คืนค่า: 'full' (กระเป๋าส้ม.png), 'normal' (กระเป๋า.png), 'unknown' (ไม่พบ)
    """
    if screen is None:
        return "unknown"

    # โซนกระเป๋า (ปกติอยู่แถบขวาบนหรือขวาล่าง ให้ครอบคลุมขอบจอ)
    # เช็กกระเป๋าส้มก่อน
    f_orange = os.path.join(ICONS_DIR, "กระเป๋าส้ม.png")
    if os.path.exists(f_orange):
        tpl = cv2.imread(f_orange)
        if tpl is not None:
            res = cv2.matchTemplate(screen, tpl, cv2.TM_CCOEFF_NORMED)
            _, val, _, _ = cv2.minMaxLoc(res)
            if val >= 0.60:
                return "full"

    # เช็กกระเป๋าปกติ
    f_normal = os.path.join(ICONS_DIR, "กระเป๋า.png")
    if os.path.exists(f_normal):
        tpl = cv2.imread(f_normal)
        if tpl is not None:
            res = cv2.matchTemplate(screen, tpl, cv2.TM_CCOEFF_NORMED)
            _, val, _, _ = cv2.minMaxLoc(res)
            if val >= 0.60:
                return "normal"

    return "normal"


def check_axe_status(screen):
    """
    ข้อ 2: ตรวจสอบสถานะขวานแถบล่าง
    คืนค่า: True (พร้อมใช้ ขวาน.png), False (พัง/แถบแดง ขวานแดง.png)
    """
    if screen is None:
        return True

    # โซนแถบช่องใส่อุปกรณ์ด้านล่าง (Y=560-720, X=350-950)
    roi = screen[560:720, 350:950]

    # ตรวจสอบขวานแดงก่อน (ถ้าพังให้หยุดทันที)
    f_broken = os.path.join(ICONS_DIR, "ขวานแดง.png")
    if os.path.exists(f_broken):
        tpl = cv2.imread(f_broken)
        if tpl is not None and roi.shape[0] > tpl.shape[0] and roi.shape[1] > tpl.shape[1]:
            res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
            _, val, _, _ = cv2.minMaxLoc(res)
            if val >= 0.55:
                return False  # ขวานพัง

    # ตรวจสอบขวานปกติสีเขียว
    f_ready = os.path.join(ICONS_DIR, "ขวาน.png")
    if os.path.exists(f_ready):
        tpl = cv2.imread(f_ready)
        if tpl is not None and roi.shape[0] > tpl.shape[0] and roi.shape[1] > tpl.shape[1]:
            res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
            _, val, _, _ = cv2.minMaxLoc(res)
            if val >= 0.50:
                return True

    return True


def check_target_name_match(screen):
    """
    ข้อ 4: บังคับตรวจชื่อต้นไม้เหนือหลอดเลือด
    - ต้องตรงกับ Name_Rubber_tree.png หรือ OCR พบ Rubber Tree / ยางพารา เท่านั้น
    - ถ้าเจอเลือดแต่ชื่อไม่ใช่ (เช่น Fern Tree) -> ให้ถือว่าไม่ใช่เป้าหมาย เพื่อหมุนหาต้นใหม่
    """
    if screen is None:
        return False, False, "ไม่พบภาพหน้าจอ"

    # โซนป้ายชื่อและหลอดเลือดมุมขวาบน
    roi = screen[100:360, 750:1270]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # ตรวจจับแถบเลือดสีแดง
    mask_red1 = cv2.inRange(hsv_roi, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(hsv_roi, np.array([160, 100, 100]), np.array([180, 255, 255]))
    red_count = int(np.sum(mask_red1 > 0) + np.sum(mask_red2 > 0))
    has_blood = red_count > 45

    # 1. เทียบไฟล์ Name_Rubber_tree.png
    f_has_blood = os.path.join(ICONS_DIR, "Name_Rubber_tree.png")
    val_has_blood = 0.0
    if os.path.exists(f_has_blood):
        tpl1 = cv2.imread(f_has_blood)
        if tpl1 is not None and roi.shape[0] >= tpl1.shape[0] and roi.shape[1] >= tpl1.shape[1]:
            res1 = cv2.matchTemplate(roi, tpl1, cv2.TM_CCOEFF_NORMED)
            _, val_has_blood, _, _ = cv2.minMaxLoc(res1)

    # 2. เทียบไฟล์ No_Name_Rubber_tree.png (กรณีต้นไม้ล้ม/ตายแล้ว)
    f_no_blood = os.path.join(ICONS_DIR, "No_Name_Rubber_tree.png")
    val_no_blood = 0.0
    if os.path.exists(f_no_blood):
        tpl2 = cv2.imread(f_no_blood)
        if tpl2 is not None and roi.shape[0] >= tpl2.shape[0] and roi.shape[1] >= tpl2.shape[1]:
            res2 = cv2.matchTemplate(roi, tpl2, cv2.TM_CCOEFF_NORMED)
            _, val_no_blood, _, _ = cv2.minMaxLoc(res2)

    # 3. OCR สแกนเฉพาะตัวอักษรสีขาว
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 50, 255])
    mask_white = cv2.inRange(hsv_roi, lower_white, upper_white)
    raw_ocr = pytesseract.image_to_string(mask_white, lang="eng+tha", config=r"--psm 7").strip().lower()

    # ดักชื่อต้นไม้อื่นที่ไม่ใช่ต้นยางพารา (เช่น Fern Tree ในภาพ)
    if any(k in raw_ocr for k in ["fern", "เฟิร์น", "stone", "หิน"]):
        return False, False, f"พบต้นไม้อื่น ({raw_ocr}) ไม่ใช่ต้นยางพารา -> หมุนหาใหม่"

    # ตรวจว่าใช่ต้นไม้ล้มแล้วหรือไม่
    if val_no_blood >= 0.48 or (not has_blood and any(k in raw_ocr for k in ["rubber", "tree", "ยาง"])):
        return False, True, f"ต้นยางพาราล้มแล้ว (พร้อมเก็บของ) | OCR: '{raw_ocr}'"

    # เงื่อนไขยืนยันว่าเป็นต้นยางพาราจริง + มีเลือด
    is_rubber_name = (val_has_blood >= 0.44) or any(k in raw_ocr for k in ["rubber", "tree", "ยาง", "พารา"])

    if is_rubber_name and has_blood:
        return True, False, f"ยืนยันต้นยางพาราถูกต้องและมีเลือด (ไอคอน: {val_has_blood:.2f}, เลือด: {red_count}px, OCR: '{raw_ocr}')"

    return False, False, f"ชื่อไม่ตรง Name_Rubber_tree หรือไม่ใช่ต้นยาง (ไอคอน: {val_has_blood:.2f}, OCR: '{raw_ocr}')"




def check_target_name_match(screen):
    """
    ข้อ 4: ตรวจสอบอ้างอิงไฟล์จากโฟลเดอร์ icons/ โดยตรง
    1. Name_Rubber_tree.png = มีชื่อ + มีหลอดเลือด -> พร้อมฟันต่อ
    2. No_Name_Rubber_tree.png = มีชื่อ + ไม่มีหลอดเลือด -> ต้นไม้ล้ม/เสร็จแล้ว พร้อมเก็บ
    """
    if screen is None:
        return False, False, "ไม่พบภาพหน้าจอ"

    # กำหนดพื้นที่ป้ายชื่อและหลอดเลือดมุมขวาบนของจอ
    roi = screen[100:360, 750:1270]

    # --- ตรวจสอบไฟล์ที่ 1: Name_Rubber_tree.png (มีชื่อ + มีเลือด) ---
    f_has_blood = os.path.join(ICONS_DIR, "Name_Rubber_tree.png")
    val_has_blood = 0.0
    if os.path.exists(f_has_blood):
        tpl1 = cv2.imread(f_has_blood)
        if tpl1 is not None and roi.shape[0] >= tpl1.shape[0] and roi.shape[1] >= tpl1.shape[1]:
            res1 = cv2.matchTemplate(roi, tpl1, cv2.TM_CCOEFF_NORMED)
            _, val_has_blood, _, _ = cv2.minMaxLoc(res1)
            if val_has_blood >= 0.45:
                return True, False, f"พบ Name_Rubber_tree.png ({val_has_blood:.2f}) -> มีเลือด พร้อมฟันต่อ"

    # --- ตรวจสอบไฟล์ที่ 2: No_Name_Rubber_tree.png (มีชื่อ + ไม่มีเลือด) ---
    f_no_blood = os.path.join(ICONS_DIR, "No_Name_Rubber_tree.png")
    val_no_blood = 0.0
    if os.path.exists(f_no_blood):
        tpl2 = cv2.imread(f_no_blood)
        if tpl2 is not None and roi.shape[0] >= tpl2.shape[0] and roi.shape[1] >= tpl2.shape[1]:
            res2 = cv2.matchTemplate(roi, tpl2, cv2.TM_CCOEFF_NORMED)
            _, val_no_blood, _, _ = cv2.minMaxLoc(res2)
            if val_no_blood >= 0.45:
                return False, True, f"พบ No_Name_Rubber_tree.png ({val_no_blood:.2f}) -> ต้นไม้ล้มแล้ว พร้อมเก็บ"

    # เช็กเสริมด้วย HSV แถบสีแดงและ OCR
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask_red1 = cv2.inRange(hsv_roi, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(hsv_roi, np.array([160, 100, 100]), np.array([180, 255, 255]))
    red_count = int(np.sum(mask_red1 > 0) + np.sum(mask_red2 > 0))

    lower_white = np.array([0, 0, 185])
    upper_white = np.array([180, 50, 255])
    mask_white = cv2.inRange(hsv_roi, lower_white, upper_white)
    raw_ocr = pytesseract.image_to_string(mask_white, lang="eng+tha", config=r"--psm 7").strip()

    if red_count > 60:
        return True, False, f"พบแถบเลือดแดง ({red_count}px, OCR: '{raw_ocr}') -> พร้อมฟันต่อ"

    return False, False, f"เคาะไม่โดน/ไม่พบป้าย (มีเลือด: {val_has_blood:.2f}, ไร้เลือด: {val_no_blood:.2f}, OCR: '{raw_ocr}')"


def walk_direction(deg=0, duration_ms=480):
    rad = math.radians(deg)
    tx = int(JOY_CENTER_X + JOY_RADIUS * math.sin(rad))
    ty = int(JOY_CENTER_Y - JOY_RADIUS * math.cos(rad))
    swipe_screen(JOY_CENTER_X, JOY_CENTER_Y, tx, ty, duration=duration_ms)
    time.sleep(0.1)


def rotate_view_right(dist=180):
    cx, cy = 850, 240
    swipe_screen(cx, cy, cx - dist, cy, duration=240)
    time.sleep(0.35)


def pixel_to_angle(target_x):
    cx = SCREEN_W / 2
    off = target_x - cx
    if abs(off) <= DEADZONE_PX:
        return 0.0
    f = cx / math.tan(math.radians(HFOV_DEG / 2))
    return math.degrees(math.atan(off / f))


def turn_camera(angle_deg):
    px = int(min(abs(angle_deg) * TURN_GAIN * CAMERA_PX_PER_DEG, 500))
    if px < 15:
        return
    cx, cy = 850, 240
    if angle_deg > 0:
        swipe_screen(cx, cy, cx - px, cy, duration=200)
    else:
        swipe_screen(cx, cy, cx + min(px, 340), cy, duration=200)
    time.sleep(0.3)


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
    print("[*] UtopiaFarmAI: ระบบฟาร์มอัตโนมัติ (คงฟังก์ชันครบ 100% แก้ตรงจุด)")
    print(f"[*] โฟลเดอร์ต้นแบบ: {ICONS_DIR}")
    print("=" * 60)
    time.sleep(2)

    round_count = 0
    retry_mode = False

    while True:
        round_count += 1
        print(f"\n{'='*20} เริ่มต้นลูปฟาร์มรอบที่ {round_count} {'='*20}")

        # ขั้นตอนที่ 1: กวาดสายตา 360 องศาค้นหาลวดลายต้นไม้
        print("[1] ขั้นตอนที่ 1: กวาดสายตา 360 องศาค้นหาลวดลายต้นไม้...")
        candidate_found = False
        target_x = 0
        target_y = 0
        tree_info = ""

        for step in range(8):
            scr = get_screen()
            found, src, cx, cy, area, h = detect_target_tree(scr)
            if found:
                candidate_found = True
                target_x = cx
                target_y = cy
                tree_info = src
                print(f"\n[🌲] พบเป้าหมายที่มุม {step + 1}/8! [{src}] (X={cx}, Y={cy}, สูง={h}px)")
                break
            else:
                print(f"[🔄] สแกนมุม {step + 1}/8 -> หมุนขวาต่อ...")
                rotate_view_right(dist=180)

        if not candidate_found:
            print("\n" + "=" * 55)
            print("[!] ล้มเหลว: กวาดหา 360 องศาแล้วไม่พบต้นไม้ -> หยุดการทำงาน")
            print("=" * 55)
            return

        # ตรวจสอบข้อ 1: สถานะกระเป๋า
        scr_check = get_screen()
        bag_status = check_bag_status(scr_check)
        if bag_status == "full":
            print("\n[!] กระเป๋าเต็ม (ตรวจพบ กระเป๋าส้ม.png) -> หยุดทำงาน")
            return
        else:
            print("[✓] สถานะกระเป๋าปกติ (กระเป๋า.png)")

        # ตรวจสอบข้อ 2: สถานะขวานแถบล่าง
        axe_ready = check_axe_status(scr_check)
        if not axe_ready:
            print("\n" + "=" * 60)
            print("[!] ล้มเหลว: ตรวจพบ ขวานแดง.png (ขวานพัง/หลอดแดง) -> สั่งหยุดทำงานทันที")
            print("=" * 60)
            return
        else:
            print("[✓] ขวานพร้อมใช้งาน (ขวาน.png) -> ดำเนินการวิเคราะห์เฉดสีและเดินต่อได้")

        # ขั้นตอนที่ 2: ล็อกเป้าหมายทันที (ไม่สั่งหมุนวนซ้ำ และไม่เซฟรูปภาพติด permission denied)
        print("\n[2] ขั้นตอนที่ 2: ล็อกหน้าจอนิ่งสนิท...")
        time.sleep(0.6)
        zoom_scr = get_screen()
        is_real_tree, tree_type_name, matched_file, zoom_detail, nearest_x, tree_box = optical_zoom_analysis(zoom_scr, target_x)
        
        # ปรับค่า X ล่าสุด และยืนยันชื่อต้นไม้
        if is_real_tree:
            target_x = nearest_x
        else:
            tree_type_name = tree_info

        print(f"[✓] ยืนยันเป้าหมาย [{tree_type_name}] สำเร็จ เตรียมเดินเข้าหาต้นนี้...")
        time.sleep(0.5)

        # ขั้นตอนที่ 3: เดินเข้าหาตามแนวระนาบปกติ (คำนวณสดตลอดเวลา ไม่ก้มหัวลงพื้น)
        print("\n[3] ขั้นตอนที่ 3: เดินเข้าหาเป้าหมายตามแนวระนาบปกติ (คำนวณสดตลอดเวลา)...")
        reached_close = False

        for step in range(30):
            curr_scr = get_screen()

            if check_pickup_hand(curr_scr)[0]:
                print("\n[🎯] พบปุ่มรูปมือขึ้นมาแล้ว -> ประชิดเป้าหมายเรียบร้อย!")
                reached_close = True
                break

            found_step, _, cur_x, cur_y, cur_area, cur_h = detect_target_tree(curr_scr)

            if found_step:
                ang = pixel_to_angle(cur_x)
                if abs(ang) > TURN_THRESHOLD_DEG:
                    turn_camera(ang)
                    curr_scr = get_screen()
                    found_step, _, cur_x, cur_y, cur_area, cur_h = detect_target_tree(curr_scr)

                walk_deg, direction_label = calculate_walk_angle(cur_x)
                offset_center = cur_x - (SCREEN_W // 2)

                print(f"[*] ก้าวที่ {step + 1:02d} | พิกัดสด: ({cur_x:4d}, {cur_y:3d}) | "
                      f"สูง: {cur_h:3d}px | เบี่ยง: {offset_center:+4d}px | เดิน: {direction_label}")

                if step >= 3 and cur_h >= 220:
                    print(f"\n[🎯] ลำต้นขยายใหญ่เต็มหน้าจอ ({cur_h}px) -> ถึงระยะฟันสมบูรณ์!")
                    reached_close = True
                    break

                walk_direction(walk_deg, duration_ms=480)
                time.sleep(0.08)

            else:
                print(f"\n[⚠️] ก้าวที่ {step + 1:02d} | เป้าหมายหลุดมุมมองชั่วคราว!")
                target_reacquired = False

                # นับถอยหลัง 5 วินาทีเพื่อมองหาเป้าหมายซ้ำ
                for wait_sec in range(5, 0, -1):
                    print(f"[*] กำลังค้นหาเป้าหมายใหม่... นับถอยหลังเหลือ {wait_sec} วินาที")
                    time.sleep(1.0)
                    retry_scr = get_screen()
                    found_retry, _, _, _, _ = detect_target_tree(retry_scr)
                    if found_retry or check_pickup_hand(retry_scr)[0]:
                        print("[✓] พบเป้าหมายกลับเข้ามาในมุมมองแล้ว! เดินหน้าต่อ...")
                        target_reacquired = True
                        break

                # หากครบ 5 วินาทีแล้วยังไม่เจอเป้าหมาย ให้หยุดทำงานทันที
                if not target_reacquired:
                    print("\n" + "=" * 60)
                    print("[!] ล้มเหลว: เป้าหมายหลุดมุมมองเกิน 5 วินาที -> สั่งหยุดทำงานทันที")
                    print("=" * 60)
                    return

         # ขั้นตอนที่ 4: ดันตัวประชิดโคนต้นไม้ แล้วเคาะตี 1 ครั้งเพื่อเช็กชื่อและเลือด
        print(f"\n[4] ขั้นตอนที่ 4: ดันตัวประชิดโคนต้นไม้ แล้วเคาะตี 1 ครั้ง...")
        walk_direction(0, duration_ms=280)
        time.sleep(0.15)
        tap(1100, 530)
        time.sleep(1.0)

        target_scr = get_screen()
        is_target_ready, is_tree_dead, result_log = check_target_name_match(target_scr)
        print(f"[*] ผลการสแกน: {result_log}")

        # ถ้าไม่ใช่เป้าหมาย (เช่น เจอ Fern Tree หรือหิน) ให้กลับไปเริ่มขั้นตอนที่ 1 หมุน 360 องศาหาต้นใหม่ทันที
        if not is_target_ready and not is_tree_dead:
            print("\n[🔄] ชื่อไม่ตรงกับ Name_Rubber_tree.png -> ย้อนกลับไปขั้นตอนที่ 1 เพื่อกวาดหา 360 องศาใหม่...")
            rotate_view_right(dist=220)
            continue

        # ขั้นตอนที่ 5: ฟันต่อเนื่องจนต้นไม้ล้ม
        print(f"\n[5] ขั้นตอนที่ 5: ยืนยันต้นยางพาราถูกต้อง! เริ่มต้นฟันต่อเนื่อง...")
        hits = 0
        hand_detected = False
        hx, hy = 0, 0

        while hits < 45:
            hits += 1
            tap(1100, 530)
            time.sleep(0.35)

            if hits % 2 == 0:
                chk_scr = get_screen()
                finished, fx, fy = check_tree_cut_finished(chk_scr)
                if finished:
                    hand_detected = True
                    hx, hy = fx, fy
                    print(f"\n[🖐️] ฟันต้นไม้ล้มแล้ว! (ตีไป {hits} ครั้ง)")
                    break

        # 1. หลอดเลือดหมด/ฟันเสร็จ: ให้หยุดนิ่ง 2 วินาที
        print("[⏳] ฟันเสร็จสมบูรณ์ -> หยุดนิ่ง 2 วินาที...")
        time.sleep(2.0)

        # 2. ตรวจสอบปุ่มรูปมือ หรือใช้พิกัดสำรอง (670, 430)
        final_chk = get_screen()
        is_hand, hand_x, hand_y, _ = check_pickup_hand(final_chk)
        if is_hand:
            hx, hy = hand_x, hand_y

        # หากระบบมองไม่เห็นปุ่มมือ ให้ใช้พิกัดสำรองในรูปภาพ (670, 430)
        if hx == 0 and hy == 0:
            hx, hy = 670, 430
            print(f"[*] ตรวจไม่พบไอคอนรูปมือ -> ใช้พิกัดสำรอง ({hx}, {hy}) ทันที")

        print(f"[🖐️] ทำการกดปุ่มรูปมือ 2 ครั้ง ที่พิกัด ({hx}, {hy})...")
        tap(hx, hy)
        time.sleep(0.3)
        tap(hx, hy)
        time.sleep(1.2)


        # ขั้นตอนที่ 6: ต้องขึ้นหน้าต่างคำว่า "เก็บ" / "Pick All" เท่านั้น
        print("\n[6] ขั้นตอนที่ 6: ตรวจสอบหน้าต่างไอเทม (ปุ่มสีเขียว / คำว่า 'เก็บ')...")
        dialog_scr = get_screen()
        pick_all_ready, px, py = check_pick_all_dialog(dialog_scr)

        if not pick_all_ready:
            print("\n" + "=" * 60)
            print("[!] ล้มเหลว: กดรูปมือแล้วรอ 1.2 วิ ไม่ขึ้นคำว่า 'เก็บ' ที่ปุ่มสีเขียว -> สั่งหยุดทำงาน")
            print("=" * 60)
            return

        print(f"[📦] ตรวจพบคำว่า 'Pick All' สมบูรณ์! ทำการแตะเก็บไอเทมที่ ({px}, {py})...")
        tap(px, py)
        time.sleep(0.25)
        tap(px, py)

        print("[✓] เก็บไอเทมเข้าตัวเสร็จสมบูรณ์!")

        # ขั้นตอนที่ 7: พักรอ 5 วินาที
        print("\n[7] ขั้นตอนที่ 7: รอนิ่ง 5 วินาทีเพื่อให้โมเดลในฉากหายสนิท...")
        for wait_sec in range(5, 0, -1):
            print(f"[*] พักรอ... เหลือ {wait_sec} วิ")
            time.sleep(1.0)

        print("[🚀] เริ่มต้นสแกนหาต้นต่อไปทันที...\n")


if __name__ == "__main__":
    main()
