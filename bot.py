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
DEADZONE_PX = 35
TURN_GAIN = 0.65
CAMERA_PX_PER_DEG = 3.8

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


def swipe_screen(x1, y1, x2, y2, duration=220):
    run_cmd(f"input swipe {x1} {y1} {x2} {y2} {duration}")


def get_screen():
    run_cmd(f"screencap -p {TMP_PATH}")
    run_cmd(f"chmod 777 {TMP_PATH}")
    if os.path.exists(TMP_PATH):
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


def check_pickup_hand(screen):
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
    if screen is None:
        return False, 0, 0

    roi = screen[480:600, 260:410]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, np.array([35, 70, 70]), np.array([85, 255, 255]))
    if np.sum(mask_green > 0) > 380:
        return True, PICK_ALL_BTN[0], PICK_ALL_BTN[1]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, lang="eng+tha", config=r"--psm 7").strip()
    if any(k in text.lower() for k in ["pick", "all", "เก็บ"]):
        return True, PICK_ALL_BTN[0], PICK_ALL_BTN[1]

    return False, 0, 0


def detect_target_tree(screen):
    """ตรวจจับหาต้นไม้รอบจอ และรายงานพิกัด X, Y ความสูง และพื้นที่อย่างแม่นยำ"""
    if screen is None:
        return False, "ไม่พบ", 0, 0, 0, 0

    target_roi = screen[40:600, 80:1200]
    sh, sw = target_roi.shape[:2]
    gray_roi = cv2.cvtColor(target_roi, cv2.COLOR_BGR2GRAY)

    tree_files = ["tree_name1.png", "tree_coconut.png", "tree_coconut2.png"]

    best_match_val = -1
    best_x = 0
    best_y = 0
    best_h = 0
    best_area = 0
    matched_filename = ""

    for fname in tree_files:
        icon_path = os.path.join(ICONS_DIR, fname)
        if not os.path.exists(icon_path):
            continue

        template = cv2.imread(icon_path)
        if template is None:
            continue

        gray_tpl = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        for scale in [0.5, 0.7, 1.0, 1.4, 1.8]:
            tw = int(gray_tpl.shape[1] * scale)
            th = int(gray_tpl.shape[0] * scale)

            if th >= sh or tw >= sw or th < 25 or tw < 25:
                continue

            resized_tpl = cv2.resize(gray_tpl, (tw, th))
            res = cv2.matchTemplate(gray_roi, resized_tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val >= 0.47 and max_val > best_match_val:
                best_match_val = max_val
                best_x = 80 + max_loc[0] + (tw // 2)
                best_y = 40 + max_loc[1] + (th // 2)
                best_h = th
                best_area = tw * th
                matched_filename = f"{fname} ({max_val:.2f})"

    if best_match_val >= 0.47:
        return True, matched_filename, best_x, best_y, best_area, best_h

    return False, "ไม่พบ", 0, 0, 0, 0


def check_target_name_match(screen):
    if screen is None:
        return False, "ไม่พบชื่อ"

    target_names = ["tree_rubber.png", "name_coconut.png"]
    roi = screen[160:320, 860:1250]

    for r_name in target_names:
        icon_path = os.path.join(ICONS_DIR, r_name)
        if os.path.exists(icon_path):
            tpl = cv2.imread(icon_path)
            if tpl is not None:
                if roi.shape[0] > tpl.shape[0] and roi.shape[1] > tpl.shape[1]:
                    res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
                    _, val, _, _ = cv2.minMaxLoc(res)
                    if val >= 0.50:
                        return True, f"ป้ายชื่อตรง {r_name} ({val:.2f})"

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh, lang="eng+tha", config=r"--psm 7").strip()

    if any(k in text.lower() for k in ["stone", "mine", "หิน"]):
        return False, f"พบก้อนหิน ({text}) ไม่ใช่ต้นไม้"

    if any(k in text.lower() for k in ["rubber", "coconut", "palm", "tree", "ยาง", "มะพร้าว"]):
        return True, f"OCR พบชื่อต้นไม้: '{text}'"

    return False, "ชื่อไม่ตรงกับ tree_rubber.png หรือ name_coconut.png"


def walk_direction(deg=0, duration_ms=680):
    rad = math.radians(deg)
    tx = int(JOY_CENTER_X + JOY_RADIUS * math.sin(rad))
    ty = int(JOY_CENTER_Y - JOY_RADIUS * math.cos(rad))
    swipe_screen(JOY_CENTER_X, JOY_CENTER_Y, tx, ty, duration=duration_ms)
    time.sleep(0.1)


def step_backward(duration_ms=420):
    """ถอยหลัง 1 ก้าวเมื่อหลุดโฟกัส"""
    tx = JOY_CENTER_X
    ty = int(JOY_CENTER_Y + JOY_RADIUS)
    swipe_screen(JOY_CENTER_X, JOY_CENTER_Y, tx, ty, duration=duration_ms)
    time.sleep(0.25)


def rotate_view_right(dist=180):
    cx, cy = 850, 240
    swipe_screen(cx, cy, cx - dist, cy, duration=220)
    time.sleep(0.35)


def pixel_to_angle(target_x):
    cx = SCREEN_W / 2
    off = target_x - cx
    if abs(off) <= DEADZONE_PX:
        return 0.0
    f = cx / math.tan(math.radians(HFOV_DEG / 2))
    return math.degrees(math.atan(off / f))


def calculate_walk_angle(target_x):
    ang = pixel_to_angle(target_x)
    label = "เอียงซ้าย" if ang < -1.5 else ("เอียงขวา" if ang > 1.5 else "ตรงกลาง")
    return ang % 360, f"{label} {ang:+.1f}°", ang


def align_camera_to_crosshair(target_x, target_y):
    """
    ข้อ 1 & 2: หมุนปรับหน้าจอทีละนิด (ซ้าย-ขวา, บน-ล่าง)
    ดึงพิกัดลำต้นไม้เข้าสู่กรอบสี่เหลี่ยมเป้าหมายกลางจอ (640, 360) ให้ใกล้ที่สุด
    """
    for _ in range(4):
        diff_x = target_x - 640
        diff_y = target_y - 360

        # อยู่ในกรอบสี่เหลี่ยมตรงกลางแล้ว (ไม่เกิน 45px)
        if abs(diff_x) <= 45 and abs(diff_y) <= 45:
            break

        move_x = int(np.clip(diff_x * 0.42, -150, 150))
        move_y = int(np.clip(diff_y * 0.40, -110, 110))

        cx, cy = 850, 240
        swipe_screen(cx, cy, cx - move_x, cy - move_y, duration=160)
        time.sleep(0.25)

        chk = get_screen()
        found, _, cur_x, cur_y, _, _ = detect_target_tree(chk)
        if found:
            target_x, target_y = cur_x, cur_y
        else:
            break

    return target_x, target_y


def main():
    print("=" * 60)
    print("[*] UtopiaFarmAI: ระบบตรวจจับคำนวณพิกัดสด (Live Real-Time Radar)")
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

        for step in range(8):
            scr = get_screen()
            found, src, cx, cy, area, h = detect_target_tree(scr)
            if found:
                candidate_found = True
                target_x = cx
                print(f"\n[🌲] พบเป้าหมายที่มุม {step + 1}/8! [{src}] (พิกัดสด: X={cx}, Y={cy}, สูง={h}px)")
                break
            else:
                print(f"[🔄] สแกนมุม {step + 1}/8 -> หมุนขวาต่อ...")
                rotate_view_right(dist=180)

        if not candidate_found:
            print("\n" + "=" * 55)
            print("[!] ล้มเหลว: กวาดหา 360 องศาแล้วไม่พบต้นไม้ -> หยุดการทำงาน")
            print("=" * 55)
            return

        # ขั้นตอนที่ 2: ตีกรอบสีเขียวครอบต้นไม้ที่ใหญ่ที่สุด
        print("\n[2] ขั้นตอนที่ 2: ล็อกหน้าจอนิ่งสนิท...")
        time.sleep(0.8)
        zoom_scr = get_screen()
        is_real_tree, tree_type_name, matched_file, zoom_detail, nearest_x, tree_box = optical_zoom_analysis(zoom_scr, target_x)
        target_x = nearest_x

        if is_real_tree and zoom_scr is not None:
            bx, by, bw, bh = tree_box
            locked_img = zoom_scr.copy()
            cv2.rectangle(locked_img, (bx, by), (bx + bw, by + bh), (0, 255, 0), 4)
            cv2.circle(locked_img, (target_x, by + (bh // 2)), 6, (0, 0, 255), -1)
            cv2.putText(locked_img, f"TARGET LOCKED: {tree_type_name}", (bx, max(30, by - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.imwrite(LOCKED_IMG_PATH, locked_img)
            print(f"[🎯] ยืนยันเป้าหมาย! ตีกรอบสีเขียวที่ X={bx}, Y={by} (กว้าง {bw}px, สูง {bh}px)")

        if not is_real_tree:
            print("[⚠️] ปฏิเสธเป้าหมาย: ไม่ตรงกับ icons -> หมุนค้นหาต่อ")
            rotate_view_right(dist=180)
            continue

        # ขั้นตอนที่ 3: ปรับศูนย์เล็งเข้ากึ่งกลางจอ (640, 360) แล้วเดินยาว
        print("\n[3] ขั้นตอนที่ 3: ระบบปรับศูนย์เล็งเข้ากึ่งกลางจอ (640, 360) แล้วเดินยาว...")
        reached_close = False

        for step in range(25):
            curr_scr = get_screen()

            # ตรวจสอบปุ่มรูปมือก่อนทุกครั้ง
            if check_pickup_hand(curr_scr)[0]:
                print("\n[🎯] ตรวจพบปุ่มรูปมือขึ้นมาแล้ว -> ถึงระยะเก็บเกี่ยวเรียบร้อย!")
                reached_close = True
                break

            # ข้อ 1: หยุดเพื่อวิเคราะห์พิกัดตำแหน่งของต้นไม้ในจอ
            found_step, src, cur_x, cur_y, cur_area, cur_h = detect_target_tree(curr_scr)

            if found_step:
                offset_x = cur_x - 640
                offset_y = cur_y - 360

                # ข้อ 4: ถ้าหลุดกึ่งกลางจอมากเกินไป (> 260px) ให้ถอยหลัง 1 ก้าวแล้วเริ่มตั้งลำใหม่
                if abs(offset_x) > 260:
                    print(f"[⚠️] เป้าหมายหลุดกึ่งกลางจอมากเกินไป (เบี่ยง {offset_x:+d}px) -> ถอยหลัง 1 ก้าวเพื่อตั้งลำใหม่...")
                    step_backward(duration_ms=420)
                    time.sleep(0.3)
                    continue

                # ข้อ 2: หมุนปรับหน้าจอให้พิกัดต้นไม้อยู่ใกล้ (640, 360) มากที่สุด
                if abs(offset_x) > 50 or abs(offset_y) > 50:
                    cur_x, cur_y = align_camera_to_crosshair(cur_x, cur_y)
                    curr_scr = get_screen()
                    _, _, cur_x, cur_y, _, cur_h = detect_target_tree(curr_scr)

                walk_deg, direction_label, _ = calculate_walk_angle(cur_x)
                print(f"[📍 TARGET LOCKED] ก้าวที่ {step + 1:02d} | พิกัด: ({cur_x:4d}, {cur_y:3d}) | "
                      f"สูง: {cur_h:3d}px | เดิน: {direction_label}")

                # เงื่อนไขระยะประชิดเนื้อไม้ (เดินอย่างน้อย 4 ก้าว และลำต้นขยายใหญ่เต็มจอ)
                if step >= 4 and cur_h >= 400:
                    print(f"\n[🎯] ลำต้นขยายใหญ่เต็มหน้าจอ ({cur_h}px) -> ถึงระยะฟันสมบูรณ์!")
                    reached_close = True
                    break

                # ข้อ 3: เมื่อเล็งตรงกึ่งกลางจอแล้ว คำนวณการเดินก้าวตรงไปข้างหน้าแบบยาว ๆ (680ms)
                walk_direction(walk_deg, duration_ms=680)
                time.sleep(0.12)
            else:
                print(f"[⚠️] ไม่พบเป้าหมายในระยะมองเห็น -> ถอยหลัง 1 ก้าวเพื่อค้นหาใหม่...")
                step_backward(duration_ms=380)
                time.sleep(0.2)

        # ขั้นตอนที่ 4: ตี 1 ครั้งเพื่อตรวจสอบชื่อ
        print(f"\n[4] ขั้นตอนที่ 4: กดตี 1 ครั้งเพื่อเช็กชื่อเป้าหมาย ({tree_type_name})...")
        tap(1100, 530)
        time.sleep(0.7)

        target_scr = get_screen()
        name_matched, name_detail = check_target_name_match(target_scr)
        print(f"[*] ผลตรวจป้ายชื่อ: {name_detail}")

        if not name_matched:
            if not retry_mode:
                print("\n[⚠️] ตี 1 ครั้งแล้วไม่พบชื่อที่ถูกต้อง -> ย้อนกลับไปค้นหาใหม่ (แก้มือ 1 รอบ)...")
                retry_mode = True
                rotate_view_right(dist=180)
                continue
            else:
                print("\n" + "=" * 60)
                print("[!] ล้มเหลว: ย้อนกลับมาตีรอบ 2 แล้วยังไม่พบชื่อ -> สั่งหยุดทำงาน")
                print("=" * 60)
                return

        retry_mode = False

        # ขั้นตอนที่ 5: ฟันต่อเนื่องจนเจอปุ่มรูปมือ -> กดปุ่มมือ 2 ครั้ง
        print(f"\n[5] ขั้นตอนที่ 5: ชื่อถูกต้อง! ฟันต่อเนื่องจนกว่าจะเจอปุ่มรูปมือ...")
        hits = 0
        hand_detected = False
        hx, hy = 0, 0

        while hits < 45:
            hits += 1
            tap(1100, 530)
            time.sleep(0.35)

            if hits % 2 == 0:
                chk_scr = get_screen()
                is_hand, hand_x, hand_y, hand_src = check_pickup_hand(chk_scr)
                if is_hand:
                    hand_detected = True
                    hx, hy = hand_x, hand_y
                    print(f"\n[🖐️] ตรวจพบปุ่มรูปมือ [{hand_src}] หลังตีไป {hits} ครั้ง!")
                    break

        if not hand_detected:
            final_chk = get_screen()
            is_hand, hand_x, hand_y, hand_src = check_pickup_hand(final_chk)
            if is_hand:
                hand_detected = True
                hx, hy = hand_x, hand_y

        if not hand_detected:
            print("\n" + "=" * 60)
            print("[!] ล้มเหลว: ฟันครบกำหนดแล้วแต่ไม่พบปุ่มรูปมือ (pickup_hand.png) -> หยุดทำงาน")
            print("=" * 60)
            return

        print(f"[🖐️] ทำการกดปุ่มรูปมือ 2 ครั้ง ที่พิกัด ({hx}, {hy})...")
        tap(hx, hy)
        time.sleep(0.3)
        tap(hx, hy)
        time.sleep(0.8)

        # ขั้นตอนที่ 6: ต้องขึ้นหน้าต่างคำว่า "เก็บ" / "Pick All" เท่านั้น
        print("\n[6] ขั้นตอนที่ 6: ตรวจสอบหน้าต่างไอเทม (ต้องมีปุ่มเขียว หรือคำว่า 'Pick All')...")
        dialog_scr = get_screen()
        pick_all_ready, px, py = check_pick_all_dialog(dialog_scr)

        if not pick_all_ready:
            print("\n" + "=" * 60)
            print("[!] ล้มเหลว (Error): ไม่พบหน้าต่างหรือคำว่า 'เก็บ / Pick All' ขึ้นมา -> สั่งหยุดทำงานทันที")
            print("=" * 60)
            return

        print(f"[📦] ตรวจพบคำว่า 'Pick All' สมบูรณ์! แตะเก็บไอเทมที่ ({px}, {py})...")
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
