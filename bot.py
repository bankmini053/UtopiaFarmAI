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

# พารามิเตอร์หน้าจอและมุมมองกล้อง
SCREEN_W = 1280
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
        img = cv2.imread(TMP_PATH)
        if img is not None:
            h, w = img.shape[:2]
            if h > w:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            return img
    return None


def match_texture(target_img, icon_filename, threshold=0.48):
    """จับคู่ลวดลายกับไฟล์ภาพต้นแบบใน icons/"""
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
    """วิเคราะห์ลวดลายต้นไม้ และเลือกเป้าหมายต้นที่ใหญ่ที่สุด"""
    if screen is None:
        return False, "ไม่ทราบ", "ไม่มี", "ไม่พบภาพ", target_x

    h, w = screen.shape[:2]
    target_roi = screen[40:540, 80:1200]
    gray_roi = cv2.cvtColor(target_roi, cv2.COLOR_BGR2GRAY)
    tree_files = ["tree_name1.png", "tree_coconut.png", "tree_coconut2.png"]

    best_area = 0
    best_target_x = target_x

    for fname in tree_files:
        icon_path = os.path.join(ICONS_DIR, fname)
        if not os.path.exists(icon_path):
            continue
        template = cv2.imread(icon_path)
        if template is None:
            continue

        gray_tpl = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        for scale in [0.6, 0.9, 1.2, 1.5]:
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

    box_w, box_h = 320, 360
    x1 = max(0, best_target_x - (box_w // 2))
    x2 = min(w, best_target_x + (box_w // 2))
    y1 = max(40, target_y - (box_h // 2))
    y2 = min(h - 80, target_y + (box_h // 2))

    cropped_view = screen[y1:y2, x1:x2]
    if cropped_view.size == 0:
        return False, "ไม่ทราบ", "ไม่มี", "พิกัดหลุดขอบจอ", best_target_x

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
                return True, tree_name, icon_fname, detail, best_target_x

    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([5, 30, 25]), np.array([28, 220, 160]))
    if np.sum(mask > 0) > 2200:
        return True, "ต้นไม้ทั่วไป", "HSV_Bark", "เฉดสีลำต้นไม้ตรงตามเกณฑ์", best_target_x

    return False, "ไม่ทราบ", "ไม่มี", "ลวดลายไม่ตรงกับ icons", best_target_x


def check_pickup_hand(screen):
    """ตรวจจับปุ่มรูปมือ pickup_hand.png ในพิกัดจอเท่านั้น"""
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
    """ตรวจจับปุ่มเขียว Pick All"""
    if screen is None:
        return False, 0, 0

    roi = screen[480:600, 260:410]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, np.array([35, 70, 70]), np.array([85, 255, 255]))
    if np.sum(mask_green > 0) > 400:
        return True, PICK_ALL_BTN[0], PICK_ALL_BTN[1]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, config=r"--psm 7").strip()
    if any(k in text.lower() for k in ["pick", "all", "เก็บ"]):
        return True, PICK_ALL_BTN[0], PICK_ALL_BTN[1]

    return False, 0, 0


def detect_target_tree(screen):
    """ตรวจหาต้นไม้จากเฉดสีและไฟล์รูปภาพ"""
    if screen is None:
        return False, "ไม่พบ", 0, 0, 0

    target_roi = screen[40:620, 80:1200]
    sh, sw = target_roi.shape[:2]
    gray_roi = cv2.cvtColor(target_roi, cv2.COLOR_BGR2GRAY)

    tree_files = ["tree_name1.png", "tree_coconut.png", "tree_coconut2.png"]

    best_match_val = -1
    best_x = 0
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

            if max_val >= 0.48 and max_val > best_match_val:
                best_match_val = max_val
                best_x = 80 + max_loc[0] + (tw // 2)
                best_h = th
                best_area = tw * th
                matched_filename = f"{fname} ({max_val:.2f})"

    if best_match_val >= 0.48:
        return True, matched_filename, best_x, best_area, best_h

    return False, "ไม่พบ", 0, 0, 0


def check_target_name_match(screen):
    """ตรวจสอบชื่อเป้าหมายจาก icons: ต้องตรงกับ tree_rubber.png หรือ name_coconut.png"""
    if screen is None:
        return False, "ไม่พบชื่อ"

    target_names = ["tree_rubber.png", "name_coconut.png"]

    for r_name in target_names:
        icon_path = os.path.join(ICONS_DIR, r_name)
        if os.path.exists(icon_path):
            tpl = cv2.imread(icon_path)
            if tpl is not None:
                roi = screen[60:260, 350:950]
                if roi.shape[0] > tpl.shape[0] and roi.shape[1] > tpl.shape[1]:
                    res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
                    _, val, _, _ = cv2.minMaxLoc(res)
                    if val >= 0.52:
                        return True, f"ป้ายชื่อตรง {r_name} ({val:.2f})"

    bar_roi = screen[70:240, 400:900]
    hsv = cv2.cvtColor(bar_roi, cv2.COLOR_BGR2HSV)
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 120, 100]), np.array([14, 255, 255])),
        cv2.inRange(hsv, np.array([168, 120, 100]), np.array([180, 255, 255]))
    )
    if np.sum(mask_red > 0) > 45:
        return True, "พบแถบเลือดเป้าหมาย"

    return False, "ชื่อไม่ตรงกับ tree_rubber.png หรือ name_coconut.png"


def pitch_camera_down():
    """รูดนิ้วจากบนลงล่าง เพื่อกดมุมกล้องให้ก้มมองพื้น (Top-Down)"""
    cx, cy = 850, 260
    swipe_screen(cx, cy, cx, cy + 340, duration=280)
    time.sleep(0.4)



def check_topdown_proximity(screen):
    """คำนวณระยะห่างระหว่างหัวตัวละคร (640, 360) กับโคนต้นไม้ตรงหน้า"""
    if screen is None:
        return False, 999

    roi = screen[180:380, 520:760]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_bark = np.array([5, 30, 25])
    upper_bark = np.array([28, 220, 160])
    mask = cv2.inRange(hsv, lower_bark, upper_bark)

    y_coords, _ = np.where(mask > 0)
    if len(y_coords) == 0:
        return False, 999

    tree_bottom_y = np.max(y_coords)
    distance_to_head = 180 - tree_bottom_y

    if distance_to_head <= 40:
        return True, distance_to_head

    return False, distance_to_head


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
    print("[*] UtopiaFarmAI: ระบบฟาร์มอัตโนมัติฉบับสมบูรณ์ (icons 100%)")
    print(f"[*] โฟลเดอร์ต้นแบบ: {ICONS_DIR}")
    print("=" * 60)
    time.sleep(2)

    round_count = 0
    retry_mode = False

    while True:
        round_count += 1
        print(f"\n{'='*20} เริ่มต้นลูปฟาร์มรอบที่ {round_count} {'='*20}")

        # ขั้นตอนที่ 1: กวาดสายตา 360 องศา
        print("[1] ขั้นตอนที่ 1: กวาดสายตา 360 องศาค้นหาลวดลายต้นไม้รอบตัว...")
        candidate_found = False
        target_x = 0

        for step in range(8):
            scr = get_screen()
            found, src, cx, area, h = detect_target_tree(scr)
            if found:
                candidate_found = True
                target_x = cx
                print(f"\n[🌲] พบเป้าหมายที่มุม {step + 1}/8! [{src}] (X={cx}, สูง={h}px)")
                break
            else:
                print(f"[🔄] สแกนมุม {step + 1}/8 -> หมุนขวาต่อ...")
                rotate_view_right(dist=180)

        if not candidate_found:
            print("\n" + "=" * 55)
            print("[!] ล้มเหลว: กวาดหา 360 องศาแล้วไม่พบต้นไม้ -> หยุดการทำงาน")
            print("=" * 55)
            return

        # ขั้นตอนที่ 2: ล็อกหน้าจอนิ่ง แล้ววิเคราะห์ต้นที่ใหญ่ที่สุด
        print("\n[2] ขั้นตอนที่ 2: ล็อกหน้าจอนิ่งสนิท...")
        time.sleep(1.0)
        print("[🔬] เปิดโหมดซูมวิเคราะห์ เลือกต้นที่ใหญ่ที่สุด (ใกล้ตัวละครที่สุด)...")

        zoom_scr = get_screen()
        is_real_tree, tree_type_name, matched_file, zoom_detail, nearest_x = optical_zoom_analysis(zoom_scr, target_x)
        target_x = nearest_x

        print("\n" + "=" * 50)
        print(f"[🔍] ผลการวิเคราะห์ผ่านเลนส์ซูม:")
        print(f"  - ชนิดเป้าหมาย : {tree_type_name}")
        print(f"  - ไฟล์ที่ตรงกัน : {matched_file}")
        print(f"  - พิกัดต้นใกล้สุด: X={target_x}")
        print(f"  - รายละเอียด  : {zoom_detail}")
        print("=" * 50)

        if not is_real_tree:
            print("[⚠️] ปฏิเสธเป้าหมาย: ไม่ตรงกับ icons -> หมุนค้นหาต่อ")
            rotate_view_right(dist=180)
            continue

        print(f"[✓] เลือกต้นไม้ขนาดใหญ่สุดเรียบร้อย ({tree_type_name})! เตรียมเดินเข้าหา...")
        time.sleep(1.0)

        # ขั้นตอนที่ 3: เดินเข้าหา และก้มหน้าจอลงพื้นวัดระยะชิดหัว
        print("\n[3] ขั้นตอนที่ 3: เดินเข้าหาเป้าหมาย (ใช้เฉพาะเฉดสีเปลือกไม้)...")
        reached_close = False

        for step in range(25):
            curr_scr = get_screen()

            if check_pickup_hand(curr_scr)[0]:
                print("\n[🎯] พบปุ่มรูปมือขึ้นมาแล้ว -> ชิดติดเนื้อไม้ 100%!")
                reached_close = True
                break

            found_step, _, cur_x, cur_area, cur_h = detect_target_tree(curr_scr)

            if found_step:
                ang = pixel_to_angle(cur_x)
                if abs(ang) > TURN_THRESHOLD_DEG:
                    turn_camera(ang)
                    curr_scr = get_screen()

                walk_deg, direction_label = calculate_walk_angle(cur_x)
                print(f"[*] ก้าวที่ {step + 1} | เดิน: {direction_label} ({walk_deg:.1f}°) | สูง={cur_h}px")

                if cur_h >= 240 or step >= 5:
                    print("[👀] เข้าใกล้ระยะโคนต้น -> กำลังก้มหน้าจอลงพื้นจนสุดเพื่อวัดระยะชิดหัว...")
                    pitch_camera_down()
                    
                    td_scr = get_screen()
                    is_snapped, dist = check_topdown_proximity(td_scr)
                    print(f"[*] ระยะห่างโคนต้นไม้จากหัวตัวละคร: {dist} px")

                    if is_snapped or dist <= 40:
                        print("\n[🎯] ลำต้นแนบชิดติดหัวตัวละครแล้ว (ระยะตัดสมบูรณ์)!")
                        reached_close = True
                        break
                    else:
                        print("[+] เดินดันชนอัดเข้าไปอีกนิด...")
                        walk_direction(0, duration_ms=380)
                        time.sleep(0.1)
                else:
                    walk_direction(walk_deg, duration_ms=480)
                    time.sleep(0.1)
            else:
                walk_direction(0, duration_ms=350)
                time.sleep(0.1)

        # ขั้นตอนที่ 4: ทดสอบกดตี 1 ครั้งเพื่อเช็กชื่อ
        print(f"\n[4] ขั้นตอนที่ 4: ทดสอบกดตี 1 ครั้งเพื่อเช็กชื่อต้นไม้ ({tree_type_name})...")
        tap(1100, 530)
        time.sleep(0.7)

        target_scr = get_screen()
        name_matched, name_detail = check_target_name_match(target_scr)
        print(f"[*] ผลตรวจป้ายชื่อ: {name_detail}")

        if not name_matched:
            if not retry_mode:
                print("\n[⚠️] ตี 1 ครั้งแล้วไม่พบชื่อที่ถูกต้อง -> ย้อนกลับไปขั้นตอนที่ 1 เพื่อมองหาใหม่...")
                retry_mode = True
                rotate_view_right(dist=180)
                continue
            else:
                print("\n" + "=" * 60)
                print("[!] ล้มเหลว: วนกลับมาตีรอบที่ 2 แล้วยังไม่มีชื่อขึ้นมา -> สั่งหยุดทำงาน")
                print("=" * 60)
                return

        retry_mode = False

        # ขั้นตอนที่ 5: ฟันต่อเนื่องจนเลือดหมด
        print(f"\n[5] ขั้นตอนที่ 5: ชื่อถูกต้อง! เริ่มต้นฟันต่อเนื่องจนกว่าหลอดเลือดจะหมด...")
        hits = 0

        while hits < 40:
            hits += 1
            tap(1100, 530)
            time.sleep(0.35)

            if hits % 2 == 0:
                chk_scr = get_screen()
                name_alive, _ = check_target_name_match(chk_scr)
                hand_ready, _, _, _ = check_pickup_hand(chk_scr)

                if not name_alive or hand_ready:
                    print(f"\n[🛑] หลอดเลือดหมด/ชื่อหาย/พบปุ่มมือ! (หยุดฟันหลังตีไป {hits} ครั้ง)")
                    break

        time.sleep(0.6)

        # ขั้นตอนที่ 6: ตรวจสอบรูปมือในพิกัดจอ แตะรูปมือ -> Pick All
        print("\n[6] ขั้นตอนที่ 6: ตรวจจับปุ่มรูปมือ pickup_hand.png ในพิกัดจอ...")
        final_scr = get_screen()
        hand_found, hx, hy, hinfo = check_pickup_hand(final_scr)

        if not hand_found:
            print("\n" + "=" * 60)
            print("[!] ล้มเหลว: เลือดต้นไม้ลดหมดแล้ว แต่ไม่พบรูปมือในพิกัดจอ -> สั่งหยุดทำงาน")
            print("=" * 60)
            return

        print(f"[🖐️] ตรวจพบรูปมือ! แตะเก็บไอเทมที่ ({hx}, {hy}) [{hinfo}]...")
        tap(hx, hy)
        time.sleep(0.5)

        dialog_scr = get_screen()
        pick_all_ready, px, py = check_pick_all_dialog(dialog_scr)

        if pick_all_ready:
            print(f"[📦] ปรากฏหน้าต่าง -> แตะปุ่มเขียว 'Pick All' ที่ ({px}, {py})")
            tap(px, py)
            time.sleep(0.3)
            tap(px, py)
        else:
            print("[📦] แตะพิกัด Pick All (335, 545)...")
            tap(PICK_ALL_BTN[0], PICK_ALL_BTN[1])
            time.sleep(0.3)
            tap(PICK_ALL_BTN[0], PICK_ALL_BTN[1])

        print("[✓] เก็บไอเทมเข้าตัวเสร็จสมบูรณ์!")

        # ขั้นตอนที่ 7: พักรอ 5 วินาที
        print("\n[7] ขั้นตอนที่ 7: รอนิ่ง 5 วินาทีเพื่อให้โมเดลในฉากหายสนิท...")
        for wait_sec in range(5, 0, -1):
            print(f"[*] พักรอ... เหลือ {wait_sec} วิ")
            time.sleep(1.0)

        print("[🚀] เริ่มต้นสแกนหาต้นต่อไปทันที...\n")


if __name__ == "__main__":
    main()
