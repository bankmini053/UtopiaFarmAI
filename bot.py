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
SCREEN_H = 720
HFOV_DEG = 60
DEADZONE_PX = 40
TURN_THRESHOLD_DEG = 20
TURN_GAIN = 0.75
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


def swipe_screen(x1, y1, x2, y2, duration=240):
    run_cmd(f"input swipe {x1} {y1} {x2} {y2} {duration}")


def get_screen():
    """แคปหน้าจอและรอให้ไฟล์เขียนเสร็จเพื่อกัน libpng error"""
    run_cmd(f"screencap -p {TMP_PATH}")
    if os.path.exists(TMP_PATH):
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


def detect_target_tree(screen):
    """สแกนหาต้นไม้รอบจอแบบไว (ทั้งยางพาราและมะพร้าว)"""
    if screen is None:
        return False, "ไม่พบ", 0, 0, 0

    target_roi = screen[40:620, 80:1200]
    sh, sw = target_roi.shape[:2]
    gray_roi = cv2.cvtColor(target_roi, cv2.COLOR_BGR2GRAY)

    tree_files = ["tree_name1.png", "tree_coconut.png", "tree_coconut2.png"]
    best_match_val = -1
    best_x = 0
    best_y = 0
    best_h = 0
    matched_filename = ""

    for fname in tree_files:
        icon_path = os.path.join(ICONS_DIR, fname)
        if not os.path.exists(icon_path):
            continue

        template = cv2.imread(icon_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            continue

        for scale in [0.5, 0.7, 1.0, 1.3, 1.6]:
            tw = int(template.shape[1] * scale)
            th = int(template.shape[0] * scale)

            if th >= sh or tw >= sw or th < 25 or tw < 25:
                continue

            resized_tpl = cv2.resize(template, (tw, th))
            res = cv2.matchTemplate(gray_roi, resized_tpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val >= 0.46 and max_val > best_match_val:
                best_match_val = max_val
                best_x = 80 + max_loc[0] + (tw // 2)
                best_y = 40 + max_loc[1] + (th // 2)
                best_h = th
                matched_filename = f"{fname} ({max_val:.2f})"

    if best_match_val >= 0.46:
        return True, matched_filename, best_x, best_y, best_h

    return False, "ไม่พบ", 0, 0, 0


def check_target_name_match(screen):
    """เช็กป้ายชื่อหลังตี 1 ครั้ง (อ่านทั้งไทยและอังกฤษ ป้องกันวนหมุนจอ)"""
    if screen is None:
        return False, "ไม่พบภาพ"

    target_names = ["tree_rubber.png", "name_coconut.png"]
    roi = screen[120:340, 780:1260]

    # 1. เช็กรูปไอคอนป้ายชื่อ
    for r_name in target_names:
        icon_path = os.path.join(ICONS_DIR, r_name)
        if os.path.exists(icon_path):
            tpl = cv2.imread(icon_path)
            if tpl is not None:
                if roi.shape[0] > tpl.shape[0] and roi.shape[1] > tpl.shape[1]:
                    res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
                    _, val, _, _ = cv2.minMaxLoc(res)
                    if val >= 0.40:
                        return True, f"ป้ายชื่อตรงไอคอน {r_name} ({val:.2f})"

    # 2. อ่านตัวหนังสือ OCR
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 175, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh, lang="eng+tha", config=r"--psm 7").strip()

    # ดักว่าถ้าเป็นหินหรือแร่ให้หยุดตี
    if any(k in text.lower() for k in ["stone", "mine", "หิน", "iron", "ore", "copper"]):
        return False, f"พบก้อนหินหรือแร่ ({text}) ไม่ใช่ต้นไม้"

    # คำค้นหาชื่อต้นไม้
    tree_keywords = ["rubber", "coconut", "palm", "tree", "wood", "ยาง", "พารา", "มะพร้าว", "ต้นไม้", "ไม้"]
    if any(k in text.lower() for k in tree_keywords):
        return True, f"OCR ตรวจพบชื่อต้นไม้: '{text}'"

    # ถ้าไม่ใช่หิน ให้ถือว่าถูกต้อง ลุยตัดต่อได้เลย
    return True, f"ยืนยันต้นไม้เป้าหมาย (ตรวจชื่อ: '{text}')"


def check_pickup_hand(screen):
    """ตรวจจับปุ่มรูปมือ pickup_hand.png"""
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
                if val >= 0.52:
                    return True, 200 + loc[0] + (tpl.shape[1] // 2), 300 + loc[1] + (tpl.shape[0] // 2), f"pickup_hand.png ({val:.2f})"

    return False, 0, 0, ""


def check_pick_all_dialog(screen):
    """ตรวจจับหน้าต่างเก็บของ: ปุ่มเขียว หรือคำว่า Pick / All / เก็บ"""
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


def walk_direction(deg=0, duration_ms=480):
    rad = math.radians(deg)
    tx = int(JOY_CENTER_X + JOY_RADIUS * math.sin(rad))
    ty = int(JOY_CENTER_Y - JOY_RADIUS * math.cos(rad))
    swipe_screen(JOY_CENTER_X, JOY_CENTER_Y, tx, ty, duration=duration_ms)
    time.sleep(0.1)


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
    print("[*] UtopiaFarmAI: บอทฟาร์มต้นไม้อัตโนมัติ (ฉบับแก้บั๊กสมบูรณ์)")
    print(f"[*] โฟลเดอร์ต้นแบบ: {ICONS_DIR}")
    print("=" * 60)
    time.sleep(2)

    round_count = 0
    retry_mode = False

    while True:
        round_count += 1
        print(f"\n{'='*20} เริ่มต้นลูปฟาร์มรอบที่ {round_count} {'='*20}")

        # ขั้นตอนที่ 1: กวาดสายตา 360 องศาค้นหาต้นไม้
        print("[1] ขั้นตอนที่ 1: กวาดสายตา 360 องศาค้นหาต้นไม้...")
        candidate_found = False
        target_x = 0
        target_y = 0
        tree_info = ""

        for step in range(8):
            scr = get_screen()
            found, src, cx, cy, h = detect_target_tree(scr)
            if found:
                candidate_found = True
                target_x = cx
                target_y = cy
                tree_info = src
                print(f"\n[🌲] พบเป้าหมายที่มุม {step + 1}/8! [{src}] (พิกัด: X={cx}, Y={cy}, สูง={h}px)")
                break
            else:
                print(f"[🔄] สแกนมุม {step + 1}/8 -> หมุนขวาต่อ...")
                rotate_view_right(dist=180)

        if not candidate_found:
            print("\n[!] ล้มเหลว: กวาดหา 360 องศาแล้วไม่พบต้นไม้ -> หยุดการทำงาน")
            return

        # ขั้นตอนที่ 2: ล็อกเป้าหมายทันที (ไม่กรองซ้ำจนหลุด)
        print(f"\n[2] ขั้นตอนที่ 2: ล็อกเป้าหมาย [{tree_info}] สำเร็จ -> เตรียมเดินเข้าหาทันที...")
        time.sleep(0.4)

        # ขั้นตอนที่ 3: เดินเข้าหาตามแนวระนาบปกติ (คำนวณสดตลอดเวลา ไม่ก้มหน้าจอ)
        print("\n[3] ขั้นตอนที่ 3: เดินเข้าหาเป้าหมายตามแนวระนาบปกติ (คำนวณสดตลอดเวลา)...")
        reached_close = False

        for step in range(30):
            curr_scr = get_screen()

            # หากขึ้นปุ่มรูปมือระหว่างเดิน แสดงว่าประชิดเป้าหมายแล้ว
            if check_pickup_hand(curr_scr)[0]:
                print("\n[🎯] พบปุ่มรูปมือขึ้นมาแล้ว -> ประชิดเป้าหมายเรียบร้อย!")
                reached_close = True
                break

            found_step, src, cur_x, cur_y, cur_h = detect_target_tree(curr_scr)

            if found_step:
                ang = pixel_to_angle(cur_x)
                if abs(ang) > TURN_THRESHOLD_DEG:
                    turn_camera(ang)
                    curr_scr = get_screen()
                    _, _, cur_x, cur_y, cur_h = detect_target_tree(curr_scr)

                walk_deg, direction_label = calculate_walk_angle(cur_x)
                offset_center = cur_x - (SCREEN_W // 2)

                print(f"[*] ก้าวที่ {step + 1:02d} | พิกัดสด: ({cur_x:4d}, {cur_y:3d}) | "
                      f"สูง: {cur_h:3d}px | เบี่ยง: {offset_center:+4d}px | เดิน: {direction_label}")

                # ถ้าลำต้นขยายใหญ่เต็มสายตา แสดงว่าถึงระยะประชิดแล้ว
                if step >= 3 and cur_h >= 360:
                    print(f"\n[🎯] ลำต้นขยายใหญ่เต็มหน้าจอ ({cur_h}px) -> ประชิดระยะฟันสมบูรณ์!")
                    reached_close = True
                    break

                walk_direction(walk_deg, duration_ms=480)
                time.sleep(0.08)
            else:
                print(f"[*] ก้าวที่ {step + 1:02d} | เป้าหมายหลุดมุมมองชั่วคราว -> ดันเดินตรงต่อ")
                walk_direction(0, duration_ms=360)
                time.sleep(0.08)

        # ขั้นตอนที่ 4: ตรวจสอบเป้าหมาย (ตี 1 ครั้งเพื่อเช็กชื่อ)
        print(f"\n[4] ขั้นตอนที่ 4: ทดสอบกดตี 1 ครั้งเพื่อเช็กชื่อเป้าหมาย...")
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
                print("\n[!] ล้มเหลว: วนกลับมาตีรอบที่ 2 แล้วยังไม่มีชื่อขึ้นมา -> สั่งหยุดทำงาน")
                return

        retry_mode = False

        # ขั้นตอนที่ 5: ฟันต่อเนื่องจนเจอปุ่มรูปมือ -> กดปุ่มมือ 2 ครั้ง
        print(f"\n[5] ขั้นตอนที่ 5: ชื่อถูกต้อง! เริ่มต้นฟันต่อเนื่องจนกว่าจะเจอปุ่มรูปมือ...")
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
                    print(f"\n[🖐️] ตรวจพบปุ่มรูปมือในจอ [{hand_src}] หลังตีไป {hits} ครั้ง!")
                    break

        if not hand_detected:
            final_chk = get_screen()
            is_hand, hand_x, hand_y, hand_src = check_pickup_hand(final_chk)
            if is_hand:
                hand_detected = True
                hx, hy = hand_x, hand_y

        if not hand_detected:
            print("\n[!] ล้มเหลว: ฟันจนครบกำหนดแล้วแต่ไม่พบปุ่มรูปมือ (pickup_hand.png) -> หยุดทำงาน")
            return

        print(f"[🖐️] ทำการกดปุ่มรูปมือ 2 ครั้ง ที่พิกัด ({hx}, {hy})...")
        tap(hx, hy)
        time.sleep(0.3)
        tap(hx, hy)
        time.sleep(0.8)

        # ขั้นตอนที่ 6: ต้องขึ้นหน้าต่างคำว่า "เก็บ" / "Pick All" เท่านั้น
        print("\n[6] ขั้นตอนที่ 6: ตรวจสอบหน้าต่างไอเทม (ต้องมีปุ่มเขียว / คำว่า 'Pick All')...")
        dialog_scr = get_screen()
        pick_all_ready, px, py = check_pick_all_dialog(dialog_scr)

        if not pick_all_ready:
            print("\n[!] ล้มเหลว (Error): ไม่พบหน้าต่างหรือคำว่า 'เก็บ / Pick All' ขึ้นมา -> สั่งหยุดทำงานทันที")
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
