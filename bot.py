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

# พารามิเตอร์หน้าจอและมุมมอง
SCREEN_W = 1280
SCREEN_H = 720
HFOV_DEG = 60
DEADZONE_PX = 35

# จุดพิกัดอ้างอิงเป้าหมายที่กำหนดใหม่
TARGET_REF_X = 472
TARGET_REF_Y = 753

# พิกัดปุ่มควบคุมในเกม
PICK_ALL_BTN = (335, 545)
ATTACK_BTN = (1100, 530)
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


def verify_tree_bark_color(screen, target_x, target_y):
    """
    ตรวจสอบเฉดสีเนื้อไม้ (Bark Color) บริเวณพิกัดเป้าหมาย
    """
    if screen is None:
        return False

    h, w = screen.shape[:2]
    # ครอปพื้นที่ขนาด 60x80 รอบจุดเป้าหมาย (ป้องกันพิกัดเกินขอบจอจริง)
    x1 = max(0, min(w - 1, target_x - 30))
    x2 = min(w, max(1, target_x + 30))
    y1 = max(0, min(h - 1, target_y - 40))
    y2 = min(h, max(1, target_y + 40))

    roi = screen[y1:y2, x1:x2]
    if roi.size == 0:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_bark = np.array([8, 25, 20])
    upper_bark = np.array([32, 230, 180])
    mask = cv2.inRange(hsv, lower_bark, upper_bark)

    pixel_count = np.sum(mask > 0)
    return pixel_count > 120


def detect_target_tree(screen):
    """ตรวจจับต้นไม้และส่งคืนพิกัด X, Y, ความสูง และชื่อไอคอนที่เจอ"""
    if screen is None:
        return False, "ไม่พบ", 0, 0, 0

    target_roi = screen[40:600, 80:1200]
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

            res = cv2.matchTemplate(gray_roi, cv2.resize(template, (tw, th)), cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val >= 0.47 and max_val > best_match_val:
                best_match_val = max_val
                best_x = 80 + max_loc[0] + (tw // 2)
                best_y = 40 + max_loc[1] + (th // 2)
                best_h = th
                matched_filename = f"{fname} ({max_val:.2f})"

    if best_match_val >= 0.47:
        return True, matched_filename, best_x, best_y, best_h

    return False, "ไม่พบ", 0, 0, 0


def align_camera_to_center(target_x, target_y):
    """หมุนปรับมุมมองกล้องให้พิกัดต้นไม้วิ่งเข้าหาจุด (472, 753)"""
    for _ in range(5):
        diff_x = target_x - TARGET_REF_X
        diff_y = target_y - TARGET_REF_Y

        if abs(diff_x) <= 45 and abs(diff_y) <= 45:
            break

        move_x = int(np.clip(diff_x * 0.45, -160, 160))
        move_y = int(np.clip(diff_y * 0.40, -110, 110))

        cx, cy = 850, 360
        swipe_screen(cx, cy, cx - move_x, cy - move_y, duration=180)
        time.sleep(0.25)

        chk = get_screen()
        found, _, cur_x, cur_y, _ = detect_target_tree(chk)
        if found:
            target_x, target_y = cur_x, cur_y
        else:
            break

    return target_x, target_y


def walk_direction(deg=0, duration_ms=650):
    rad = math.radians(deg)
    tx = int(JOY_CENTER_X + JOY_RADIUS * math.sin(rad))
    ty = int(JOY_CENTER_Y - JOY_RADIUS * math.cos(rad))
    swipe_screen(JOY_CENTER_X, JOY_CENTER_Y, tx, ty, duration=duration_ms)
    time.sleep(0.1)


def step_backward(duration_ms=420):
    """ถอยหลัง 1 ก้าวเพื่อตั้งหลักใหม่"""
    tx = JOY_CENTER_X
    ty = int(JOY_CENTER_Y + JOY_RADIUS)
    swipe_screen(JOY_CENTER_X, JOY_CENTER_Y, tx, ty, duration=duration_ms)
    time.sleep(0.25)


def rotate_view_right(dist=180):
    cx, cy = 850, 240
    swipe_screen(cx, cy, cx - dist, cy, duration=220)
    time.sleep(0.35)


def pixel_to_angle(target_x):
    off = target_x - TARGET_REF_X
    if abs(off) <= DEADZONE_PX:
        return 0.0
    f = (SCREEN_W / 2) / math.tan(math.radians(HFOV_DEG / 2))
    return math.degrees(math.atan(off / f))


def check_target_name_match(screen):
    """ตรวจสอบชื่อเป้าหมายในโซนขวาบน (X=860-1250, Y=160-320)"""
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


def main():
    print("=" * 60)
    print(f"[*] UtopiaFarmAI: บอทฟาร์มต้นไม้อัตโนมัติ (จุดอ้างอิงเป้าหมาย: {TARGET_REF_X}, {TARGET_REF_Y})")
    print(f"[*] โฟลเดอร์ไอคอน: {ICONS_DIR}")
    print("=" * 60)
    time.sleep(2)

    round_count = 0
    retry_mode = False

    while True:
        round_count += 1
        print(f"\n{'='*20} เริ่มต้นลูปฟาร์มรอบที่ {round_count} {'='*20}")

        # ขั้นตอนที่ 1 & 2: หมุนหาต้นไม้แล้วหยุดนิ่งเพื่อวิเคราะห์พิกัด
        print("[1] ขั้นตอนที่ 1: กวาดสายตา 360 องศาค้นหาต้นไม้...")
        candidate_found = False
        target_x = 0
        target_y = 0

        for step in range(8):
            scr = get_screen()
            found, src, cx, cy, h = detect_target_tree(scr)
            if found:
                candidate_found = True
                target_x = cx
                target_y = cy
                print(f"\n[🛑] ขั้นตอนที่ 2: พบต้นไม้แล้วสั่งหยุดทันที! พิกัด: X={cx}, Y={cy} (สูง={h}px, {src})")
                break
            else:
                print(f"[🔄] สแกนมุม {step + 1}/8 -> หมุนขวาต่อ...")
                rotate_view_right(dist=180)

        if not candidate_found:
            print("\n[!] ล้มเหลว: กวาดหา 360 องศาแล้วไม่พบต้นไม้ -> หยุดการทำงาน")
            return

        # ขั้นตอนที่ 3: หันหน้าจอเล็งพิกัดเป้าหมายเข้าหาจุด (472, 753) ก่อนเดินเสมอ
        print(f"\n[3] ขั้นตอนที่ 3: ปรับมุมกล้องดึงเป้าหมาย (X={target_x}, Y={target_y}) เข้าสู่พิกัดอ้างอิง ({TARGET_REF_X}, {TARGET_REF_Y})...")
        target_x, target_y = align_camera_to_center(target_x, target_y)
        print(f"[✓] หันหน้าจอล็อกเป้าหมายเรียบร้อย (พิกัดล่าสุด: X={target_x}, Y={target_y})")

        # ตรวจสอบสีเนื้อไม้หลังการหันจอ
        center_scr = get_screen()
        is_bark_color = verify_tree_bark_color(center_scr, target_x, target_y)

        if not is_bark_color:
            print("[⚠️] ตรวจสอบเฉดสีแล้วไม่ใช่เนื้อไม้ (อาจเป็นหินหรือฉากหลัง) -> หมุนค้นหาใหม่...")
            rotate_view_right(dist=180)
            continue
        else:
            print("[✓] ยืนยันเฉดสีเนื้อไม้ถูกต้องสมบูรณ์! เตรียมคำนวณการเดิน...")

        # ขั้นตอนที่ 4: คำนวณการเดินพุ่งตรงเข้าหาเป้าหมาย
        print("\n[4] ขั้นตอนที่ 4: คำนวณการเดินพุ่งตรงเข้าหาเป้าหมาย (ตรวจสอบภาพสดทุกก้าว)...")
        reached_close = False

        for step in range(25):
            curr_scr = get_screen()

            # ตรวจจับปุ่มรูปมือก่อนทุกก้าว
            if check_pickup_hand(curr_scr)[0]:
                print("\n[🎯] ตรวจพบปุ่มรูปมือขึ้นมาแล้ว -> ประชิดเป้าหมายเรียบร้อย!")
                reached_close = True
                break

            found_step, src, cur_x, cur_y, cur_h = detect_target_tree(curr_scr)

            if found_step:
                offset_x = cur_x - TARGET_REF_X
                offset_y = cur_y - TARGET_REF_Y

                # ปรับแต่งมุมกล้องให้อยู่ตรงเป้าหมายอ้างอิงเสมอ
                if abs(offset_x) > 50 or abs(offset_y) > 50:
                    cur_x, cur_y = align_camera_to_center(cur_x, cur_y)
                    curr_scr = get_screen()
                    _, _, cur_x, cur_y, cur_h = detect_target_tree(curr_scr)

                ang = pixel_to_angle(cur_x)
                walk_deg = ang % 360
                direction_label = "เอียงซ้าย" if ang < -1.5 else ("เอียงขวา" if ang > 1.5 else "ตรงกลาง")

                print(f"[📍 TARGET LOCKED] ก้าวที่ {step + 1:02d} | พิกัด: ({cur_x:4d}, {cur_y:3d}) | สูง: {cur_h:3d}px | เดิน: {direction_label}")

                # เงื่อนไขระยะประชิดเนื้อไม้ (เดินอย่างน้อย 3 ก้าว และลำต้นขยายใหญ่ >= 380px)
                if step >= 3 and cur_h >= 380:
                    print(f"\n[🎯] ลำต้นขยายใหญ่เต็มหน้าจอ ({cur_h}px) -> ถึงระยะฟันสมบูรณ์!")
                    reached_close = True
                    break

                walk_direction(walk_deg, duration_ms=650)
                time.sleep(0.1)
            else:
                print(f"[⚠️] หลุดมุมมองเป้าหมายชั่วคราว -> ถอยหลัง 1 ก้าวเพื่อค้นหาใหม่...")
                step_backward(duration_ms=380)
                time.sleep(0.2)

        # ขั้นตอนที่ 5: ตี 1 ครั้งเพื่อตรวจสอบชื่อ
        print(f"\n[5] ขั้นตอนที่ 5: กดตี 1 ครั้งเพื่อเช็กชื่อเป้าหมาย...")
        tap(ATTACK_BTN[0], ATTACK_BTN[1])
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
                print("\n[!] ล้มเหลว: ย้อนกลับมาตีรอบ 2 แล้วยังไม่พบชื่อ -> สั่งหยุดทำงาน")
                return

        retry_mode = False

        # ขั้นตอนที่ 6: ฟันต่อเนื่องจนเจอปุ่มรูปมือ -> แตะปุ่มรูปมือ 2 ครั้ง
        print(f"\n[6] ขั้นตอนที่ 6: ป้ายชื่อถูกต้อง! ฟันต่อเนื่องจนกว่าจะเจอปุ่มรูปมือ...")
        hits = 0
        hand_detected = False
        hx, hy = 0, 0

        while hits < 45:
            hits += 1
            tap(ATTACK_BTN[0], ATTACK_BTN[1])
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
            print("\n[!] ล้มเหลว: ฟันครบกำหนดแล้วแต่ไม่พบปุ่มรูปมือ (pickup_hand.png) -> หยุดทำงาน")
            return

        print(f"[🖐️] ทำการกดปุ่มรูปมือ 2 ครั้ง ที่พิกัด ({hx}, {hy})...")
        tap(hx, hy)
        time.sleep(0.3)
        tap(hx, hy)
        time.sleep(0.8)

        # ขั้นตอนที่ 7: ตรวจสอบหน้าต่างไอเทม (Pick All) แล้วกดเก็บ
        print("\n[7] ขั้นตอนที่ 7: ตรวจสอบหน้าต่างไอเทม (ต้องมีปุ่มเขียว หรือคำว่า 'Pick All')...")
        dialog_scr = get_screen()
        pick_all_ready, px, py = check_pick_all_dialog(dialog_scr)

        if not pick_all_ready:
            print("\n[!] ล้มเหลว (Error): ไม่พบหน้าต่างหรือคำว่า 'เก็บ / Pick All' ขึ้นมา -> สั่งหยุดทำงานทันที")
            return

        print(f"[📦] ตรวจพบคำว่า 'Pick All' สมบูรณ์! แตะเก็บไอเทมที่ ({px}, {py})...")
        tap(px, py)
        time.sleep(0.25)
        tap(px, py)
        print("[✓] เก็บไอเทมเข้าตัวเสร็จสมบูรณ์!")

        # พักรอ 5 วินาทีก่อนเริ่มรอบถัดไป
        print("\n[*] พักรอ 5 วินาทีเพื่อให้โมเดลในฉากหายสนิท...")
        for wait_sec in range(5, 0, -1):
            print(f"[*] พักรอ... เหลือ {wait_sec} วิ")
            time.sleep(1.0)

        print("[🚀] เริ่มต้นสแกนหาต้นต่อไปทันที...\n")


if __name__ == "__main__":
    main()
