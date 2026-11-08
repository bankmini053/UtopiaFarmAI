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
OUTPUT_LOCAL_IMG = os.path.expanduser("~/UtopiaFarmAI/bot_locked_target.png")
OUTPUT_GALLERY_IMG = "/sdcard/Download/bot_locked_target.png"
ICONS_DIR = os.path.expanduser("~/UtopiaFarmAI/icons")

# พารามิเตอร์หน้าจอและมุมมองกล้อง
SCREEN_W = 1280
HFOV_DEG = 60
DEADZONE_PX = 35
TURN_THRESHOLD_DEG = 18
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
    run_cmd(f"chmod 777 {TMP_PATH}")
    if os.path.exists(TMP_PATH):
        img = cv2.imread(TMP_PATH)
        if img is not None:
            h, w = img.shape[:2]
            if h > w:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            return img
    return None


def scan_all_trees_radar(screen):
    """
    สแกนตรวจจับต้นไม้ทุกมุมจอ คัดกรองจุดซ้ำซ้อน 
    และคัดเลือกเฉพาะ ต้นไม้ที่ใหญ่ที่สุดและใกล้ที่สุด (ตัดขอบจอและปุ่ม UI ออก)
    """
    if screen is None:
        return None, []

    h, w = screen.shape[:2]
    gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    tree_files = ["tree_coconut.png", "tree_coconut2.png", "tree_name1.png"]
    
    all_candidates = []

    for fname in tree_files:
        fpath = os.path.join(ICONS_DIR, fname)
        if not os.path.exists(fpath):
            continue

        tpl = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if tpl is None:
            continue

        for scale in [0.6, 0.8, 1.0, 1.3, 1.6]:
            tw = int(tpl.shape[1] * scale)
            th = int(tpl.shape[0] * scale)
            if th >= h or tw >= w or th < 25 or tw < 25:
                continue

            res = cv2.matchTemplate(gray_screen, cv2.resize(tpl, (tw, th)), cv2.TM_CCOEFF_NORMED)
            locs = np.where(res >= 0.48)

            for pt in zip(*locs[::-1]):
                bx, by = pt[0], pt[1]
                cx = bx + (tw // 2)
                cy = by + (th // 2)

                # Safe Zone: ตัดขอบจอที่มักเป็นปุ่ม UI/จอยสติ๊ก
                if cy > 580 or (cx < 200 and cy > 420) or (cx > 1050 and cy > 420) or cy < 70:
                    continue

                # ตัดจุดซ้ำซ้อนใกล้กันเกิน 50px
                if not any(abs(cx - item["cx"]) < 50 and abs(cy - item["cy"]) < 50 for item in all_candidates):
                    all_candidates.append({
                        "file": fname,
                        "cx": cx,
                        "cy": cy,
                        "bx": bx,
                        "by": by,
                        "w": tw,
                        "h": th,
                        "area": tw * th
                    })

    if not all_candidates:
        return None, []

    # เลือกต้นที่ใหญ่ที่สุด (Area มากที่สุด)
    all_candidates.sort(key=lambda x: x["area"], reverse=True)
    return all_candidates[0], all_candidates


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
                if val >= 0.54:
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

    # 2. เช็ก OCR
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, lang="eng+tha", config=r"--psm 7").strip()
    if any(k in text.lower() for k in ["pick", "all", "เก็บ"]):
        return True, PICK_ALL_BTN[0], PICK_ALL_BTN[1]

    return False, 0, 0


def check_target_name_match(screen):
    """ตรวจสอบชื่อเป้าหมาย: ตรวจในโซนขวาบน (X=860-1250, Y=160-320)"""
    if screen is None:
        return False, "ไม่พบภาพ"

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
                    if val >= 0.48:
                        return True, f"ป้ายชื่อตรง {r_name} ({val:.2f})"

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh, lang="eng+tha", config=r"--psm 7").strip()

    if any(k in text.lower() for k in ["stone", "mine", "หิน"]):
        return False, f"พบก้อนหิน ({text}) ไม่ใช่ต้นไม้"

    if any(k in text.lower() for k in ["rubber", "coconut", "palm", "tree", "ยาง", "มะพร้าว"]):
        return True, f"OCR พบชื่อ: '{text}'"

    return False, "ชื่อไม่ตรงกับ tree_rubber.png หรือ name_coconut.png"


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
    label = "เอียงซ้าย" if ang < -1.5 else ("เอียงขวา" if ang > 1.5 else "ตรง")
    return ang % 360, f"{label} {ang:+.1f}°"


def main():
    print("=" * 60)
    print("[*] UtopiaFarmAI: ระบบตรวจจับเรดาร์ & ตัดต้นไม้อัตโนมัติ (ฉบับสมบูรณ์)")
    print(f"[*] โฟลเดอร์ต้นแบบ: {ICONS_DIR}")
    print("=" * 60)
    time.sleep(2)

    round_count = 0
    retry_mode = False

    while True:
        round_count += 1
        print(f"\n{'='*20} เริ่มต้นลูปฟาร์มรอบที่ {round_count} {'='*20}")

        # ขั้นตอนที่ 1: กวาดสายตา 360 องศา สแกนวัตถุต้นไม้รอบตัว
        print("[1] ขั้นตอนที่ 1: กวาดสายตา 360 องศา สแกนวัตถุต้นไม้รอบตัว...")
        candidate_found = False
        selected_tree = None

        for step in range(8):
            scr = get_screen()
            best_t, all_t = scan_all_trees_radar(scr)
            if best_t is not None:
                candidate_found = True
                selected_tree = best_t
                print(f"\n[🌲] มุม {step + 1}/8: ตรวจพบ {len(all_t)} จุด -> เลือกต้นเป้าหมาย [{best_t['file']}] (X={best_t['cx']}, พื้นที่={best_t['area']}px)")
                break
            else:
                print(f"[🔄] สแกนมุม {step + 1}/8 (ไม่พบ) -> หมุนขวาต่อ...")
                rotate_view_right(dist=180)

        if not candidate_found or selected_tree is None:
            print("\n" + "=" * 55)
            print("[!] ล้มเหลว: กวาดหา 360 องศาแล้วไม่พบต้นไม้ -> หยุดการทำงาน")
            print("=" * 55)
            return

        # ขั้นตอนที่ 2: ตีกรอบสีเขียวล็อกเป้าหมาย (Lock-On Visual)
        print("\n[2] ขั้นตอนที่ 2: ล็อกหน้าจอนิ่ง และบันทึกภาพกรอบเขียวล็อกเป้าหมาย...")
        time.sleep(0.8)
        lock_scr = get_screen()
        if lock_scr is not None:
            bx, by, bw, bh = selected_tree["bx"], selected_tree["by"], selected_tree["w"], selected_tree["h"]
            debug_view = lock_scr.copy()
            cv2.rectangle(debug_view, (bx, by), (bx + bw, by + bh), (0, 255, 0), 3)
            cv2.circle(debug_view, (selected_tree["cx"], selected_tree["cy"]), 6, (0, 0, 255), -1)
            cv2.putText(debug_view, f"TARGET: {selected_tree['file']}", (bx, max(30, by - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imwrite(OUTPUT_LOCAL_IMG, debug_view)
            cv2.imwrite(OUTPUT_GALLERY_IMG, debug_view)
            print(f"[🎯] ล็อกกรอบเขียวสำเร็จ! (เซฟภาพที่: {OUTPUT_GALLERY_IMG})")

        # ขั้นตอนที่ 3: เดินเข้าหาตามแนวราบ (ตัดการก้มมองพื้นออก)
        print("\n[3] ขั้นตอนที่ 3: เดินเข้าหาเป้าหมายตามแนวระนาบปกติ...")
        reached_close = False

        for step in range(25):
            curr_scr = get_screen()
            if check_pickup_hand(curr_scr)[0]:
                print("\n[🎯] พบปุ่มรูปมือขึ้นมาแล้ว -> ประชิดเป้าหมายเรียบร้อย!")
                reached_close = True
                break

            cur_best, _ = scan_all_trees_radar(curr_scr)
            if cur_best is not None:
                cur_x, cur_h = cur_best["cx"], cur_best["h"]
                ang = pixel_to_angle(cur_x)
                if abs(ang) > TURN_THRESHOLD_DEG:
                    turn_camera(ang)
                    curr_scr = get_screen()

                walk_deg, direction_label = calculate_walk_angle(cur_x)
                print(f"[*] ก้าวที่ {step + 1} | เดิน: {direction_label} ({walk_deg:.1f}°) | สูง={cur_h}px")

                if cur_h >= 260:
                    print(f"\n[🎯] ลำต้นขยายใหญ่ ({cur_h}px) -> ถึงระยะฟันสมบูรณ์!")
                    reached_close = True
                    break

                walk_direction(walk_deg, duration_ms=460)
                time.sleep(0.1)
            else:
                walk_direction(0, duration_ms=350)
                time.sleep(0.1)

        # ขั้นตอนที่ 4: ตี 1 ครั้งเพื่อตรวจสอบป้ายชื่อ
        print(f"\n[4] ขั้นตอนที่ 4: กดตี 1 ครั้งเพื่อตรวจสอบป้ายชื่อเป้าหมาย...")
        tap(1100, 530)
        time.sleep(0.7)

        target_scr = get_screen()
        name_matched, name_detail = check_target_name_match(target_scr)
        print(f"[*] ผลตรวจป้ายชื่อ: {name_detail}")

        if not name_matched:
            if not retry_mode:
                print("\n[⚠️] ตี 1 ครั้งแล้วไม่พบชื่อ -> ย้อนกลับไปค้นหาใหม่ (แก้มือ 1 รอบ)...")
                retry_mode = True
                rotate_view_right(dist=180)
                continue
            else:
                print("\n" + "=" * 60)
                print("[!] ล้มเหลว: ย้อนกลับมาตีรอบ 2 แล้วยังไม่พบชื่อ -> หยุดทำงานทันที")
                print("=" * 60)
                return

        retry_mode = False

        # ขั้นตอนที่ 5: ฟันต่อเนื่องจนเจอปุ่มมือ -> กดรูปมือ 2 ครั้ง
        print(f"\n[5] ขั้นตอนที่ 5: ป้ายชื่อถูกต้อง! ฟันต่อเนื่องจนกว่าจะเจอปุ่มรูปมือ...")
        hits, hand_detected, hx, hy = 0, False, 0, 0

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
            print("[!] ล้มเหลว: เลือดต้นไม้หมดแล้ว แต่ไม่เจอปุ่มรูปมือ -> หยุดทำงาน")
            print("=" * 60)
            return

        print(f"[🖐️] แตะปุ่มรูปมือ 2 ครั้ง ที่พิกัด ({hx}, {hy})...")
        tap(hx, hy)
        time.sleep(0.3)
        tap(hx, hy)
        time.sleep(0.8)

        # ขั้นตอนที่ 6: ตรวจสอบหน้าต่างไอเทม (ต้องมีปุ่มเขียว หรือคำว่า Pick All)
        print("\n[6] ขั้นตอนที่ 6: ตรวจสอบหน้าต่างไอเทม (ต้องมีปุ่มเขียว หรือคำว่า 'Pick All')...")
        dialog_scr = get_screen()
        pick_all_ready, px, py = check_pick_all_dialog(dialog_scr)

        if not pick_all_ready:
            print("\n" + "=" * 60)
            print("[!] ล้มเหลว (Error): ไม่พบหน้าต่าง 'Pick All / เก็บ' -> สั่งหยุดทำงานทันที")
            print("=" * 60)
            return

        print(f"[📦] พบคำว่า 'Pick All' สมบูรณ์! กดเก็บไอเทมที่ ({px}, {py})...")
        tap(px, py)
        time.sleep(0.25)
        tap(px, py)
        print("[✓] เก็บไอเทมเรียบร้อย!")

        # ขั้นตอนที่ 7: หน่วงเวลารอ 5 วินาที
        print("\n[7] ขั้นตอนที่ 7: หน่วงเวลารอ 5 วินาทีเพื่อให้โมเดลต้นไม้หายสนิท...")
        for wait_sec in range(5, 0, -1):
            print(f"[*] พักรอ... {wait_sec} วิ")
            time.sleep(1.0)

        print("[🚀] เริ่มต้นสแกนหาต้นต่อไปทันที...\n")


if __name__ == "__main__":
    main()
