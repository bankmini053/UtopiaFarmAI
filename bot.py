import os
import re
import sys
import time
import math
import subprocess
import cv2
import numpy as np
import pytesseract

ROOT_SU = "/system/xbin/su"
TMP_PATH = "/data/local/tmp/screen.png"

TARGET_CHEST = (17529, 15339)
NO_COORD_TIMEOUT = 5      # ไม่เห็นพิกัดเกินกี่วินาทีให้ล้มเหลว
STOP_DURATION = 2         # หยุดนิ่งก่อนอ่านพิกัดทุกรอบ (วินาที)
ARRIVE_DIST = 3.0         # ระยะที่ถือว่าถึงกล่อง
FINE_DIST = 8.0           # ใกล้กว่านี้ = โหมดขยับทีละนิด
MAX_WORSE = 3             # ห่างขึ้นติดกันกี่รอบให้ล้มเหลว
MAX_STALL = 8             # ไม่ใกล้ขึ้นติดกันกี่รอบให้หยุด (ใกล้สุดแล้ว)
NEAR_TAP_DIST = 6.0       # ถ้าหยุดเพราะ stall และห่างไม่เกินนี้ ให้ลองกดเปิดกล่อง
MAX_STEPS = 100
CAL_MS = 900              # หน่วยก้าวอ้างอิงของโมเดล
CAL_WALKS = (1500, 2200, 900)   # เวลาเดินตอนสอบเทียบ (ลองตามลำดับ)
LEARN_RATE = 0.6          # ความเร็วในการปรับโมเดลจากผลเดินจริง
DIAGONAL = False          # False = เดินสี่เหลี่ยม, True = เดินเฉียง (เฉพาะตอนไกล)
TAP_OPEN = (820, 430)     # ปุ่มเปิดกล่อง

# --- การอ่านภาพ (OCR) ---
VERBOSE_OCR = True        # พิมพ์ผลอ่านทุกครั้ง (กี่โหวต/ค่าไหนบ้าง)
EARLY_VOTES = 3           # ค่าไหนได้ครบกี่โหวตให้หยุดอ่านทันที
MAX_JUMP = 25             # อ่านได้โหวตเดียว: ต้องไม่ต่างจากตำแหน่งก่อนหน้าเกินนี้
# กรอบอ่านพิกัด (y1, y2, x1, x2) บนจอแนวนอน ลองหลายกรอบ
ROIS = [(180, 300, 40, 270), (170, 310, 20, 300), (190, 290, 60, 260)]

JOY_CENTER_X = 250
JOY_CENTER_Y = 530
JOY_RADIUS = 145

_last_screen = None
LAST_POS = None


# ------------------------------------------------------------ พื้นฐาน
def tap(x, y):
    subprocess.run(f"{ROOT_SU} -c 'input tap {x} {y}'", shell=True)


def swipe(x1, y1, x2, y2, duration=250):
    subprocess.run(
        f"{ROOT_SU} -c 'input swipe {x1} {y1} {x2} {y2} {duration}'", shell=True
    )


def get_screen():
    subprocess.run(f"{ROOT_SU} -c 'screencap -p {TMP_PATH}'", shell=True)
    if os.path.exists(TMP_PATH):
        return cv2.imread(TMP_PATH)
    return None


# ------------------------------------------------------------ อ่านพิกัด (OCR หลายวิธี + โหวต)
def crop_roi(screen, idx):
    y1, y2, x1, x2 = ROIS[idx]
    h, w = screen.shape[:2]
    if w > h:
        return screen[y1:y2, x1:x2]
    roi = screen[x1:x2, y1:y2]            # จอแนวตั้ง: สลับแกนแล้วหมุน
    return cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)


def variants(roi):
    """ภาพหลายแบบให้ OCR ลอง (ชื่อ, ภาพ, psm)"""
    big = cv2.resize(roi, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(big, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 70, 255]))
    k = np.ones((2, 2), np.uint8)
    yield "white", white, 6
    yield "white+dilate", cv2.dilate(white, k), 6
    yield "white-psm11", white, 11
    for th in (195, 175, 215, 235, 150):
        yield f"th{th}", cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)[1], 6
    yield "otsu", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1], 6
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    yield "adaptive", cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -15
    ), 6


def ocr_raw(img, psm):
    return pytesseract.image_to_string(
        img, config=f"--psm {psm} -c tessedit_char_whitelist=0123456789,."
    )


def parse(raw):
    for pattern in (r"\d{5}", r"\d{4,6}"):
        nums = re.findall(pattern, raw)
        if len(nums) >= 2:
            x, y = int(nums[0]), int(nums[1])
            if 17400 <= x <= 17600 and 15250 <= y <= 15400:
                return (x, y)
    return None


def vote_ocr(screen):
    """อ่านด้วยหลายกรอบ x หลายวิธี แล้วนับโหวตแต่ละค่า"""
    votes = {}
    total = 0
    for idx in range(len(ROIS)):
        roi = crop_roi(screen, idx)
        if roi.size == 0:
            continue
        for _name, img, psm in variants(roi):
            total += 1
            p = parse(ocr_raw(img, psm))
            if p:
                votes[p] = votes.get(p, 0) + 1
            if votes and max(votes.values()) >= EARLY_VOTES:
                return votes, total
    return votes, total


def pick(votes):
    """เลือกค่าที่เชื่อถือได้ ไม่งั้นคืน None"""
    if not votes:
        return None
    best_cnt = max(votes.values())
    cands = [p for p, c in votes.items() if c == best_cnt]
    if best_cnt >= 2:
        return cands[0]
    # โหวตเดียว: เชื่อเมื่อไม่กระโดดไกลจากตำแหน่งก่อนหน้า
    if LAST_POS is not None:
        cands.sort(key=lambda p: math.hypot(p[0] - LAST_POS[0], p[1] - LAST_POS[1]))
        p = cands[0]
        if math.hypot(p[0] - LAST_POS[0], p[1] - LAST_POS[1]) <= MAX_JUMP:
            return p
    return None


def save_debug():
    try:
        if _last_screen is not None:
            cv2.imwrite("ocr_fail.png", _last_screen)
            print("[i] บันทึกภาพตอนอ่านพิกัดไม่ได้ไว้ที่ ocr_fail.png")
    except Exception:
        pass


def read_pos(timeout=NO_COORD_TIMEOUT):
    """อ่านพิกัดจากหน้าจอ (โหวตหลายวิธี) ไม่เห็นเกิน timeout คืน None"""
    global _last_screen, LAST_POS
    start = time.time()
    while True:
        t0 = time.time()
        _last_screen = get_screen()
        if _last_screen is not None:
            votes, total = vote_ocr(_last_screen)
        else:
            votes, total = {}, 0
        pos = pick(votes)
        if VERBOSE_OCR:
            vs = {f"{p[0]},{p[1]}": c for p, c in votes.items()}
            if pos:
                print(f"    [ocr] อ่านได้ ({pos[0]},{pos[1]}) โหวต {votes[pos]}/{total} "
                      f"ทั้งหมด {vs} ({time.time() - t0:.1f}s)")
            else:
                print(f"    [ocr] อ่านไม่ได้/ไม่มั่นใจ ลอง {total} แบบ ได้ {vs} "
                      f"({time.time() - t0:.1f}s)")
        if pos:
            LAST_POS = pos
            return pos
        if time.time() - start >= timeout:
            break
        time.sleep(0.1)
    save_debug()
    return None


def ocr_test(path=None):
    """python utopia_walk.py ocr [ไฟล์ภาพ] : ดูว่าแต่ละวิธีอ่านได้อะไร"""
    scr = cv2.imread(path) if path else get_screen()
    if scr is None:
        print("[!] ไม่มีภาพ")
        return
    print(f"ภาพขนาด {scr.shape[1]}x{scr.shape[0]}")
    for idx in range(len(ROIS)):
        roi = crop_roi(scr, idx)
        cv2.imwrite(f"roi{idx}.png", roi)
        print(f"--- กรอบ {idx} {ROIS[idx]} (บันทึก roi{idx}.png) ---")
        for name, img, psm in variants(roi):
            raw = ocr_raw(img, psm).strip().replace("\n", " | ")
            print(f"{name:13} psm{psm:<2}: {raw!r} -> {parse(raw)}")
    votes, total = vote_ocr(scr)
    print("โหวตรวม:", {f"{p[0]},{p[1]}": c for p, c in votes.items()}, f"จาก {total} แบบ")


# ------------------------------------------------------------ เดิน
def walk_in_direction(angle_deg, duration_ms=450):
    """0°=หน้า, 90°=ขวา, 180°=หลัง, 270°=ซ้าย"""
    rad = math.radians(angle_deg)
    tx = int(JOY_CENTER_X + JOY_RADIUS * math.sin(rad))
    ty = int(JOY_CENTER_Y - JOY_RADIUS * math.cos(rad))
    swipe(JOY_CENTER_X, JOY_CENTER_Y, tx, ty, duration=int(duration_ms))
    time.sleep(0.2)


def measure_step(deg):
    """เดินสอบเทียบ คืนค่าพิกัดที่เปลี่ยนต่อ CAL_MS (dx, dy)"""
    for ms in CAL_WALKS:
        time.sleep(1.0)
        p1 = read_pos()
        walk_in_direction(deg, ms)
        time.sleep(STOP_DURATION)
        p2 = read_pos()
        print(f"[cal] deg={deg} ms={ms} p1={p1} p2={p2}")
        if p1 and p2:
            v = (p2[0] - p1[0], p2[1] - p1[1])
            d = math.hypot(*v)
            if d > 80:
                print(f"[cal] ค่ากระโดดผิดปกติ ({d:.0f} หน่วย) ข้าม ลองใหม่")
                continue
            if d > 2:
                k = CAL_MS / ms
                return (v[0] * k, v[1] * k)
    return None


def calibrate():
    """เทียบแกนแผนที่กับทิศหน้าจอ (รองรับแกนหมุน/สลับ/กลับด้าน)"""
    f = measure_step(0)    # เดินหน้า
    r = measure_step(90)   # เดินขวา
    if not f or not r:
        return None
    M = np.array([[f[0], r[0]], [f[1], r[1]]], dtype=float)
    if abs(np.linalg.det(M)) < 1e-3:
        return None
    print("[✓] สอบเทียบแล้ว", M.round(2).tolist())
    return M


def plan_move(M, dx, dy, fine):
    """คืน (องศา, มิลลิวินาที, u=(หน้า,ขวา) หน่วย CAL_MS, a, b)"""
    a, b = np.linalg.solve(M, np.array([dx, dy], dtype=float))  # หน้า, ขวา
    if DIAGONAL and not fine:
        deg = math.degrees(math.atan2(b, a)) % 360
        comp = math.hypot(a, b)
    elif abs(a) >= abs(b):
        deg = 0 if a > 0 else 180
        comp = abs(a)
    else:
        deg = 90 if b > 0 else 270
        comp = abs(b)

    if fine:   # ขยับทีละนิด
        ms = int(np.clip(comp * CAL_MS * 0.5, 120, 400))
    else:
        ms = int(np.clip(comp * CAL_MS * 0.7, 150, 1500))

    rad = math.radians(deg)
    k = ms / CAL_MS
    u = np.array([k * math.cos(rad), k * math.sin(rad)])
    return deg, ms, u, a, b


# ------------------------------------------------------------ main
def fail(msg):
    print(f"\n[✗] ล้มเหลว: {msg}")


def main():
    print("=" * 60)
    print("[*] UtopiaFarmAI: หยุดอ่านพิกัด -> คำนวณ -> เดิน (ต้องใกล้ขึ้นเรื่อยๆ)")
    print(f"[*] พิกัดเป้าหมาย: {TARGET_CHEST}")
    print("=" * 60)
    time.sleep(3)

    if read_pos() is None:
        return fail(f"ไม่เห็นพิกัดเกิน {NO_COORD_TIMEOUT} วินาที")

    M = calibrate()
    if M is None:
        return fail("สอบเทียบแกนไม่ได้ (ตัวละครไม่ขยับ/ติดสิ่งกีดขวาง หรือ OCR อ่านไม่ได้)")

    pending = None     # ข้อมูลก้าวที่เพิ่งเดิน เพื่อเทียบ "คาด" กับ "จริง"
    worse = 0
    recals = 0
    best = None
    best_pos = None
    stall = 0

    for step in range(MAX_STEPS):
        time.sleep(STOP_DURATION)
        pos = read_pos()
        if pos is None:
            return fail(f"ไม่เห็นพิกัดเกิน {NO_COORD_TIMEOUT} วินาที")

        dx = TARGET_CHEST[0] - pos[0]
        dy = TARGET_CHEST[1] - pos[1]
        dist = math.hypot(dx, dy)
        fine = dist <= FINE_DIST

        print("-" * 55)
        print(f"[{step}] ตอนนี้ ({pos[0]:.0f},{pos[1]:.0f}) | เป้า {TARGET_CHEST} | "
              f"ห่าง {dist:.1f} (dx={dx:.0f}, dy={dy:.0f})"
              f"{' | โหมดละเอียด' if fine else ''}")

        # ---- รายงานผลก้าวที่แล้ว: คาด vs จริง, ใกล้หรือห่าง
        if pending:
            prev = pending["dist"]
            pred = pending["pred"]
            pred_err = math.hypot(pos[0] - pred[0], pos[1] - pred[1])
            diff = prev - dist
            if diff > 1.0:
                trend = f"ใกล้ขึ้น ✔ (-{diff:.1f})"
            elif diff < -1.0:
                trend = f"ห่างขึ้น ✘ (+{-diff:.1f})"
            else:
                trend = "แทบไม่เปลี่ยน"
            print(f"    ผลจริง: คาดไว้ ({pred[0]:.0f},{pred[1]:.0f}) "
                  f"จริง ({pos[0]:.0f},{pos[1]:.0f}) คลาด {pred_err:.1f} | "
                  f"ห่าง {prev:.1f} -> {dist:.1f} {trend}")

            # เรียนรู้ระยะ/ทิศจากผลจริง (กันค่า OCR กระโดดผิดปกติ)
            obs = np.array([pos[0] - pending["pos"][0], pos[1] - pending["pos"][1]])
            if np.hypot(*obs) < 40:
                u = pending["u"]
                err = obs - M @ u
                M = M + LEARN_RATE * np.outer(err, u) / float(u @ u)
                print(f"    [learn] ปรับโมเดล M={M.round(2).tolist()}")
                if abs(np.linalg.det(M)) < 1e-3:
                    return fail("โมเดลการเดินเพี้ยน (ติดสิ่งกีดขวางหรืออ่านพิกัดผิดบ่อย)")

            if diff < -1.0:
                worse += 1
                print(f"    [!] ห่างขึ้นติดกัน {worse}/{MAX_WORSE}")
                if worse >= MAX_WORSE:
                    return fail("เดินแล้วห่างจากกล่องขึ้นเรื่อยๆ")
                if worse == 2 and recals < 1:
                    print("    [*] สอบเทียบแกนใหม่ 1 ครั้ง")
                    M = calibrate()
                    recals += 1
                    worse = 0
                    pending = None
                    if M is None:
                        return fail("สอบเทียบใหม่ไม่ได้")
                    continue
            else:
                worse = 0

        # ---- ถึงเป้าหมายไหม
        if dist <= ARRIVE_DIST:
            print("\n[✓] ถึงหน้ากล่องแล้ว!")
            time.sleep(0.5)
            tap(*TAP_OPEN)
            return

        # ---- ติดตามจุดที่ใกล้ที่สุด
        if best is None or dist < best - 0.5:
            best, best_pos, stall = dist, pos, 0
        else:
            stall += 1
            if stall >= MAX_STALL:
                print(f"\n[~] ใกล้ที่สุดเท่าที่ทำได้: ห่าง {best:.1f} ที่ {best_pos}")
                if best <= NEAR_TAP_DIST:
                    tap(*TAP_OPEN)
                return

        # ---- วางแผนก้าวถัดไป
        deg, ms, u, a, b = plan_move(M, dx, dy, fine)
        pred = (pos[0] + (M @ u)[0], pos[1] + (M @ u)[1])
        pred_dist = math.hypot(TARGET_CHEST[0] - pred[0], TARGET_CHEST[1] - pred[1])
        print(f"    [calc] ต้องไปอีก dx={dx:.0f}, dy={dy:.0f} => แก้สมการได้ "
              f"หน้า {a:+.1f} ขวา {b:+.1f} (หน่วยก้าว {CAL_MS}ms)")
        print(f"    สั่งเดิน ทิศ {deg:.0f}° นาน {ms} ms | "
              f"คาดว่าจะไปที่ ({pred[0]:.0f},{pred[1]:.0f}) -> ห่าง {pred_dist:.1f}")
        pending = {"pos": pos, "pred": pred, "dist": dist, "u": u}
        walk_in_direction(deg, ms)

    fail(f"ครบ {MAX_STEPS} ก้าวแล้วยังไม่ถึงกล่อง")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ocr":
        ocr_test(sys.argv[2] if len(sys.argv) > 2 else None)
    else:
        main()
