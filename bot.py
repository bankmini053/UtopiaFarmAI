import os
import re
import time
import math
import statistics
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

JOY_CENTER_X = 250
JOY_CENTER_Y = 530
JOY_RADIUS = 145

_last_screen = None


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


# ------------------------------------------------------------ อ่านพิกัด (OCR)
def ocr_candidates(roi):
    """สร้างภาพหลายแบบให้ OCR ลอง เรียงจากเร็ว/ได้ผลบ่อยก่อน"""
    big_bgr = cv2.resize(roi, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(big_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(big_bgr, cv2.COLOR_BGR2HSV)
    # ตัวเลขสีขาว: อิ่มตัวต่ำ + สว่างสูง
    yield cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 70, 255]))
    for th in (195, 175, 215, 235, 150):
        yield cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)[1]
    yield cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def extract_coords(screen):
    if screen is None:
        return None
    h, w = screen.shape[:2]
    if w > h:
        roi = screen[180:300, 40:270]
    else:
        roi = screen[40:270, 180:300]
        roi = cv2.rotate(roi, cv2.ROTATE_90_CLOCKWISE)
    if roi.size == 0:
        return None

    cfg = r"--psm 6 -c tessedit_char_whitelist=0123456789,."
    for img in ocr_candidates(roi):
        raw = pytesseract.image_to_string(img, config=cfg)
        for pattern in (r"\d{5}", r"\d{4,6}"):
            nums = re.findall(pattern, raw)
            if len(nums) >= 2:
                x, y = int(nums[0]), int(nums[1])
                if 17400 <= x <= 17600 and 15250 <= y <= 15400:
                    return (x, y)
    return None


def save_debug():
    try:
        if _last_screen is not None:
            cv2.imwrite("ocr_fail.png", _last_screen)
            print("[i] บันทึกภาพตอนอ่านพิกัดไม่ได้ไว้ที่ ocr_fail.png")
    except Exception:
        pass


def read_pos(timeout=NO_COORD_TIMEOUT):
    """อ่านพิกัดสูงสุด 3 ครั้งแล้วเอามัธยฐาน ถ้าไม่เห็นเลยเกิน timeout คืน None"""
    global _last_screen
    start = time.time()
    xs, ys = [], []
    while time.time() - start < timeout:
        _last_screen = get_screen()
        p = extract_coords(_last_screen)
        if p:
            xs.append(p[0])
            ys.append(p[1])
            if len(xs) >= 3:
                break
        else:
            time.sleep(0.1)
    if not xs:
        save_debug()
        return None
    return (statistics.median(xs), statistics.median(ys))


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
            if math.hypot(*v) > 2:
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
    """คืน (องศาจอยสติ๊ก, มิลลิวินาที, เวกเตอร์สั่ง u = (หน้า, ขวา) หน่วย CAL_MS)"""
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
    return deg, ms, u


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
        deg, ms, u = plan_move(M, dx, dy, fine)
        pred = (pos[0] + (M @ u)[0], pos[1] + (M @ u)[1])
        pred_dist = math.hypot(TARGET_CHEST[0] - pred[0], TARGET_CHEST[1] - pred[1])
        print(f"    สั่งเดิน ทิศ {deg:.0f}° นาน {ms} ms | "
              f"คาดว่าจะไปที่ ({pred[0]:.0f},{pred[1]:.0f}) -> ห่าง {pred_dist:.1f}")
        pending = {"pos": pos, "pred": pred, "dist": dist, "u": u}
        walk_in_direction(deg, ms)

    fail(f"ครบ {MAX_STEPS} ก้าวแล้วยังไม่ถึงกล่อง")


if __name__ == "__main__":
    main()
