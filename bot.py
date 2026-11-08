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
NO_COORD_TIMEOUT = 5     # ไม่เห็นพิกัดเกินกี่วินาทีให้ล้มเหลว
STOP_DURATION = 2        # หยุดนิ่งก่อนอ่านพิกัดทุกรอบ (วินาที)
ARRIVE_DIST = 3.0        # ระยะที่ถือว่าถึงกล่อง
MAX_WORSE = 3            # ห่างขึ้นติดกันกี่รอบให้ล้มเหลว
MAX_STEPS = 80
CAL_MS = 900             # เวลาก้าวสอบเทียบแกน
DIAGONAL = False         # False = เดินสี่เหลี่ยม (ตรงทีละแกน), True = เดินเฉียง
TAP_OPEN = (820, 430)    # ปุ่มเปิดกล่องเมื่อถึงที่

JOY_CENTER_X = 250
JOY_CENTER_Y = 530
JOY_RADIUS = 145


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

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (0, 0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    cfg = r"--psm 6 -c tessedit_char_whitelist=0123456789,."

    for th_val in [175, 195, 215]:
        _, thresh = cv2.threshold(resized, th_val, 255, cv2.THRESH_BINARY)
        raw = pytesseract.image_to_string(thresh, config=cfg)
        for pattern in (r"\d{5}", r"\d{4,6}"):
            nums = re.findall(pattern, raw)
            if len(nums) >= 2:
                x, y = int(nums[0]), int(nums[1])
                if 17400 <= x <= 17600 and 15250 <= y <= 15400:
                    return (x, y)
    return None


def read_pos(timeout=NO_COORD_TIMEOUT):
    """อ่านพิกัดสูงสุด 3 ครั้งแล้วเอามัธยฐาน ถ้าไม่เห็นเลยเกิน timeout คืน None"""
    start = time.time()
    xs, ys = [], []
    while time.time() - start < timeout:
        p = extract_coords(get_screen())
        if p:
            xs.append(p[0])
            ys.append(p[1])
            if len(xs) >= 3:
                break
        else:
            time.sleep(0.2)
    if not xs:
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
    """เดินสอบเทียบ 1 ก้าว คืนค่าพิกัดที่เปลี่ยน (dx, dy) ต่อ CAL_MS"""
    for ms in (CAL_MS, 1500, 2200):
        p1 = read_pos()
        walk_in_direction(deg, ms)
        time.sleep(0.5)
        p2 = read_pos()
        print(f"[cal] deg={deg} ms={ms} p1={p1} p2={p2}")
        if p1 and p2:
            v = (p2[0] - p1[0], p2[1] - p1[1])
            if math.hypot(*v) > 1:
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


def plan_move(M, dx, dy):
    """คืน (องศาจอยสติ๊ก, มิลลิวินาที)"""
    a, b = np.linalg.solve(M, np.array([dx, dy], dtype=float))  # หน้า, ขวา
    if DIAGONAL:
        deg = math.degrees(math.atan2(b, a)) % 360
        comp = math.hypot(a, b)
    elif abs(a) >= abs(b):
        deg = 0 if a > 0 else 180
        comp = abs(a)
    else:
        deg = 90 if b > 0 else 270
        comp = abs(b)
    ms = int(np.clip(comp * CAL_MS * 0.7, 150, 1500))  # 0.7 กันเดินเลย
    return deg, ms


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
        return fail("สอบเทียบแกนไม่ได้ (ตัวละครไม่ขยับ หรือ OCR อ่านไม่ได้ตอนเดิน)")

    prev_dist = None
    worse = 0
    recals = 0

    for step in range(MAX_STEPS):
        print(f"[⏸] หยุดนิ่ง {STOP_DURATION} วิ แล้วอ่านพิกัด...")
        time.sleep(STOP_DURATION)

        pos = read_pos()
        if pos is None:
            return fail(f"ไม่เห็นพิกัดเกิน {NO_COORD_TIMEOUT} วินาที")

        dx = TARGET_CHEST[0] - pos[0]
        dy = TARGET_CHEST[1] - pos[1]
        dist = math.hypot(dx, dy)
        print("-" * 55)
        print(f"[{step}] พิกัด {pos} | ห่าง {dist:.1f} (dx={dx:.0f}, dy={dy:.0f})")

        if dist <= ARRIVE_DIST:
            print("[✓] ถึงหน้ากล่องแล้ว!")
            time.sleep(0.5)
            tap(*TAP_OPEN)
            return

        # ตรวจว่าใกล้ขึ้นหรือห่างขึ้น
        if prev_dist is not None:
            if dist > prev_dist + 0.5:
                worse += 1
                print(f"[!] ห่างขึ้น ({prev_dist:.1f} -> {dist:.1f}) ติดกัน {worse}/{MAX_WORSE}")
                if worse >= MAX_WORSE:
                    return fail("เดินแล้วห่างจากกล่องขึ้นเรื่อยๆ")
                if worse == 2 and recals < 1:
                    print("[*] สอบเทียบแกนใหม่ 1 ครั้ง")
                    M = calibrate()
                    recals += 1
                    worse = 0
                    prev_dist = None
                    if M is None:
                        return fail("สอบเทียบใหม่ไม่ได้")
                    continue
            else:
                worse = 0
        prev_dist = dist

        deg, ms = plan_move(M, dx, dy)
        print(f"[->] เดินทิศ {deg:.0f}° นาน {ms} ms")
        walk_in_direction(deg, ms)

    fail(f"ครบ {MAX_STEPS} ก้าวแล้วยังไม่ถึงกล่อง")


if __name__ == "__main__":
    main()
