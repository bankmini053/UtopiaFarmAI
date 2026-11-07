import os
import glob
import math
import random
import subprocess
import time
import cv2
import numpy as np

os.environ.setdefault("RISH_APPLICATION_ID", "com.termux")

# =================== ตั้งค่า (ปรับตามเครื่องคุณ) ===================
ATTACK = (2060, 893)          # ปุ่มดาบ (โจมตี/ตัด) อ่านจากภาพขนาด 2576x1189
BAG = (1988, 1113)            # ปุ่มกระเป๋า (เดาจากภาพ ตรวจสอบก่อนใช้)

TAP_TO_WALK = False           # เกมนี้เดินด้วยจอยสติ๊ก
JOY_CENTER = (630, 893)       # กลางจอยสติ๊ก
JOY_RADIUS = 90               # ลากไกลจากกลางจอยกี่พิกเซล
CLOSE_PX = 220                # ใกล้เป้าหมายกว่านี้ (บนจอ) ให้หยุดเดินแล้วตีเลย
HIT_TO_REVEAL = True          # ชื่อจะโผล่หลังตี จึงตีทีละครั้งระหว่างเข้าใกล้
WALK_PX_PER_SEC = 250         # ความเร็วเดินโดยประมาณ (พิกเซล/วินาที) ใช้โหมดจอยสติ๊ก

CHAR_POS = (1290, 850)        # ตำแหน่งตัวละครบนจอ
NAME_ROI = (1070, 0, 560, 95)  # บริเวณแถบชื่อเป้าหมายด้านบนจอ (x, y, w, h)
HP_ROI = (1070, 48, 560, 30)   # บริเวณแถบเลือดสีแดงของเป้าหมาย

TOOL_SLOTS = {"tree": (718, 1113), "rock": (845, 1113)}  # ช่อง 1 และ 2 ของแถบเครื่องมือ ตรวจสอบว่าถูกชนิด

TARGET_THR = 0.80             # ความแม่นการหาต้นไม้/หินในฉาก
NAME_THR = 0.80               # ความแม่นการเช็กชื่อ
ICON_THR = 0.78               # ความแม่นไอคอน UI

APPROACH_MAX_SEC = 8          # เวลาสูงสุดที่รอให้เดินไปถึง
MAX_CHOP_SEC = 90             # ฟันเป้าหมายเดียวนานสุดกี่วินาที
MISSING_LIMIT = 3             # ชื่อหายกี่รอบติดกัน ถือว่าของหมดแล้ว
STOP_AFTER_HOME = True        # True = หยุดหลังวาร์ปกลับบ้าน
DEBUG = False
# ====================================================================

ICON_HAND = "icons/pickup_hand.png"
ICON_FULL = "icons/weight_full.png"
ICON_HOME = "icons/hearthstone.png"
ICON_PICKALL = "icons/pick_all.png"   # ปุ่ม Pick All ในหน้าต่างของที่ได้


# ---------- ควบคุมเครื่อง ----------
def tap(x, y, jitter=4):
    x += random.randint(-jitter, jitter)
    y += random.randint(-jitter, jitter)
    subprocess.run(f"rish -c 'input tap {x} {y}'", shell=True)


def swipe(x1, y1, x2, y2, ms):
    subprocess.run(f"rish -c 'input swipe {x1} {y1} {x2} {y2} {ms}'", shell=True)


def get_screen():
    raw = subprocess.run("rish -c 'screencap -p'", shell=True,
                         capture_output=True).stdout
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raw = raw.replace(b"\r\n", b"\n")
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    return img


# ---------- โหลดรูปต้นแบบ ----------
def load_group(pattern):
    items = []
    for path in sorted(glob.glob(pattern)):
        img = cv2.imread(path)
        if img is not None:
            kind = os.path.basename(path).split("_")[0]  # tree_name1.png -> tree
            items.append((kind, img))
    return items


def load_one(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"ไม่พบไฟล์รูป: {path}")
    return img


# ---------- จับภาพ ----------
def nms(points, min_dist):
    keep = []
    for v, x, y in sorted(points, reverse=True):
        if all(math.hypot(x - kx, y - ky) >= min_dist for _, kx, ky in keep):
            keep.append((v, x, y))
    return keep


def match_all(screen, tpl, thr):
    res = cv2.matchTemplate(screen, tpl, cv2.TM_CCOEFF_NORMED)
    ys, xs = np.where(res >= thr)
    h, w = tpl.shape[:2]
    pts = [(float(res[y, x]), int(x + w // 2), int(y + h // 2))
           for y, x in zip(ys, xs)]
    pts.sort(reverse=True)
    return nms(pts[:300], min(w, h) * 0.8)


def find_icon(screen, tpl, thr=ICON_THR):
    res = cv2.matchTemplate(screen, tpl, cv2.TM_CCOEFF_NORMED)
    _, val, _, loc = cv2.minMaxLoc(res)
    if DEBUG:
        print(f"   icon score {val:.2f}")
    if val >= thr:
        h, w = tpl.shape[:2]
        return loc[0] + w // 2, loc[1] + h // 2
    return None


def char_pos(screen):
    if CHAR_POS:
        return CHAR_POS
    h, w = screen.shape[:2]
    return w // 2, h // 2


def find_candidates(screen, targets):
    """หาต้นไม้/หินทั้งหมดในฉาก เรียงจากใกล้ตัวละครไปไกล"""
    pts = []
    for _, tpl in targets:
        pts += match_all(screen, tpl, TARGET_THR)
    pts = nms(pts, 60)
    cx, cy = char_pos(screen)
    pts.sort(key=lambda p: math.hypot(p[1] - cx, p[2] - cy))
    return [(x, y) for _, x, y in pts]


def name_visible(screen, names):
    """ถ้าเห็นชื่อที่ต้องการ คืนชนิด (tree/rock) ไม่เห็นคืน None"""
    if NAME_ROI:
        x, y, w, h = NAME_ROI
        screen = screen[y:y + h, x:x + w]
    for kind, tpl in names:
        if tpl.shape[0] > screen.shape[0] or tpl.shape[1] > screen.shape[1]:
            continue
        res = cv2.matchTemplate(screen, tpl, cv2.TM_CCOEFF_NORMED)
        _, val, _, _ = cv2.minMaxLoc(res)
        if DEBUG:
            print(f"   name {kind} score {val:.2f}")
        if val >= NAME_THR:
            return kind
    return None


def hp_bar_present(screen):
    """แถบเลือดสีแดงของเป้าหมายยังอยู่ไหม (ตัดหมดแล้วแถบแดงจะหาย)"""
    x, y, w, h = HP_ROI
    c = screen[y:y + h, x:x + w].astype(int)
    b, g, r = c[:, :, 0], c[:, :, 1], c[:, :, 2]
    frac = float(((r > 190) & (g < 120) & (b < 120)).mean())
    if DEBUG:
        print(f"   hp bar {frac:.2f}")
    return frac > 0.02


PICKALL = None


def collect_loot(screen, tpl_hand):
    """เก็บของ: กด Pick All ถ้ามีหน้าต่างของ ไม่งั้นกดรูปมือ คืน True ถ้าได้กด"""
    if PICKALL is not None:
        btn = find_icon(screen, PICKALL)
        if btn:
            print("[+] Pick All")
            tap(*btn)
            time.sleep(random.uniform(0.6, 0.9))
            return True
    hand = find_icon(screen, tpl_hand)
    if hand:
        print("[+] พบไอเทมตก แตะเก็บของ")
        tap(*hand)
        time.sleep(random.uniform(0.5, 0.8))
        return True
    return False


# ---------- การเคลื่อนที่ ----------
def joystick_walk(dx, dy, seconds):
    if not JOY_CENTER:
        return
    dist = math.hypot(dx, dy) or 1
    ux, uy = dx / dist, dy / dist
    jx, jy = JOY_CENTER
    swipe(jx, jy, int(jx + ux * JOY_RADIUS), int(jy + uy * JOY_RADIUS),
          int(seconds * 1000))


def wander():
    """เดินไปทิศสุ่มเพื่อหาเป้าหมายใหม่"""
    if JOY_CENTER:
        ang = random.uniform(0, 2 * math.pi)
        joystick_walk(math.cos(ang), math.sin(ang), random.uniform(1.5, 3.0))
    else:
        time.sleep(2.0)


def approach(target, names):
    """เดินเข้าหาเป้าหมาย ตีให้ชื่อโผล่ แล้วเช็กชื่อ
    คืนชนิดถ้าชื่อตรง ไม่ตรงหรือไปไม่ถึงคืน None"""
    start = time.time()
    first = True
    while time.time() - start < APPROACH_MAX_SEC:
        screen = get_screen()
        if screen is None:
            time.sleep(1)
            continue
        kind = name_visible(screen, names)
        if kind:
            return kind
        cx, cy = char_pos(screen)
        if TAP_TO_WALK:
            if first:
                tap(*target)
                first = False
            time.sleep(0.8)
        else:
            dx, dy = target[0] - cx, target[1] - cy
            dist = math.hypot(dx, dy)
            if dist > CLOSE_PX:
                joystick_walk(dx, dy, max(min(dist / WALK_PX_PER_SEC, 2.0), 0.4))
        if HIT_TO_REVEAL:
            tap(*ATTACK)
            time.sleep(0.7)
        if not TAP_TO_WALK:
            screen2 = get_screen()
            if screen2 is not None:
                cands = find_candidates(screen2, TARGETS)
                if cands:
                    target = cands[0]
    return None


# ---------- ฟันและเก็บของ ----------
def chop(kind, names, tpl_hand, tpl_full):
    if kind in TOOL_SLOTS:
        tap(*TOOL_SLOTS[kind])
        time.sleep(0.7)
    start = time.time()
    missing = 0
    while time.time() - start < MAX_CHOP_SEC:
        screen = get_screen()
        if screen is None:
            time.sleep(1)
            continue
        if collect_loot(screen, tpl_hand):
            continue
        if find_icon(screen, tpl_full):
            return "full"
        # ยังตัดอยู่ = ชื่อยังอยู่ และแถบเลือดแดงยังอยู่
        if name_visible(screen, names) and hp_bar_present(screen):
            missing = 0
        else:
            missing += 1
            if missing >= MISSING_LIMIT:
                return "done"
        tap(*ATTACK)
        time.sleep(random.uniform(1.15, 1.35))
    return "timeout"


def go_home(tpl_home):
    print("[!] กระเป๋าเต็ม กำลังวาร์ปกลับบ้าน...")
    tap(*BAG)
    time.sleep(1.2)
    screen = get_screen()
    btn = find_icon(screen, tpl_home) if screen is not None else None
    if not btn:
        print("[x] ไม่เจอปุ่มวาร์ป")
        return False
    tap(*btn)
    time.sleep(15.0)
    return True


# ---------- ลูปหลัก ----------
TARGETS = []


def start_bot():
    global TARGETS, PICKALL
    TARGETS = load_group("icons/targets/*.png")
    names = load_group("icons/names/*.png")
    tpl_hand, tpl_full, tpl_home = (load_one(ICON_HAND), load_one(ICON_FULL),
                                    load_one(ICON_HOME))
    PICKALL = cv2.imread(ICON_PICKALL)
    if PICKALL is None:
        print("[!] ไม่พบ icons/pick_all.png จะใช้รูปมือเก็บของแทน")
    if not names:
        print("[x] ต้องมีรูปป้ายชื่อที่ต้องการใน icons/names/")
        return
    if not TARGETS:
        print("[!] ไม่มีรูปใน icons/targets/ จะใช้โหมดเดินสุ่มแล้วตีเพื่อหาเป้าหมาย")
    print(f"[*] เริ่มทำงาน: เป้าหมาย {len(TARGETS)} รูป, ชื่อที่ต้องการ {len(names)} รูป "
          "(Ctrl+C เพื่อหยุด)")

    while True:
        screen = get_screen()
        if screen is None:
            print("[x] จับภาพหน้าจอไม่ได้ ลองใหม่...")
            time.sleep(2)
            continue

        # 1) กระเป๋าเต็ม -> กลับบ้าน
        if find_icon(screen, tpl_full):
            if go_home(tpl_home) and STOP_AFTER_HOME:
                print("[*] กลับบ้านแล้ว หยุดทำงาน")
                break
            continue

        # 2) มีของตกให้เก็บ
        if collect_loot(screen, tpl_hand):
            continue

        # 3) หาต้นไม้/หินในฉาก (ถ้าไม่มีรูปเป้าหมาย ใช้โหมดเดินสุ่ม + ตีเพื่อให้ชื่อโผล่)
        cands = find_candidates(screen, TARGETS) if TARGETS else []
        if cands:
            print(f"[>] เจอเป้าหมาย {len(cands)} จุด เข้าไปตรวจสอบจุดที่ใกล้สุด")
            kind = approach(cands[0], names)
        else:
            print("[-] ไม่เจอเป้าหมายในฉาก เดินสุ่มแล้วลองตี")
            wander()
            tap(*ATTACK)
            time.sleep(0.7)
            s2 = get_screen()
            kind = name_visible(s2, names) if s2 is not None else None

        # 4) ชื่อไม่ตรง -> หาที่อื่น
        if not kind:
            print("[x] ชื่อไม่ตรงหรือไม่พบเป้าหมาย หาที่อื่น")
            wander()
            continue

        # 5) ฟันจนของหมด
        print(f"[+] ชื่อตรง ({kind}) เริ่มฟัน")
        result = chop(kind, names, tpl_hand, tpl_full)
        print(f"[*] ผลการฟัน: {result}")
        if result == "full":
            if go_home(tpl_home) and STOP_AFTER_HOME:
                print("[*] กลับบ้านแล้ว หยุดทำงาน")
                break


if __name__ == "__main__":
    try:
        start_bot()
    except KeyboardInterrupt:
        print("\n[*] หยุดบอทแล้ว")
