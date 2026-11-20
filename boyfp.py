"""
boyfp.py - บอทตกปลา (แตะหน้าจอจริง, อ่านภาพต่อเนื่อง, โหลดไอคอนจาก GitHub)

ติดตั้ง:  pip install opencv-python numpy
โหมดใช้งาน:
  MODE = "adb"    รันบนคอม คุมมือถือ/คลาวด์โฟนผ่าน ADB (แนะนำ ลง OpenCV ง่าย)
  MODE = "local"  รันบนตัวมือถือที่รูทแล้ว (Termux + su)

ไอคอน: บอทหาในโฟลเดอร์ icons/ ข้างไฟล์นี้ก่อน ถ้าไม่มีจะโหลดทั้งโฟลเดอร์ icons/ จาก GitHub (GITHUB_REPO)
ชื่อไฟล์ที่บอทมองหา (ใช้ชื่อมีคำนี้ก็ได้ เช่น rod_green.png, เบ็ดเขียว_green.png):
  rod_green  เบ็ดสีเขียว    rod_red  เบ็ดสีแดง    repair  ปุ่มซ่อม (ไม่บังคับ แต่ถ้าเบ็ดแดงแล้วไม่มี บอทจะหยุด)
  cast       ปุ่มคันเบ็ดวงแหวนบาง (ตอนตก/รอปลา)
  reel       ปุ่มคันเบ็ดเรืองแสง (ปลากิน -> กด)
  bait       ข้อความ/ไอคอน "Brand-new Type of Bait -1" (ไม่บังคับ ใช้ยืนยันว่าติดปลา)
ถ้าชื่อไฟล์ใน repo ไม่ตรง ให้แก้ ICON_FILES ด้านล่าง

มินิเกม (ปลาวิ่งบนเส้นด้านบน + ขีดขาว) อ่านจากสีโดยตรง ไม่ต้องใช้ไอคอน:
  วงแหวนเหลือง = ปลา   แถบขาวบนเส้น = โซนที่ต้องกด   แถบเปลี่ยนเป็นสีแดง = กดพลาด
"""
import os
import sys
import json
import time
import subprocess
import urllib.request

import cv2
import numpy as np

# ================= CONFIG =================
MODE = "auto"                 # "auto" = เจอ /system/bin/screencap ใช้ local (รันบนมือถือ) ไม่งั้น adb
ADB_SERIAL = ""               # เช่น "192.168.1.50:5555" ปล่อยว่าง = เครื่องเดียว
COUNTDOWN = 60                # วินาทีให้สลับเกม (1 นาที)

GITHUB_REPO = "bankmini053/UtopiaFarmAI"   # ดึงไอคอนจากโฟลเดอร์ icons/ ของ repo นี้
GITHUB_BRANCH = "main"
ICON_DIR = "icons"            # โฟลเดอร์ไอคอนในเครื่อง (ใช้เป็นแคชด้วย)
FORCE_DOWNLOAD = False        # True = โหลดใหม่ทุกครั้งที่เปิดบอท
# ถ้าชื่อไฟล์ไม่ตรง ระบุเองได้ เช่น {"cast": "fishing_btn.png", "reel": "fishing_glow.png"}
ICON_FILES = {}
REQUIRED = ["rod_green", "rod_red", "cast", "reel"]
OPTIONAL = ["repair", "bait"]
# คำที่ใช้เดาชื่อไฟล์ ถ้าไม่ระบุใน ICON_FILES และไม่มีชื่อตรงตัว
ALIASES = {
    "rod_green": ["rod_green", "green", "เขียว"],
    "rod_red": ["rod_red", "red", "แดง"],
    "repair": ["repair", "fix", "ซ่อม"],
    "cast": ["cast", "fishing_btn", "rod_btn", "idle"],
    "reel": ["reel", "bite", "glow", "hook"],
    "bait": ["bait", "เหยื่อ"],
}

THRESHOLD = 0.80              # ความเหมือนภาพของไอคอน 0-1

# ---- มินิเกม ----
TAP_DELAY = 0.08              # หน่วงจากสั่งแตะถึงเกมรับ (วินาที)
USER_OFFSET = 0.0             # กดช้าไป -> เพิ่มค่า / กดเร็วไป -> ลดค่า (เช่น 0.05)
INNER_MARGIN = 0.10           # ไม่กดใกล้ขอบแถบขาว (สัดส่วนความกว้างแถบ)
HIT_COOLDOWN = 1.0            # กันกดซ้ำหลังกดแล้ว (วินาที)
MAX_MISS = 3                  # กดพลาด (แถบแดง) ครบกี่ครั้งต่อปลา 1 ตัว ให้หยุดบอท
MISS_COOLDOWN = 1.0           # นับพลาดห่างกันอย่างน้อยกี่วินาที (กันนับซ้ำจากแถบแดงค้าง)
MINIGAME_LOST_AFTER = 1.5     # ไม่เห็นปลานานเท่านี้ = จบมินิเกม (ได้ปลาแล้ว)
HIT_BUTTON = None             # พิกัดปุ่มกด เช่น (2212, 960) ถ้า None จะหาจาก cast.png
HIT_BUTTON_FRAC = (0.80, 0.75)  # สำรอง: ตำแหน่งปุ่มเป็นสัดส่วนจอ (ปุ่มคันเบ็ดขวา)

# ---- ขั้นตอนอื่น ----
ACTION_COOLDOWN = 1.5         # พักหลังกดปุ่มตก/ซ่อม/reel
BITE_TIMEOUT = 90             # ตกแล้วรอปลากินนานสุดกี่วินาที เกินนี้กลับไปเช็กเบ็ดใหม่
NO_ROD_ASSUME_FISHING = 8     # ไม่เจอไอคอนเบ็ดนานเท่านี้ ถือว่ากำลังตกอยู่แล้ว (แถบล่างถูกซ่อน)
DEBUG = True
# ==========================================


def log(*a):
    if DEBUG:
        print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------- ไอคอน (ในเครื่อง + GitHub) ----------
IMG_EXT = (".png", ".jpg", ".jpeg")


def download_icons_from_github():
    api = f"https://api.github.com/repos/{GITHUB_REPO}/contents/icons?ref={GITHUB_BRANCH}"
    try:
        req = urllib.request.Request(api, headers={"User-Agent": "boyfp-bot"})
        items = json.load(urllib.request.urlopen(req, timeout=20))
    except Exception as e:
        log("โหลดรายชื่อไอคอนจาก GitHub ไม่ได้:", e)
        return
    os.makedirs(ICON_DIR, exist_ok=True)
    for it in items:
        name = it.get("name", "")
        if it.get("type") != "file" or not name.lower().endswith(IMG_EXT):
            continue
        path = os.path.join(ICON_DIR, name)
        if FORCE_DOWNLOAD or not os.path.exists(path):
            try:
                urllib.request.urlretrieve(it["download_url"], path)
                log("โหลด", name)
            except Exception as e:
                log("โหลดไม่ได้", name, e)


def resolve_icon(logical, files):
    """หาไฟล์ของไอคอนชื่อ logical จากรายชื่อไฟล์ในโฟลเดอร์"""
    if logical in ICON_FILES:
        return ICON_FILES[logical] if ICON_FILES[logical] in files else None
    stems = {os.path.splitext(f)[0].lower(): f for f in files}
    if logical in stems:
        return stems[logical]
    for kw in ALIASES.get(logical, []):
        for stem, f in stems.items():
            if kw.lower() in stem:
                return f
    return None


def list_icon_files():
    if not os.path.isdir(ICON_DIR):
        return []
    return sorted(f for f in os.listdir(ICON_DIR) if f.lower().endswith(IMG_EXT))


def load_icons():
    files = list_icon_files()
    if any(resolve_icon(n, files) is None for n in REQUIRED) or FORCE_DOWNLOAD:
        download_icons_from_github()
        files = list_icon_files()
    icons, missing = {}, []
    for name in REQUIRED + OPTIONAL:
        f = resolve_icon(name, files)
        if f is None:
            (missing if name in REQUIRED else []).append(name)
            continue
        img = cv2.imread(os.path.join(ICON_DIR, f))
        if img is None:
            sys.exit(f"อ่านไฟล์ {f} ไม่ได้")
        icons[name] = img
        log(f"ไอคอน {name} <- {f}")
    if missing:
        sys.exit(f"หาไอคอนไม่เจอ: {missing}\nไฟล์ที่มีใน {ICON_DIR}/: {files}\n"
                 f"แก้ชื่อไฟล์ให้มีคำว่าชื่อไอคอน หรือกำหนดใน ICON_FILES")
    return icons


# ---------- ควบคุมเครื่อง ----------
def _use_local():
    if MODE == "local":
        return True
    if MODE == "adb":
        return False
    return os.path.exists("/system/bin/screencap")   # รันบนมือถือ (Termux)


class Device:
    def _run(self, cmd):
        if not _use_local():
            base = ["adb"] + (["-s", ADB_SERIAL] if ADB_SERIAL else [])
            return subprocess.run(base + ["exec-out"] + cmd, capture_output=True).stdout
        return subprocess.run(["su", "-c", " ".join(cmd)], capture_output=True).stdout

    def screen(self):
        data = self._run(["screencap", "-p"])
        if not data:
            return None
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    def tap(self, x, y):
        self._run(["input", "tap", str(int(x)), str(int(y))])


# ตำแหน่งที่ไอคอนอยู่บนจอ (สัดส่วน x0,y0,x1,y1) ช่วยให้เร็วและแม่น
ROI = {
    "cast": (0.60, 0.45, 1.0, 1.0),     # ปุ่มคันเบ็ดฝั่งขวา
    "reel": (0.60, 0.45, 1.0, 1.0),     # ปุ่มเรืองแสงตอนปลากิน
    "rod_green": (0.0, 0.65, 1.0, 1.0),  # แถบล่าง
    "rod_red": (0.0, 0.65, 1.0, 1.0),
    "bait": (0.30, 0.15, 1.0, 0.85),    # ข้อความเหยื่อ -1 ลอยขึ้นกลางจอ
}
_scale_cache = {}


def find(frame, icons, name):
    """หาไอคอนแบบหลายขนาด (ไอคอนที่ครอปมาจะใหญ่/เล็กต่างจากหน้าจอจริงได้)
    คืน (score, cx, cy) ขนาดที่เจอครั้งแรกจะถูกจำไว้ ครั้งต่อไปเร็วขึ้น"""
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = ROI.get(name, (0, 0, 1, 1))
    ox, oy = int(x0 * W), int(y0 * H)
    roi = frame[oy:int(y1 * H), ox:int(x1 * W)]
    tpl = icons[name]
    scales = ([_scale_cache[name] * k for k in (0.85, 0.92, 1.0, 1.08, 1.15)]
              if name in _scale_cache else np.geomspace(0.1, 0.9, 28))
    best = (-1.0, 0, 0, 1.0)
    for sc in scales:
        t = cv2.resize(tpl, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        if t.shape[0] < 8 or t.shape[0] >= roi.shape[0] or t.shape[1] >= roi.shape[1]:
            continue
        res = cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if score > best[0]:
            best = (score, ox + loc[0] + t.shape[1] // 2, oy + loc[1] + t.shape[0] // 2, sc)
    if name not in _scale_cache and best[0] >= THRESHOLD:
        _scale_cache[name] = best[3]
    return best[0], best[1], best[2]


def detect_minigame(frame):
    """คืน (pin_x, bar_l, bar_r) หรือ None  -- พิกัดพิกเซลของภาพเต็ม ใช้สัดส่วนจอ ไม่ผูกกับความละเอียด"""
    H, W = frame.shape[:2]
    # --- ปลา (วงแหวนเหลือง) ---
    y0, y1, x0, x1 = int(0.04*H), int(0.17*H), int(0.30*W), int(0.70*W)
    roi = frame[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    yel = cv2.inRange(hsv, (20, 110, 170), (36, 255, 255))
    n, lab, stats, cen = cv2.connectedComponentsWithStats(yel)
    best = None
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        if w > 0.02*W and 0.6 < w/h < 1.6 and (best is None or a > stats[best][4]):
            best = i
    if best is None:
        return None
    x, y, w, h, a = stats[best]
    pin_x = x0 + x + w/2
    # --- ขีดขาว (โซนเป้าหมาย) บนเส้น ---
    by0, by1 = int(0.1375*H), int(0.153*H)
    band = frame[by0:by1, x0:x1]
    white = (band.min(axis=2) > 225)
    cols = white.sum(axis=0) >= max(2, int(0.003*H))
    # หา run ที่ยาวที่สุด
    best_run, run_s = (0, 0), None
    for i, c in enumerate(list(cols) + [False]):
        if c and run_s is None: run_s = i
        if not c and run_s is not None:
            if i - run_s > best_run[1] - best_run[0]: best_run = (run_s, i)
            run_s = None
    if best_run[1] - best_run[0] < 0.025*W:
        return None
    return pin_x, x0 + best_run[0], x0 + best_run[1]


def detect_miss(frame):
    """แถบบนเส้นกลายเป็นสีแดง = กดพลาด คืน True/False"""
    H, W = frame.shape[:2]
    x0, x1 = int(0.30 * W), int(0.70 * W)
    y0, y1 = int(0.12 * H), int(0.17 * H)
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    h, s_, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = ((h <= 8) | (h >= 172)) & (s_ >= 110) & (v >= 110)
    cols = red.sum(axis=0) >= max(2, int(0.004 * H))
    run = best = 0
    for c in cols:
        run = run + 1 if c else 0
        best = max(best, run)
    return best >= 0.02 * W


class Timing:
    """ตัดสินใจจังหวะกด จากตำแหน่งปลาที่อ่านได้เป็นช่วงๆ"""
    def __init__(self, tap_delay=0.08, user_offset=0.0, inner=0.10, cooldown=1.0):
        self.tr = []            # (t, x)
        self.tap_delay, self.user_offset = tap_delay, user_offset
        self.inner, self.cooldown = inner, cooldown
        self.last_hit = -9.0
        self.cap = 0.25         # เวลาแคปภาพเฉลี่ย (วัดเอง)
        self.last_t = None
        self.dt = 0.3

    def reset(self):
        self.tr.clear()

    def update(self, t, cap_time, pin_x, bar_l, bar_r):
        """คืน True ถ้าควรกดตอนนี้"""
        self.cap = 0.8*self.cap + 0.2*cap_time
        if self.last_t is not None:
            self.dt = 0.8*self.dt + 0.2*(t - self.last_t)
        self.last_t = t
        self.tr = [(a, b) for a, b in self.tr if t - a < 1.0] + [(t, pin_x)]
        if len(self.tr) < 2 or t - self.last_hit < self.cooldown:
            return False
        ts = np.array([a for a, _ in self.tr]); xs = np.array([b for _, b in self.tr])
        v = np.polyfit(ts - ts[0], xs, 1)[0]
        lead = self.cap + self.tap_delay + self.user_offset
        p = pin_x + v*lead                      # ตำแหน่งตอนเกมรับคำสั่งแตะ
        c, w = (bar_l + bar_r)/2, bar_r - bar_l
        if not (bar_l + self.inner*w <= p <= bar_r - self.inner*w):
            return False
        nxt = p + v*self.dt                     # ตำแหน่งตอนรอบถัดไป
        if abs(nxt - c) < abs(p - c) and bar_l + self.inner*w <= nxt <= bar_r - self.inner*w:
            return False                        # รอบหน้าใกล้กลางกว่า รอก่อน
        self.last_hit = t
        self.tr.clear()
        return True


def hit_point(frame, icons):
    if HIT_BUTTON:
        return HIT_BUTTON
    if "cast" in icons:
        s, x, y = find(frame, icons, "cast")
        if s >= THRESHOLD:
            return x, y
    h, w = frame.shape[:2]
    return int(w * HIT_BUTTON_FRAC[0]), int(h * HIT_BUTTON_FRAC[1])


def main():
    icons = load_icons()
    dev = Device()

    for i in range(COUNTDOWN, 0, -1):
        print(f"เริ่มใน {i} วิ (สลับไปหน้าเกม)...", flush=True)
        time.sleep(1)
    log("เริ่มทำงาน (Ctrl+C เพื่อหยุด)")

    timing = Timing(TAP_DELAY, USER_OFFSET, INNER_MARGIN, HIT_COOLDOWN)
    state = "IDLE"            # IDLE = เช็กเบ็ด/ตก, WAIT_BITE = รอปลากิน
    cast_time = 0.0
    last_action = 0.0
    last_seen_game = 0.0
    in_game = False
    misses = 0
    last_miss = -9.0
    last_bait = -9.0
    last_bait_check = 0.0
    no_rod_since = None
    caught = 0

    while True:
        t0 = time.time()
        frame = dev.screen()
        if frame is None:
            log("แคปจอไม่ได้ ลองใหม่")
            time.sleep(1)
            continue
        t = time.time()

        # ===== 4) มินิเกม: ปลาวิ่งบนเส้น เมื่อถึงขีดขาวให้กด =====
        mg = detect_minigame(frame)
        if mg:
            if not in_game:
                log("เข้ามินิเกม")
                in_game = True
                misses = 0
            last_seen_game = t
            state = "IDLE"

            # กดพลาด -> แถบเป็นสีแดง
            if detect_miss(frame) and t - last_miss > MISS_COOLDOWN:
                last_miss = t
                misses += 1
                log(f"กดพลาด (แถบแดง) ครั้งที่ {misses}/{MAX_MISS}")
                if misses >= MAX_MISS:
                    print(f"\nจับปลาไม่ได้ (กดพลาด {misses} ครั้ง) หยุดบอท", flush=True)
                    sys.exit(1)

            # เหยื่อ -1 = ติดปลาแล้ว (เช็กทุก 1 วินาที ไม่ให้หน่วงจังหวะกด)
            if "bait" in icons and t - last_bait_check > 1.0:
                last_bait_check = t
                bs, _, _ = find(frame, icons, "bait")
                if bs >= THRESHOLD and t - last_bait > 3.0:
                    last_bait = t
                    log("เหยื่อ -1 (ติดปลาแล้ว)")

            pin_x, bar_l, bar_r = mg
            if timing.update(t, t - t0, pin_x, bar_l, bar_r):
                dev.tap(*hit_point(frame, icons))
                log(f"กด! ปลา={pin_x:.0f} โซน={bar_l:.0f}-{bar_r:.0f}")
            continue
        if in_game:
            if t - last_seen_game < MINIGAME_LOST_AFTER:
                continue          # ภาพกะพริบตอนกดโดน รอดูก่อน
            in_game = False
            timing.reset()
            caught += 1
            last_action = time.time()
            log(f"จบมินิเกม จับปลาแล้ว (รอบที่ {caught}) พลาด {misses} ครั้ง")
            misses = 0

        if time.time() - last_action < ACTION_COOLDOWN:
            continue

        # ===== ปลากินเบ็ด (ปุ่มเรืองแสง) -> กด =====
        rs, rx, ry = find(frame, icons, "reel")
        if rs >= THRESHOLD:
            cs0, _, _ = find(frame, icons, "cast")
            if rs > cs0:
                log("ปลากิน (ปุ่มเรืองแสง) -> กด")
                dev.tap(rx, ry)
                last_action = time.time()
                state = "WAIT_BITE"
                cast_time = time.time()
                continue

        # ===== รอปลากิน =====
        if state == "WAIT_BITE":
            if time.time() - cast_time > BITE_TIMEOUT:
                log("รอปลาเกินเวลา กลับไปเช็กเบ็ด")
                state = "IDLE"
            continue

        # ===== 1) เช็กเบ็ด เขียว/แดง ก่อนตกทุกครั้ง =====
        gs, _, _ = find(frame, icons, "rod_green")
        rds, _, _ = find(frame, icons, "rod_red")
        if gs >= THRESHOLD or rds >= THRESHOLD:
            no_rod_since = None
        elif no_rod_since is None:
            no_rod_since = time.time()

        if rds >= THRESHOLD and rds > gs:
            if "repair" not in icons:
                print("\nเบ็ดแดง แต่ยังไม่มีไอคอนปุ่มซ่อม (repair.png) หยุดบอท", flush=True)
                sys.exit(1)
            ps, px, py = find(frame, icons, "repair")
            if ps >= THRESHOLD:
                log("เบ็ดแดง -> ซ่อม")
                dev.tap(px, py)
                last_action = time.time()
            else:
                log("เบ็ดแดง แต่ไม่เจอปุ่มซ่อม")
            continue

        # ===== 2) เบ็ดเขียว -> กดตกปลา แล้วรอปลากิน =====
        if gs >= THRESHOLD:
            cs, cx, cy = find(frame, icons, "cast")
            if cs >= THRESHOLD:
                log("เบ็ดเขียว -> กดตกปลา")
                dev.tap(cx, cy)
                last_action = time.time()
                state = "WAIT_BITE"
                cast_time = time.time()
            continue

        # ไม่เจอไอคอนเบ็ดเลย: แถบล่างซ่อมอยู่ = น่าจะกำลังตกอยู่แล้ว
        if no_rod_since and time.time() - no_rod_since > NO_ROD_ASSUME_FISHING:
            log("ไม่เจอไอคอนเบ็ด -> ถือว่ากำลังตกอยู่ รอปลากิน")
            state = "WAIT_BITE"
            cast_time = time.time()
            no_rod_since = None


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nหยุดบอท")
      
