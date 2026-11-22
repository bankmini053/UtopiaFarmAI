"""
boyfp.py - บอทตกปลา (แตะหน้าจอจริง, อ่านภาพต่อเนื่อง, โหลดไอคอนจาก GitHub)

ติดตั้ง:  pip install opencv-python numpy
โหมดใช้งาน:
  MODE = "adb"    รันบนคอม คุมมือถือ/คลาวด์โฟนผ่าน ADB (แนะนำ ลง OpenCV ง่าย)
  MODE = "local"  รันบนตัวมือถือที่รูทแล้ว (Termux + su)

ไอคอน: บอทหาในโฟลเดอร์ icons/ ข้างไฟล์นี้ก่อน ถ้าไม่มีจะโหลดทั้งโฟลเดอร์ icons/ จาก GitHub (GITHUB_REPO)
ชื่อไฟล์ที่บอทมองหา (ใช้ชื่อมีคำนี้ก็ได้ เช่น rod_green.png, เบ็ดเขียว_green.png):
  rod_green  เบ็ดสีเขียว    rod_red  เบ็ดสีแดง    repair  ปุ่มซ่อม (ไม่บังคับ แต่ถ้าเบ็ดแดงแล้วไม่มี บอทจะหยุด)
  sword      ปุ่มดาบฝั่งขวา (ตอนแถบล่างยังอยู่ กดเพื่อเริ่มตก)
  cast       ปุ่มคันเบ็ดวงแหวนบาง (ตอนตก/รอปลา และใช้กดจังหวะมินิเกม)
  reel       ปุ่มคันเบ็ดเรืองแสง (ปลากิน -> กด)
  bait       ข้อความ/ไอคอน "Brand-new Type of Bait -1" (ไม่บังคับ ใช้ยืนยันว่าติดปลา)
  hunger_full / hunger_yellow / hunger_red  หลอดอาหารมุมขวาบน เต็ม(เขียว) / เริ่มหิว(เหลือง) / หิวข้าว(แดง)
  hunger_danger  กระเพาะสีแดง = อันตรายถึงชีวิต
  food, food2, food3 ...  ไอคอนอาหารบนแถบล่าง (เหลือง/แดง/อันตราย -> รอตกจบ แถบล่างขึ้นแล้วกดจนเขียวเต็มก่อนตกต่อ ไม่มีอาหารก็ตกต่อ)
ถ้าชื่อไฟล์ใน repo ไม่ตรง ให้แก้ ICON_FILES ด้านล่าง

มินิเกม (ปลาวิ่งบนเส้นด้านบน + ขีดขาว) อ่านจากสีโดยตรง ไม่ต้องใช้ไอคอน:
  วงแหวนเหลือง = ปลา   แถบขาวบนเส้น = โซนที่ต้องกด   แถบเปลี่ยนเป็นสีแดง = กดพลาด
"""
import os
import sys
import json
import struct
import time
import subprocess
import urllib.request
from collections import deque

import cv2
import numpy as np

# ================= CONFIG =================
MODE = "auto"                 # "auto" = เจอ /system/bin/screencap ใช้ local (รันบนมือถือ) ไม่งั้น adb
ADB_SERIAL = ""               # เช่น "192.168.1.50:5555" ปล่อยว่าง = เครื่องเดียว
COUNTDOWN = 10                # วินาทีให้สลับเกม

GITHUB_REPO = "bankmini053/UtopiaFarmAI"   # ดึงไอคอนจากโฟลเดอร์ icons/ ของ repo นี้
GITHUB_BRANCH = "main"
ICON_DIR = "icons"            # โฟลเดอร์ไอคอนในเครื่อง (ใช้เป็นแคชด้วย)
FORCE_DOWNLOAD = False        # True = โหลดใหม่ทุกครั้งที่เปิดบอท
# ถ้าชื่อไฟล์ไม่ตรง ระบุเองได้ เช่น {"cast": "fishing_btn.png", "reel": "fishing_glow.png"}
ICON_FILES = {}
REQUIRED = ["cast", "reel"]
HUNGER_NAMES = ("hunger_full", "hunger_yellow", "hunger_red")     # ไอคอนกระเพาะ 3 ระดับที่มีวงแหวน
OPTIONAL = ["sword", "fix", "equip_no", "rod_green", "rod_red", "repair", "bait", "hunger_full", "hunger_yellow", "hunger_red", "hunger_danger"]
# คำที่ใช้เดาชื่อไฟล์ ถ้าไม่ระบุใน ICON_FILES และไม่มีชื่อตรงตัว
ALIASES = {
    "rod_green": ["rod_green"],
    "rod_red": ["rod_red"],
    "repair": ["repair", "fix", "ซ่อม"],
    "cast": ["cast", "fishing_btn", "rod_btn", "idle"],
    "reel": ["reel", "bite", "glow", "hook"],
    "bait": ["bait", "เหยื่อ"],
    "sword": ["sword", "ดาบ", "dagger", "attack"],
    "fix": ["fix"],
    "equip_no": ["equip_no", "equip"],
    "hunger_full": ["hunger_full"],
    "hunger_yellow": ["hunger_yellow"],
    "hunger_red": ["hunger_red"],
    "hunger_danger": ["hunger_danger", "danger"],
}
FOOD_PREFIX = ("food", "อาหาร")   # ไฟล์อาหารบนแถบล่าง ตั้งชื่อ food.png, food2.png, food3.png ... ได้หลายอัน

THRESHOLD = 0.80              # ความเหมือนภาพของไอคอน 0-1
REEL_MIN = 0.70               # ปุ่มเรืองแสง: คะแนนขั้นต่ำ (ภาพเรืองแสงกะพริบ คะแนนแกว่ง)
REEL_MARGIN = 0.05            # และต้องสูงกว่าคะแนนปุ่มปกติอย่างน้อยเท่านี้
BITE_RING_MIN = 90            # ปลากิน: วงแสงรอบปุ่มต้องสว่างอย่างน้อยเท่านี้ (0-255) และสว่างกว่าปกติ 1.7 เท่า
BITE_RING_RATIO = 1.7
SCREENCAP_RAW = True          # แคปแบบดิบ (ไม่บีบ PNG) เร็วกว่ามาก ถ้าใช้ไม่ได้บอทสลับไป PNG เอง

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
REPAIR_DELAY = 1.2            # หน่วงระหว่างขั้นตอนซ่อมเบ็ด (กดเบ็ด -> Fix -> Equip No) วินาที
REPAIR_WAIT = 6.0             # รอให้หน้าต่าง/ปุ่มขึ้นสูงสุดกี่วินาที
MAX_REPAIR_FAIL = 3           # ซ่อมไม่สำเร็จติดกันกี่ครั้งให้หยุดบอท
EAT_COOLDOWN = 2.5            # กดอาหารซ้ำห่างกันอย่างน้อยกี่วินาที
MAX_EAT_TRIES = 20            # กดอาหารสูงสุดกี่ครั้งต่อรอบ (กินจนกระเพาะเต็ม) เกินนี้ก็ตกต่อ กันค้าง
FOOD_THRESHOLD = 0.65         # ความเหมือนของไอคอนอาหาร (ช่องอาหารมีพื้นหลัง/ตัวเลขต่างกันได้)
FOOD_MISSING_WAIT = 60        # ไม่เจออาหารแล้ว รอกี่วินาทีค่อยหาใหม่
ACTION_COOLDOWN = 1.5         # พักหลังกดปุ่มตก/ซ่อม/reel
BITE_TIMEOUT = 90             # ตกแล้วรอปลากินนานสุดกี่วินาที เกินนี้กลับไปเช็กเบ็ดใหม่
NO_ROD_ASSUME_FISHING = 8     # ไม่เจอไอคอนเบ็ดนานเท่านี้ ถือว่ากำลังตกอยู่แล้ว (แถบล่างถูกซ่อน)
DEBUG = True
DASHBOARD = True              # แสดงหน้าสถานะสวยๆ (ปิด = พิมพ์เป็นบรรทัดแบบเดิม)
# ==========================================


def _fmt_time(sec):
    sec = int(sec)
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


class Status:
    """หน้าสถานะ: กำลังทำอะไร / ตกกี่รอบ / ได้ปลากี่ตัว / พลาดกี่ครั้ง + เหตุการณ์ล่าสุด"""
    W = 34

    def __init__(self):
        self.t0 = time.time()
        self.activity = "เริ่มต้น"
        self.casts = self.fish = self.misses = self.eats = self.repairs = 0
        self.hunger = "-"
        self.events = deque(maxlen=6)
        self.last_draw = 0.0
        self.live = DASHBOARD and sys.stdout.isatty()

    def event(self, text):
        ts = time.strftime("%H:%M:%S")
        self.events.append(f"{ts} {text}")
        if self.live:
            self.draw()
        elif DEBUG:
            print(f"{ts} {text}", flush=True)

    def set(self, activity):
        if activity != self.activity:
            self.activity = activity
            if self.live:
                self.draw()

    def tick(self):
        if self.live and time.time() - self.last_draw >= 1.0:
            self.draw(force=True)

    def lines(self, final=False):
        el = time.time() - self.t0
        rate = f"{el / self.fish:.0f} วิ/ตัว" if self.fish else "-"
        bar = "━" * self.W
        out = [bar, ("  🎣 สรุปผลการตกปลา" if final else "  🎣 บอทตกปลา  (Ctrl+C เพื่อหยุด)"), bar]
        if not final:
            out.append(f" สถานะ    : {self.activity}")
        out += [
            f" เวลาทำงาน : {_fmt_time(el)}",
            "─" * self.W,
            f" 🔄 ตกรวม   : {self.casts} รอบ",
            f" 🐟 ได้ปลา  : {self.fish} ตัว  ({rate})",
            f" ❌ พลาดรวม : {self.misses} ครั้ง",
            f" 🍖 กินอาหาร: {self.eats} ครั้ง",
            f" 🔧 ซ่อมเบ็ด: {self.repairs} ครั้ง",
            f" 🥣 กระเพาะ : {self.hunger}",
        ]
        if not final:
            out.append("─" * self.W)
            out.append(" เหตุการณ์ล่าสุด")
            out += [" " + e[:44] for e in self.events]
        out.append(bar)
        return out

    def draw(self, force=False):
        now = time.time()
        if not force and now - self.last_draw < 0.25:
            return
        self.last_draw = now
        sys.stdout.write("\033[H\033[2J" + "\n".join(self.lines()) + "\n")
        sys.stdout.flush()

    def summary(self):
        return "\n".join(self.lines(final=True))


STATUS = Status()


def log(*a):
    STATUS.event(" ".join(str(x) for x in a))


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
    for f in files:                                   # ไอคอนอาหาร (หลายไฟล์ได้)
        if os.path.splitext(f)[0].lower().startswith(FOOD_PREFIX):
            img = cv2.imread(os.path.join(ICON_DIR, f))
            if img is not None:
                icons["food:" + f] = img
                log(f"อาหาร <- {f}")
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


def parse_raw(data):
    """screencap แบบดิบ: หัว 12/16 ไบต์ (กว้าง,สูง,รูปแบบ) + พิกเซล RGBA ถ้าขนาดไม่ลงตัว (ภาพขาด) คืน None"""
    if not data or len(data) < 16:
        return None
    w, h = struct.unpack("<II", data[:8])
    px = w * h * 4
    hdr = len(data) - px
    if not (0 < w < 8000 and 0 < h < 8000) or hdr not in (12, 16):
        return None
    arr = np.frombuffer(data, np.uint8, count=px, offset=hdr).reshape(h, w, 4)
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)


def ring_p90(frame, btn):
    """ความสว่างของวงแหวนรอบปุ่ม (เปอร์เซ็นไทล์ 90) ปุ่มเรืองแสง = วงขาวสว่าง ปุ่มปกติ = วงเทาเข้ม"""
    cx, cy, r = btn
    pad = int(1.3 * r)
    x0, y0 = max(0, int(cx - pad)), max(0, int(cy - pad))
    g = cv2.cvtColor(frame[y0:int(cy + pad), x0:int(cx + pad)], cv2.COLOR_BGR2GRAY)
    yy, xx = np.ogrid[:g.shape[0], :g.shape[1]]
    d = np.hypot(xx - (cx - x0), yy - (cy - y0))
    m = (d >= 0.80 * r) & (d <= 1.12 * r)
    return float(np.percentile(g[m], 90)) if m.any() else 0.0


class Device:
    def _run(self, cmd):
        if not _use_local():
            base = ["adb"] + (["-s", ADB_SERIAL] if ADB_SERIAL else [])
            return subprocess.run(base + ["exec-out"] + cmd, capture_output=True).stdout
        return subprocess.run(["su", "-c", " ".join(cmd)], capture_output=True).stdout

    raw_ok = SCREENCAP_RAW
    raw_seen = False
    raw_fail = 0

    def screen(self):
        if self.raw_ok:
            frame = parse_raw(self._run(["screencap"]))
            if frame is not None:
                self.raw_seen = True
                return frame
            self.raw_fail += 1
            if not self.raw_seen and self.raw_fail >= 5:
                self.raw_ok = False
                log("แคปแบบดิบใช้ไม่ได้ สลับไปใช้ PNG")
            return None
        data = self._run(["screencap", "-p"])
        if not data or b"IEND" not in data[-16:]:
            return None                      # ภาพขาดครึ่ง (แคปไม่ทัน) ข้ามเฟรมนี้
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    def tap(self, x, y):
        self._run(["input", "tap", str(int(x)), str(int(y))])


# ตำแหน่งที่ไอคอนอยู่บนจอ (สัดส่วน x0,y0,x1,y1) ช่วยให้เร็วและแม่น
ROI = {
    "cast": (0.60, 0.45, 1.0, 1.0),     # ปุ่มคันเบ็ดฝั่งขวา
    "reel": (0.60, 0.45, 1.0, 1.0),     # ปุ่มเรืองแสงตอนปลากิน
    "sword": (0.60, 0.45, 1.0, 1.0),    # ปุ่มดาบ (ตอนแถบล่างยังอยู่)
    "rod_green": (0.0, 0.65, 1.0, 1.0),  # แถบล่าง
    "rod_red": (0.0, 0.65, 1.0, 1.0),
    "bait": (0.30, 0.15, 1.0, 0.85),    # ข้อความเหยื่อ -1 ลอยขึ้นกลางจอ
    "fix": (0.25, 0.45, 0.75, 1.0),     # ปุ่ม Fix ในหน้าต่างซ่อมของ
    "equip_no": (0.25, 0.45, 0.75, 1.0),
}
HUNGER_ROI = (0.55, 0.0, 1.0, 0.22)       # หลอดอาหาร (กระเพาะ) มุมขวาบน
FOOD_ROI = (0.0, 0.70, 0.85, 1.0)        # แถบล่าง
_scale_cache = {}
_last_scale = {}
DS = 0.5                      # ย่อภาพครึ่งหนึ่งก่อนจับคู่ ให้เร็วขึ้น (มือถือ CPU ช้า)
BUTTON_FRAC_H = 0.19          # ปุ่มคันเบ็ดโดยประมาณ = 19% ของความสูงจอ (ใช้เดาขนาดไอคอนเริ่มต้น)


def find(frame, icons, name):
    """หาไอคอนแบบหลายขนาด คืน (score, cx, cy)  ขนาดที่เจอครั้งแรกจะถูกจำไว้ ครั้งต่อไปเร็วขึ้น"""
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = ROI.get(name) or (FOOD_ROI if name.startswith("food:") else HUNGER_ROI if name.startswith("hunger_") else (0, 0, 1, 1))
    ox, oy = int(x0 * W), int(y0 * H)
    roi = frame[oy:int(y1 * H), ox:int(x1 * W)]
    roi = cv2.resize(roi, None, fx=DS, fy=DS, interpolation=cv2.INTER_AREA)
    tpl = icons[name]
    if name in _scale_cache:
        scales = [_scale_cache[name] * k for k in (0.85, 0.92, 1.0, 1.08, 1.15)]
    elif name in ("cast", "reel", "sword"):
        sc0 = BUTTON_FRAC_H * H / tpl.shape[1]
        scales = [sc0 * k for k in np.geomspace(0.55, 1.8, 12)]
    else:
        scales = np.geomspace(0.15, 1.6, 24)
    best = (-1.0, 0, 0, 1.0)
    for sc in scales:
        t = cv2.resize(tpl, None, fx=sc * DS, fy=sc * DS, interpolation=cv2.INTER_AREA)
        if t.shape[0] < 6 or t.shape[0] >= roi.shape[0] or t.shape[1] >= roi.shape[1]:
            continue
        res = cv2.matchTemplate(roi, t, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if score > best[0]:
            best = (score, ox + (loc[0] + t.shape[1] / 2) / DS, oy + (loc[1] + t.shape[0] / 2) / DS, sc)
    _last_scale[name] = best[3]
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


def rod_from_bar(frame):
    """ไม่มีกรอบเหลือง: หาแท่งความทนทานแนวตั้งสีแดง/เขียวสว่างข้างช่องเบ็ด (ช่องซ้ายสุดของแถบล่าง)
    คืน ("green"|"red", x, y กึ่งกลางช่องเบ็ด) หรือ None"""
    H, W = frame.shape[:2]
    x0, y0 = 0, int(0.80 * H)
    hsv = cv2.cvtColor(frame[y0:, x0:int(0.6 * W)], cv2.COLOR_BGR2HSV)
    h_, s_, v_ = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    masks = {
        "red": (((h_ <= 8) | (h_ >= 172)) & (s_ >= 130) & (v_ >= 140)),
        "green": ((h_ >= 45) & (h_ <= 85) & (s_ >= 90) & (v_ >= 190)),
    }
    found = []
    for color, m in masks.items():
        n, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8) * 255)
        for x, y, w, h, a in st[1:]:
            if (0.0025 * W <= w <= 0.012 * W and 0.04 * H <= h <= 0.09 * H
                    and h >= 2.5 * w and a >= 0.6 * w * h):
                found.append((x, y, w, h, color))
    if not found:
        return None
    x, y, w, h, color = min(found, key=lambda f: f[0])      # ซ้ายสุด = ช่องเบ็ด
    bx, by = x0 + x + w / 2, y0 + y + h / 2
    return color, bx - 0.0162 * W, by - 0.0172 * H           # เลื่อนจากแท่งไปกลางช่อง


def detect_rod(frame):
    """เช็กเบ็ดจากสี: หากรอบเหลือง 4 มุมของช่องเบ็ดที่เลือกอยู่บนแถบล่างซ้าย
    แล้วดูแท่งความทนทานข้างในช่อง  คืน ("green"|"red", กึ่งกลางช่อง x, y) หรือ None (ไม่เจอแถบล่าง)"""
    H, W = frame.shape[:2]
    x0, y0 = 0, int(0.75 * H)
    x1 = int(0.65 * W)
    hsv = cv2.cvtColor(frame[y0:, x0:x1], cv2.COLOR_BGR2HSV)
    yel = cv2.inRange(hsv, (20, 120, 170), (36, 255, 255))
    n, _, st, _ = cv2.connectedComponentsWithStats(yel)
    corners = [(x, y, w, h) for x, y, w, h, a in st[1:]
               if 0.006 * W <= w <= 0.02 * W and 0.006 * W <= h <= 0.02 * W]
    if len(corners) < 3:
        return rod_from_bar(frame)          # ไม่มีกรอบเหลือง (เบ็ดไม่ได้ถูกเลือก/เบ็ดพัง) ดูจากแท่งความทนทาน
    # เอาเฉพาะกลุ่มที่ขนาดใกล้เคียงกัน (มุมของกรอบเดียวกัน)
    corners.sort(key=lambda c: c[2])
    ref = corners[len(corners) // 2]
    corners = [c for c in corners if abs(c[2] - ref[2]) <= 0.3 * ref[2] and abs(c[3] - ref[3]) <= 0.3 * ref[3]]
    if len(corners) < 3:
        return rod_from_bar(frame)
    bx0 = min(c[0] for c in corners)
    by0 = min(c[1] for c in corners)
    bx1 = max(c[0] + c[2] for c in corners)
    by1 = max(c[1] + c[3] for c in corners)
    box = hsv[by0:by1, bx0:bx1]
    h_, s_, v_ = box[..., 0], box[..., 1], box[..., 2]
    green = int(((h_ >= 45) & (h_ <= 85) & (s_ >= 90) & (v_ >= 190)).sum())
    red = int(((((h_ <= 8) | (h_ >= 172)) & (s_ >= 130) & (v_ >= 140))).sum())
    need = 0.00006 * W * H
    cx, cy = x0 + (bx0 + bx1) / 2, y0 + (by0 + by1) / 2
    if green >= need and green > red:
        return "green", cx, cy
    return "red", cx, cy    # เจอช่องเบ็ดแต่แท่งไม่เขียว = ต้องซ่อม/ใช้ไม่ได้


LEVEL_TEXT = {
    "full": "เต็ม (เขียว)",
    "yellow": "เริ่มหิว (เหลือง)",
    "red": "หิวข้าว (แดง) -- เตือน",
    "danger": "อันตรายถึงชีวิต (กระเพาะสีแดง) !!!",
}


_orange_ref = {}


def _orange_frac(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h_, s_, v_ = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return float(((h_ >= 9) & (h_ <= 24) & (s_ >= 120) & (v_ >= 150)).mean())   # ส้ม (ไม่รวมสีแดงของหัวใจ)


def hunger_level(frame, icons):
    """อ่านระดับหลอดอาหาร 4 ระดับ: "full" เต็ม / "yellow" เริ่มหิว / "red" หิวข้าว / "danger" อันตราย
    ใช้รูปหาตำแหน่ง แล้วยืนยันด้วยสี (กันเดาผิดจากภาพอื่น) คืน None ถ้าอ่านไม่ได้"""
    # --- หาไอคอนกระเพาะมีวงแหวน (เต็ม/เหลือง/แดง) ---
    best = (0.0, None, 0.0, 0.0)
    for n in HUNGER_NAMES:
        if n in icons:
            sc, cx, cy = find(frame, icons, n)
            if sc > best[0]:
                best = (sc, n, cx, cy)
    # --- ระดับอันตราย: ไอคอนกระเพาะสีส้มแดง ต้องเหมือนกว่าไอคอนวงแหวนชัดเจน และสียืนยันได้ ---
    if "hunger_danger" in icons:
        sd, dx, dy = find(frame, icons, "hunger_danger")
        if sd >= 0.85 and sd > best[0] + 0.10:
            tpl = icons["hunger_danger"]
            if "d" not in _orange_ref:
                _orange_ref["d"] = _orange_frac(tpl)
            k = _last_scale["hunger_danger"]
            hw, hh = int(tpl.shape[1] * k / 2), int(tpl.shape[0] * k / 2)
            box = frame[max(0, int(dy) - hh):int(dy) + hh, max(0, int(dx) - hw):int(dx) + hw]
            if box.size and _orange_frac(box) >= 0.6 * _orange_ref["d"]:
                return "danger"
    if best[1] is None or best[0] < 0.55:
        return None
    sc, n, cx, cy = best
    w = icons[n].shape[1] * _last_scale[n]                 # ความกว้างไอคอนบนจอ
    pad = int(0.7 * w)
    x0, y0 = max(0, int(cx - pad)), max(0, int(cy - pad))
    hsv = cv2.cvtColor(frame[y0:int(cy + pad), x0:int(cx + pad)], cv2.COLOR_BGR2HSV)
    yy, xx = np.ogrid[:hsv.shape[0], :hsv.shape[1]]
    d = np.hypot(xx - (cx - x0), yy - (cy - y0))
    h_, s_, v_ = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # ตัวกระเพาะสีเหลืองตรงกลางต้องมี ไม่งั้นไม่ใช่ไอคอนนี้
    glyph = (d <= 0.30 * w) & (h_ >= 18) & (h_ <= 34) & (s_ >= 90) & (v_ >= 140)
    if int(glyph.sum()) < 0.02 * w * w:
        return None
    ring = (d >= 0.34 * w) & (d <= 0.54 * w)
    cnt = {
        "full": int((ring & (h_ >= 40) & (h_ <= 90) & (s_ >= 100) & (v_ >= 120)).sum()),
        "yellow": int((ring & (h_ >= 18) & (h_ <= 34) & (s_ >= 120) & (v_ >= 170)).sum()),
        "red": int((ring & ((h_ <= 8) | (h_ >= 172)) & (s_ >= 110) & (v_ >= 110)).sum()),
    }
    top = max(cnt, key=cnt.get)
    if cnt[top] >= max(20, 0.004 * w * w):
        return top
    return None


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


def wait_icon(dev, icons, name, timeout, thr=THRESHOLD):
    """รอจนไอคอนขึ้นบนจอ คืน (x, y) หรือ None ถ้าหมดเวลา"""
    end = time.time() + timeout
    while time.time() < end:
        fr = dev.screen()
        if fr is not None:
            sc, x, y = find(fr, icons, name)
            if sc >= thr:
                return x, y
        time.sleep(0.15)
    return None


def repair_rod(dev, icons, slot_x, slot_y):
    """ซ่อมเบ็ด: กดเบ็ดแดง 1 ครั้ง -> ขึ้น Fix กด -> ขึ้น Equip No กดอีก (มีหน่วงระหว่างขั้น) คืน True/False"""
    STATUS.set("ซ่อมเบ็ด: กดเบ็ดสีแดง")
    log("เบ็ดแดง -> กดเบ็ดเพื่อซ่อม")
    dev.tap(slot_x, slot_y)
    time.sleep(REPAIR_DELAY)
    pos = wait_icon(dev, icons, "fix", REPAIR_WAIT)
    if pos is None:
        log("ไม่เจอปุ่ม Fix")
        return False
    STATUS.set("ซ่อมเบ็ด: กด Fix")
    log("กด Fix")
    dev.tap(*pos)
    time.sleep(REPAIR_DELAY)
    pos = wait_icon(dev, icons, "equip_no", REPAIR_WAIT) if "equip_no" in icons else None
    if pos is None:
        log("ไม่เจอปุ่ม Equip No")
        return False
    STATUS.set("ซ่อมเบ็ด: กด Equip No")
    log("กด Equip No")
    dev.tap(*pos)
    time.sleep(REPAIR_DELAY)
    log("ซ่อมเบ็ดเสร็จ")
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

    wait = int(sys.argv[1]) if len(sys.argv) > 1 else COUNTDOWN   # python boyfp.py 30 = นับ 30 วินาที
    for i in range(wait, 0, -1):
        STATUS.set(f"เริ่มใน {i} วิ (สลับไปหน้าเกม)")
        STATUS.draw(force=True)
        time.sleep(1)

    # ไม่เริ่มจนกว่าจะเห็นหน้าเกมจริง (ปุ่มคันเบ็ด/แถบเบ็ด) กันกดผิดหน้าจอ
    shown = 0.0
    while True:
        frame = dev.screen()
        if frame is not None and (detect_rod(frame) is not None or any(
                find(frame, icons, n)[0] >= THRESHOLD for n in ("cast", "reel"))):
            break
        STATUS.set("รอเข้าหน้าเกม (ยังไม่เห็นปุ่มคันเบ็ด)")
        STATUS.tick()
        time.sleep(0.5)
    log("เจอหน้าเกม เริ่มทำงาน")
    STATUS.set("ตรวจเบ็ด")
    repair_fail = 0

    timing = Timing(TAP_DELAY, USER_OFFSET, INNER_MARGIN, HIT_COOLDOWN)
    state = "IDLE"            # IDLE = รอแถบล่าง/เช็กเบ็ด/ตก   WAIT_BITE = ตกแล้ว รอปลากิน
    cast_time = 0.0
    last_action = 0.0
    last_seen_game = 0.0
    in_game = False
    missing = 0               # จำนวนเฟรมติดกันที่ไม่เห็นมินิเกม (เฉพาะเฟรมที่แคปได้)
    misses = 0
    last_miss = -9.0
    last_bait = -9.0
    last_bait_check = 0.0
    rod_seen = False          # เคยเห็นแถบล่างแล้ว
    no_rod_since = None
    red_count = 0
    green_while_wait = 0
    last_warn = 0.0
    caught = 0
    foods = [k for k in icons if k.startswith("food:")]
    hunger_on = any(n in icons for n in HUNGER_NAMES) or "hunger_danger" in icons
    eating = False
    eat_tries = 0
    last_eat = -99.0
    food_missing_until = 0.0
    last_level = None
    last_level_log = 0.0
    btn = None                # (x, y, รัศมี) ของปุ่มคันเบ็ดฝั่งขวา
    ring_hist = []
    bite_tapped = -99.0
    last_rod_check = 0.0

    while True:
        t0 = time.time()
        frame = dev.screen()
        if frame is None:
            time.sleep(0.05)      # แคปไม่สมบูรณ์ ลองเฟรมถัดไปทันที
            continue
        t = time.time()
        STATUS.tick()

        # ===== ทางด่วนรอปลากิน: ดูแค่วงแสงรอบปุ่ม (เร็วกว่าเทียบรูป) =====
        if state == "WAIT_BITE" and not in_game and time.time() - bite_tapped > 6.0:
            if btn is None:
                cs_, bx, by = find(frame, icons, "cast")
                if cs_ >= THRESHOLD and "cast" in _scale_cache:
                    btn = (bx, by, _scale_cache["cast"] * icons["cast"].shape[1] / 2)
                    log("เจอปุ่มคันเบ็ด ใช้วงแสงจับปลากิน")
            if btn is not None and time.time() - cast_time >= 0.5:
                p = ring_p90(frame, btn)
                ring_hist = (ring_hist + [p])[-20:]
                base = min(ring_hist)
                if p >= max(BITE_RING_MIN, base * BITE_RING_RATIO) and len(ring_hist) >= 3:
                    STATUS.set("ปลากิน! กดทันที")
                    log(f"ปลากิน (วงแสง {p:.0f} / ปกติ {base:.0f}) -> กดทันที")
                    dev.tap(btn[0], btn[1])
                    bite_tapped = last_action = time.time()
                    ring_hist = []
                    continue
                now = time.time()
                if now - cast_time > BITE_TIMEOUT:
                    log("รอปลาเกินเวลา กลับไปเช็กเบ็ดใหม่")
                    state = "IDLE"
                elif now - last_rod_check > 1.0:
                    last_rod_check = now
                    ri = detect_rod(frame)
                    if ri and ri[0] == "green" and now - cast_time > 3:
                        green_while_wait += 1
                        if green_while_wait >= 3:
                            log("แถบล่างกลับมาโดยไม่มีปลา -> ตกใหม่")
                            state = "IDLE"
                            green_while_wait = 0
                    else:
                        green_while_wait = 0
                if state == "WAIT_BITE":
                    continue

        # ===== 4) มินิเกม: ปลาวิ่งบนเส้น เมื่อถึงขีดขาวให้กด =====
        mg = detect_minigame(frame)
        if mg:
            missing = 0
            if not in_game:
                log("เข้ามินิเกม")
                in_game = True
                misses = 0
            STATUS.set(f"มินิเกม: กดตามจังหวะ (พลาด {misses}/{MAX_MISS})")
            last_seen_game = t
            state = "IDLE"

            # กดพลาด -> แถบเป็นสีแดง
            if detect_miss(frame) and t - last_miss > MISS_COOLDOWN:
                last_miss = t
                misses += 1
                STATUS.misses += 1
                STATUS.set(f"มินิเกม: กดตามจังหวะ (พลาด {misses}/{MAX_MISS})")
                log(f"กดพลาด (แถบแดง) ครั้งที่ {misses}/{MAX_MISS}")
                if misses >= MAX_MISS:
                    STATUS.set("หยุด: จับปลาไม่ได้")
                    log(f"จับปลาไม่ได้ (กดพลาด {misses} ครั้ง) หยุดบอท")
                    STATUS.stop_reason = f"จับปลาไม่ได้ (กดพลาด {misses} ครั้ง)"
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
            missing += 1
            if t - last_seen_game < MINIGAME_LOST_AFTER or missing < 3:
                continue          # ภาพกะพริบตอนกดโดน/แคปพลาด รอดูก่อน
            in_game = False
            timing.reset()
            caught += 1
            STATUS.fish += 1
            STATUS.set("จับปลาได้ -> รอแถบล่างกลับมา")
            last_action = time.time()
            rod_seen = True       # หลังจับปลา แถบล่างจะกลับมา ห้ามเดาว่ากำลังตกอยู่
            no_rod_since = None
            log(f"จบมินิเกม จับปลาแล้ว (รอบที่ {caught}) พลาด {misses} ครั้ง -> รอแถบล่างกลับมาแล้วตกใหม่")
            misses = 0
            eat_tries = 0

        if time.time() - last_action < ACTION_COOLDOWN:
            continue

        # ===== ปลากินเบ็ด (ปุ่มเรืองแสง) -> กด =====
        rs, rx, ry = find(frame, icons, "reel")
        if rs >= REEL_MIN:
            cs0, _, _ = find(frame, icons, "cast")
            if rs - cs0 >= REEL_MARGIN:          # ปุ่มเรืองแสงต้องเหมือน reel มากกว่า cast ชัดเจน
                STATUS.set("ปลากิน! กดทันที")
                log("ปลากิน (ปุ่มเรืองแสง) -> กด")
                dev.tap(rx, ry)
                last_action = time.time()
                state = "WAIT_BITE"
                cast_time = time.time()
                continue

        rod_info = detect_rod(frame)  # ("green"|"red", x, y) หรือ None (แถบล่างซ่อนอยู่)
        rod = rod_info[0] if rod_info else None

        # ===== รอปลากิน (แถบล่างจะหายไปตอนตก) =====
        if state == "WAIT_BITE":
            if rod == "green" and time.time() - cast_time > 3:
                green_while_wait += 1      # แถบกลับมาทั้งที่ยังไม่มีมินิเกม = ตกไม่ติด
                if green_while_wait >= 3:
                    log("แถบล่างกลับมาโดยไม่มีปลา -> ตกใหม่")
                    state = "IDLE"
                    green_while_wait = 0
            else:
                green_while_wait = 0
            if state == "WAIT_BITE" and time.time() - cast_time > BITE_TIMEOUT:
                log("รอปลาเกินเวลา กลับไปเช็กเบ็ดใหม่")
                state = "IDLE"
            continue

        # ===== 1) เช็กเบ็ดเขียว/แดงก่อนตกทุกครั้ง =====
        if rod is None and "rod_green" in icons:      # สำรอง: เทียบรูปไอคอน ถ้าหาจากสีไม่เจอ
            gs, _, _ = find(frame, icons, "rod_green")
            rds, _, _ = find(frame, icons, "rod_red") if "rod_red" in icons else (0, 0, 0)
            if gs >= THRESHOLD and gs >= rds:
                rod = "green"
            elif rds >= THRESHOLD:
                rod = "red"

        if rod:
            rod_seen = True
            no_rod_since = None
        elif no_rod_since is None:
            no_rod_since = time.time()

        if rod == "red":
            red_count += 1
            if red_count < 3:
                continue          # อ่านซ้ำให้แน่ใจก่อน
            red_count = 0
            if "fix" not in icons:
                STATUS.stop_reason = "เบ็ดแดง แต่ยังไม่มีไอคอน fix.png"
                log("เบ็ดแดง แต่ยังไม่มีไอคอน fix.png (และ equip_no.png) หยุดบอท")
                sys.exit(1)
            if repair_rod(dev, icons, rod_info[1], rod_info[2]):
                STATUS.repairs += 1
                repair_fail = 0
            else:
                repair_fail += 1
                if repair_fail >= MAX_REPAIR_FAIL:
                    STATUS.stop_reason = f"ซ่อมเบ็ดไม่สำเร็จติดกัน {repair_fail} ครั้ง"
                    log(f"ซ่อมเบ็ดไม่สำเร็จติดกัน {repair_fail} ครั้ง หยุดบอท")
                    sys.exit(1)
            STATUS.set("ตรวจเบ็ด")
            last_action = time.time()
            continue
        red_count = 0

        # ===== 2) เบ็ดเขียว -> กดตกปลา แล้วรอปลากิน =====
        if rod == "green" and hunger_on:
            now = time.time()
            level = hunger_level(frame, icons)
            if level:
                STATUS.hunger = LEVEL_TEXT[level]
            if level and (level != last_level or (level in ("red", "danger") and now - last_level_log > 60)):
                log(f"สถานะกระเพาะ: {LEVEL_TEXT[level]}")
                last_level, last_level_log = level, now
            if level in ("yellow", "red", "danger"):
                if not eating:
                    log("หลอดอาหารไม่เต็ม -> กินจนเต็มก่อนตกรอบนี้")
                eating = True                  # เหลือง/แดง/อันตราย -> กินจนเขียวเต็มค่อยตก (ไม่ขัดจังหวะตอนกำลังตก)
            elif level == "full":
                eating = False
                eat_tries = 0
            if eating and foods and now >= food_missing_until:
                if now - last_eat <= EAT_COOLDOWN:
                    continue                   # รอให้กระเพาะอัปเดต ยังไม่ตก
                if eat_tries >= MAX_EAT_TRIES:
                    log(f"กินครบ {MAX_EAT_TRIES} ครั้งแล้วกระเพาะยังไม่เต็ม -> ตกต่อ")
                    eating = False
                    eat_tries = 0
                else:
                    tapped = False
                    for name in foods:
                        fs_, fx, fy = find(frame, icons, name)
                        if fs_ >= FOOD_THRESHOLD:
                            eat_tries += 1
                            STATUS.eats += 1
                            STATUS.set(f"กินอาหารจนเต็ม (ครั้งที่ {eat_tries})")
                            log(f"กดอาหาร ({name[5:]}) ครั้งที่ {eat_tries} เพื่อให้กระเพาะเต็มก่อนตก")
                            dev.tap(fx, fy)
                            last_eat = last_action = time.time()
                            tapped = True
                            break
                    if tapped:
                        continue
                    log("ต้องกินแต่ไม่เจอไอคอนอาหารบนแถบล่าง -> ตกต่อ")
                    eating = False
                    food_missing_until = now + FOOD_MISSING_WAIT

        if rod == "green":
            # เบ็ดเขียว -> กดปุ่มดาบ -> ปุ่มเปลี่ยนเป็น cast.png และแถบล่างหาย = กำลังตก
            tx = ty = None
            for name, label in (("sword", "ปุ่มดาบ"), ("cast", "cast.png")):
                if name in icons:
                    sc_, x_, y_ = find(frame, icons, name)
                    if sc_ >= THRESHOLD:
                        log(f"เบ็ดเขียว -> กด{label}")
                        tx, ty = x_, y_
                        break
            if tx is None:
                log("เบ็ดเขียว แต่ไม่เจอปุ่มดาบ/cast.png -> กดที่ช่องเบ็ดแทน")
                tx, ty = rod_info[1], rod_info[2]
            dev.tap(tx, ty)
            STATUS.casts += 1
            STATUS.set("รอปลากิน")
            last_action = time.time()
            state = "WAIT_BITE"
            cast_time = time.time()
            green_while_wait = 0
            ring_hist = []
            bite_tapped = -99.0
            continue

        # ===== ไม่เห็นแถบล่าง =====
        if not rod_seen and no_rod_since and time.time() - no_rod_since > NO_ROD_ASSUME_FISHING:
            log("เริ่มบอทตอนแถบล่างซ่อนอยู่ -> ถือว่ากำลังตกอยู่ รอปลากิน")
            state = "WAIT_BITE"
            cast_time = time.time()
            no_rod_since = None
            STATUS.set("รอปลากิน")
        elif rod_seen:
            STATUS.set("รอแถบล่างกลับมา")


if __name__ == "__main__":
    reason = "หยุดโดยผู้ใช้ (Ctrl+C)"
    try:
        main()
    except KeyboardInterrupt:
        pass
    except SystemExit:
        reason = getattr(STATUS, "stop_reason", "หยุดการทำงาน")
    finally:
        print("\n" + STATUS.summary())
        print(f" หยุดเพราะ : {reason}")
        print("━" * Status.W)
