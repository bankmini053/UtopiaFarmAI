import os
import re
import time
import subprocess
import numpy as np
from PIL import Image
import pytesseract

# ==========================================
# 1. ตัวจำลองโมดูล OpenCV ในตัว (Standalone CV2 Engine)
# รองรับการทำงานบน Python 3.14 ใน Termux โดยไม่ต้องลง cv2 เพิ่ม
# ==========================================
class MiniCV:
    COLOR_BGR2GRAY = 6
    COLOR_RGB2GRAY = 7
    COLOR_GRAY2BGR = 8
    COLOR_GRAY2RGB = 8
    COLOR_BGR2HSV = 40
    COLOR_RGB2HSV = 41

    ROTATE_90_CLOCKWISE = 0
    ROTATE_180 = 1
    ROTATE_90_COUNTERCLOCKWISE = 2

    RETR_EXTERNAL = 0
    CHAIN_APPROX_SIMPLE = 1
    THRESH_BINARY = 0

    @staticmethod
    def rotate(src, rotateCode):
        if rotateCode == MiniCV.ROTATE_90_CLOCKWISE:
            return np.rot90(src, -1)
        elif rotateCode == MiniCV.ROTATE_180:
            return np.rot90(src, 2)
        elif rotateCode == MiniCV.ROTATE_90_COUNTERCLOCKWISE:
            return np.rot90(src, 1)
        return src

    @staticmethod
    def imread(path, flags=1):
        try:
            img = Image.open(path)
            if flags == 0:
                return np.array(img.convert('L'))
            arr = np.array(img.convert('RGB'))
            return arr[:, :, ::-1].copy()
        except Exception:
            return None

    @staticmethod
    def imwrite(path, img):
        try:
            if len(img.shape) == 2:
                Image.fromarray(img).save(path)
            else:
                Image.fromarray(img[:, :, ::-1]).save(path)
            return True
        except Exception:
            return False

    @staticmethod
    def _bgr2hsv(bgr):
        img = bgr.astype(np.float32) / 255.0
        b, g, r = img[..., 0], img[..., 1], img[..., 2]
        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        df = mx - mn
        h = np.zeros_like(mx)
        mask = df > 1e-6
        mask_r = mask & (mx == r)
        mask_g = mask & (mx == g) & (~mask_r)
        mask_b = mask & (mx == b) & (~mask_r) & (~mask_g)
        h[mask_r] = ((g[mask_r] - b[mask_r]) / df[mask_r]) % 6.0
        h[mask_g] = ((b[mask_g] - r[mask_g]) / df[mask_g]) + 2.0
        h[mask_b] = ((r[mask_b] - g[mask_b]) / df[mask_b]) + 4.0
        h = (h * 30.0) % 180.0
        s = np.zeros_like(mx)
        s_mask = mx > 1e-6
        s[s_mask] = (df[s_mask] / mx[s_mask]) * 255.0
        v = mx * 255.0
        return np.stack([h, s, v], axis=-1).astype(np.uint8)

    @staticmethod
    def cvtColor(src, code):
        if code in (MiniCV.COLOR_BGR2HSV, MiniCV.COLOR_RGB2HSV):
            return MiniCV._bgr2hsv(src)
        elif code == MiniCV.COLOR_BGR2GRAY:
            if len(src.shape) == 3:
                return np.dot(src[..., :3], [0.114, 0.587, 0.299]).astype(np.uint8)
            return src
        elif code in (MiniCV.COLOR_GRAY2BGR, MiniCV.COLOR_GRAY2RGB):
            if len(src.shape) == 2:
                return np.stack((src,)*3, axis=-1)
            return src
        return src

    @staticmethod
    def inRange(src, lowerb, upperb):
        lowerb = np.asarray(lowerb, dtype=src.dtype)
        upperb = np.asarray(upperb, dtype=src.dtype)
        mask = np.all((src >= lowerb) & (src <= upperb), axis=-1)
        return np.where(mask, 255, 0).astype(np.uint8)

    @staticmethod
    def findContours(image, mode=0, method=0):
        y_idx, x_idx = np.where(image > 0)
        if len(x_idx) == 0:
            return [], None
        pts = np.stack([x_idx, y_idx], axis=1).reshape(-1, 1, 2)
        return [pts], None

    @staticmethod
    def contourArea(contour):
        if contour is None:
            return 0.0
        pts = np.asarray(contour).reshape(-1, 2)
        return float(len(pts))

    @staticmethod
    def moments(array):
        pts = np.asarray(array).reshape(-1, 2)
        m = {"m00": float(len(pts)), "m10": 0.0, "m01": 0.0}
        if len(pts) > 0:
            m["m10"] = float(np.sum(pts[:, 0]))
            m["m01"] = float(np.sum(pts[:, 1]))
        return m

    @staticmethod
    def boundingRect(array):
        pts = np.asarray(array).reshape(-1, 2)
        if len(pts) == 0:
            return 0, 0, 0, 0
        x_min, y_min = int(pts[:, 0].min()), int(pts[:, 1].min())
        x_max, y_max = int(pts[:, 0].max()), int(pts[:, 1].max())
        return x_min, y_min, (x_max - x_min + 1), (y_max - y_min + 1)

    @staticmethod
    def threshold(src, thresh, maxval, type=0):
        res = np.where(src > thresh, maxval, 0).astype(np.uint8)
        return float(thresh), res

cv2 = MiniCV()

# ==========================================
# 2. คอนฟิกหลักของระบบบอทตกปลา
# ==========================================
ROOT_SU = "su"
TMP_PATH = "/data/local/tmp/screen.png"
FISH_BTN = (1280, 550)  # พิกัดปุ่มตกปลา (อิงจอ 1600x720)

def run_cmd(cmd):
    subprocess.run(f"{ROOT_SU} -c '{cmd}'", shell=True)

def tap(x, y):
    run_cmd(f"input tap {x} {y}")

def get_screen():
    run_cmd(f"screencap -p {TMP_PATH}")
    if os.path.exists(TMP_PATH):
        for _ in range(6):
            if os.path.getsize(TMP_PATH) > 50000:
                break
            time.sleep(0.03)

        img = cv2.imread(TMP_PATH)
        if img is not None:
            h, w = img.shape[:2]
            if h > w:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            return img
    return None

def check_minigame_active(screen):
    """
    ตรวจสอบว่าหลอดมินิเกมตกปลาด้านบนจอแสดงขึ้นมาหรือยัง (สเกล 1600x720)
    """
    if screen is None:
        return False, None
    h, w = screen.shape[:2]
    scale_x = w / 1600.0
    scale_y = h / 720.0
    
    x1, x2 = int(500 * scale_x), int(1100 * scale_x)
    y1, y2 = int(50 * scale_y), int(130 * scale_y)
    
    bar_roi = screen[y1:y2, x1:x2]
    gray = cv2.cvtColor(bar_roi, cv2.COLOR_BGR2GRAY)
    
    white_pixels = np.sum(gray > 210)
    if white_pixels > 150:
        return True, bar_roi
    return False, None

def analyze_fish_and_bar(bar_roi):
    """
    วิเคราะห์ตำแหน่งตัวปลาและเส้นตัวชี้ขาว คืนค่า (ควรดึงเบ็ด, ระยะห่าง)
    """
    if bar_roi is None:
        return False, 999

    hsv = cv2.cvtColor(bar_roi, cv2.COLOR_BGR2HSV)

    # 1. ค้นหาไอคอนวงกลมปลาสีทอง/เหลือง
    lower_gold = np.array([15, 100, 150])
    upper_gold = np.array([35, 255, 255])
    mask_fish = cv2.inRange(hsv, lower_gold, upper_gold)
    
    fish_x = None
    fish_contours, _ = cv2.findContours(mask_fish, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if fish_contours:
        largest_cnt = max(fish_contours, key=cv2.contourArea)
        if cv2.contourArea(largest_cnt) > 20:
            M = cv2.moments(largest_cnt)
            if M["m00"] != 0:
                fish_x = int(M["m10"] / M["m00"])

    # 2. ค้นหาตำแหน่งเส้นสีขาวตัวชี้ (Cursor)
    lower_white = np.array([0, 0, 220])
    upper_white = np.array([180, 40, 255])
    mask_cursor = cv2.inRange(hsv, lower_white, upper_white)
    
    cursor_x = None
    cursor_contours, _ = cv2.findContours(mask_cursor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cursor_contours:
        for cnt in cursor_contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if h >= 12 and w <= 30:
                cursor_x = x + (w // 2)
                break

    # 3. คำนวณระยะห่างระหว่างตัวชี้กับปลา
    if fish_x is not None and cursor_x is not None:
        diff = abs(fish_x - cursor_x)
        if diff <= 45:
            return True, diff
        return False, diff

    return False, 999

def check_fish_reward(screen):
    """
    ตรวจสอบชื่อปลาที่ได้รับบริเวณกลางจอด้วย Tesseract OCR
    """
    if screen is None:
        return ""

    h, w = screen.shape[:2]
    scale_x = w / 1600.0
    scale_y = h / 720.0
    
    x1, x2 = int(750 * scale_x), int(1200 * scale_x)
    y1, y2 = int(450 * scale_y), int(630 * scale_y)
    
    roi = screen[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    
    try:
        text = pytesseract.image_to_string(Image.fromarray(thresh), lang="eng", config=r"--psm 6").strip()
    except Exception:
        text = ""
    return text

def main():
    print("=" * 60)
    print("[🎣] เริ่มต้นระบบบอทตกปลาอัตโนมัติ (Utopia: Origin)")
    print(f"[*] พิกัดปุ่มตกปลา (ปุ่มตี): {FISH_BTN}")
    print("=" * 60)

    # นับถอยหลัง 10 วินาทีให้สลับหน้าต่างกลับเข้าเกม
    print("\n[⏱️] กรุณาสลับหน้าจอกลับเข้าเกม Utopia: Origin...")
    for remaining in range(10, 0, -1):
        print(f"[*] บอทจะเริ่มทำงานในอีก {remaining} วินาที...", end="\r", flush=True)
        time.sleep(1.0)
    print("\n[🚀] หมดเวลาเตรียมตัว -> เริ่มต้นระบบบอทตกปลาทันที!\n")

    elfish_count = 0
    total_fish_count = 0
    cast_round = 0

    while True:
        cast_round += 1
        print(f"\n{'='*15} เริ่มต้นเหวี่ยงเบ็ดรอบที่ {cast_round} {'='*15}")
        
        # 1. กดปุ่มเหวี่ยงเบ็ด
        print("[*] กดปุ่มเริ่มตกปลา...")
        tap(FISH_BTN[0], FISH_BTN[1])
        time.sleep(1.5)

        # 2. รอปลากินเบ็ด
        print("[⏳] กำลังรอปลากินเบ็ด...")
        minigame_started = False
        wait_start = time.time()

        while time.time() - wait_start < 20.0:
            scr = get_screen()
            is_active, bar_roi = check_minigame_active(scr)
            if is_active:
                minigame_started = True
                print("[🐟] ปลากินเบ็ดแล้ว! หลอดมินิเกมปรากฏ -> เริ่มเล่นจังหวะดึง...")
                break
            time.sleep(0.3)

        if not minigame_started:
            print("[⚠️] ปลารอบนี้ไม่กินเบ็ด หรือเหยื่อหมด -> ลองกดเหวี่ยงใหม่")
            continue

        # 3. ลูปมินิเกม: กดย้ำเมื่อเส้นขาวทับปลา
        minigame_timer = time.time()
        while time.time() - minigame_timer < 12.0:
            scr = get_screen()
            is_active, bar_roi = check_minigame_active(scr)
            if not is_active:
                print("[✓] จบมินิเกมตกปลาแล้ว!")
                break

            should_pull, diff = analyze_fish_and_bar(bar_roi)
            if should_pull:
                print(f"[🎯] เส้นทับตัวปลาพอดี (ต่างกัน {diff}px) -> กดดึงเบ็ด!")
                tap(FISH_BTN[0], FISH_BTN[1])
                time.sleep(0.12)
            else:
                time.sleep(0.04)

        # 4. หยุดรอแอนิเมชันและตรวจเช็กชื่อปลา
        time.sleep(1.2)
        end_scr = get_screen()
        reward_text = check_fish_reward(end_scr)
        
        if reward_text:
            print(f"[📦] ได้รับไอเทม: '{reward_text}'")
            total_fish_count += 1
            if any(k in reward_text.lower() for k in ["elfish", "eife", "el"]):
                elfish_count += 1
                print(f"\n[🌟] ตกได้ Elfish สำเร็จ! (สะสมรวม: {elfish_count} ตัว)")
        else:
            print("[*] ตกปลาสำเร็จรอบนี้")

        print(f"[📊] สถิติ: ตกได้ทั้งหมด {total_fish_count} ตัว | Elfish: {elfish_count} ตัว")
        time.sleep(2.0)

if __name__ == "__main__":
    main()
                
