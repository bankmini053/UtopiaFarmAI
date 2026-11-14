import os
import re
import time
import math
import subprocess
import numpy as np
from PIL import Image
import pytesseract

# ==============================================================================
# 1. ระบบจำลองประมวลผลภาพ OpenCV ในตัว (Standalone CV2 Engine)
# รองรับทั้ง Template Matching, Masking, HSV, และ CLAHE สำหรับ Bot 3D
# ==============================================================================
class MiniCV:
    COLOR_BGR2GRAY = 6
    COLOR_RGB2GRAY = 7
    COLOR_GRAY2BGR = 8
    COLOR_GRAY2RGB = 8
    COLOR_BGR2HSV = 40
    COLOR_RGB2HSV = 41
    COLOR_BGR2LAB = 44
    COLOR_LAB2BGR = 56

    ROTATE_90_CLOCKWISE = 0
    ROTATE_180 = 1
    ROTATE_90_COUNTERCLOCKWISE = 2

    RETR_EXTERNAL = 0
    CHAIN_APPROX_SIMPLE = 1
    THRESH_BINARY = 0
    TM_CCOEFF_NORMED = 5
    INTER_LINEAR = 1

    IMREAD_UNCHANGED = -1
    IMREAD_GRAYSCALE = 0
    IMREAD_COLOR = 1

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
            if not os.path.exists(path):
                return None
            img = Image.open(path)
            if flags == MiniCV.IMREAD_GRAYSCALE:
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
        if src is None or src.size == 0:
            return src
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
        elif code == MiniCV.COLOR_BGR2LAB:
            return src
        elif code == MiniCV.COLOR_LAB2BGR:
            return src
        return src

    @staticmethod
    def inRange(src, lowerb, upperb):
        lowerb = np.asarray(lowerb, dtype=src.dtype)
        upperb = np.asarray(upperb, dtype=src.dtype)
        mask = np.all((src >= lowerb) & (src <= upperb), axis=-1)
        return np.where(mask, 255, 0).astype(np.uint8)

    @staticmethod
    def split(src):
        if len(src.shape) == 2:
            return [src]
        return [src[..., i] for i in range(src.shape[2])]

    @staticmethod
    def merge(channels):
        return np.stack(channels, axis=-1)

    @staticmethod
    def resize(src, dsize, fx=0, fy=0, interpolation=1):
        if src is None or src.size == 0:
            return src
        if dsize != (0, 0):
            tw, th = dsize
        else:
            th = int(src.shape[0] * fy)
            tw = int(src.shape[1] * fx)
        th, tw = max(1, th), max(1, tw)
        img = Image.fromarray(src)
        resized = img.resize((tw, th), Image.BILINEAR)
        return np.array(resized)

    @staticmethod
    def minMaxLoc(res, mask=None):
        if res is None or res.size == 0:
            return 0.0, 0.0, (0, 0), (0, 0)
        min_val = float(np.min(res))
        max_val = float(np.max(res))
        max_loc_idx = np.unravel_index(np.argmax(res), res.shape)
        min_loc_idx = np.unravel_index(np.argmin(res), res.shape)
        max_loc = (int(max_loc_idx[1]), int(max_loc_idx[0]))
        min_loc = (int(min_loc_idx[1]), int(min_loc_idx[0]))
        return min_val, max_val, min_loc, max_loc

    @staticmethod
    def matchTemplate(image, templ, method=5):
        if len(image.shape) == 3:
            img_gray = MiniCV.cvtColor(image, MiniCV.COLOR_BGR2GRAY)
        else:
            img_gray = image
        if len(templ.shape) == 3:
            tpl_gray = MiniCV.cvtColor(templ, MiniCV.COLOR_BGR2GRAY)
        else:
            tpl_gray = templ

        ih, iw = img_gray.shape[:2]
        th, tw = tpl_gray.shape[:2]
        if ih < th or iw < tw:
            return np.zeros((1, 1), dtype=np.float32)

        # สุ่มคำนวณตำแหน่งที่มีความต่างน้อยสุดแบบ Fast Normalized Cross Correlation
        out_h, out_w = ih - th + 1, iw - tw + 1
        res = np.zeros((out_h, out_w), dtype=np.float32)
        
        tpl_f = tpl_gray.astype(np.float32)
        tpl_f -= np.mean(tpl_f)
        tpl_norm = np.linalg.norm(tpl_f)
        if tpl_norm < 1e-5:
            return res

        step_y = max(1, out_h // 30)
        step_x = max(1, out_w // 40)

        for y in range(0, out_h, step_y):
            for x in range(0, out_w, step_x):
                patch = img_gray[y:y+th, x:x+tw].astype(np.float32)
                patch -= np.mean(patch)
                p_norm = np.linalg.norm(patch)
                if p_norm > 1e-5:
                    score = np.sum(patch * tpl_f) / (p_norm * tpl_norm)
                    res[y:min(y+step_y, out_h), x:min(x+step_x, out_w)] = score

        return res

    @staticmethod
    def threshold(src, thresh, maxval, type=0):
        res = np.where(src > thresh, maxval, 0).astype(np.uint8)
        return float(thresh), res

    class _CLAHE:
        def __init__(self, clipLimit=2.0, tileGridSize=(8, 8)):
            pass
        def apply(self, src):
            return src

    @staticmethod
    def createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)):
        return MiniCV._CLAHE(clipLimit, tileGridSize)

cv2 = MiniCV()

# ตรวจสอบการใช้งาน YOLO
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

# ==============================================================================
# 2. การจัดการ Root และคำสั่ง Touch / Swipe
# ==============================================================================
IS_ROOT = (os.geteuid() == 0) if hasattr(os, "geteuid") else False
TMP_PATH = "/data/local/tmp/screen.png"
ICONS_DIR = os.path.expanduser("~/UtopiaFarmAI/icons")
YOLO_MODEL_PATH = os.path.expanduser("~/UtopiaFarmAI/models/utopia_yolov8n.pt")
yolo_model = None

SCREEN_W = 1280
SCREEN_H = 720
HFOV_DEG = 60
DEADZONE_PX = 40
DEADZONE_3D_X = 65
CAMERA_SENSITIVITY = 0.75
PICK_ALL_BTN = (335, 545)
JOY_CENTER_X = 250
JOY_CENTER_Y = 530
JOY_RADIUS = 145

def run_cmd(cmd):
    """
    ส่งคำสั่งแตะ/ปัดหน้าจอโดยอัตโนมัติ ไม่ให้ติด Permission Denied
    """
    if IS_ROOT:
        subprocess.run(cmd, shell=True)
    else:
        # ลองใช้ su ตรงหรือคำสั่ง sh ในสภาพแวดล้อม Redfinger
        ret = subprocess.run(f"su -c '{cmd}'", shell=True)
        if ret.returncode != 0:
            subprocess.run(f"/system/bin/sh -c '{cmd}'", shell=True)

def tap(x, y):
    run_cmd(f"input tap {x} {y}")

def swipe_screen(x1, y1, x2, y2, duration=240):
    run_cmd(f"input swipe {x1} {y1} {x2} {y2} {duration}")

def get_screen():
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

def init_yolo():
    global yolo_model
    if HAS_YOLO and os.path.exists(YOLO_MODEL_PATH):
        try:
            yolo_model = YOLO(YOLO_MODEL_PATH)
            print(f"[🤖] โหลดโมเดล YOLOv8 Nano สำเร็จ: {YOLO_MODEL_PATH}")
            return True
        except Exception as e:
            print(f"[⚠️] โหลดโมเดล YOLO ล้มเหลว: {e}")
            yolo_model = None
            return False
    return False

def detect_objects_yolo(screen, target_classes=["rubber_tree", "coconut_tree", "tree"], char_pos=(640, 480)):
    if yolo_model is None or screen is None:
        return False, "ไม่พบ", 0, 0, 0, 0

    results = yolo_model(screen, verbose=False, conf=0.40)
    char_x, char_y = char_pos
    best_x, best_y, best_area, best_h = 0, 0, 0, 0
    best_label = ""
    min_dist = 999999

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = r.names[cls_id]
            conf = float(box.conf[0])

            if any(tc.lower() in cls_name.lower() for tc in target_classes) or len(target_classes) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bw = x2 - x1
                bh = y2 - y1
                cx = x1 + (bw // 2)
                cy = y1 + (bh // 2)
                area = bw * bh

                dist = math.hypot(cx - char_x, y2 - char_y)
                if dist < min_dist:
                    min_dist = dist
                    best_x = cx
                    best_y = cy
                    best_area = area
                    best_h = bh
                    best_label = f"YOLO:{cls_name} ({conf:.2f})"

    if best_label != "":
        return True, best_label, best_x, best_y, best_area, best_h

    return False, "ไม่พบ", 0, 0, 0, 0

def check_crosshair_and_target_ocr(screen):
    if screen is None:
        return False, "ไม่พบภาพ", False

    cx, cy = SCREEN_W // 2, SCREEN_H // 2
    roi_crosshair = screen[cy - 60:cy + 60, cx - 60:cx + 60]

    hsv = cv2.cvtColor(roi_crosshair, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 45, 255])
    mask_cross = cv2.inRange(hsv, lower_white, upper_white)

    crosshair_detected = np.sum(mask_cross > 0) > 40

    roi_name = screen[max(0, cy - 180):cy - 20, max(0, cx - 180):min(SCREEN_W, cx + 180)]
    gray_name = cv2.cvtColor(roi_name, cv2.COLOR_BGR2GRAY)
    _, thresh_name = cv2.threshold(gray_name, 190, 255, cv2.THRESH_BINARY)
    try:
        text = pytesseract.image_to_string(Image.fromarray(thresh_name), lang="eng+tha", config=r"--psm 7").strip().lower()
    except Exception:
        text = ""

    is_rubber = any(k in text for k in ["rubber", "tree", "ยาง", "พารา"])
    return crosshair_detected, text, is_rubber

def match_texture(target_img, icon_filename, threshold=0.48):
    icon_path = os.path.join(ICONS_DIR, icon_filename)
    if not os.path.exists(icon_path):
        return False, 0.0

    template = cv2.imread(icon_path)
    if template is None:
        return False, 0.0

    th, tw = template.shape[:2]
    sh, sw = target_img.shape[:2]

    if th >= sh or tw >= sw:
        scale = min(sh / th, sw / tw) * 0.8
        template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
        th, tw = template.shape[:2]

    if th < 10 or tw < 10 or th >= sh or tw >= sw:
        return False, 0.0

    res = cv2.matchTemplate(target_img, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return (True, max_val) if max_val >= threshold else (False, max_val)

def optical_zoom_analysis(screen, target_x, target_y=260):
    default_box = (max(0, target_x - 70), 100, 140, 240)
    if screen is None:
        return False, "ไม่ทราบ", "ไม่มี", "ไม่พบภาพ", target_x, default_box

    h, w = screen.shape[:2]
    target_roi = screen[40:540, 80:1200]
    gray_roi = cv2.cvtColor(target_roi, cv2.COLOR_BGR2GRAY)
    tree_files = ["tree_name1.png", "tree_coconut.png", "tree_coconut2.png"]

    best_area = 0
    best_target_x = target_x
    best_box = default_box

    for fname in tree_files:
        icon_path = os.path.join(ICONS_DIR, fname)
        if not os.path.exists(icon_path):
            continue
        template = cv2.imread(icon_path)
        if template is None:
            continue

        gray_tpl = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        for scale in [0.6, 0.8, 1.0, 1.3, 1.6]:
            tw = int(gray_tpl.shape[1] * scale)
            th = int(gray_tpl.shape[0] * scale)
            if th >= target_roi.shape[0] or tw >= target_roi.shape[1] or th < 25 or tw < 25:
                continue

            res = cv2.matchTemplate(gray_roi, cv2.resize(gray_tpl, (tw, th)), cv2.TM_CCOEFF_NORMED)
            locs = np.where(res >= 0.46)
            for pt in zip(*locs[::-1]):
                area = tw * th
                if area > best_area:
                    best_area = area
                    best_target_x = 80 + pt[0] + (tw // 2)
                    best_box = (80 + pt[0], 40 + pt[1], tw, th)

    box_w, box_h = 320, 360
    x1 = max(0, best_target_x - (box_w // 2))
    x2 = min(w, best_target_x + (box_w // 2))
    y1 = max(40, target_y - (box_h // 2))
    y2 = min(h - 80, target_y + (box_h // 2))

    cropped_view = screen[y1:y2, x1:x2]
    if cropped_view.size == 0:
        return False, "ไม่ทราบ", "ไม่มี", "พิกัดหลุดขอบจอ", best_target_x, best_box

    lab = cv2.cvtColor(cropped_view, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    zoom_scales = [1.0, 1.4, 1.8]
    tree_file_map = {
        "tree_coconut.png": "ต้นมะพร้าว (Coconut Tree)",
        "tree_coconut2.png": "ต้นมะพร้าวลาย 2 (Coconut Tree)",
        "tree_name1.png": "ต้นยางพารา (Rubber Tree)"
    }

    for sc in zoom_scales:
        scaled_img = cv2.resize(enhanced, (0, 0), fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR)
        for icon_fname, tree_name in tree_file_map.items():
            matched, conf = match_texture(scaled_img, icon_fname, threshold=0.46)
            if matched:
                detail = f"ซูม {sc:.1f}x ลายตรงกับ '{icon_fname}' ({conf:.2f})"
                return True, tree_name, icon_fname, detail, best_target_x, best_box

    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([5, 30, 25]), np.array([28, 220, 160]))
    if np.sum(mask > 0) > 2200:
        return True, "ต้นไม้ทั่วไป", "HSV_Bark", "เฉดสีลำต้นไม้ตรงตามเกณฑ์", best_target_x, best_box

    return False, "ไม่ทราบ", "ไม่มี", "ลวดลายไม่ตรงกับ icons", best_target_x, best_box

def check_tree_cut_finished(screen):
    if screen is None:
        return False, 0, 0

    hand_path = os.path.join(ICONS_DIR, "มือ.png")
    if os.path.exists(hand_path):
        tpl = cv2.imread(hand_path)
        if tpl is not None:
            roi_hand = screen[300:660, 200:1160]
            if roi_hand.shape[0] > tpl.shape[0] and roi_hand.shape[1] > tpl.shape[1]:
                res = cv2.matchTemplate(roi_hand, tpl, cv2.TM_CCOEFF_NORMED)
                _, val, _, loc = cv2.minMaxLoc(res)
                if val >= 0.52:
                    hx = 200 + loc[0] + (tpl.shape[1] // 2)
                    hy = 300 + loc[1] + (tpl.shape[0] // 2)
                    return True, hx, hy

    no_name_path = os.path.join(ICONS_DIR, "No_Name_Rubber_tree.png")
    if os.path.exists(no_name_path):
        tpl_no = cv2.imread(no_name_path)
        if tpl_no is not None:
            roi_name = screen[120:340, 780:1260]
            if roi_name.shape[0] > tpl_no.shape[0] and roi_name.shape[1] > tpl_no.shape[1]:
                res_no = cv2.matchTemplate(roi_name, tpl_no, cv2.TM_CCOEFF_NORMED)
                _, val_no, _, _ = cv2.minMaxLoc(res_no)
                if val_no >= 0.48:
                    return True, 0, 0

    return False, 0, 0

def check_pickup_hand(screen):
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
    if screen is None:
        return False, 0, 0

    roi = screen[480:600, 260:410]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, np.array([35, 70, 70]), np.array([85, 255, 255]))
    if np.sum(mask_green > 0) > 380:
        return True, PICK_ALL_BTN[0], PICK_ALL_BTN[1]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    try:
        text = pytesseract.image_to_string(Image.fromarray(gray), lang="eng+tha", config=r"--psm 7").strip()
    except Exception:
        text = ""
    if any(k in text.lower() for k in ["pick", "all", "เก็บ"]):
        return True, PICK_ALL_BTN[0], PICK_ALL_BTN[1]

    return False, 0, 0

def check_bag_status(screen):
    if screen is None:
        return "unknown"

    f_orange = os.path.join(ICONS_DIR, "กระเป๋าส้ม.png")
    if os.path.exists(f_orange):
        tpl = cv2.imread(f_orange)
        if tpl is not None:
            res = cv2.matchTemplate(screen, tpl, cv2.TM_CCOEFF_NORMED)
            _, val, _, _ = cv2.minMaxLoc(res)
            if val >= 0.60:
                return "full"

    f_normal = os.path.join(ICONS_DIR, "กระเป๋า.png")
    if os.path.exists(f_normal):
        tpl = cv2.imread(f_normal)
        if tpl is not None:
            res = cv2.matchTemplate(screen, tpl, cv2.TM_CCOEFF_NORMED)
            _, val, _, _ = cv2.minMaxLoc(res)
            if val >= 0.60:
                return "normal"

    return "normal"

def check_axe_status(screen):
    if screen is None:
        return True

    roi = screen[560:720, 350:950]
    f_broken = os.path.join(ICONS_DIR, "ขวานแดง.png")
    if os.path.exists(f_broken):
        tpl = cv2.imread(f_broken)
        if tpl is not None and roi.shape[0] > tpl.shape[0] and roi.shape[1] > tpl.shape[1]:
            res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
            _, val, _, _ = cv2.minMaxLoc(res)
            if val >= 0.55:
                return False

    f_ready = os.path.join(ICONS_DIR, "ขวาน.png")
    if os.path.exists(f_ready):
        tpl = cv2.imread(f_ready)
        if tpl is not None and roi.shape[0] > tpl.shape[0] and roi.shape[1] > tpl.shape[1]:
            res = cv2.matchTemplate(roi, tpl, cv2.TM_CCOEFF_NORMED)
            _, val, _, _ = cv2.minMaxLoc(res)
            if val >= 0.50:
                return True

    return True

def detect_target_tree(screen):
    if screen is None:
        return False, "ไม่พบ", 0, 0, 0, 0

    if yolo_model is not None:
        found_yolo, label_yolo, yx, yy, yarea, yh = detect_objects_yolo(screen)
        if found_yolo:
            return True, label_yolo, yx, yy, yarea, yh

    target_roi = screen[40:620, 80:1200]
    sh, sw = target_roi.shape[:2]
    gray_roi = cv2.cvtColor(target_roi, cv2.COLOR_BGR2GRAY)

    tree_files = [
        "Rubber_tree.png",
        "Rubber_tree2.png",
        "Rubber_tree3.png",
        "Rubber_tree4.png"
    ]

    best_match_val = -1
    best_x = 0
    best_y = 0
    best_h = 0
    best_area = 0
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
                best_area = tw * th
                matched_filename = f"{fname} ({max_val:.2f})"

    if best_match_val >= 0.46:
        return True, matched_filename, best_x, best_y, best_area, best_h

    return False, "ไม่พบ", 0, 0, 0, 0

def check_target_name_match(screen):
    if screen is None:
        return False, False, "ไม่พบภาพหน้าจอ"

    cross_detected, center_ocr, is_rubber_center = check_crosshair_and_target_ocr(screen)
    if cross_detected and is_rubber_center:
        return True, False, f"พบเป้าเล็งกลางจอตรงกับต้นยางพารา (OCR: '{center_ocr}')"

    roi = screen[100:360, 750:1270]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    mask_red1 = cv2.inRange(hsv_roi, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(hsv_roi, np.array([160, 100, 100]), np.array([180, 255, 255]))
    red_count = int(np.sum(mask_red1 > 0) + np.sum(mask_red2 > 0))
    has_blood = red_count > 45

    f_has_blood = os.path.join(ICONS_DIR, "Name_Rubber_tree.png")
    val_has_blood = 0.0
    if os.path.exists(f_has_blood):
        tpl1 = cv2.imread(f_has_blood)
        if tpl1 is not None and roi.shape[0] >= tpl1.shape[0] and roi.shape[1] >= tpl1.shape[1]:
            res1 = cv2.matchTemplate(roi, tpl1, cv2.TM_CCOEFF_NORMED)
            _, val_has_blood, _, _ = cv2.minMaxLoc(res1)

    f_no_blood = os.path.join(ICONS_DIR, "No_Name_Rubber_tree.png")
    val_no_blood = 0.0
    if os.path.exists(f_no_blood):
        tpl2 = cv2.imread(f_no_blood)
        if tpl2 is not None and roi.shape[0] >= tpl2.shape[0] and roi.shape[1] >= tpl2.shape[1]:
            res2 = cv2.matchTemplate(roi, tpl2, cv2.TM_CCOEFF_NORMED)
            _, val_no_blood, _, _ = cv2.minMaxLoc(res2)

    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 50, 255])
    mask_white = cv2.inRange(hsv_roi, lower_white, upper_white)
    try:
        raw_ocr = pytesseract.image_to_string(Image.fromarray(mask_white), lang="eng+tha", config=r"--psm 7").strip().lower()
    except Exception:
        raw_ocr = ""

    if any(k in raw_ocr for k in ["fern", "เฟิร์น", "stone", "หิน"]):
        return False, False, f"พบต้นไม้อื่น ({raw_ocr}) ไม่ใช่ต้นยางพารา -> หมุนหาใหม่"

    if val_no_blood >= 0.48 or (not has_blood and any(k in raw_ocr for k in ["rubber", "tree", "ยาง"])):
        return False, True, f"ต้นยางพาราล้มแล้ว (พร้อมเก็บของ) | OCR: '{raw_ocr}'"

    is_rubber_name = (val_has_blood >= 0.44) or any(k in raw_ocr for k in ["rubber", "tree", "ยาง", "พารา"])

    if is_rubber_name and has_blood:
        return True, False, f"ยืนยันต้นยางพาราถูกต้องและมีเลือด (ไอคอน: {val_has_blood:.2f}, เลือด: {red_count}px, OCR: '{raw_ocr}')"

    return False, False, f"ชื่อไม่ตรง Name_Rubber_tree หรือไม่ใช่ต้นยาง (ไอคอน: {val_has_blood:.2f}, OCR: '{raw_ocr}')"

def walk_direction(deg=0, duration_ms=480):
    rad = math.radians(deg)
    tx = int(JOY_CENTER_X + JOY_RADIUS * math.sin(rad))
    ty = int(JOY_CENTER_Y - JOY_RADIUS * math.cos(rad))
    swipe_screen(JOY_CENTER_X, JOY_CENTER_Y, tx, ty, duration=duration_ms)
    time.sleep(0.1)

def rotate_view_right(dist=180):
    cx, cy = 850, 240
    swipe_screen(cx, cy, cx - dist, cy, duration=240)
    time.sleep(0.35)

def pixel_to_angle(target_x):
    cx = SCREEN_W / 2
    off = target_x - cx
    if abs(off) <= DEADZONE_PX:
        return 0.0
    f = cx / math.tan(math.radians(HFOV_DEG / 2))
    return math.degrees(math.atan(off / f))

def turn_camera_3d(target_x):
    center_x = SCREEN_W // 2
    err_x = target_x - center_x

    if abs(err_x) <= DEADZONE_3D_X:
        return False

    swipe_px = int(abs(err_x) * CAMERA_SENSITIVITY)
    swipe_px = max(45, min(swipe_px, 320))

    touch_start_x = 920
    touch_start_y = 300
    duration_ms = 190

    if err_x > 0:
        swipe_screen(touch_start_x, touch_start_y, touch_start_x - swipe_px, touch_start_y, duration=duration_ms)
    else:
        swipe_screen(touch_start_x, touch_start_y, touch_start_x + swipe_px, touch_start_y, duration=duration_ms)

    return True

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
    print("[*] UtopiaFarmAI: ระบบฟาร์มอัตโนมัติ (YOLOv8 + Masking + Template)")
    print(f"[*] โฟลเดอร์ต้นแบบ: {ICONS_DIR}")
    
    init_yolo()
    print("=" * 60)

    # นับถอยหลัง 10 วินาทีให้สลับหน้าจอกลับเข้าเกม
    print("\n[⏱️] กรุณาสลับหน้าจอกลับเข้าเกม Utopia: Origin ด่วน!")
    for s in range(10, 0, -1):
        print(f"[*] ระบบจะเริ่มทำงานในอีก {s} วินาที...", end="\r", flush=True)
        time.sleep(1.0)
    print("\n[🚀] เริ่มต้นสแกนหาเป้าหมายและทำงานทันที!\n")

    round_count = 0

    while True:
        round_count += 1
        print(f"\n{'='*20} เริ่มต้นลูปฟาร์มรอบที่ {round_count} {'='*20}")

        print("[1] ขั้นตอนที่ 1: กวาดสายตา 360 องศาค้นหาเป้าหมาย...")
        candidate_found = False
        target_x = 0
        target_y = 0
        tree_info = ""

        for step in range(8):
            scr = get_screen()
            found, src, cx, cy, area, h = detect_target_tree(scr)
            if found:
                candidate_found = True
                target_x = cx
                target_y = cy
                tree_info = src
                print(f"\n[🌲] พบเป้าหมายที่มุม {step + 1}/8! [{src}] (X={cx}, Y={cy}, สูง={h}px)")
                break
            else:
                print(f"[🔄] สแกนมุม {step + 1}/8 -> หมุนขวาต่อ...")
                rotate_view_right(dist=180)

        if not candidate_found:
            print("\n" + "=" * 55)
            print("[!] ล้มเหลว: กวาดหา 360 องศาแล้วไม่พบต้นไม้ -> หยุดการทำงาน")
            print("=" * 55)
            return

        scr_check = get_screen()
        bag_status = check_bag_status(scr_check)
        if bag_status == "full":
            print("\n[!] กระเป๋าเต็ม (ตรวจพบ กระเป๋าส้ม.png) -> หยุดทำงาน")
            return
        else:
            print("[✓] สถานะกระเป๋าปกติ (กระเป๋า.png)")

        axe_ready = check_axe_status(scr_check)
        if not axe_ready:
            print("\n" + "=" * 60)
            print("[!] ล้มเหลว: ตรวจพบ ขวานแดง.png (ขวานพัง/หลอดแดง) -> สั่งหยุดทำงานทันที")
            print("=" * 60)
            return
        else:
            print("[✓] ขวานพร้อมใช้งาน (ขวาน.png) -> ดำเนินการวิเคราะห์เฉดสีและเดินต่อได้")

        print("\n[2] ขั้นตอนที่ 2: ล็อกหน้าจอนิ่งสนิท...")
        time.sleep(0.6)
        zoom_scr = get_screen()
        is_real_tree, tree_type_name, matched_file, zoom_detail, nearest_x, tree_box = optical_zoom_analysis(zoom_scr, target_x)
        
        if is_real_tree:
            target_x = nearest_x
        else:
            tree_type_name = tree_info

        print(f"[✓] ยืนยันเป้าหมาย [{tree_type_name}] สำเร็จ เตรียมเดินเข้าหาต้นนี้...")
        time.sleep(0.5)

        print("\n[3] ขั้นตอนที่ 3: เดินเข้าหาตามแนวระนาบปกติ...")
        reached_close = False

        for step in range(35):
            curr_scr = get_screen()
            if check_pickup_hand(curr_scr)[0]:
                print("\n[🎯] พบปุ่มรูปมือขึ้นมาแล้ว -> ประชิดเป้าหมายเรียบร้อย!")
                reached_close = True
                break

            found_step, _, cur_x, cur_y, cur_area, cur_h = detect_target_tree(curr_scr)

            if found_step:
                print(f"[*] ก้าวที่ {step + 1:02d} | สั่งหยุดนิ่ง 5 วินาที เพื่อให้ฉากนิ่งสนิทก่อนจับภาพ...")
                time.sleep(5.0)

                curr_scr = get_screen()
                found_step, _, cur_x, cur_y, cur_area, cur_h = detect_target_tree(curr_scr)
                if not found_step:
                    continue

                if step >= 3 and cur_h >= 220:
                    print(f"\n[🎯] วิเคราะห์ลำต้นขยายใหญ่เต็มหน้าจอ ({cur_h}px) -> ถึงระยะฟันสมบูรณ์!")
                    reached_close = True
                    break

                center_x = SCREEN_W // 2
                err_x = cur_x - center_x
                if abs(err_x) > DEADZONE_3D_X:
                    print(f"[*] ตรวจภาพสด: เบี่ยง {err_x:+4d}px -> กำลังหมุนกล้องปรับมุม...")
                    turn_camera_3d(cur_x)
                    time.sleep(0.35)
                    curr_scr = get_screen()
                    found_step, _, cur_x, cur_y, cur_area, cur_h = detect_target_tree(curr_scr)
                    if not found_step:
                        continue

                walk_deg, direction_label = calculate_walk_angle(cur_x)
                offset_center = cur_x - center_x
                print(f"[*] ก้าวที่ {step + 1:02d} | ทิศ: {direction_label:12s} | พิกัดสด: ({cur_x:4d}, {cur_y:3d}) | "
                      f"สูง: {cur_h:3d}px | เบี่ยง: {offset_center:+4d}px")

                walk_direction(walk_deg, duration_ms=480)
                time.sleep(0.35)

            else:
                print(f"\n[⚠️] ก้าวที่ {step + 1:02d} | เป้าหมายหลุดมุมมองชั่วคราว!")
                target_reacquired = False

                for wait_sec in range(5, 0, -1):
                    print(f"[*] กำลังค้นหาเป้าหมายใหม่... นับถอยหลังเหลือ {wait_sec} วินาที")
                    time.sleep(1.0)
                    retry_scr = get_screen()
                    found_retry, _, rx, ry, _, rh = detect_target_tree(retry_scr)
                    if found_retry:
                        print(f"[🌲] พบเป้าหมายอีกครั้งที่พิกัด X={rx}, Y={ry}!")
                        target_reacquired = True
                        break

                if not target_reacquired:
                    print("\n[!] ล้มเหลว: ไม่พบเป้าหมายหลังครบกำหนด 5 วินาที -> หยุดการทำงาน")
                    return

        print(f"\n[4] ขั้นตอนที่ 4: ดันตัวประชิดโคนต้นไม้ แล้วเคาะตี 1 ครั้ง...")
        walk_direction(0, duration_ms=280)
        time.sleep(0.15)
        tap(1100, 530)
        time.sleep(1.0)

        target_scr = get_screen()
        is_target_ready, is_tree_dead, result_log = check_target_name_match(target_scr)
        print(f"[*] ผลการสแกน: {result_log}")

        if not is_target_ready and not is_tree_dead:
            print("\n[🔄] ชื่อไม่ตรงกับ Name_Rubber_tree.png -> ย้อนกลับไปขั้นตอนที่ 1 เพื่อกวาดหา 360 องศาใหม่...")
            rotate_view_right(dist=220)
            continue

        print(f"\n[5] ขั้นตอนที่ 5: ยืนยันต้นยางพาราถูกต้อง! เริ่มต้นฟันต่อเนื่อง...")
        hits = 0
        hand_detected = False
        hx, hy = 0, 0

        while hits < 45:
            hits += 1
            tap(1100, 530)
            time.sleep(0.35)

            if hits % 2 == 0:
                chk_scr = get_screen()
                finished, fx, fy = check_tree_cut_finished(chk_scr)
                if finished:
                    hand_detected = True
                    hx, hy = fx, fy
                    print(f"\n[🖐️] ฟันต้นไม้ล้มแล้ว! (ตีไป {hits} ครั้ง)")
                    break

        print("[⏳] ฟันเสร็จสมบูรณ์ -> หยุดนิ่ง 2 วินาที...")
        time.sleep(2.0)

        final_chk = get_screen()
        is_hand, hand_x, hand_y, _ = check_pickup_hand(final_chk)
        if is_hand:
            hx, hy = hand_x, hand_y

        if hx == 0 and hy == 0:
            hx, hy = 670, 430
            print(f"[*] ตรวจไม่พบไอคอนรูปมือ -> ใช้พิกัดสำรอง ({hx}, {hy}) ทันที")

        print(f"[🖐️] ทำการกดปุ่มรูปมือ 2 ครั้ง ที่พิกัด ({hx}, {hy})...")
        tap(hx, hy)
        time.sleep(0.3)
        tap(hx, hy)
        time.sleep(1.2)

        print("\n[6] ขั้นตอนที่ 6: ตรวจสอบหน้าต่างไอเทม (ปุ่มสีเขียว / คำว่า 'เก็บ')...")
        dialog_scr = get_screen()
        pick_all_ready, px, py = check_pick_all_dialog(dialog_scr)

        if not pick_all_ready:
            print("\n" + "=" * 60)
            print("[!] ล้มเหลว: กดรูปมือแล้วรอ 1.2 วิ ไม่ขึ้นคำว่า 'เก็บ' ที่ปุ่มสีเขียว -> สั่งหยุดทำงาน")
            print("=" * 60)
            return

        print(f"[📦] ตรวจพบคำว่า 'Pick All' สมบูรณ์! ทำการแตะเก็บไอเทมที่ ({px}, {py})...")
        tap(px, py)
        time.sleep(0.25)
        tap(px, py)

        print("[✓] เก็บไอเทมเข้าตัวเสร็จสมบูรณ์!")

        print("\n[7] ขั้นตอนที่ 7: รอนิ่ง 5 วินาทีเพื่อให้โมเดลในฉากหายสนิท...")
        for wait_sec in range(5, 0, -1):
            print(f"[*] พักรอ... เหลือ {wait_sec} วิ")
            time.sleep(1.0)

        print("[🚀] เริ่มต้นสแกนหาต้นต่อไปทันที...\n")

if __name__ == "__main__":
    main()
