
import os
import time
import subprocess
import cv2
import numpy as np

ROOT_SU = "/system/xbin/su"
TMP_PATH = "/data/local/tmp/radar_scr.png"
OUTPUT_RADAR = os.path.expanduser("~/UtopiaFarmAI/monster_esp.png")
ICONS_DIR = os.path.expanduser("~/UtopiaFarmAI/icons")

def run_cmd(cmd):
    subprocess.run(f"{ROOT_SU} -c '{cmd}'", shell=True)

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

def main():
    print("=" * 60)
    print("[*] UtopiaFarmAI: ตรวจจับเรดาร์ มอนสเตอร์/สัตว์/ทรัพยากร")
    print("=" * 60)

    print("\n[⏳] กำลังเริ่มใน 5 วินาที... (กรุณาสลับหน้าจอเข้าเกม)")
    for i in range(5, 0, -1):
        print(f"[*] นับถอยหลัง... {i} วินาที")
        time.sleep(1.0)

    screen = get_screen()
    if screen is None:
        print("[!] ไม่สามารถบันทึกภาพหน้าจอได้")
        return

    h, w = screen.shape[:2]
    result_img = screen.copy()
    gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

    detect_database = {
        "MONSTER/ANIMAL": ["mob_turtle.png", "mob_bear.png", "mob_wolf.png"],
        "RESOURCE_TREE": ["tree_coconut.png", "tree_coconut2.png", "tree_name1.png"],
        "STONE_MINE": ["mine_stone.png", "mine_iron.png"]
    }

    detected_targets = []

    for category, file_list in detect_database.items():
        for fname in file_list:
            fpath = os.path.join(ICONS_DIR, fname)
            if not os.path.exists(fpath):
                continue

            tpl = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if tpl is None:
                continue

            for scale in [0.6, 0.9, 1.2, 1.5]:
                tw = int(tpl.shape[1] * scale)
                th = int(tpl.shape[0] * scale)
                if th >= h or tw >= w or th < 25 or tw < 25:
                    continue

                res = cv2.matchTemplate(gray_screen, cv2.resize(tpl, (tw, th)), cv2.TM_CCOEFF_NORMED)
                locs = np.where(res >= 0.50)

                for pt in zip(*locs[::-1]):
                    cx = pt[0] + (tw // 2)
                    cy = pt[1] + (th // 2)

                    duplicate = False
                    for item in detected_targets:
                        if abs(cx - item['cx']) < 40 and abs(cy - item['cy']) < 40:
                            duplicate = True
                            break

                    if not duplicate:
                        detected_targets.append({
                            "category": category,
                            "filename": fname,
                            "x1": pt[0],
                            "y1": pt[1],
                            "x2": pt[0] + tw,
                            "y2": pt[1] + th,
                            "cx": cx,
                            "cy": cy
                        })

    print(f"\n[🎯] ตรวจพบเป้าหมายรอบตัวทั้งหมด: {len(detected_targets)} ตัว")
    print("-" * 55)

    for idx, target in enumerate(detected_targets, 1):
        color = (0, 0, 255) if target["category"] == "MONSTER/ANIMAL" else (0, 255, 0)
        
        cv2.rectangle(result_img, (target["x1"], target["y1"]), (target["x2"], target["y2"]), color, 2)
        cv2.line(result_img, (640, 360), (target["cx"], target["cy"]), color, 1)
        
        tag = f"{target['filename'].split('.')[0]}"
        cv2.putText(result_img, tag, (target["x1"], max(25, target["y1"] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        print(f"[{idx}] {target['category']} -> {tag} | พิกัดกลาง: ({target['cx']}, {target['cy']})")

    cv2.circle(result_img, (640, 360), 6, (255, 255, 0), -1)
    
    cv2.imwrite(OUTPUT_RADAR, result_img)
    print("-" * 55)
    print(f"[✓] เซฟภาพผลลัพธ์สำเร็จ ไม่ติด Permission Denied!")
    print(f"[*] ภาพอยู่ที่: {OUTPUT_RADAR}")

if __name__ == "__main__":
    main()
EOF
