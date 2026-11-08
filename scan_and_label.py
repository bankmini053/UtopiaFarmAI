
import os
import time
import subprocess
import cv2
import numpy as np

ROOT_SU = "/system/xbin/su"
TMP_PATH = "/data/local/tmp/screen_scan.png"
OUTPUT_DETECTED = "/data/local/tmp/detected_objects.png"
ICONS_DIR = os.path.expanduser("~/UtopiaFarmAI/icons")


def run_cmd(cmd):
    subprocess.run(f"{ROOT_SU} -c '{cmd}'", shell=True)


def get_screen():
    run_cmd(f"screencap -p {TMP_PATH}")
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
    print("[*] UtopiaFarmAI: ระบบตรวจจับและระบุวัตถุรอบตัว (Multi-Object Lock)")
    print("=" * 60)

    print("\n[⏳] เริ่มทำงานใน 5 วินาที... (สลับไปหน้าจอเกมตอนนี้)")
    for i in range(5, 0, -1):
        print(f"[*] นับถอยหลัง... {i} วินาที")
        time.sleep(1.0)

    screen = get_screen()
    if screen is None:
        print("[!] ไม่สามารถบันทึกภาพหน้าจอได้")
        return

    result_img = screen.copy()
    gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)

    # รายชื่อวัตถุและไฟล์อ้างอิงใน icons
    targets = {
        "Coconut Tree": ["tree_coconut.png", "tree_coconut2.png"],
        "Rubber Tree": ["tree_name1.png"],
        "Target Name": ["tree_rubber.png", "name_coconut.png"],
        "Hand Pickup": ["pickup_hand.png"]
    }

    detected_list = []

    for label, files in targets.items():
        for fname in files:
            fpath = os.path.join(ICONS_DIR, fname)
            if not os.path.exists(fpath):
                continue

            tpl = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if tpl is None:
                continue

            for sc in [0.7, 1.0, 1.3]:
                tw = int(tpl.shape[1] * sc)
                th = int(tpl.shape[0] * sc)
                if th >= gray_screen.shape[0] or tw >= gray_screen.shape[1] or th < 25 or tw < 25:
                    continue

                res = cv2.matchTemplate(gray_screen, cv2.resize(tpl, (tw, th)), cv2.TM_CCOEFF_NORMED)
                locs = np.where(res >= 0.50)

                for pt in zip(*locs[::-1]):
                    cx = pt[0] + (tw // 2)
                    cy = pt[1] + (th // 2)

                    # ตัดตำแหน่งซ้ำซ้อน
                    duplicate = False
                    for d in detected_list:
                        if abs(cx - d['cx']) < 45 and abs(cy - d['cy']) < 45:
                            duplicate = True
                            break

                    if not duplicate:
                        detected_list.append({
                            "name": label,
                            "file": fname,
                            "x1": pt[0],
                            "y1": pt[1],
                            "x2": pt[0] + tw,
                            "y2": pt[1] + th,
                            "cx": cx,
                            "cy": cy
                        })

    print(f"\n[✓] ประมวลผลเสร็จสิ้น ตรวจพบทั้งหมด {len(detected_list)} จุด:")
    print("-" * 50)

    for i, item in enumerate(detected_list, 1):
        print(f"[{i}] วัตถุ: {item['name']} | อ้างอิง: {item['file']} | จุดศูนย์กลาง: ({item['cx']}, {item['cy']})")
        
        # วาดกรอบสี่เหลี่ยมสีเขียว
        cv2.rectangle(result_img, (item['x1'], item['y1']), (item['x2'], item['y2']), (0, 255, 0), 2)
        cv2.circle(result_img, (item['cx'], item['cy']), 4, (0, 0, 255), -1)
        
        # เขียนป้ายชื่อวัตถุเหนือกรอบ
        tag = f"{item['name']}"
        cv2.putText(result_img, tag, (item['x1'], max(20, item['y1'] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite(OUTPUT_DETECTED, result_img)
    print("-" * 50)
    print(f"[*] บันทึกภาพที่ตีกรอบและระบุชื่อวัตถุไว้ที่: {OUTPUT_DETECTED}")


if __name__ == "__main__":
    main()
EOF
