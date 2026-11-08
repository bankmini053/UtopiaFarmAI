import os
import time
import subprocess
import cv2
import numpy as np

ROOT_SU = "/system/xbin/su"
TMP_PATH = "/data/local/tmp/debug_screen.png"
OUTPUT_DEBUG_IMG = "/data/local/tmp/bot_vision.png"
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
    print("[*] UtopiaFarmAI: เครื่องมือตรวจจับสายตาบอท (Visual Debug)")
    print("=" * 60)

    # นับถอยหลัง 5 วินาทีเพื่อให้สลับหน้าจอเข้าเกม
    print("\n[⏳] กำลังเริ่มทำงานใน 5 วินาที... (สลับหน้าจอเข้าเกมตอนนี้)")
    for i in range(5, 0, -1):
        print(f"[*] นับถอยหลัง... {i} วินาที")
        time.sleep(1.0)

    print("\n[📸] กำลังแคปหน้าจอและวิเคราะห์เป้าหมาย...")
    screen = get_screen()
    if screen is None:
        print("[!] ไม่สามารถแคปภาพหน้าจอได้")
        return

    debug_img = screen.copy()
    target_roi = screen[40:620, 80:1200]
    gray_roi = cv2.cvtColor(target_roi, cv2.COLOR_BGR2GRAY)

    tree_files = ["tree_name1.png", "tree_coconut.png", "tree_coconut2.png"]
    found_any = False

    print("\n" + "=" * 55)
    print("  รายชื่อและพิกัดต้นไม้ที่บอทตรวจพบในจอ (แบบกรองกรอบซ้ำ):")
    print("=" * 55)

    tree_count = 0

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
            locs = np.where(res >= 0.48)

            raw_boxes = []
            for pt in zip(*locs[::-1]):
                bx = int(80 + pt[0])
                by = int(40 + pt[1])
                raw_boxes.append([bx, by, tw, th])
                raw_boxes.append([bx, by, tw, th])

            if len(raw_boxes) == 0:
                continue

            # รวมกรอบพิกเซลที่ซ้อนทับกันให้เหลือเพียง 1 กรอบต่อ 1 จุด
            grouped_boxes, _ = cv2.groupRectangles(raw_boxes, groupThreshold=1, eps=0.25)

            for (rx, ry, bw, bh) in grouped_boxes:
                tree_count += 1
                found_any = True
                cx = rx + (bw // 2)
                cy = ry + (bh // 2)

                print(f"[🌲 ต้นที่ {tree_count}] ไฟล์: {fname} (ขนาดสเกล {scale:.1f}x)")
                print(f"    - พิกัดกึ่งกลางเป้าหมาย : X = {cx}, Y = {cy}")
                print(f"    - ขนาดกรอบ (Bounding Box) : กว้าง {bw}px, สูง {bh}px")
                print(f"    - ขอบเขตพื้นที่ : ({rx}, {ry}) ถึง ({rx + bw}, {ry + bh})")
                print("-" * 50)

                # วาดกรอบสี่เหลี่ยมสีเขียวและจุดแดงลงบนภาพ
                cv2.rectangle(debug_img, (rx, ry), (rx + bw, ry + bh), (0, 255, 0), 3)
                cv2.circle(debug_img, (cx, cy), 6, (0, 0, 255), -1)
                cv2.putText(debug_img, f"{fname}", (rx, max(25, ry - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # วาดจุดกึ่งกลางจอ (ตำแหน่งสายตา/หัวตัวละคร)
    cv2.circle(debug_img, (640, 360), 8, (255, 255, 0), -1)
    cv2.putText(debug_img, "Center (640, 360)", (540, 395),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

    if not found_any:
        print("[!] ไม่พบต้นไม้ใด ๆ ในหน้าจอนี้ตามไฟล์ต้นแบบใน icons")
    else:
        cv2.imwrite(OUTPUT_DEBUG_IMG, debug_img)
        print(f"\n[✓] ประมวลผลเสร็จสิ้น! พบต้นไม้ทั้งหมด {tree_count} จุด")
        print(f"[*] บันทึกภาพผลลัพธ์พร้อมตีกรอบไว้ที่: {OUTPUT_DEBUG_IMG}")


if __name__ == "__main__":
    main()
