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

    # นับถอยหลัง 5 วินาทีเพื่อให้ผู้ใช้สลับหน้าจอเข้าเกม
    print("\n[⏳] กำลังเริ่มทำงานใน 5 วินาที... (กรุณาสลับหน้าจอเข้าเกมตอนนี้)")
    for i in range(5, 0, -1):
        print(f"[*] นับถอยหลัง... {i} วินาที")
        time.sleep(1.0)

    print("\n[📸] กำลังแคปหน้าจอและวิเคราะห์สิ่งที่บอทมองเห็น...")
    screen = get_screen()
    if screen is None:
        print("[!] ไม่สามารถแคปภาพหน้าจอได้")
        return

    debug_img = screen.copy()
    target_roi = screen[40:620, 80:1200]
    gray_roi = cv2.cvtColor(target_roi, cv2.COLOR_BGR2GRAY)

    tree_files = ["tree_name1.png", "tree_coconut.png", "tree_coconut2.png"]
    found_any = False

    print("\n" + "=" * 50)
    print("  พิกัดและเป้าหมายที่บอทตรวจพบในจอขณะนี้:")
    print("=" * 50)

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

            for pt in zip(*locs[::-1]):
                real_x1 = 80 + pt[0]
                real_y1 = 40 + pt[1]
                real_x2 = real_x1 + tw
                real_y2 = real_y1 + th
                center_x = real_x1 + (tw // 2)
                center_y = real_y1 + (th // 2)

                found_any = True
                print(f"[🌲 พบเป้าหมาย] ไฟล์: {fname} | สเกล: {scale:.1f}x")
                print(f"    - พิกัดกึ่งกลางเป้าหมาย : X = {center_x}, Y = {center_y}")
                print(f"    - กรอบพื้นที่ (Bounding Box) : กว้าง {tw}px, สูง {th}px")
                print(f"    - พิกัดมุมกล่อง : ({real_x1}, {real_y1}) ถึง ({real_x2}, {real_y2})")
                print("-" * 45)

                # วาดกรอบสี่เหลี่ยมสีเขียวครอบต้นไม้ และจุดกึ่งกลางสีแดง
                cv2.rectangle(debug_img, (real_x1, real_y1), (real_x2, real_y2), (0, 255, 0), 3)
                cv2.circle(debug_img, (center_x, center_y), 6, (0, 0, 255), -1)
                cv2.putText(debug_img, f"{fname} ({scale:.1f}x)", (real_x1, max(30, real_y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # วาดจุดกึ่งกลางจอ (ตำแหน่งสายตา/หัวตัวละคร)
    cv2.circle(debug_img, (640, 360), 8, (255, 255, 0), -1)
    cv2.putText(debug_img, "Center Screen (640, 360)", (540, 395),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    if not found_any:
        print("[!] บอทมองไม่พบต้นไม้ใด ๆ จากไฟล์ใน icons เลย")
    else:
        # บันทึกภาพผลลัพธ์
        cv2.imwrite(OUTPUT_DEBUG_IMG, debug_img)
        print("\n[✓] วาดกรอบสี่เหลี่ยมสีเขียวระบุพิกัดสำเร็จ!")
        print(f"[*] ภาพผลลัพธ์ถูกเซฟไว้ที่: {OUTPUT_DEBUG_IMG}")


if __name__ == "__main__":
    main()
  
