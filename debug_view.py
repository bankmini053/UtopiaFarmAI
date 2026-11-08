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
    print("[*] UtopiaFarmAI: ตรวจจับสายตาบอท (จำกัด 5 ตำแหน่ง)")
    print("=" * 60)

    print("\n[⏳] เริ่มทำงานใน 5 วินาที... (สลับหน้าจอเข้าเกมตอนนี้)")
    for i in range(5, 0, -1):
        print(f"[*] นับถอยหลัง... {i} วินาที")
        time.sleep(1.0)

    print("\n[📸] แคปภาพหน้าจอและกำลังวิเคราะห์...")
    screen = get_screen()
    if screen is None:
        print("[!] แคปภาพหน้าจอไม่สำเร็จ")
        return

    debug_img = screen.copy()
    target_roi = screen[40:620, 80:1200]
    gray_roi = cv2.cvtColor(target_roi, cv2.COLOR_BGR2GRAY)

    tree_files = ["tree_name1.png", "tree_coconut.png", "tree_coconut2.png"]
    found_targets = []

    for fname in tree_files:
        icon_path = os.path.join(ICONS_DIR, fname)
        if not os.path.exists(icon_path):
            continue

        template = cv2.imread(icon_path)
        if template is None:
            continue

        gray_tpl = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        for scale in [0.7, 1.0, 1.3]:
            tw = int(gray_tpl.shape[1] * scale)
            th = int(gray_tpl.shape[0] * scale)

            if th >= target_roi.shape[0] or tw >= target_roi.shape[1] or th < 30 or tw < 30:
                continue

            res = cv2.matchTemplate(gray_roi, cv2.resize(gray_tpl, (tw, th)), cv2.TM_CCOEFF_NORMED)
            locs = np.where(res >= 0.50)

            for pt in zip(*locs[::-1]):
                cx = 80 + pt[0] + (tw // 2)
                cy = 40 + pt[1] + (th // 2)

                # ดักจับไม่ให้เก็บตำแหน่งที่อยู่ใกล้กันเกิน 60px (ตัดจุดซ้ำ)
                is_duplicate = False
                for t in found_targets:
                    if abs(cx - t['x']) < 60 and abs(cy - t['y']) < 60:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    found_targets.append({
                        'file': fname,
                        'scale': scale,
                        'x': cx,
                        'y': cy,
                        'w': tw,
                        'h': th,
                        'rx': 80 + pt[0],
                        'ry': 40 + pt[1]
                    })

                # เมื่อเก็บครบ 5 ตำแหน่ง ให้ตัดการค้นหาทันที
                if len(found_targets) >= 5:
                    break
            if len(found_targets) >= 5:
                break
        if len(found_targets) >= 5:
            break

    print("\n" + "=" * 55)
    print(f"  รายงาน 5 ตำแหน่งต้นไม้ที่ตรวจพบ (ตัดจุดซ้ำซ้อนแล้ว):")
    print("=" * 55)

    if len(found_targets) == 0:
        print("[!] ไม่พบต้นไม้ในระยะสายตานี้")
    else:
        for idx, t in enumerate(found_targets, 1):
            print(f"\n[🌲 ตำแหน่งที่ {idx}/5] ไฟล์: {t['file']} (สเกล {t['scale']:.1f}x)")
            print(f"    - พิกัดกึ่งกลาง : X = {t['x']}, Y = {t['y']}")
            print(f"    - ขนาดกรอบ     : กว้าง {t['w']}px, สูง {t['h']}px")
            print(f"    - ขอบเขตภาพ    : ({t['rx']}, {t['ry']}) ถึง ({t['rx'] + t['w']}, {t['ry'] + t['h']})")

            # วาดกรอบลงรูปภาพ
            cv2.rectangle(debug_img, (t['rx'], t['ry']), (t['rx'] + t['w'], t['ry'] + t['h']), (0, 255, 0), 3)
            cv2.circle(debug_img, (t['x'], t['y']), 6, (0, 0, 255), -1)
            cv2.putText(debug_img, f"#{idx} {t['file']}", (t['rx'], max(25, t['ry'] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        # จุดกึ่งกลางจอ
        cv2.circle(debug_img, (640, 360), 8, (255, 255, 0), -1)
        cv2.putText(debug_img, "Center Screen (640, 360)", (520, 395),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        cv2.imwrite(OUTPUT_DEBUG_IMG, debug_img)
        print("\n" + "=" * 55)
        print(f"[✓] แสดงผลครบ 5 ตำแหน่งและหยุดการทำงานเรียบร้อย")
        print(f"[*] บันทึกภาพที่: {OUTPUT_DEBUG_IMG}")
        print("=" * 55)


if __name__ == "__main__":
    main()
