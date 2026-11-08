import os
import time
import subprocess
import cv2
import numpy as np
import pytesseract

ROOT_SU = "/system/xbin/su"
TMP_PATH = "/data/local/tmp/name_debug.png"
OUTPUT_CROP_PATH = "/data/local/tmp/name_crop.png"
OUTPUT_BOX_PATH = "/data/local/tmp/name_screen_box.png"
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
    print("[*] UtopiaFarmAI: ตรวจสอบพิกัดป้ายชื่อต้นไม้ในจอ")
    print("=" * 60)

    print("\n[⏳] เริ่มทำงานใน 5 วินาที... (สลับไปหน้าจอเกมที่มีป้ายชื่อขึ้นตอนนี้)")
    for i in range(5, 0, -1):
        print(f"[*] นับถอยหลัง... {i} วินาที")
        time.sleep(1.0)

    print("\n[📸] กำลังแคปหน้าจอและตรวจเช็กพิกัดป้ายชื่อ...")
    screen = get_screen()
    if screen is None:
        print("[!] แคปภาพหน้าจอไม่สำเร็จ")
        return

    # พิกัดแถบป้ายชื่อจริงในเกม (ฝั่งขวาบนของจอ)
    # X: 880 ถึง 1240, Y: 180 ถึง 340
    rx1, ry1, rx2, ry2 = 880, 180, 1240, 340
    name_roi = screen[ry1:ry2, rx1:rx2]

    # บันทึกภาพครอปเฉพาะป้ายชื่อ
    cv2.imwrite(OUTPUT_CROP_PATH, name_roi)

    # วาดกรอบสีแดงลงบนภาพเต็มหน้าจอเพื่อดูตำแหน่งจริง
    full_debug = screen.copy()
    cv2.rectangle(full_debug, (rx1, ry1), (rx2, ry2), (0, 0, 255), 3)
    cv2.putText(full_debug, "TARGET NAME ZONE (880-1240, 180-340)", (rx1 - 80, ry1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite(OUTPUT_BOX_PATH, full_debug)

    print("\n" + "=" * 55)
    print("  ข้อมูลตำแหน่งที่บอทใช้ตรวจป้ายชื่อต้นไม้:")
    print("=" * 55)
    print(f"  - ขอบเขตพิกัด : X={rx1} ถึง {rx2} | Y={ry1} ถึง {ry2}")
    print(f"  - ขนาดพื้นที่ : กว้าง {rx2 - rx1}px, สูง {ry2 - ry1}px")

    # อ่านตัวอักษร OCR
    gray = cv2.cvtColor(name_roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(thresh, lang="eng", config=r"--psm 7").strip()
    print(f"  - ตัวอักษรที่อ่านได้ (OCR) : '{text}'")

    # ตรวจสอบกับไฟล์ใน icons
    for icon_name in ["tree_rubber.png", "name_coconut.png"]:
        icon_path = os.path.join(ICONS_DIR, icon_name)
        if os.path.exists(icon_path):
            tpl = cv2.imread(icon_path)
            if tpl is not None:
                res = cv2.matchTemplate(name_roi, tpl, cv2.TM_CCOEFF_NORMED)
                _, val, _, _ = cv2.minMaxLoc(res)
                print(f"  - เทียบกับ {icon_name} : ค่าความตรงกัน = {val:.2f}")

    print("\n[✓] บันทึกไฟล์ภาพตรวจสอบเรียบร้อย:")
    print(f"  1. ภาพเฉพาะป้ายชื่อ : {OUTPUT_CROP_PATH}")
    print(f"  2. ภาพเต็มหน้าจอพร้อมตีกรอบ : {OUTPUT_BOX_PATH}")


if __name__ == "__main__":
    main()

