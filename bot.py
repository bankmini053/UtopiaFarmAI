import os
import time
import subprocess
import cv2
import numpy as np

# พาธคำสั่ง Root ในเครื่อง Cloud Phone
ROOT_SU = "/system/xbin/su"

def run_cmd(cmd):
    """ส่งคำสั่งเชลล์ผ่าน Root"""
    subprocess.run(f"{ROOT_SU} -c '{cmd}'", shell=True)

def tap(x, y):
    """ส่งคำสั่งแตะสัมผัสหน้าจอ"""
    run_cmd(f"input tap {x} {y}")

def swipe(x1, y1, x2, y2, duration=200):
    """ส่งคำสั่งลากนิ้ว (สำหรับเดินคันโยกหรือหมุนมุมมอง)"""
    run_cmd(f"input swipe {x1} {y1} {x2} {y2} {duration}")

def get_screen():
    """จับภาพหน้าจอเกมตรงเข้าหน่วยความจำแบบ Real-time"""
    cmd = f"{ROOT_SU} -c 'screencap -p'"
    pipe = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    raw, _ = pipe.communicate()
    if not raw:
        return None
    raw = raw.replace(b'\r\n', b'\n')
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    return image

def match(screen, template, thr=0.65):
    """ค้นหาตำแหน่งรูปภาพด้วย OpenCV Template Matching คืนค่าพิกัดจุดกึ่งกลาง (x, y)"""
    if screen is None or template is None:
        return None
    res = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if max_val >= thr:
        h, w = template.shape[:2]
        return (max_loc[0] + w // 2, max_loc[1] + h // 2)
    return None

def load_img(path):
    """โหลดไฟล์เทมเพลตรูปภาพจากโฟลเดอร์ icons"""
    if os.path.exists(path):
        img = cv2.imread(path)
        if img is not None:
            return img
    print(f"[!] คำเตือน: ไม่พบไฟล์รูปภาพที่ '{path}'")
    return None

def main():
    print("=" * 45)
    print("   UtopiaFarmAI - Automated Farming Bot")
    print("   รันด้วยสิทธิ์: Root (/system/xbin/su)")
    print("=" * 45)

    # โหลดไฟล์ภาพไอคอนทั้งหมดจากโฟลเดอร์ icons/
    tpl_pick_all = load_img("icons/pick_all.png")
    tpl_hand     = load_img("icons/pickup_hand.png")
    tpl_rubber   = load_img("icons/tree_rubber.png")
    tpl_weight   = load_img("icons/weight_full.png")
    tpl_home     = load_img("icons/hearthstone.png")

    print("[*] กำลังเตรียมความพร้อม... สลับไปที่หน้าต่างเกม Utopia: Origin")
    time.sleep(3)
    print("[*] เริ่มต้นลูปตรวจจับหน้าจออัตโนมัติ...")

    while True:
        screen = get_screen()
        if screen is None:
            time.sleep(0.5)
            continue

        h, w = screen.shape[:2]
        # คำนวณพิกัดปุ่มดาบ/โจมตีหลัก (บริเวณมุมขวาล่างของจอ)
        attack_x = int(w * 0.86)
        attack_y = int(h * 0.78)

        # -----------------------------------------------------------------
        # สถานะที่ 1: ตรวจสอบกระเป๋าเต็ม (น้ำหนักสัมภาระเกิน)
        # -----------------------------------------------------------------
        if tpl_weight is not None:
            pos_weight = match(screen, tpl_weight, thr=0.75)
            if pos_weight:
                print("[!] ตรวจพบกระเป๋าเต็ม (Weight Full) -> เตรียมวาร์ปกลับ")
                if tpl_home is not None:
                    pos_home = match(screen, tpl_home, thr=0.70)
                    if pos_home:
                        tap(pos_home[0], pos_home[1])
                        print("[+] กดหินวาร์ปกลับบ้านสำเร็จ (พักระบบ 15 วินาที)")
                        time.sleep(15)
                        continue

        # -----------------------------------------------------------------
        # สถานะที่ 2: ตรวจพบหน้าต่าง Pick All (ต้นไม้ล้มแล้ว มีของดรอป)
        # -----------------------------------------------------------------
        if tpl_pick_all is not None:
            pos_pick = match(screen, tpl_pick_all, thr=0.70)
            if pos_pick:
                print("[+] พบหน้าต่างดรอปของ -> กดปุ่ม 'Pick All'")
                tap(pos_pick[0], pos_pick[1])
                time.sleep(1.0)
                continue

        # -----------------------------------------------------------------
        # สถานะที่ 3: ตรวจสอบเป้าหมายว่าเป็นต้นยาง (Rubber Tree)
        # -----------------------------------------------------------------
        if tpl_rubber is not None:
            pos_rubber = match(screen, tpl_rubber, thr=0.65)
            if pos_rubber:
                print("[+] ล็อกเป้าหมาย 'Rubber Tree' -> ฟันตัดไม้อย่างต่อเนื่อง")
                tap(attack_x, attack_y)
                # หน่วงเวลา 1.2 วินาทีให้สอดคล้องกับแอนิเมชันการฟัน
                time.sleep(1.2)
                continue

        # -----------------------------------------------------------------
        # สถานะที่ 4: ตรวจพบไอคอนรูปมือ (เริ่มตัดหรือโต้ตอบกับวัตถุ)
        # -----------------------------------------------------------------
        if tpl_hand is not None:
            pos_hand = match(screen, tpl_hand, thr=0.65)
            if pos_hand:
                print("[+] พบไอคอนรูปมือ -> กดเริ่มโต้ตอบ")
                tap(pos_hand[0], pos_hand[1])
                time.sleep(0.8)
                continue

        # -----------------------------------------------------------------
        # สถานะที่ 5: เดินตรวจหา หรือฟัน 1 ครั้งเพื่อดึงป้ายชื่อต้นไม้
        # -----------------------------------------------------------------
        # แตะปุ่มโจมตีเบาๆ 1 ครั้งเพื่อเช็กวัตถุด้านหน้า
        tap(attack_x, attack_y)
        time.sleep(0.6)

if __name__ == "__main__":
    main()
