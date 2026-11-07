import os
import time
import math
import subprocess

ROOT_SU = "/system/xbin/su"

def tap(x, y):
    subprocess.run(f"{ROOT_SU} -c 'input tap {x} {y}'", shell=True)

def swipe(x1, y1, x2, y2, duration=1500):
    subprocess.run(f"{ROOT_SU} -c 'input swipe {x1} {y1} {x2} {y2} {duration}'", shell=True)

# พิกัดกึ่งกลางจอยสติ๊กบนจอ 1280x720
JOY_CENTER_X = 195
JOY_CENTER_Y = 520
JOY_RADIUS   = 160

# พิกัดจุดเริ่มและกล่อง
START_POS = (17514, 15323)
CHEST_POS = (17529, 15339)

def walk_to_target(dx, dy, total_steps=3):
    """
    บังคับจอยสติ๊กเดินตามทิศทางเวกเตอร์โดยแบ่งเป็นก้าวสั้นๆ
    dx > 0 คือเลี้ยวขวา, dy > 0 คือเดินตรงไปข้างหน้า
    """
    angle_rad = math.atan2(dx, dy)
    target_x = int(JOY_CENTER_X + JOY_RADIUS * math.sin(angle_rad))
    target_y = int(JOY_CENTER_Y - JOY_RADIUS * math.cos(angle_rad))
    
    print(f"[*] คำนวณมุมเดิน: {math.degrees(angle_rad):.1f}° | เริ่มเดิน...")
    
    for step in range(total_steps):
        print(f"[->] กำลังเดินก้าวที่ {step + 1}/{total_steps}...")
        swipe(JOY_CENTER_X, JOY_CENTER_Y, target_x, target_y, duration=1200)
        time.sleep(0.3)

def main():
    print("=" * 50)
    print(f"[*] UtopiaFarmAI: เดินจากจุดเกิด {START_POS} ไปยังกล่อง {CHEST_POS}")
    print("=" * 50)
    
    # หน่วงเวลา 3 วินาทีเพื่อให้สลับเข้าเกม
    print("[*] เตรียมตัว... กรุณาสลับเข้าหน้าจอเกมใน 3 วินาที")
    time.sleep(3)

    dx = CHEST_POS[0] - START_POS[0]  # +15
    dy = CHEST_POS[1] - START_POS[1]  # +16

    # สั่งเดินเข้าหากล่อง (เดิน 3 จังหวะ จังหวะละ 1.2 วินาที)
    walk_to_target(dx, dy, total_steps=3)

    print("[✓] เดินถึงตำแหน่งหน้ากล่องเรียบร้อยแล้ว!")
    
    # แตะเปิดกล่อง (พิกัดกลางจอขวาสำหรับปุ่มมือ/เปิดกล่อง)
    time.sleep(0.5)
    tap(820, 430)

if __name__ == "__main__":
    main()
