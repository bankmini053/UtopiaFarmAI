import os
import time
import subprocess
import cv2

ROOT_SU = "/system/xbin/su"
TMP_PATH = "/data/local/tmp/screen.png"
DEBUG_OUT = "/data/local/tmp/screen_debug.png"

def run_cmd(cmd):
    subprocess.run(f"{ROOT_SU} -c '{cmd}'", shell=True)

def get_screen():
    run_cmd(f"screencap -p {TMP_PATH}")
    if os.path.exists(TMP_PATH):
        img = cv2.imread(TMP_PATH)
        h, w = img.shape[:2]
        # บังคับให้อยู่ในระนาบแนวนอน 1280x720 เสมอ
        if h > w:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return img
    return None

def test_swipe_camera(direction="right", dist=200):
    """
    พิกัดโซนหมุนมุมมองที่ปลอดภัยที่สุดบนจอ 1280x720:
    จุดศูนย์กลาง X: 900, Y: 320 (อยู่ฝั่งขวาบน ห่างจากปุ่มสกิลและจอยสติ๊ก)
    """
    cx, cy = 900, 320
    if direction == "right":
        # รูดจากขวามาซ้าย -> กล้องหันขวา
        cmd = f"input swipe {cx} {cy} {cx - dist} {cy} 200"
    else:
        # รูดจากซ้ายไปขวา -> กล้องหันซ้าย
        cmd = f"input swipe {cx - dist} {cy} {cx} {cy} 200"
    
    print(f"[*] กำลังส่งคำสั่งรูดจอ: {cmd}")
    run_cmd(cmd)

def capture_and_mark_zones():
    """แคปภาพพร้อมตีกรอบพิกัดสายตาและจุดรูดจอ เพื่อเอาไว้เปิดดูตรวจสอบ"""
    img = get_screen()
    if img is None:
        print("[!] แคปภาพหน้าจอไม่สำเร็จ")
        return

    # 1. วาดกรอบพื้นที่สแกนหากล่อง (Zone มองเห็น: X 100-1180, Y 80-560)
    cv2.rectangle(img, (100, 80), (1180, 560), (0, 255, 0), 2)
    cv2.putText(img, "Vision Zone (Search Box)", (110, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 2. วาดเส้นกึ่งกลางจอ (เส้นเล็งเป้า X=640)
    cv2.line(img, (640, 0), (640, 720), (0, 0, 255), 2)
    cv2.putText(img, "Center X=640", (650, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 3. วาดจุดและแนวรูดหมุนกล้อง (จุด X=900, Y=320)
    cv2.circle(img, (900, 320), 8, (255, 0, 0), -1)
    cv2.line(img, (900, 320), (700, 320), (255, 0, 0), 2)
    cv2.putText(img, "Camera Swipe Point (900, 320)", (720, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # บันทึกไฟล์ภาพ
    cv2.imwrite(DEBUG_OUT, img)
    print(f"[✓] บันทึกภาพมาร์กพิกัดเรียบร้อยที่: {DEBUG_OUT}")

def main():
    print("=" * 60)
    print("[*] ทดสอบพิกัดมองภาพเกม และพิกัดหันจอ (ไม่มีโค้ดเดิน)")
    print("=" * 60)
    time.sleep(1)

    print("\n1. แคปภาพและมาร์กจุดตำแหน่งมอง/หันจอ...")
    capture_and_mark_zones()

    print("\n2. ทดสอบรูดกล้องไปทางขวา (กวาด 200px)...")
    test_swipe_camera("right", 200)
    time.sleep(1)

    print("\n3. ทดสอบรูดกล้องกลับไปทางซ้าย (กวาด 200px)...")
    test_swipe_camera("left", 200)
    time.sleep(1)

    print("\n[✓] ทดสอบเสร็จสิ้น ไม่มีคำสั่งเดินใดๆ ทั้งสิ้น")

if __name__ == "__main__":
    main()
