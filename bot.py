import subprocess
import time
import random
import cv2
import numpy as np

# ฟังก์ชันแตะหน้าจอผ่านสิทธิ์ Shizuku (Non-Root)
def run_touch(x, y):
    # สุ่มพิกัดเล็กน้อย ±4 พิกเซล เพื่อความเป็นธรรมชาติ
    rx = x + random.randint(-4, 4)
    ry = y + random.randint(-4, 4)
    subprocess.run(f"rish -c 'input tap {rx} {ry}'", shell=True)

# ฟังก์ชันดึงภาพหน้าจอปัจจุบัน
def get_screen():
    pipe = subprocess.Popen("rish -c 'screencap -p'", shell=True, stdout=subprocess.PIPE)
    raw = pipe.stdout.read().replace(b'\r\n', b'\n')
    arr = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

# ฟังก์ชันตรวจหาไอคอนบนหน้าจอ
def find_image(template_path, threshold=0.78):
    screen = get_screen()
    template = cv2.imread(template_path)
    if screen is None or template is None:
        return None
    res = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val >= threshold:
        h, w = template.shape[:2]
        return (max_loc[0] + w // 2, max_loc[1] + h // 2)
    return None

# ลูปทำงานหลัก
def start_bot():
    print("[*] UtopiaFarmAI กำลังเริ่มทำงาน...")
    
    # พิกัดปุ่มโจมตี/ฟันขวาน (ปรับตามขนาดหน้าจอมือถือของคุณ)
    ATTACK_X, ATTACK_Y = 1680, 850
    chop_counter = 0

    while True:
        # ตรวจสอบว่ามีของตกให้เก็บหรือไม่
        loot_btn = find_image("icons/pickup_hand.png")
        if loot_btn:
            print("[+] พบไอเทมตก แตะเก็บของ")
            run_touch(loot_btn[0], loot_btn[1])
            time.sleep(random.uniform(0.5, 0.8))
            continue

        # ตรวจสอบสถานะกระเป๋าเต็ม ทุกๆ 25 ครั้งที่ฟัน
        if chop_counter >= 25:
            if find_image("icons/weight_full.png"):
                print("[!] กระเป๋าเต็ม กำลังวาร์ปกลับบ้าน...")
                run_touch(1800, 100) # แตะเปิดกระเป๋า
                time.sleep(1.0)
                
                tp_btn = find_image("icons/hearthstone.png")
                if tp_btn:
                    run_touch(tp_btn[0], tp_btn[1])
                    time.sleep(15.0) # รอนับถอยหลังวาร์ปกลับบ้าน
                break
            chop_counter = 0

        # กดฟันต้นไม้
        run_touch(ATTACK_X, ATTACK_Y)
        chop_counter += 1
        time.sleep(random.uniform(1.15, 1.35))

if __name__ == "__main__":
    start_bot()
