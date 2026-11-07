
import os, time, subprocess, cv2
import numpy as np

ROOT_SU = "/system/xbin/su"
TMP_PATH = "/data/local/tmp/screen.png"

def tap(x, y):
    subprocess.run(f"{ROOT_SU} -c 'input tap {x} {y}'", shell=True)

def swipe(x1, y1, x2, y2, duration=300):
    subprocess.run(f"{ROOT_SU} -c 'input swipe {x1} {y1} {x2} {y2} {duration}'", shell=True)

def get_screen():
    subprocess.run(f"{ROOT_SU} -c 'screencap -p {TMP_PATH}'", shell=True)
    if os.path.exists(TMP_PATH):
        return cv2.imread(TMP_PATH)
    return None

def match(screen, template, thr=0.55):
    if screen is None or template is None:
        return None
    res = cv2.matchTemplate(screen, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if max_val >= thr:
        h, w = template.shape[:2]
        return (max_loc[0] + w // 2, max_loc[1] + h // 2)
    return None

def load_img(path):
    return cv2.imread(path) if os.path.exists(path) else None

# โหลดไฟล์ภาพไอคอน
tpl_hand   = load_img("icons/pickup_hand.png")
tpl_rubber = load_img("icons/tree_rubber.png")
tpl_pick   = load_img("icons/pick_all.png")
tpl_home   = load_img("icons/hearthstone.png")

# พิกัดหน้าจอ 1280x720
attack_x, attack_y = 1100, 560
joy_x, joy_y = 210, 530
joy_forward_y = 420

def move_forward(seconds=1.5):
    """เดินไปข้างหน้า"""
    swipe(joy_x, joy_y, joy_x, joy_forward_y, int(seconds * 1000))
    time.sleep(0.3)

def find_next_tree():
    """หมุนมุมกล้องและเดินมองหาต้นถัดไป"""
    print("[->] กำลังมองหาต้นไม้ต้นถัดไป...")
    swipe(850, 360, 600, 360, 300)  # หมุนมุมกล้อง
    time.sleep(0.3)
    move_forward(2.0)               # เดินหน้าเข้าหาจุดใหม่
    time.sleep(0.5)

def is_bag_orange(screen):
    """
    ตรวจสอบสีแถบกระเป๋า/น้ำหนัก (มุมบนขวาหรือจุดแสดงน้ำหนัก)
    ตรวจจับเฉดสีส้ม/แดงในระบบ HSV
    """
    if screen is None:
        return False
    # ตรวจสอบบริเวณแถบไอคอนกระเป๋า (ปรับตำแหน่ง x, y ตามจุดกระเป๋าจริงบนจอ)
    hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
    # ช่วงสีส้ม/แดงเตือนน้ำหนัก
    lower_orange = np.array([5, 150, 150])
    upper_orange = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    # ถ้ามีพิกเซลสีส้มเกินเกณฑ์ ถือว่ากระเป๋าเตือนสีส้ม
    orange_ratio = np.sum(mask > 0)
    return orange_ratio > 300

print("[*] UtopiaFarmAI: รันระบบฟันจนล้ม + เช็กรูปมือ + ตรวจสีกระเป๋า...")
time.sleep(2)

while True:
    screen = get_screen()
    if screen is None:
        time.sleep(0.3)
        continue

    # ขั้นที่ 1: ตีกระตุ้น 1 ครั้งเพื่อเรียกดูชื่อต้นไม้
    tap(attack_x, attack_y)
    time.sleep(0.7)

    # ขั้นที่ 2: แคปหน้าจอเพื่อตรวจป้ายชื่อ
    screen_check = get_screen()
    pos_rubber = match(screen_check, tpl_rubber, 0.50)

    if pos_rubber:
        print("[+] ตรวจพบ Rubber Tree! เริ่มต้นฟันจนกว่าจะล้ม...")
        
        # ฟันต่อเนื่องจนกว่าต้นไม้จะล้ม (จะปรากฏปุ่มรูปมือ หรือปุ่ม Pick All)
        chopping = True
        hit_count = 0
        while chopping and hit_count < 25:
            tap(attack_x, attack_y)
            hit_count += 1
            time.sleep(1.1)

            scr_now = get_screen()
            pos_hand = match(scr_now, tpl_hand, 0.55)
            pos_pick = match(scr_now, tpl_pick, 0.60)

            # ตรวจพบว่าต้นไม้ล้มและมีปุ่มรูปมือ หรือ Pick All ขึ้นมา
            if pos_hand or pos_pick:
                print("[+] ต้นไม้ล้มแล้ว! พบไอคอนเก็บของ")
                if pos_hand:
                    tap(pos_hand[0], pos_hand[1])
                if pos_pick:
                    tap(pos_pick[0], pos_pick[1])
                time.sleep(1.2)
                chopping = False

        # ขั้นที่ 3: เช็กสถานะกระเป๋าหลังเก็บของ
        screen_after_loot = get_screen()
        if is_bag_orange(screen_after_loot):
            print("[!] กระเป๋าขึ้นสีส้ม (น้ำหนักใกล้เต็ม/เต็มแล้ว) -> พักระบบหรือวาร์ปกลับ")
            if tpl_home is not None:
                pos_home = match(screen_after_loot, tpl_home, 0.65)
                if pos_home:
                    tap(pos_home[0], pos_home[1])
            time.sleep(15)
            break
        else:
            print("[✓] กระเป๋ายังเป็นสีเขียว (ยังมีที่ว่าง) -> ออกเดินหาต้นไม้ต่อไป")
            find_next_tree()

    else:
        # ถ้าชื่อไม่ตรง ให้หมุนกล้องหาต้นใหม่
        print("[x] ชื่อไม่ตรง -> หมุนกล้องและหาต้นถัดไป")
        find_next_tree()
EOF
  
