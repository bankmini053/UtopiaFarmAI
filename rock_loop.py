#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rock_loop.py - บอทวนทดสอบ: จุดมองหา -> หาหิน -> ตี -> จุดมองหา -> กล่อง(บ้าน) -> จุดมองหา -> วนซ้ำ

ขั้นตอน (ตามที่กำหนด):
  1) เดินไปใกล้ "พิกัดจุดมองหา" แล้วมองหาหิน (เลือกโทนสีหินตามช่วงเวลาในเกม จากรูปในโฟลเดอร์ icons)
  2) เจอหิน -> เดินเข้าไปชิดตัวละคร -> ตี
  3) ตีแล้วชื่อบนแถบตรงกับแร่ -> พิมพ์ "ตีสำเร็จ" -> กลับไปจุดมองหา
  4) เดินไปใกล้ "พิกัดกล่อง" -> ถ้าเห็นรูปมือขึ้น -> พิมพ์ "กลับบ้านสำเร็จ"
  5) กลับไปจุดมองหา -> วนอยู่อย่างนั้น

เป้าหมายของไฟล์นี้คือทดสอบ: อ่านเวลา/เลือกรูปหินตามช่วงเวลาถูกไหม (เช้า/เย็น)
ทุกครั้งที่หาหินจะพิมพ์ผลต่อโปรไฟล์ + เซฟ debug/time_log.csv ไว้ดูย้อนหลัง

ใส่พิกัดที่หัวไฟล์: SEARCH_COORD และ BOX_COORD (ถ้ายังเป็น None: ใช้ตำแหน่งปัจจุบันเป็นจุดมองหา / ข้ามขั้นไปกล่อง)

คำสั่ง:
  python rock_loop.py run                       # เริ่มวน
  python rock_loop.py run --max-rounds 3        # วนแค่ 3 รอบแล้วหยุด
  python rock_loop.py calibrate                 # จับภาพ 1 เฟรม -> debug/calibrate.png + พิมพ์สิ่งที่บอทอ่านได้
  python rock_loop.py calibrate --image ภาพ.png --landscape
  python rock_loop.py test-input joystick|hold|camera [--backend input|sendevent]
  python rock_loop.py camera-cal                # วัดว่ากล้องหมุนไหม/ทิศไหน แล้วเซฟค่าไว้ใช้
  python rock_loop.py doctor                    # ตรวจ root/tesseract/จับภาพ/อ่านเวลา-พิกัด-ชื่อ
  python rock_loop.py stop                      # สั่งหยุดบอทที่รันอยู่ (หรือกด q + Enter / Ctrl+C ที่หน้าต่างบอท)
"""
import argparse
import difflib
import glob
import json
import math
import os
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import Counter

import cv2
import numpy as np

try:
    import pytesseract
    HAS_TESS = shutil.which("tesseract") is not None      # ต้องมีโปรแกรม tesseract จริง ไม่ใช่แค่ pytesseract
except Exception:          # ยังไม่ได้ลง pytesseract/tesseract: บอททำงานได้ แต่อ่านชื่อ/นาฬิกาไม่ได้
    pytesseract = None
    HAS_TESS = False

# =============================================================================
# CONFIG  (แก้ที่นี่)
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
DEBUG_DIR = os.path.join(BASE_DIR, "debug")

# พิกัด UI ทั้งหมดเขียนเป็น "พิกเซลอ้างอิง" ของภาพแนวนอน 1210x720 (เกมบนเครื่อง cloud 720x1280 หลังหมุนกลับ)
REF_W, REF_H = 1210, 720

# screencap ของเครื่อง cloud เป็นจอตั้ง แต่เกมถูกหมุนตามเข็ม 90° อยู่ในจอ
SCREEN_ROTATED = "auto"      # "auto" | True | False
VIEW_HEIGHT = 1210           # ความสูงส่วนเกมในภาพจอตั้ง (ตัดแถบปุ่ม back/home) None = ประมาณเอง

ANCHORS = {
    "joy_center":    (227, 540),    # กลางจอยสติ๊ก
    "joy_radius":    65,
    "sword":         (1041, 539),   # ปุ่มดาบ (กดค้างเพื่อตี)
    "hand_guess":    (867, 440),    # ปุ่มมือเก็บของ (วงกลมเหนือปุ่มวิ่ง ขึ้นเมื่อมีของให้เก็บ)
    "char_feet":     (605, 548),    # เท้าตัวละคร
    "char_head":     (605, 330),    # หัวตัวละคร (ตัวละครนอนยาวตามแนวตั้งในภาพแนวนอน เท้าล่าง หัวบน)
    "camera_start":  (929, 360),    # จุดเริ่มลากกล้อง (เท่ากับที่ลากจริง)
    "pickaxe_slot":  (370, 676),    # ช่องของที่ 1 = ที่ขุด (กดตอนขวานแดง = เปิดหน้าซ่อม)
    "repair_btn":    (605, 513),    # ปุ่ม Fix (ซ่อมเสร็จจะเปลี่ยนเป็น Equip Now ตำแหน่งเดิม)
    "repair_close":  (808, 172),    # ปุ่ม X ปิดหน้าซ่อม
}
ROI = {                              # (x1, y1, x2, y2)
    "clock":      (0, 4, 64, 32),
    "durability": (335, 698, 405, 712),   # แถบความทนทานใต้ไอคอนที่ขุด (แดง = ขวานพัง)
    "bag_ring":   (968, 636, 1046, 712),  # วงแหวนรอบไอคอนกระเป๋า
    "target_bar": (240, 0, 970, 110),     # แถบชื่อ+เลือดเป้าหมายกลางบน
    "dialog":     (330, 290, 880, 560),   # หน้าต่าง Pick All
    "repair":     (385, 150, 832, 552),    # หน้าต่างซ่อมที่ขุด
}
UI_EXCLUDE = [                        # โซน UI ที่ไม่มองหาหิน
    (0, 0, 195, 205), (30, 205, 260, 280), (195, 0, 480, 50), (200, 55, 425, 230),
    (0, 0, 1210, 60), (880, 0, 1210, 170), (0, 420, 380, 720), (900, 380, 1210, 720),
    (230, 575, 1050, 720), (930, 180, 1010, 280),
]
CHAR_EXCLUDE = (572, 300, 642, 572)   # ตัวละครเรา (ผม/เสื้อผ้าไม่ให้นับเป็นหิน)
ROCK_MIN_Y = 150
ROCK_V_FLOOR = 40

# ช่วงเวลาในเกม (นาทีของวัน): เช้ามืด 05:20-06:20 / กลางวัน 06:20-19:50 / กลางคืน 19:50-05:20
PERIODS = {"dawn": (5 * 60 + 20, 6 * 60 + 20), "day": (6 * 60 + 20, 19 * 60 + 50), "night": (19 * 60 + 50, 5 * 60 + 20)}
TWILIGHT_BLEND_MIN = 40       # ใกล้รอยต่อ ±นาที ใช้โทนม่วง 'dawn' ร่วมด้วย (แสงเปลี่ยนไม่ตรงเวลาเป๊ะ)
CLOCK_READ_INTERVAL = 3.0     # อ่านนาฬิกาทุกกี่วินาที
CLOCK_FALLBACK_VISUAL = True  # อ่านไม่ได้เลย -> เดาจากแสงของฉาก

# ชื่อบนแถบเป้าหมาย: ต้องมีคำเหล่านี้ (ตัวพิมพ์เล็ก) จึงถือว่าเป็นแร่ / ชื่อที่ห้ามตี
ORE_NAME_KEYWORDS = {"dawn": ["mine", "iron", "ore"], "day": ["mine", "iron", "ore"], "night": ["mine", "iron", "ore"]}
NAME_STRICT = True            # อ่านชื่อได้แต่ไม่ใช่แร่ (เช่น ต้นไม้/มอน) = ไม่ใช่หิน ให้ออกจากเป้า (อ่านไม่ออกเลย = ถือว่าตีได้)
NAME_BLOCKLIST = ["cow", "chicken", "pig", "sheep", "goblin"]

# การสั่งเครื่อง
SHELL_PREFIX = ["su", "-c"]            # ไม่ได้รูท: ["sh", "-c"]
TOUCH_BACKEND = {
    "hold": "input",                   # กดค้างดาบ: "input" | "sendevent"
    "joystick": "input",
    "camera": "input",                 # "input" | "sendevent" | "none" = ไม่หมุนกล้อง ใช้จอยเดินหาแทน
}
TOUCH_EVENT_DEVICE = "/dev/input/event0"   # อุปกรณ์สัมผัสของเครื่องนี้ (จาก getevent)

# จังหวะ/เกณฑ์
HOLD_MS = 1200                # กดค้างดาบครั้งละ
HIT_TIMEOUT = 150             # ตีหินก้อนเดียวไม่เกินกี่วินาที
HP_NO_DROP_RETRY = 2          # เลือดไม่ลดกี่รอบติดกัน -> ขยับเข้าไปอีกนิดแล้วกดใหม่
TIRED_REST_SEC = 12           # รอสแตมิน่าฟื้น
STEP_MS_MIN, STEP_MS_MAX = 350, 1500
NEAR_TRUST_DIST = 170          # ใช้ 'สีหินรอบตัวเยอะ = ชิดแล้ว' ได้ก็ต่อเมื่อหินที่เห็นอยู่ไม่เกินระยะนี้ (กันนับพื้น/ตัวละครผิด)
WALK_MS_PER_PX = 18            # เดินเข้าหาหินตอนตีไม่โดน: ms ต่อ px ที่ยังขาด (ตัวละครเดินช้า ~55px/วิ)
NEAR_DIST = 105               # ระยะ (px อ้างอิง) เท้า->ฐานหิน ที่ลองตีได้
APPROACH_MAX_STEPS = 40
ENGAGE_TRIES = 8              # ตีไม่โดนเป้ากี่ครั้ง (แต่ละครั้งขยับเข้าใกล้หินมากขึ้น) ก่อนยอมออก
SEARCH_MAX_ROUNDS = 25
AFTER_PICK_WAIT = 0.8         # รอหลังกด Pick All ก่อนทำอย่างอื่น
COLLECT_DELAY = 0.0           # รอกี่วินาทีหลังเลือดหมดก่อนกดปุ่มมือครั้งแรก (ปรับด้วย run --collect-delay)
PICK_WAIT = 0.4               # กดปุ่มมือแล้วรอกี่วินาทีก่อนเช็กว่าหน้าต่าง Pick All ขึ้นหรือยัง (run --pick-wait)
PICK_TRIES = 8                # กดปุ่มมือ/หา Pick All ซ้ำกี่รอบ
CAM_SCAN_SLICES = 6           # มองหาหิน: หมุนรอบตัวแบ่งกี่ช่วง (6 = ช่วงละ 60°) ซ้าย -> เช็กภาพ -> ซ้าย -> เช็กภาพ
CAM_SCAN_SHIFT = 850          # ฉากต้องเลื่อนกี่ px ต่อ 1 ช่วง (ประมาณ 60°) ปรับด้วย run --scan-shift
SWEEP_STEPS = 5               # เดินติด: หมุนกล้องไปขวาทีละนิดกี่ก้าวเพื่อดูว่าชิดหินแล้วหรือยัง (แล้วหมุนกลับที่เดิม)
SWEEP_STEP_PX = 90            # ก้าวละกี่ px ของฉาก
HAND_ZONE = (330, 230, 880, 560)   # รูปมือลอยขึ้นใกล้ตัวละคร (เหนือของที่ดรอป)
HAND_TPL_B64 = "iVBORw0KGgoAAAANSUhEUgAAAB4AAAAhCAAAAADrd0QGAAAAoElEQVQoFXXBiYGbAAAEMU3/RW+Mf3NEymGEya8sjDCxvDXCCBPLS0OY3CwsTxnC5GYxeYghjLCY3IUhjDCHHHIYYcQccsjdCPORmzyMmI/c5Gk030Je5iTkbU4ib3MS+ZhfkY/5FfmYk+TL/Eq+zK+Eyd38Sgw5zElhCHNWDvMfuZtreZhLeZoreZkLeZu/8jFn5cI8lSvzUC7NIbk2wj9KWTsc628B7QAAAABJRU5ErkJggg=="
HAND_TPL2_B64 = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAdCAAAAADu9TFbAAAAkklEQVQoFXXBgWGDMAADMPn/oz0CJM0KlWJo/JISKt5FCRWvQgkVp4pNUEKFOsQmDiVU1BAfMVSoJZY4ldQSS1xqF0vcahdTTLWJKabaxBRTbWKKpT5iiqU2cQslDrWJW1BC/ROXONSbGGKoV0Gc6ofEpX6IW70IMdVTiC+1CfGtliAe6hSneKghLvFUxC1eNJY/cZwpHVTDrZQAAAAASUVORK5CYII="   # มือตอนกลางคืน (จางกว่า)
HAND_LOOT_POS = (531, 365)     # รูปมือลอยตำแหน่งนี้เสมอ (เทียบตัวละคร) ใช้กดสำรองเมื่อตรวจไม่เจอ
ATTACK_TRY_DIST = 70          # ฐานหินห่างจาก 'ตัวละคร (เส้นเท้า->หัว)' ไม่เกินนี้ (px อ้างอิง) = ชิดตัว/ชิดหน้า หยุดเดินแล้วกดตีเลย ไม่หมุนกล้อง
SLOT1_AT_HP = 0.30            # เลือดเป้าเหลือไม่เกินนี้: กดช่อง 1 (ขวาน) 1 ครั้งแบบช้าๆ แล้วตีปกติต่อ (0 = ปิด)
SLOT1_PRESS_MS = 450          # กดช่อง 1 ค้างกี่ ms (ช้าๆ แต่ไม่ช้ามาก)
PEEK_EVERY = 4                # เดินตรงทุกกี่ก้าวหันกล้องส่องหินข้างๆ ว่าใกล้กว่าเป้าไหม
PEEK_MAX = 4                  # ส่องไม่เกินกี่รอบต่อหินหนึ่งก้อน
PEEK_NEARER = 0.85            # หินข้างๆ ต้องใกล้กว่าเป้าอย่างน้อยเท่านี้ (ระยะ/ระยะเป้า) ถึงเปลี่ยนเป้า
ALIGN_STEP_CAP = 260          # หมุนกล้องทีละไม่เกินกี่ px ของฉาก (หมุนทีละนิดไปเรื่อยๆ)
STALL_NEED = 2                # เดินแล้วภาพไม่ขยับติดกันกี่ครั้งถึงนับว่าติด (ของนิ่งไม่ขวางทางตรง)
JUMP_UP = 0.30                # เลือดเป้าพุ่งขึ้นเกินนี้ = ก้อนเดิมแตกแล้ว ระบบล็อกก้อนใหม่เอง
SINGLE_TAP_HP = 0.30          # เลือดเหลือน้อยกว่านี้ตีทีละครั้ง กันตีเลยไปโดนก้อนข้างๆ
REPAIR_WAIT = 0.9             # รอหน้าต่างซ่อมขึ้นหลังกดแต่ละครั้ง
LOCK_RADIUS = 200             # ล็อกเป้า: หินก้อนเดิมต้องอยู่ห่างจากตำแหน่งที่ติดตามไว้ไม่เกินนี้ (px อ้างอิง) ไม่งั้นนับว่าเสียเป้า ไม่สลับไปก้อนอื่น
ALIGN_IN = 0.6                # หมุนกล้องจนหินห่างกลางจอไม่เกิน (CAM_CENTER_BAND x ค่านี้)
ALIGN_OUT = 1.3               # หินเยื้องเกิน (CAM_CENTER_BAND x ค่านี้) ถึงจะหมุนกล้องใหม่ (กันหมุนส่ายซ้ายขวา)
SIDE_TRY_DIST = 90             # ระหว่างเดิน ถ้าเห็นหินก้อนอื่นชิดข้างตัวใกล้ขนาดนี้ ลองกดตีก้อนนั้นเลย
LANE_HALF = 90                # 'เส้นแดง' ทางตรงหน้าตัวละคร: หินต้องอยู่ในช่วงกลางจอ ±นี้ (px อ้างอิง) ถึงจะเดินตรง/ตีได้
LANE_IN = 60                  # หมุนกล้องจนหินเข้ามาอยู่ห่างกลางจอไม่เกินนี้ (ในเส้นสบายๆ)
ALIGN_MAX_TURNS = 8           # หมุนกล้องปรับเข้าเส้นไม่เกินกี่ครั้ง
ALIGN_GAIN = 1.0             # หมุนทีละ 85% ของที่เยื้อง กันหมุนเลยเป้า
VIZ = True                    # เซฟภาพ debug/live.jpg ขีดเส้นแดง/เขียวแสดงเป้าที่ล็อก (ปิดด้วย run --no-viz)
NAME_TILT = True              # อ่านชื่อไม่ได้ -> ก้มกล้องลง 1 ระยะแล้วอ่านใหม่ แล้วเงยกลับที่เดิม
NAME_TILT_PX = 140            # ระยะลากก้มกล้อง (px) ต่อ 1 ระยะ
BAR_LOST_RETRY = 4            # แถบเป้าหายตอนเลือดยังเหลือเยอะ: ตีต่ออีกกี่ครั้งก่อนสรุปว่าหลุดเป้า
LOST_NEAR_PIX = 3500          # หินหายตอนเข้าใกล้: ต้องมีสีหินรอบตัวอย่างน้อยเท่านี้ถึงนับว่า 'หินบังกล้อง' (ไม่งั้น = เห็นผิด)
BURST_TAPS = 3                # ตีรัว: แตะดาบกี่ครั้งต่อรอบก่อนอ่านเลือด (ขวานไม้ -6 ต่อครั้ง)
BURST_GAP = 0.30              # เว้นระหว่างแตะ (วินาที)
BURST_PRESS_MS = 140          # แตะแต่ละครั้งกดค้างกี่ ms
STALE_CYCLES = 3              # เลือดไม่ลดติดกันกี่รอบ(รอบละ BURST_TAPS ครั้ง) ถึงนับว่าตีไม่เข้า
GUARD = True                  # ยามเฝ้า: ไม่เห็นปุ่มตี / อ่านพิกัดไม่ได้ ติดกันหลายครั้ง = หยุดทันที (ปิดด้วย run --no-guard)
GUARD_SWORD_FAILS = 4         # ไม่เห็นปุ่มตีติดกันกี่เฟรมถึงหยุด
GUARD_COORD_FAILS = 4         # อ่านพิกัดไม่ได้ติดกันกี่ครั้งถึงหยุด
GUARD_COORD_EVERY = 8.0       # ตรวจพิกัดทุกกี่วินาที
KILL_AFTER_STOP = 4.0         # สั่งหยุดแล้วโปรแกรมยังไม่ปิดภายในกี่วินาที = บังคับปิดเอง
STUCK_FRAME_DIFF = 2.0
NEAR_BOX = (430, 330, 790, 640)     # รอบตัวละคร: ถ้ามีสีหินเยอะ = หินอยู่ติดตัวแล้ว (หินใหญ่จนบังตัว/ถูกตัดทิ้ง)
NEAR_PIX = 9000                     # พื้นที่สีหิน (px อ้างอิง) ในกรอบด้านบนที่ถือว่า 'ถึงหินแล้ว'
LOST_NEAR_DIST = 330                # หินหายตอนห่างไม่เกินนี้ = เดินชนติดแล้ว (ไม่ใช่หลุดจอ) -> ตีเลย
PROGRESS_MIN = 10                   # เดินแล้วระยะต้องลดอย่างน้อยกี่ px ไม่งั้นนับว่า 'ไม่คืบหน้า'
UNSTICK_MAX = 4                     # ติดสิ่งกีดขวางกี่ครั้งติดกันแล้วยอมข้ามหินก้อนนี้
PERIOD_TH = {"dawn": "เช้ามืด", "day": "กลางวัน", "night": "กลางคืน"}
BLACKLIST_SEC = 90
BAG_FULL_RATIO = 0.25         # สัดส่วนสีเหลือง/ส้ม/แดงในวงแหวนกระเป๋าที่ถือว่า 'เต็ม'
REPAIR_MAX_FAILS = 3
COUNTDOWN_SEC = 10

# กล้อง: ลากนิ้วที่ camera_start ไปซ้าย/ขวา (จอแนวนอน) ระบบวัดเองว่าลาก 1px แล้วฉากขยับกี่px (มีเครื่องหมาย)
CAM_DRAG_PX = 260             # ระยะลากต่อครั้งตอนกวาด
CAM_DRAG_MS = 380
CAM_SWEEP_STEPS = 14          # กวาดซ้ายกี่ครั้งต่อ 1 รอบการมองหา
CAM_CENTER_BAND = 85          # หินอยู่ห่างกลางจอไม่เกินนี้ (px อ้างอิง) = 'อยู่กลาง' เดินตรงได้
CAM_ALIGN_MAX = 4             # หมุนปรับเข้าหาหินไม่เกินกี่ครั้งต่อก้าว
CAM_MAX_SHIFT = 260           # ฉากขยับต่อครั้งไม่เกินนี้ (px) กันวัดสลับรอบ (ROI กว้าง ~750)
CAM_MIN_SHIFT = 6.0           # ฉากต้องขยับอย่างน้อยกี่px ถึงนับว่ากล้องหมุน
CAM_FILE = os.path.join(DEBUG_DIR, "camera_cal.json")
CAM_SCENE_ROI = (250, 70, 1000, 290)     # ส่วนฉากไกล (ท้องฟ้า/ต้นไม้) ใช้วัดการหมุน ไม่โดน UI/ตัวละคร

# เลย์เอาต์ HUD: "A" = นาฬิกา/พิกัดมุมซ้ายบน (เครื่อง cloud ที่ทดสอบ)  "B" = นาฬิกาขวาบนข้างมินิแมพ + พิกัดใต้มินิแมพ (เครื่องจริง)  "auto" = ลองทั้งคู่
LAYOUT = "auto"
LAYOUT_ROI = {
    "A": {"clock": (0, 4, 64, 32), "coords": (40, 246, 290, 276)},
    "B": {"clock": (888, 3, 958, 25), "coords": (1072, 130, 1205, 160)},
}

# โปรไฟล์สีหินสำรอง (ถ้าไม่มีโฟลเดอร์ templates/rocks จะใช้ชุดนี้)
# ถ้ามีรูปใน templates/rocks/<dawn|day|night>/ บอทจะสร้างโปรไฟล์ใหม่จากรูปเองทุกครั้งที่เริ่ม
DEFAULT_PROFILES = {
    "dawn": {"s": [28, 158], "v": [40, 141], "hue": [[0, 9], [151, 179]]},
    "day": {"s": [0, 83], "v": [40, 172], "any_hue": True},
    "night": {"s": [4, 173], "v": [40, 143], "hue": [[99, 119]]},
}

DAY_MIN = 24 * 60
DEBUG = False


_real_sleep = time.sleep
STOP_FLAG = threading.Event()
STOP_FILE = os.path.join(BASE_DIR, "STOP")
STOP_WORDS = {"q", "quit", "stop", "s", "x", "หยุด"}


class Stop(BaseException):      # ไม่ใช่ Exception: กัน except Exception ที่ไหนก็ตามกลืนคำสั่งหยุด
    pass


QUIET = False                          # True ตอน run: log() ลงไฟล์อย่างเดียว หน้าจอแสดงเฉพาะ ui() (สั้น อ่านง่าย)
BOT_LOG = os.path.join(DEBUG_DIR, "bot.log")


def _dw(t):
    import unicodedata
    return sum(0 if unicodedata.category(c) in ("Mn", "Me", "Cf") else (2 if unicodedata.east_asian_width(c) in ("W", "F") else 1)
               for c in t)


def _fit(t, cols):
    out, w = "", 0
    for c in t:
        cw = _dw(c)
        if w + cw > cols - 1:
            return out.rstrip() + "…"
        out, w = out + c, w + cw
    return out


def _to_file(prefix, text):
    try:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        with open(BOT_LOG, "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {prefix}{text}\n")
    except Exception:
        pass


def log(msg):
    """รายละเอียดเต็ม: ลงไฟล์ debug/bot.log เสมอ  หน้าจอแสดงเมื่อไม่ได้อยู่โหมด run (คำสั่ง doctor/test-input ฯลฯ)"""
    text = str(msg).lstrip(" ")
    _to_file("", text)
    if not QUIET:
        sys.stdout.write(f"\r{time.strftime('%H:%M:%S')}  {text}\n")
        sys.stdout.flush()


def ui(msg):
    """บรรทัดสั้นๆ สำหรับหน้าจอ (ชิดซ้าย ตัดให้พอดีความกว้างจอ ไม่มีเวลา) + เก็บลงไฟล์ด้วย"""
    text = str(msg).strip()
    _to_file("▸ ", text)
    cols = shutil.get_terminal_size((40, 20)).columns
    sys.stdout.write("\r" + _fit(text, cols) + "\n")
    sys.stdout.flush()


def stop_requested():
    if STOP_FLAG.is_set():
        return True
    if os.path.exists(STOP_FILE):
        STOP_FLAG.set()
        _arm_watchdog()
        try:
            os.remove(STOP_FILE)                 # ใช้แล้วลบทิ้ง ไม่ให้ค้างไปหยุดรอบหน้า
        except OSError:
            pass
        return True
    return False


def check_stop():
    if stop_requested():
        raise Stop("ผู้ใช้สั่งหยุด")


def nap(sec):
    """sleep ที่ถูกขัดจังหวะได้ทันทีเมื่อสั่งหยุด"""
    end = time.time() + sec
    while True:
        check_stop()
        left = end - time.time()
        if left <= 0:
            return
        _real_sleep(min(0.1, left))


def _arm_watchdog():
    """สั่งหยุดแล้วโปรแกรมยังไม่ปิดภายใน KILL_AFTER_STOP วินาที (เช่น ค้างในคำสั่งเครื่อง) = ปิดตัวเองทันที"""
    if getattr(_arm_watchdog, "armed", False):
        return
    _arm_watchdog.armed = True

    def kill():
        try:
            sys.stdout.write("\r\n⛔ บังคับปิดโปรแกรม\r\n")
            sys.stdout.flush()
        finally:
            os._exit(130)
    t = threading.Timer(KILL_AFTER_STOP, kill)
    t.daemon = True
    t.start()


def install_stop_handlers():
    """หยุดได้ตลอด: Ctrl+C / พิมพ์ q แล้ว Enter / สร้างไฟล์ STOP (คำสั่ง `stop`) / SIGTERM"""
    try:
        os.remove(STOP_FILE)
    except OSError:
        pass
    STOP_FLAG.clear()
    _arm_watchdog.armed = False
    hits = {"n": 0}

    def on_sig(signum, frame):
        hits["n"] += 1
        STOP_FLAG.set()
        _arm_watchdog()
        if hits["n"] >= 2:
            os._exit(130)
        try:
            ui("⏹ หยุดแล้ว (กด Ctrl+C อีกครั้ง=บังคับ)")
        except Exception:
            pass

    for sg in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sg, on_sig)
        except Exception:
            pass

    def reader():
        while not STOP_FLAG.is_set():
            try:
                line = sys.stdin.readline()
            except Exception:
                return
            if not line:
                return
            if line.strip().lower() in STOP_WORDS:
                STOP_FLAG.set()
                _arm_watchdog()
                ui("⏹ รับ q -> หยุด")
                return

    if sys.stdin is not None and sys.stdin.isatty():
        threading.Thread(target=reader, daemon=True).start()
    ui("⏹ หยุดได้: Ctrl+C / พิมพ์ q+Enter")
    ui("   หรือหน้าต่างอื่น: python rock_loop.py stop")


# =============================================================================
# เครื่อง: จับภาพ + แปลงพิกัด + สั่งแตะ
# =============================================================================
class Frame:
    """หนึ่งเฟรมในแนวนอน + ตัวแปลงพิกัด"""

    def __init__(self, img, raw_shape, rotated):
        self.img = img
        self.h, self.w = img.shape[:2]
        self.raw_shape = raw_shape
        self.rotated = rotated
        self.sx, self.sy = self.w / REF_W, self.h / REF_H

    def A(self, name):
        x, y = ANCHORS[name]
        return int(x * self.sx), int(y * self.sy)

    def box(self, b):
        x1, y1, x2, y2 = ROI[b] if isinstance(b, str) else b
        return int(x1 * self.sx), int(y1 * self.sy), int(x2 * self.sx), int(y2 * self.sy)

    def crop(self, b):
        x1, y1, x2, y2 = self.box(b)
        return self.img[max(0, y1):y2, max(0, x1):x2]

    def to_touch(self, xl, yl):
        """พิกัดบนภาพแนวนอน -> พิกัดสัมผัสบนจอจริง (จอตั้ง: x = กว้าง-1-yl, y = xl)"""
        if self.rotated:
            return int(round(self.raw_shape[1] - 1 - yl)), int(round(xl))
        return int(round(xl)), int(round(yl))


def make_frame(raw):
    h, w = raw.shape[:2]
    rotated = (h > w) if SCREEN_ROTATED == "auto" else bool(SCREEN_ROTATED)
    if rotated:
        vh = min(h, VIEW_HEIGHT or int(h * 1210 / 1280))
        land = np.rot90(raw[:vh], 1).copy()        # หมุนทวนเข็มคืนเป็นแนวนอน
    else:
        land = raw
    return Frame(land, (h, w), rotated)


class Device:
    def __init__(self, dry=False):
        self.dry = dry
        self.last = None

    def sh(self, cmd, binary=False, timeout=25):
        if self.dry:
            log(f"[dry] {cmd}")
            return b"" if binary else ""
        check_stop()
        p = subprocess.Popen(SHELL_PREFIX + [cmd], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, start_new_session=True)
        out = {}
        th = threading.Thread(target=lambda: out.update(r=p.communicate()), daemon=True)
        th.start()
        t0 = time.time()
        while th.is_alive():
            if stop_requested() or time.time() - t0 > timeout:
                try:
                    p.kill()
                except Exception:
                    pass
                th.join(1.0)
                check_stop()
                raise subprocess.TimeoutExpired(cmd, timeout)
            th.join(0.05)
        stdout = out.get("r", (b"", b""))[0] or b""
        return stdout if binary else stdout.decode("utf-8", "ignore")

    def root_ok(self):
        out = self.sh("id")
        return "uid=0" in out, out.strip()

    def grab(self):
        raw = self.sh("screencap -p", binary=True)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR) if arr.size else None
        if img is not None:
            self.last = make_frame(img)
        return self.last

    # ---- แตะ / กดค้าง
    def tap(self, xl, yl):
        x, y = self.last.to_touch(xl, yl)
        log(f"  แตะ ({int(xl)},{int(yl)}) -> touch ({x},{y})")
        self.sh(f"input tap {x} {y}")

    def hold(self, xl, yl, ms):
        x, y = self.last.to_touch(xl, yl)
        if TOUCH_BACKEND["hold"] == "sendevent":
            self._events([(x, y)], ms)
        else:
            self.sh(f"input swipe {x} {y} {x} {y} {int(ms)}", timeout=ms / 1000 + 12)

    def burst(self, xl, yl, n, gap=BURST_GAP, press_ms=BURST_PRESS_MS):
        """แตะรัว n ครั้งในคำสั่งเดียว (ตีต่อเนื่อง ไม่เสียเวลาเรียกเครื่องทีละครั้ง)"""
        x, y = self.last.to_touch(xl, yl)
        if TOUCH_BACKEND["hold"] == "sendevent":
            for _ in range(n):
                self.hold(xl, yl, press_ms)
                nap(gap)
            return
        one = f"input swipe {x} {y} {x} {y} {int(press_ms)}"
        self.sh(f"; sleep {gap}; ".join([one] * n), timeout=n * (press_ms / 1000 + gap) + 12)

    def _swipe(self, p1, p2, ms, backend):
        x1, y1 = self.last.to_touch(*p1)
        x2, y2 = self.last.to_touch(*p2)
        if backend == "sendevent":
            n = max(6, int(ms / 25))
            pts = [(int(x1 + (x2 - x1) * i / n), int(y1 + (y2 - y1) * i / n)) for i in range(n + 1)]
            self._events(pts, ms)
        else:
            self.sh(f"input swipe {x1} {y1} {x2} {y2} {int(ms)}", timeout=ms / 1000 + 12)

    def joystick(self, dx, dy, ms):
        """ดันจอยไปทิศ (dx,dy) บนภาพแนวนอน (ขึ้น = dy ลบ) นาน ms"""
        n = math.hypot(dx, dy)
        if n < 1e-6:
            return
        ux, uy = dx / n, dy / n
        cx, cy = self.last.A("joy_center")
        r = ANCHORS["joy_radius"] * self.last.sy
        self._swipe((cx + ux * r * 0.45, cy + uy * r * 0.45), (cx + ux * r * 0.85, cy + uy * r * 0.85),
                    ms, TOUCH_BACKEND["joystick"])

    def camera_available(self):
        return TOUCH_BACKEND["camera"] != "none"

    def camera_drag(self, dx_px, dy_px=0, ms=260):
        if not self.camera_available():
            return False
        sx, sy = self.last.A("camera_start")
        self._swipe((sx, sy), (sx + dx_px, sy + dy_px), ms, TOUCH_BACKEND["camera"])
        return True

    def _events(self, pts, ms):
        """ยิงสัญญาณแตะเข้าอุปกรณ์ตรงๆ ตามลำดับที่ getevent บันทึกจากนิ้วจริง (ทดลอง)"""
        dev = TOUCH_EVENT_DEVICE
        tid = 3000 + int(time.time()) % 5000
        dt = max(0.01, ms / 1000.0 / max(1, len(pts)))
        ev = lambda t, c, v: f"sendevent {dev} {t} {c} {v};"
        s = ""
        for i, (x, y) in enumerate(pts):
            s += ev(3, 57, tid) + ev(3, 53, x) + ev(3, 54, y)
            if i == 0:
                s += ev(1, 330, 1)
            s += ev(0, 2, 0) + ev(0, 0, 0) + f"sleep {dt:.3f};"
        s += ev(3, 57, 4294967295) + ev(0, 2, 0) + ev(1, 330, 0) + ev(0, 0, 0)
        self.sh(s, timeout=ms / 1000 + 25)


# =============================================================================
# มองหาหิน: โปรไฟล์สีตามช่วงเวลา
# =============================================================================
def _rock_pixels(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(int)
    H, S, V = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    grass = (H >= 33) & (H <= 95) & (S > 70)
    tan = (H >= 3) & (H <= 28) & (S > 70) & (V > 90)
    return hsv[~grass & ~tan]


def _hue_ranges(h, min_frac=0.015, pad=6):
    hist = np.bincount(h, minlength=180).astype(float)
    ext = np.concatenate([hist[-4:], hist, hist[:4]])
    sm = np.convolve(ext, np.ones(5) / 5.0, mode="same")[4:-4]
    on = sm >= min_frac * hist.sum()
    rng, i = [], 0
    while i < 180:
        if on[i]:
            j = i
            while j + 1 < 180 and on[j + 1]:
                j += 1
            rng.append([max(0, i - pad), min(179, j + pad)])
            i = j + 1
        else:
            i += 1
    merged = []
    for r in sorted(rng):
        if merged and r[0] <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], r[1])
        else:
            merged.append(r)
    return merged


def build_profiles(verbose=True):
    """สร้างโปรไฟล์สีหินจากรูปใน templates/rocks/<ช่วงเวลา>/ (ตัดภาพหินเป็นหลัก ขอบมีหญ้านิดหน่อยได้)"""
    profiles = {}
    for p in PERIODS:
        files = sorted(glob.glob(os.path.join(TEMPLATE_DIR, "rocks", p, "*.*")))
        pix = [_rock_pixels(im) for im in (cv2.imread(f) for f in files) if im is not None]
        if not pix:
            if p in DEFAULT_PROFILES:
                profiles[p] = DEFAULT_PROFILES[p]
            continue
        a = np.vstack(pix)
        s_lo, s_hi = np.percentile(a[:, 1], [2, 97])
        v_lo, v_hi = np.percentile(a[:, 2], [2, 97])
        prof = {"s": [int(max(0, s_lo - 12)), int(min(255, s_hi + 14))],
                "v": [int(max(ROCK_V_FLOOR, v_lo - 12)), int(min(255, v_hi + 18))],
                "images": len(files)}
        if float((a[:, 1] < 35).mean()) > 0.4:
            prof["any_hue"] = True
        else:
            prof["hue"] = _hue_ranges(a[a[:, 1] >= 35, 0])
        profiles[p] = prof
        if verbose:
            log(f"โปรไฟล์ {p}: {len(files)} รูป -> {json.dumps({k: v for k, v in prof.items() if k != 'images'})}")
    return profiles or dict(DEFAULT_PROFILES)


def rock_mask(img, names, profiles):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    out = np.zeros(H.shape, np.uint8)
    for n in names:
        p = profiles.get(n)
        if not p:
            continue
        hm = np.ones(H.shape, bool) if p.get("any_hue") else np.zeros(H.shape, bool)
        if not p.get("any_hue"):
            for h0, h1 in p["hue"]:
                hm |= (H >= h0) & (H <= h1)
        m = hm & (S >= p["s"][0]) & (S <= p["s"][1]) & (V >= p["v"][0]) & (V <= p["v"][1])
        out |= m.astype(np.uint8) * 255
    return out


def find_rocks(fr, names, profiles):
    """-> (รายการหินเรียงใกล้->ไกล, มาสก์) แต่ละก้อน {'base','bbox','area','dist'}"""
    m = rock_mask(fr.img, names, profiles)
    for b in list(UI_EXCLUDE) + [CHAR_EXCLUDE]:
        x1, y1, x2, y2 = fr.box(b)
        m[max(0, y1):y2, max(0, x1):x2] = 0
    k = max(3, int(5 * fr.sy))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    feet = fr.A("char_feet")
    rocks = []
    for i in range(1, n):
        x, y, w, h, a = [int(v) for v in st[i]]
        if a < 260 * fr.sx * fr.sy or a > 90000 * fr.sx * fr.sy:
            continue
        if h < 0.45 * w or h > 2.0 * w:                 # หินสูงพอๆ กับกว้าง (ตัดลำต้น/เสา)
            continue
        if (x <= 1 or x + w >= fr.w - 1) and w < 70 * fr.sx:
            continue
        if a / float(w * h) < 0.28:
            continue
        base = (x + w // 2, y + h)
        if base[1] < ROCK_MIN_Y * fr.sy:
            continue
        rocks.append({"base": base, "bbox": (x, y, w, h), "area": a,
                      "dist": float(math.hypot(base[0] - feet[0], base[1] - feet[1]))})
    rocks.sort(key=lambda r: r["dist"])
    return rocks, m


# =============================================================================
# นาฬิกาในเกม -> ช่วงเวลา -> โปรไฟล์
# =============================================================================
def fmt_min(m):
    return f"{int(m) // 60:02d}:{int(m) % 60:02d}"


def minutes_to_period(m):
    m = int(m) % DAY_MIN
    for name, (a, b) in PERIODS.items():
        if (a <= m < b) if a <= b else (m >= a or m < b):
            return name
    return "day"


def profiles_for(m):
    """ใกล้รอยต่อของช่วง ใช้โทนม่วง 'dawn' ร่วมด้วย"""
    m = int(m) % DAY_MIN
    names = [minutes_to_period(m)]
    for bnd in [a for a, _ in PERIODS.values()] + [b for _, b in PERIODS.values()]:
        if min((m - bnd) % DAY_MIN, (bnd - m) % DAY_MIN) <= TWILIGHT_BLEND_MIN and "dawn" not in names:
            names.append("dawn")
    return names


def light_ok(best, img):
    """เวลาที่อ่านได้สอดคล้องกับความสว่างของฉากไหม (ตอนเช้า OCR มักเพี้ยน): ตรงกัน หรือฉากเป็น dawn (รอยต่อ) = ผ่าน"""
    v = visual_period(img)
    return v in profiles_for(best) or v == "dawn" or minutes_to_period(best) == "dawn"


def visual_period(img):
    """เดาช่วงเวลาจากแสงพื้นหญ้าหน้าตัวละคร (ทดสอบกับภาพจริง 30 ภาพ) -> day | dawn(รวมค่ำ) | night"""
    h, w = img.shape[:2]
    reg = np.concatenate([img[int(h * .50):int(h * .62), int(w * .30):int(w * .45)].reshape(-1, 3),
                          img[int(h * .50):int(h * .62), int(w * .58):int(w * .78)].reshape(-1, 3)])
    b, g, r = [float(x) for x in reg.mean(0)]
    if r - b < 12:
        return "night"
    return "day" if max(b, g, r) >= 112 else "dawn"


# =============================================================================
# OCR ที่ไม่ขึ้นกับพื้นหลัง: white top-hat (เก็บเฉพาะ "เส้นขาวบางๆ" ตัดท้องฟ้า/เมฆ/พื้นหลังสว่างทิ้ง)
# =============================================================================
_LAST_LAYOUT = [None]


def layouts_to_try():
    order = ["A", "B"] if LAYOUT == "auto" else [LAYOUT]
    if _LAST_LAYOUT[0] in order:
        order.remove(_LAST_LAYOUT[0])
        order.insert(0, _LAST_LAYOUT[0])
    return order


def tophat_mask(roi, sc=4, ks=17, blur=True):
    """-> ภาพขาวดำ (ตัวหนังสือ=ขาว) ขยาย sc เท่า"""
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gb = cv2.resize(g, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
    k = int(ks * sc / 6) | 1
    th = cv2.morphologyEx(gb, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if blur:
        th = cv2.GaussianBlur(th, (3, 3), 0)
    _, b = cv2.threshold(th, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return b


def tess(img_black_on_white, whitelist, psm=7):
    if not HAS_TESS:
        return ""
    v = cv2.copyMakeBorder(img_black_on_white, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=255)
    try:
        return pytesseract.image_to_string(v, config=f"--psm {psm} -c tessedit_char_whitelist={whitelist}").strip()
    except Exception:
        return ""


def segment_glyphs(mask):
    """แยกตัวอักษรตามคอลัมน์ -> [[x0,x1,y0,y1],...] ตัดจุด/ลูกน้ำ (เตี้ย) ทิ้ง"""
    H, W = mask.shape
    col = (mask > 0).sum(axis=0)
    runs, x = [], 0
    while x < W:
        if col[x] > 0:
            x0 = x
            while x < W and col[x] > 0:
                x += 1
            runs.append([x0, x])
        else:
            x += 1
    out = []
    for x0, x1 in runs:
        ys = np.where(mask[:, x0:x1].any(axis=1))[0]
        if len(ys):
            out.append([x0, x1, int(ys.min()), int(ys.max()) + 1])
    if not out:
        return []
    hmax = max(o[3] - o[2] for o in out)
    keep = [o for o in out if (o[3] - o[2]) >= 0.62 * hmax and (o[1] - o[0]) >= 0.10 * hmax]
    res = []
    for x0, x1, y0, y1 in keep:
        w, h = x1 - x0, y1 - y0
        if w > 1.15 * h:                      # ตัวเลขสองตัวติดกัน -> ตัดตรงจุดที่บางสุดกลางๆ
            seg = (mask[:, x0:x1] > 0).sum(axis=0).astype(float)
            lo, hi = int(0.35 * w), int(0.65 * w)
            cut = x0 + lo + int(np.argmin(seg[lo:hi]))
            res += [[x0, cut, y0, y1], [cut, x1, y0, y1]]
        else:
            res.append([x0, x1, y0, y1])
    return res


def read_digits_by_glyph(roi, kernels=(15, 17, 21)):
    """อ่านทีละตัวเลข (tesseract psm 10) หลายขนาดเคอร์เนล แล้วโหวตรายหลัก -> (x, y) หรือ None
    ใช้เมื่อแบ่งได้ 9 ตัวพอดี (xxxx,yyyyy) ตำแหน่งลูกน้ำ = ช่องว่างกว้างสุด"""
    votes, boxes_ref = None, None
    for ks in kernels:
        m = tophat_mask(roi, 4, ks)
        gl = segment_glyphs(m)
        if len(gl) > 9:
            gl = gl[-9:]                  # เลย์เอาต์ B มีไอคอน/เลขเกาะนำหน้า เอา 9 ตัวท้าย
        if len(gl) != 9:
            continue
        if votes is None:
            votes = [Counter() for _ in gl]
            gaps = [gl[i + 1][0] - gl[i][1] for i in range(8)]
            boxes_ref = int(np.argmax(gaps)) + 1
        for i, (x0, x1, y0, y1) in enumerate(gl):
            g = 255 - m[max(0, y0 - 2):y1 + 2, max(0, x0 - 2):x1 + 2]
            t = tess(g, "0123456789", psm=10)
            if t[:1].isdigit():
                votes[i][t[0]] += 1
    if votes is None or any(not v for v in votes):
        return None
    d = "".join(v.most_common(1)[0][0] for v in votes)
    k = boxes_ref
    if not (3 <= k <= 5):
        k = 4
    xy = int(d[:k]), int(d[k:])
    return xy if (xy[0] >= 1000 and xy[1] >= 1000) else None      # พิกัดจริงของแผนที่นี้เป็นเลข 4-5 หลักทั้งคู่ (กันอ่านผิดโซน)


class ClockReader:
    """อ่านเวลาในเกม: OCR หลายแบบ+โหวต -> กรองความสมเหตุสมผล -> เดินนาฬิกาเองด้วยอัตราที่วัดได้ -> สำรอง: ดูแสงฉาก"""

    def __init__(self):
        self.last_min = self.last_real = None
        self.pairs, self.rate, self.unwrapped = [], None, 0.0
        self.source, self._t = "none", 0.0
        self.override = None

    @staticmethod
    def _variants(roi):
        """สร้างภาพให้ tesseract ทีละแบบ (top-hat ก่อน ทนพื้นหลังท้องฟ้า แล้วค่อยแบบเดิมสำหรับพื้นหลังเข้ม)"""
        for ks in (17, 13, 21):
            for bl in (True, False):
                yield 255 - tophat_mask(roi, 5, ks, bl)
        g = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), None, fx=6, fy=6, interpolation=cv2.INTER_CUBIC)
        _, o = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        a = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, -8)
        yield 255 - o
        yield o
        yield 255 - a

    def _ocr(self, fr):
        if not HAS_TESS:
            return []
        out = []
        for lay in layouts_to_try():
            roi = fr.crop(LAYOUT_ROI[lay]["clock"])
            if roi.size == 0:
                continue
            got = []
            for v in self._variants(roi):
                d = "".join(c for c in tess(v, "0123456789:") if c.isdigit())
                if len(d) == 4 and int(d[:2]) < 24 and int(d[2:]) < 60:
                    got.append(int(d[:2]) * 60 + int(d[2:]))
                    if Counter(got).most_common(1)[0][1] >= 3:
                        break
            if got and Counter(got).most_common(1)[0][1] >= 2:
                _LAST_LAYOUT[0] = lay
                return got
            out += got
        return out

    def _consistent(self, cand, now):
        if self.last_min is None:
            return False
        start, width = ((self.last_min + self.rate * (now - self.last_real)) % DAY_MIN, 25.0) if self.rate \
            else (self.last_min, 180.0)
        return (cand - start) % DAY_MIN <= width or (start - cand) % DAY_MIN <= 6

    def _accept(self, minutes, now):
        if self.last_min is not None:
            delta = (minutes - self.last_min) % DAY_MIN
            if delta > DAY_MIN / 2:                    # ถอยหลัง = อ่านเพี้ยน
                return
            self.unwrapped += delta
        self.last_min, self.last_real = minutes, now
        self.pairs = (self.pairs + [(now, self.unwrapped)])[-12:]
        old = [p for p in self.pairs if now - p[0] >= 25]
        if old and now > old[0][0] and self.unwrapped > old[0][1]:
            self.rate = (self.unwrapped - old[0][1]) / (now - old[0][0])

    def update(self, fr, force=False):
        now = time.time()
        if not force and now - self._t < CLOCK_READ_INTERVAL:
            return
        self._t = now
        cands = self._ocr(fr)
        if not cands:
            return
        best, n = Counter(cands).most_common(1)[0]
        if n < 4 and not light_ok(best, fr.img) and not (self.last_min is not None and self._consistent(best, now) and n >= 3):
            log(f"[เวลา] อ่านได้ {fmt_min(best)} แต่ไม่ตรงกับความสว่างฉาก ({visual_period(fr.img)}) -> ไม่เชื่อ")
            return
        if n >= 2 and (self.last_min is None or self._consistent(best, now) or n >= 3):
            self._accept(best, now)
            self.source = "ocr"
        elif n == 1:
            if self.last_min is not None:
                ok = [c for c in set(cands) if self._consistent(c, now)]
                if ok:
                    self._accept(ok[0], now)
                    self.source = "ocr?"
            elif visual_period(fr.img) in profiles_for(best):     # ครั้งแรก: เชื่อเมื่อตรงกับแสงฉาก
                self._accept(best, now)
                self.source = "ocr?"

    def now_minutes(self):
        if self.last_min is None:
            return None
        if self.rate:
            return (self.last_min + self.rate * (time.time() - self.last_real)) % DAY_MIN
        return self.last_min

    def current(self, fr):
        """-> (โปรไฟล์ที่ใช้, ช่วงหลัก, ข้อความเวลา, แหล่งที่มา)"""
        if self.override:
            return [self.override], self.override, "--:--", "override"
        self.update(fr)
        m = self.now_minutes()
        if m is not None:
            names, per, src = profiles_for(m), minutes_to_period(m), self.source
            v = visual_period(fr.img)
            if v not in names:                       # เวลากับแสงฉากขัดกัน (ตอนเช้า/ค่ำ) ใช้โทนของแสงจริงร่วมด้วย
                names = names + [v]
                if v == "night" and per == "day":
                    per, src = "night", "แสง"
            return names, per, fmt_min(m), src
        if CLOCK_FALLBACK_VISUAL:
            p = visual_period(fr.img)
            return ([p, "dawn"] if p != "dawn" else ["dawn"]), p, "--:--", "visual"
        return ["day"], "day", "--:--", "default"


# =============================================================================
# ตัวตรวจ UI (สี/รูปทรง ไม่พึ่งแสงกลางวัน-กลางคืน)
# =============================================================================
CYAN = ((82, 90, 170), (102, 255, 255))                # เส้นฟ้าใต้แถบเป้าหมาย
RED_LO, RED_HI = ((0, 110, 130), (9, 255, 255)), ((171, 110, 130), (179, 255, 255))
GREEN_BTN = ((40, 70, 120), (88, 255, 235))            # Pick All / Fix
YELLOW_BTN = ((15, 130, 170), (32, 255, 255))          # Equip Now


def hsv_range(img, lo, hi):
    return cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), np.array(lo, np.uint8), np.array(hi, np.uint8))


def largest_blob(mask, min_area=50, min_aspect=0.0, max_aspect=1e9, min_fill=0.0):
    n, _, st, cen = cv2.connectedComponentsWithStats(mask, connectivity=8)
    best = None
    for i in range(1, n):
        x, y, w, h, a = [int(v) for v in st[i]]
        if a < min_area or not (min_aspect <= w / float(max(1, h)) <= max_aspect) or a / float(w * h) < min_fill:
            continue
        if best is None or a > best[4]:
            best = (x + w // 2, y + h // 2, w, h, a)
    return best


def frame_diff(a, b):
    """ค่าต่างเฉลี่ยของสองเฟรม ใช้ตรวจว่าเดินแล้ว 'ติด' หรือไม่"""
    if a is None or b is None or a.shape != b.shape:
        return 99.0
    sa = cv2.resize(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.int16)
    sb = cv2.resize(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.int16)
    return float(np.mean(np.abs(sa - sb)))


def find_button(fr, kind, roi, min_area_ref, min_aspect=1.8, max_aspect=6.0):
    x1, y1, x2, y2 = fr.box(roi)
    x1, y1 = max(0, x1), max(0, y1)
    sub = fr.img[y1:y2, x1:x2]
    m = cv2.morphologyEx(hsv_range(sub, *(GREEN_BTN if kind == "green" else YELLOW_BTN)),
                         cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    b = largest_blob(m, min_area_ref * fr.sx * fr.sy, min_aspect, max_aspect, 0.6)
    return (x1 + b[0], y1 + b[1]) if b else None


def _runs_longest(m):
    """ช่วง True ต่อเนื่องยาวสุดของเวกเตอร์ -> (ความยาว, เริ่ม, จบ)"""
    best, i, n = (0, 0, 0), 0, len(m)
    while i < n:
        if m[i]:
            j = i
            while j + 1 < n and m[j + 1]:
                j += 1
            if j - i + 1 > best[0]:
                best = (j - i + 1, i, j)
            i = j + 1
        else:
            i += 1
    return best


def _white_line_bar(fr, strip, x1, y1, out):
    """แถบเป้าหมายแบบเส้นขาว (เครื่องจริง): ชื่อ -> เส้นขาวบางยาว ~310px -> แถบเลือดแดงใต้เส้น"""
    hsv = cv2.cvtColor(strip, cv2.COLOR_BGR2HSV)
    S, V, H = hsv[..., 1], hsv[..., 2], hsv[..., 0]
    white = (S < 60) & (V > 200)
    red = (((H <= 9) | (H >= 171)) & (S > 110) & (V > 130))
    rows = [(_runs_longest(white[y]), y) for y in range(white.shape[0])]
    lo, hi = 230 * fr.sx, 380 * fr.sx
    good = [(r, y) for r, y in rows if lo <= r[0] <= hi]
    if not good:
        return out
    (n, xa, xb), ly = max(good, key=lambda t: t[0][0])
    thick = sum(1 for r, y in good if abs(y - ly) <= 8)
    if thick > 8:                                       # หนาเกิน = เมฆ/ขอบอย่างอื่น ไม่ใช่เส้นแถบ
        return out
    ya, yb = ly + int(3 * fr.sy), min(strip.shape[0], ly + int(22 * fr.sy))
    if yb <= ya:
        return out
    below_red = red[ya:yb, xa:xb + 1]
    cols = np.where(below_red.sum(axis=0) >= max(2, int(3 * fr.sy)))[0]
    dark = float((V[ya:yb, xa:xb + 1].mean(axis=0) < 140).mean())
    if cols.size == 0 and dark < 0.7:
        return out
    hp = (float(cols.max()) + 1.0) / float(xb - xa + 1) if cols.size else 0.0
    out.update(engaged=True, hp=0.0 if hp < 0.03 else min(1.0, hp), x0=x1 + xa, x1=x1 + xb, y=y1 + ly)
    return out


def read_target_bar(fr):
    """-> dict(engaged, hp, x0, x1, y)  engaged = แถบเป้าหมายขึ้น (ตีโดนแล้ว), hp = เลือดที่เหลือ 0..1 (0 = หมด)
    หาแถบจากเส้นสีฟ้ายาวๆ ใต้ชื่อ (ชิ้นส่วนที่ต่อเนื่องยาวสุด) แล้ววัดความยาวสีแดงเหนือเส้น"""
    out = {"engaged": False, "hp": 1.0, "x0": 0, "x1": 0, "y": 0}
    x1, y1, x2, y2 = fr.box("target_bar")
    x1, y1 = max(0, x1), max(0, y1)
    strip = fr.img[y1:y2, x1:x2]
    if strip.size == 0:
        return out
    cy = hsv_range(strip, *CYAN) > 0
    rows = cy.sum(axis=1)
    cand = np.where(rows >= 120 * fr.sx)[0]
    if cand.size == 0:
        return _white_line_bar(fr, strip, x1, y1, out)
    ly = int(cand[np.argmax(rows[cand])])
    line = cy[max(0, ly - 1):ly + 2].any(axis=0)
    best, i = (0, 0), 0                                   # ช่วงต่อเนื่องยาวสุด (ยอมขาดได้ 3px)
    while i < len(line):
        if line[i]:
            j, gap = i, 0
            while j + 1 < len(line) and (line[j + 1] or gap < 3):
                gap = 0 if line[j + 1] else gap + 1
                j += 1
            e = j - gap
            if e - i > best[1] - best[0]:
                best = (i, e)
            i = j + 1
        else:
            i += 1
    xa, xb = best
    if xb - xa < 150 * fr.sx:
        return out
    ya, yb = max(0, ly - int(28 * fr.sy)), max(1, ly - int(2 * fr.sy))
    band = strip[ya:yb, xa:xb + 1]
    red = hsv_range(band, *RED_LO) | hsv_range(band, *RED_HI)
    hit = np.where((red > 0).sum(axis=0) >= max(2, int(3 * fr.sy)))[0]
    hp = (float(hit.max()) + 1.0) / float(xb - xa + 1) if hit.size else 0.0
    out.update(engaged=True, hp=0.0 if hp < 0.03 else min(1.0, hp), x0=x1 + xa, x1=x1 + xb, y=y1 + ly)
    return out


SWORD_TPL_B64 = {'A': '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', 'B': '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'}   # ต้นแบบปุ่มตี 2 แบบ (ดาบสั้นเลย์เอาต์เก่า / ดาบใหญ่เครื่องล่าสุด)
_SWORD_TPL = []


def _dog2(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.GaussianBlur(g, (0, 0), 1.0) - cv2.GaussianBlur(g, (0, 0), 5.0)


def sword_present(fr, thr=0.6):
    """เห็นปุ่มตี (วงกลมมีดาบขาว) ที่ตำแหน่ง ANCHORS['sword'] ไหม -> (เห็น, คะแนน) ทนตอนปุ่มมืดลงระหว่างตี"""
    if not _SWORD_TPL:
        import base64
        for b in SWORD_TPL_B64.values():
            im = cv2.imdecode(np.frombuffer(base64.b64decode(b), np.uint8), cv2.IMREAD_COLOR)
            t = _dog2(im)
            _SWORD_TPL.append((t - t.mean()) / (t.std() + 1e-6))
    cx, cy = fr.A("sword")
    r, pad = int(48 * fr.sy), int(30 * fr.sy)
    win = _dog2(fr.img[max(0, cy - r - pad):cy + r + pad, max(0, cx - r - pad):cx + r + pad])
    best = 0.0
    for t in _SWORD_TPL:
        tt = cv2.resize(t, None, fx=fr.sx, fy=fr.sy, interpolation=cv2.INTER_AREA) if (fr.sx, fr.sy) != (1.0, 1.0) else t
        if tt.shape[0] > win.shape[0] or tt.shape[1] > win.shape[1]:
            continue
        best = max(best, float(cv2.matchTemplate(win, tt.astype(np.float32), cv2.TM_CCOEFF_NORMED).max()))
    return best >= thr, best


NAME_TPL_B64 = {"iron mine": [
    "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",
]}   # ต้นแบบตัวหนังสือชื่อ (ฝังในไฟล์ ไม่ต้องมีโฟลเดอร์ templates)


_NAME_TPL = {}


def _dog(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.GaussianBlur(g, (0, 0), 0.8) - cv2.GaussianBlur(g, (0, 0), 3.0)


def name_templates():
    """ภาพต้นแบบตัวหนังสือชื่อเป้าหมาย templates/ui/name_<ชื่อ>_<n>.png (ชื่อขึ้นต้น bad- = ห้ามตี) -> {ชื่อ: [dog]}"""
    if not _NAME_TPL:
        import base64
        for label, lst in NAME_TPL_B64.items():                  # ต้นแบบที่ฝังในไฟล์
            for b64 in lst:
                im = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
                t = _dog(im)
                _NAME_TPL.setdefault(label, []).append((t - t.mean()) / (t.std() + 1e-6))
        files = glob.glob(os.path.join(TEMPLATE_DIR, "ui", "name_*.png")) + glob.glob(os.path.join(ICONS_DIR, "name_*.png"))
        for f in sorted(files):                                  # เพิ่มเองได้: name_<ชื่อ>_<n>.png (ชื่อขึ้นต้น bad- = ห้ามตี)
            im = cv2.imread(f)
            if im is None:
                continue
            label = os.path.basename(f)[5:-4].rsplit("_", 1)[0].replace("_", " ").replace("bad-", "bad ")
            t = _dog(im)
            _NAME_TPL.setdefault(label, []).append((t - t.mean()) / (t.std() + 1e-6))
    return _NAME_TPL


def name_by_template(fr, bar, thr=0.5):
    """จับคู่ภาพตัวหนังสือชื่อกับต้นแบบ (ทนพื้นหลังท้องฟ้า/หญ้า) -> (ชื่อ, คะแนน) หรือ ('', 0)"""
    tpls = name_templates()
    if not tpls or not bar["engaged"]:
        return "", 0.0
    ya, yb = max(0, bar["y"] - int(40 * fr.sy)), min(fr.h, bar["y"] + int(4 * fr.sy))
    xa, xb = max(0, bar["x0"] - int(10 * fr.sx)), min(fr.w, bar["x1"] + int(10 * fr.sx))
    band = _dog(fr.img[ya:yb, xa:xb])
    best = ("", 0.0)
    for label, ts in tpls.items():
        for t in ts:
            for sc in (0.9, 1.0, 1.1):
                tt = cv2.resize(t, None, fx=sc * fr.sx, fy=sc * fr.sy, interpolation=cv2.INTER_AREA)
                if tt.shape[0] > band.shape[0] or tt.shape[1] > band.shape[1]:
                    continue
                v = float(cv2.matchTemplate(band, tt.astype(np.float32), cv2.TM_CCOEFF_NORMED).max())
                if v > best[1]:
                    best = (label, v)
    return (best[0], best[1]) if best[1] >= thr else ("", best[1])


def read_target_name(fr, bar):
    """อ่านชื่อเป้าหมายบนแถบ (ตัวพิมพ์เล็ก): จับคู่ภาพต้นแบบก่อน (แม่นกว่า) แล้ว OCR เสริม"""
    if not bar["engaged"]:
        return ""
    texts = []
    lab, sc = name_by_template(fr, bar)
    if lab:
        texts.append(lab)
    if HAS_TESS:
        ya, yb = max(0, bar["y"] - int(40 * fr.sy)), max(1, bar["y"] - int(1 * fr.sy))
        w = bar["x1"] - bar["x0"]
        roi = fr.img[ya:yb, bar["x0"] + int(w * 0.18):bar["x1"] - int(w * 0.18)]
        if roi.size:
            g = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            imgs = []
            _, o = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            imgs += [o, 255 - o]
            bg = cv2.GaussianBlur(g, (0, 0), 25)
            d = cv2.normalize(cv2.subtract(g, bg), None, 0, 255, cv2.NORM_MINMAX)
            imgs.append(255 - cv2.threshold(d, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1])
            imgs.append(255 - cv2.createCLAHE(3.0, (2, 2)).apply(g))
            for ks in (13, 17):
                imgs.append(255 - tophat_mask(roi, 4, ks, True))
            for t in imgs:
                t = cv2.copyMakeBorder(t, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
                try:
                    texts.append(pytesseract.image_to_string(t, config="--psm 7").strip().lower())
                except Exception:
                    pass
    return " | ".join(x for x in texts if x)


def name_verdict(text, period, strict=None):
    """-> 'ore' | 'bad' | 'unknown' ตาม ORE_NAME_KEYWORDS / NAME_BLOCKLIST (เทียบแบบเผื่อ OCR เพี้ยน)"""
    if not text:
        return "unknown"
    words = [w for w in text.replace("|", " ").split() if w]
    close = lambda k, thr: k in text or any(difflib.SequenceMatcher(None, k, w).ratio() >= thr for w in words)
    if any(close(b, 0.75) for b in NAME_BLOCKLIST):
        return "bad"
    if any(close(k, 0.7) for k in ORE_NAME_KEYWORDS.get(period, [])):
        return "ore"
    if (NAME_STRICT if strict is None else strict) and any(sum(ch.isalpha() for ch in seg) >= 4 for seg in text.split("|")):
        return "bad"                                    # อ่านชื่อออกเป็นตัวหนังสือ แต่ไม่ใช่แร่ = ไม่ใช่หิน
    return "unknown"


def find_pick_all_white(fr):
    """ปุ่ม Pick All แบบขาว (หน้าต่างเก็บของเครื่องใหม่): ปุ่มสี่เหลี่ยมขาวคู่ข้างกัน Cancel(ซ้าย) | Pick All(ขวา) -> (x, y) ของ Pick All"""
    x1, y1, x2, y2 = fr.box((300, 300, 920, 620))
    sub = fr.img[y1:y2, x1:x2]
    if sub.size == 0:
        return None
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 2] >= 205) & (hsv[..., 1] <= 45)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5)))
    n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    boxes = []
    for i in range(1, n):
        x, y, w, h, a = [int(v) for v in st[i]]
        rw, rh = w / fr.sx, h / fr.sy
        if 120 <= rw <= 230 and 36 <= rh <= 72 and a / float(w * h) >= 0.72 and 2.0 <= w / float(h) <= 5.0:
            boxes.append((x1 + x + w // 2, y1 + y + h // 2, w, h))
    for bx in boxes:                                     # ต้องมีปุ่มคู่ซ้ายมือแนวเดียวกัน (Cancel)
        for bl in boxes:
            dx = (bx[0] - bl[0]) / fr.sx
            if bl is not bx and 140 <= dx <= 290 and abs(bx[1] - bl[1]) <= 14 * fr.sy:
                return bx[0], bx[1]
    return None


def find_pick_all(fr):
    """เจอปุ่ม Pick All ได้ 2 แบบ: เขียว(+Cancel แดง) หรือขาว(+Cancel ขาว) -> (x, y)"""
    return _find_pick_all_green(fr) or find_pick_all_white(fr)


def _find_pick_all_green(fr):
    """ปุ่ม Pick All (เขียว) + ต้องมีปุ่ม Cancel สีแดงอยู่ซ้ายมือแนวเดียวกัน -> (x, y)"""
    p = find_button(fr, "green", "dialog", 3500)
    if not p:
        return None
    x, y = p
    sub = fr.img[max(0, int(y - 35 * fr.sy)):int(y + 35 * fr.sy), max(0, int(x - 330 * fr.sx)):max(1, int(x - 80 * fr.sx))]
    if sub.size == 0:
        return None
    red = hsv_range(sub, (165, 90, 150), (179, 255, 255)) | hsv_range(sub, (0, 90, 150), (8, 255, 255))
    return p if largest_blob(red, 2500 * fr.sx * fr.sy, 1.8, 6.0, 0.6) else None


def find_close_x(fr):
    x1, y1, x2, y2 = fr.box("repair")
    x1, y1 = max(0, x1), max(0, y1)
    sub = fr.img[y1:y2, x1:x2]
    m = hsv_range(sub, (0, 80, 120), (10, 255, 255)) | hsv_range(sub, (165, 80, 120), (179, 255, 255))
    b = largest_blob(m, 300 * fr.sx * fr.sy, 0.6, 1.5, 0.35)       # วงกลมแดง (มี X สีขาวอยู่ข้างใน)
    return (x1 + b[0], y1 + b[1]) if b else None


def find_fix(fr):
    return find_button(fr, "green", "repair", 6000) if find_close_x(fr) else None


def find_equip(fr):
    return find_button(fr, "yellow", "repair", 6000) if find_close_x(fr) else None


def banner_present(fr):
    """แถบข้อความมืดโปร่งแสงพาดกลางจอ (เช่น 'ฉันเหนื่อยแล้ว ต้องพักสักครู่')"""
    h, w = fr.h, fr.w
    y0, y1 = int(h * 0.25), int(h * 0.52)
    v = cv2.cvtColor(fr.img[y0:y1, int(w * 0.04):int(w * 0.96)], cv2.COLOR_BGR2HSV)[..., 2].astype(np.float32).mean(axis=1)
    k, lo, hi = max(2, int(10 * fr.sy)), int(30 * fr.sy), int(75 * fr.sy)
    for top in range(k, len(v) - lo - k):
        for hh in (lo, (lo + hi) // 2, hi):
            bot = top + hh
            if bot + k > len(v):
                continue
            inside, above, below = v[top + 3:bot - 3].mean(), v[top - k:top - 1].mean(), v[bot + 1:bot + k].mean()
            if inside < 0.68 * above and inside < 0.68 * below and above > 40 and below > 40 \
                    and v[top + 3:bot - 3].std() < 0.55 * inside + 12:
                return True
    return False


def pickaxe_state(fr):
    """'ok' = แถบเขียว | 'broken' = แดง/หมด | 'unknown'"""
    roi = fr.crop("durability")
    if roi.size == 0:
        return "unknown"
    H, S, V = [cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[..., i] for i in range(3)]
    if ((H >= 40) & (H <= 90) & (S > 130) & (V > 120)).sum() >= 6:
        return "ok"
    if ((((H <= 9) | (H >= 171)) & (S > 130) & (V > 130)).sum()) >= 6:
        return "broken"
    return "unknown"


_HAND_TPL = []


def hand_templates():
    if not _HAND_TPL:
        import base64
        for b in (HAND_TPL_B64, HAND_TPL2_B64):
            _HAND_TPL.append(cv2.imdecode(np.frombuffer(base64.b64decode(b), np.uint8), cv2.IMREAD_GRAYSCALE))
    return _HAND_TPL


def hand_template():
    return hand_templates()[0]


def find_hand(fr):
    """รูปมือสีขาวที่ลอยใกล้ตัวละคร (ของที่ดรอป/เก็บได้) -> (x, y) กลางรูปมือ | None"""
    x1, y1, x2, y2 = fr.box(HAND_ZONE)
    x1, y1 = max(0, x1), max(0, y1)
    sub = fr.img[y1:y2, x1:x2]
    if sub.size == 0:
        return None
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 1] < 65) & (hsv[..., 2] > 190)).astype(np.uint8) * 255
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    tpls = hand_templates()
    best = None
    for i in range(1, n):
        x, y, w, h, a = [int(v) for v in st[i]]
        rw, rh = w / fr.sx, h / fr.sy
        if not (22 <= rw <= 62 and 22 <= rh <= 62) or a / float(w * h) < 0.30:
            continue
        blob = (lab[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        iou = 0.0
        for tpl in tpls:
            t = cv2.resize(tpl, (w, h), interpolation=cv2.INTER_AREA)
            union = float(((blob > 0) | (t > 127)).sum())
            iou = max(iou, float(((blob > 0) & (t > 127)).sum()) / union if union else 0.0)
        sat = float(hsv[y:y + h, x:x + w, 1][lab[y:y + h, x:x + w] == i].mean())     # มือขาวบริสุทธิ์ (เมฆ/ฟ้ามีสีอมฟ้า)
        if iou >= 0.70 and sat <= 22 and (best is None or iou > best[0]):
            best = (iou, x1 + x + w // 2, y1 + y + h // 2)
    return (best[1], best[2]) if best else None


def repair_dialog_open(fr):
    """หน้าต่าง Equipment Repair: กรอบสีครีมสว่างตรงกลางจอ"""
    x1, y1, x2, y2 = fr.box((420, 215, 800, 475))
    sub = fr.img[max(0, y1):y2, max(0, x1):x2]
    if sub.size == 0:
        return False
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    return float(((hsv[..., 2] > 195) & (hsv[..., 1] < 45)).mean()) >= 0.55


def bag_state(fr):
    """'ok' = วงแหวนเขียว/ฟ้า | 'full' = เหลือง-ส้ม-แดง (ตามที่กำหนด: เหลือง = หยุด) | 'unknown'"""
    roi = fr.crop("bag_ring")
    if roi.size == 0:
        return "unknown"
    h, w = roi.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - w / 2.0, yy - h / 2.0)
    ring = (r >= 0.30 * w) & (r <= 0.48 * w)              # เฉพาะวงแหวน ไม่นับไอคอนทองตรงกลาง
    H, S, V = [cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[..., i] for i in range(3)]
    warm = ring & (S > 110) & (V > 140) & (((H >= 10) & (H <= 34)) | (H <= 6) | (H >= 172))
    cool = ring & (S > 90) & (V > 120) & (H >= 45) & (H <= 100)
    nw, nc = int(warm.sum()), int(cool.sum())
    if nw + nc < 20:
        return "unknown"
    return "full" if nw / float(nw + nc) >= BAG_FULL_RATIO else "ok"


# =============================================================================
# CONFIG ของบอทวนทดสอบ (ใส่พิกัดที่นี่)
# =============================================================================
SEARCH_COORD = None            # (x, y) พิกัดจุดมองหาหิน   เช่น (6345, 14165)   None = ใช้ตำแหน่งตอนเริ่มเป็นจุดมองหา
BOX_COORD = None               # (x, y) พิกัดกล่อง(บ้าน)   เช่น (6300, 14100)   None = ข้ามขั้นตอนไปกล่อง
GOTO_TOL = 5                   # ถึงพิกัดเมื่อห่างไม่เกินกี่หน่วย
BOX_TOL = 4
GOTO_MAX_STEPS = 60
LEASH = 45                     # หาหินแล้วเดินหลุดจากจุดมองหาเกินกี่หน่วย -> กลับไปจุดมองหาก่อน
CAL_MS = 1200                  # เดินทดสอบเพื่อเรียนรู้ทิศ (พิกัด<->จอย) ครั้งละกี่ ms
COORD_SETTLE = 1.2             # รอพิกัดบนจอ "ดีเล" ก่อนอ่านหลังเดิน (วินาที)
ROCKS_PER_VISIT = 6            # ไม่เจอหินกี่รอบ(เดินสุ่ม) แล้วกลับไปจุดมองหาใหม่
ROCK_TRIES_PER_ROUND = 5       # ตีไม่สำเร็จ/ไม่ใช่แร่ ลองหินกี่ก้อนต่อรอบ
SEARCH_LOOK_WAIT = 0.6

# ไอคอนที่ผู้ใช้เก็บไว้ (รูปหินตามช่วงเวลา + รูปมือ) ชื่อไฟล์ไม่ต้องบอกช่วงเวลา บอทจัดกลุ่มจากสีให้เอง
ICONS_DIR = os.path.expanduser("~/UtopiaFarmAI/icons")
ROCK_ICON_PATTERNS = ["*rock*", "*stone*", "*หิน*", "*ore*", "*mine*"]    # ไฟล์ใน ICONS_DIR ที่ถือเป็นรูปหิน
ROCK_ICON_PERIOD = {}          # บังคับช่วงเวลาของไฟล์ เช่น {"rock_night.png": "night"}
HAND_ICON_PATTERNS = ["*hand*", "*มือ*", "*pickup*"]
HAND_THRESHOLD = 0.6           # คะแนนจับไอคอนมือ (ทดสอบ: เห็นมือ 0.87-1.0 / ไม่เห็น <= 0.51)
HAND_SCALES = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0, 1.15]
COORD_REFS = []                # พิกัดอ้างอิง [ตามตัว, ...]: ใช้ซ่อมเลข OCR ที่เพี้ยน (เช่น 14178 อ่านเป็น 11173)  ตั้งด้วย run --pos X,Y
COORD_SNAP_R = 400             # ซ่อมเฉพาะเมื่ออยู่ห่างจากอ้างอิงไม่เกินนี้
POS_REF_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pos_ref.json")
COORD_MAX_JUMP = 120           # พิกัดที่อ่านได้กระโดดเกินนี้จากค่าก่อน ถือว่าอ่านผิด (ยกเว้นยืนยันซ้ำ)
ROI["coords"] = LAYOUT_ROI["A"]["coords"]
TIME_LOG = os.path.join(DEBUG_DIR, "time_log.csv")


# =============================================================================
# รูปหินจาก icons -> โปรไฟล์สีตามช่วงเวลา
# =============================================================================
def classify_icon(img):
    """จัดรูปหินเข้ากลุ่มช่วงเวลาจากสี: ฟ้า=night, ม่วง(อิ่มสี)=dawn, เทา(อิ่มสีต่ำ)=day"""
    a = _rock_pixels(img)
    if len(a) < 50:
        return "day"
    H, S = a[:, 0], a[:, 1]
    if float(((H >= 95) & (H <= 125) & (S > 90)).mean()) >= 0.35:
        return "night"
    if float(np.median(S)) < 40:
        return "day"
    purple = ((H >= 145) | (H <= 12)) & (S >= 30)
    return "dawn" if float(purple.mean()) >= 0.4 else "day"


def _match_files(folder, patterns):
    out = []
    if os.path.isdir(folder):
        for p in patterns:
            out += glob.glob(os.path.join(folder, p))
            out += glob.glob(os.path.join(folder, p.upper()))
    seen, res = set(), []
    for f in out:
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")) and f not in seen:
            seen.add(f)
            res.append(f)
    return sorted(res)


def _profile_from_pixels(a, n_images):
    s_lo, s_hi = np.percentile(a[:, 1], [2, 97])
    v_lo, v_hi = np.percentile(a[:, 2], [2, 97])
    prof = {"s": [int(max(0, s_lo - 12)), int(min(255, s_hi + 14))],
            "v": [int(max(ROCK_V_FLOOR, v_lo - 12)), int(min(255, v_hi + 18))], "images": n_images}
    if float((a[:, 1] < 35).mean()) > 0.4:
        prof["any_hue"] = True
    else:
        prof["hue"] = _hue_ranges(a[a[:, 1] >= 35, 0])
    return prof


def load_profiles(verbose=True):
    """รวมรูปหินจาก ICONS_DIR (จัดกลุ่มตามสีอัตโนมัติ) + templates/rocks/<ช่วงเวลา>/ -> โปรไฟล์ต่อช่วงเวลา"""
    groups = {p: [] for p in PERIODS}
    names = {p: [] for p in PERIODS}
    for f in _match_files(ICONS_DIR, ROCK_ICON_PATTERNS):
        im = cv2.imread(f)
        if im is None:
            continue
        p = ROCK_ICON_PERIOD.get(os.path.basename(f)) or classify_icon(im)
        groups[p].append(_rock_pixels(im))
        names[p].append(os.path.basename(f))
    for p in PERIODS:
        for f in sorted(glob.glob(os.path.join(TEMPLATE_DIR, "rocks", p, "*.*"))):
            im = cv2.imread(f)
            if im is not None:
                groups[p].append(_rock_pixels(im))
                names[p].append("templates/" + os.path.basename(f))
    profiles = {}
    for p in PERIODS:
        if groups[p]:
            profiles[p] = _profile_from_pixels(np.vstack(groups[p]), len(groups[p]))
        elif p in DEFAULT_PROFILES:
            profiles[p] = DEFAULT_PROFILES[p]
            names[p].append("(ค่าสำรองในโค้ด)")
        if verbose:
            log(f"รูปหินช่วง {p:5s}: {len(names[p])} รูป {names[p]}")
    if verbose and not _match_files(ICONS_DIR, ROCK_ICON_PATTERNS):
        log(f"หมายเหตุ: ไม่พบรูปหินใน {ICONS_DIR} (ชื่อไฟล์ต้องตรง {ROCK_ICON_PATTERNS}) -> ใช้รูปใน templates/rocks")
    return profiles


# =============================================================================
# ไอคอนมือ (ขึ้นเมื่อเข้าใกล้กล่อง/ของที่เก็บได้)
# =============================================================================
def _lowS_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return (((hsv[..., 1] < 55) & (hsv[..., 2] > 115) & (hsv[..., 2] < 225)) * 255).astype(np.uint8)


_HAND_MASKS = None


def hand_masks():
    global _HAND_MASKS
    if _HAND_MASKS is None:
        files = _match_files(ICONS_DIR, HAND_ICON_PATTERNS) + glob.glob(os.path.join(TEMPLATE_DIR, "ui", "hand*.png"))
        _HAND_MASKS = []
        for f in files:
            im = cv2.imread(f)
            if im is not None and im.shape[0] >= 20:
                _HAND_MASKS.append(_lowS_mask(im))
    return _HAND_MASKS


def hand_present(fr):
    """-> (เห็นรูปมือไหม, คะแนน)  ค้นรอบตำแหน่งปุ่มมือ (ปุ่มขึ้นตรงเดิมเสมอ) หลายขนาด"""
    cx, cy = fr.A("hand_guess")
    rx, ry = int(130 * fr.sx), int(110 * fr.sy)
    roi = fr.img[max(0, cy - ry):cy + ry, max(0, cx - rx):cx + rx]
    if roi.size == 0:
        return False, 0.0
    m = _lowS_mask(roi).astype(np.float32)
    best = 0.0
    for t in hand_masks():
        for s in HAND_SCALES:
            tt = cv2.resize(t, None, fx=s, fy=s, interpolation=cv2.INTER_AREA).astype(np.float32)
            if tt.shape[0] >= m.shape[0] or tt.shape[1] >= m.shape[1] or tt.std() < 1e-3:
                continue
            best = max(best, float(cv2.matchTemplate(m, tt, cv2.TM_CCOEFF_NORMED).max()))
    return best >= HAND_THRESHOLD, best


# =============================================================================
# พิกัดบนจอ (OCR) + เดินไปตามพิกัด
# =============================================================================
def _coord_variants(roi):
    for ks in (17, 21, 13):
        yield 255 - tophat_mask(roi, 4, ks, True)
    g = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    _, o = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield 255 - o
    yield o


def _parse_coord(text):
    import re
    m = re.findall(r"\d+", text)
    if len(m) >= 3:
        m = m[-2:]                      # เลย์เอาต์ B มีเลขเกาะ/โซนนำหน้า เอาสองกลุ่มท้าย
    if len(m) >= 2 and 4 <= len(m[0]) <= 5 and 4 <= len(m[1]) <= 6:
        return int(m[0]), int(m[1])
    d = "".join(m)
    if len(m) == 1 and len(d) == 9:
        return int(d[:4]), int(d[4:])
    return None


def _ham(a, b):
    return sum(1 for p, q in zip(a, b) if p != q) + abs(len(a) - len(b))


def snap_coords(c, refs=None):
    """ซ่อมเลขพิกัดที่ OCR อ่านเพี้ยนบางหลัก: หาพิกัดใกล้จุดอ้างอิงที่ตัวเลขต่างจากที่อ่านได้น้อยหลักที่สุด
    -> ((x, y), จำนวนหลักที่แก้) | (c, None) ถ้าไม่มีอ้างอิง/อ่านต่างเกินไป (เช่น เทเลพอร์ตไปไกล)"""
    refs = COORD_REFS if refs is None else refs
    if not c or not refs:
        return c, None
    xs, ys, best = f"{c[0]:04d}", f"{c[1]:05d}", None
    r = COORD_SNAP_R
    for rx, ry in refs:
        bx = min((_ham(xs, f"{x:04d}"), abs(x - rx), x) for x in range(max(0, rx - r), rx + r + 1))
        by = min((_ham(ys, f"{y:05d}"), abs(y - ry), y) for y in range(max(0, ry - r), ry + r + 1))
        h = bx[0] + by[0]
        if bx[0] <= 1 and by[0] <= 2 and h <= 3:
            key = (h, bx[1] + by[1])
            if best is None or key < best[0]:
                best = (key, (bx[2], by[2]), h)
    return (best[1], best[2]) if best else (c, None)


def set_coord_ref(pos):
    COORD_REFS[:] = [tuple(pos), tuple(pos)]


def load_pos_ref():
    try:
        with open(POS_REF_FILE, encoding="utf-8") as f:
            d = json.load(f)
        set_coord_ref((int(d["x"]), int(d["y"])))
    except Exception:
        pass


def save_pos_ref():
    if COORD_REFS:
        try:
            with open(POS_REF_FILE, "w", encoding="utf-8") as f:
                json.dump({"x": COORD_REFS[0][0], "y": COORD_REFS[0][1]}, f)
        except Exception:
            pass


def vote_coords(reads):
    """โหวตทีละหลักจากการอ่านหลายรอบ (OCR เพี้ยนบางหลักเป็นครั้งคราว เช่น 14→12) -> (x, y) หรือ None"""
    ok = [r for r in reads if r]
    if not ok:
        return None
    if len(ok) < 3:
        return ok[0]
    strs = [f"{x:04d}{y:05d}" for x, y in ok if 1000 <= x <= 9999 and 10000 <= y <= 99999] or [f"{x:04d}{y:05d}" for x, y in ok if 0 <= x <= 9999 and 0 <= y <= 99999]
    if not strs:
        return ok[0]
    n = max(len(t) for t in strs)
    strs = [t for t in strs if len(t) == n] or strs
    digs = "".join(Counter(t[i] for t in strs).most_common(1)[0][0] for i in range(len(strs[0])))
    return int(digs[:4]), int(digs[4:])


def read_coords_once(fr):
    """อ่านพิกัด แล้วซ่อมเลขเพี้ยนด้วยจุดอ้างอิง (ถ้ามี) -> (x, y) หรือ None"""
    c = _read_coords_raw(fr)
    if c and COORD_REFS:
        c2, h = snap_coords(c)
        if h is not None:
            if h:
                log(f"[พิกัด] ซ่อมเลข {c[0]},{c[1]} -> {c2[0]},{c2[1]} ({h} หลัก)")
            if h <= 2:
                COORD_REFS[0] = c2                    # จุดอ้างอิงขยับตามตัวละคร
            return c2
    return c


def _read_coords_raw(fr):
    """อ่านพิกัด 'x, y': (1) อ่านทีละตัวเลขแล้วโหวต (2) ทั้งบรรทัดหลายแบบแล้วโหวต -> (x, y) หรือ None"""
    if not HAS_TESS:
        return None
    for lay in layouts_to_try():
        roi = fr.crop(LAYOUT_ROI[lay]["coords"])
        if roi.size == 0:
            continue
        c = read_digits_by_glyph(roi)
        if c:
            _LAST_LAYOUT[0] = lay
            return c
        votes = []
        for v in _coord_variants(roi):
            c = _parse_coord(tess(v, "0123456789,. "))
            if c:
                votes.append(c)
        if votes:
            best, n = Counter(votes).most_common(1)[0]
            if n >= 2 or len(set(votes)) == 1:
                _LAST_LAYOUT[0] = lay
                return best
    return None


# =============================================================================
# กล้อง: วัดการหมุนจากการขยับของฉาก (phase correlation) แล้วควบคุมแบบวงปิด
# =============================================================================
def scene_gray(fr):
    x1, y1, x2, y2 = fr.box(CAM_SCENE_ROI)
    g = cv2.cvtColor(fr.img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(g, (5, 5), 0).astype(np.float32)


def scene_shift(a, b):
    """ฉากใน b ขยับเทียบกับ a -> (dx, dy, ความมั่นใจ)  dx บวก = เนื้อฉากเลื่อนขวา = กล้องหันซ้าย / dy = เงย-ก้ม"""
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, win)
    return float(dx), float(dy), float(resp)


class Camera:
    """ratio = ฉากขยับแนวนอนกี่ px ต่อการลากนิ้ว 1 px (มีเครื่องหมาย วัดจริง)
    axis = ลากแนวไหนของจอแนวนอนแล้วกล้อง 'หมุนซ้าย-ขวา' ("x" ลากซ้าย-ขวา / "y" ลากขึ้น-ลง) วัดเอง
    (บางเครื่องลากแนวนอนแล้วกล้องเงย-ก้มแทน จึงต้องลองทั้งสองแกน)"""

    def __init__(self, dev):
        self.dev = dev
        self.ratio = None
        self.axis = "x"
        self.ok = False

    def _snap(self):
        fr = self.dev.grab()
        if fr is None:
            raise Stop("จับภาพหน้าจอไม่ได้ (เช็คสิทธิ์รูท / su)")
        return fr

    def _drag_axis(self, d, axis):
        if axis == "x":
            self.dev.camera_drag(d, 0, CAM_DRAG_MS)
        else:
            self.dev.camera_drag(0, d, CAM_DRAG_MS)

    def drag(self, d, axis=None):
        """ลากกล้อง d px ตามแกน -> (ฉากขยับแนวนอน px, แนวตั้ง px)"""
        a = scene_gray(self._snap())
        self._drag_axis(d, axis or self.axis)
        nap(0.35)
        dx, dy, _ = scene_shift(a, scene_gray(self._snap()))
        return dx, dy

    def try_calibrate(self, verbose=True):
        """ลองลากสองแกน (ซ้าย-ขวา แล้วขึ้น-ลง) หาแกนที่ทำให้ฉากเลื่อนแนวนอน (= หมุนกล้อง)"""
        for axis in ("x", "y"):
            rs, pitch = [], 0.0
            for d in (-140, 140, -140, 140):
                dx, dy = self.drag(d, axis)
                pitch = max(pitch, abs(dy))
                if verbose:
                    log(f"[กล้อง] ลาก{'ซ้าย-ขวา' if axis == 'x' else 'ขึ้น-ลง'} {d:+d}px -> ฉากขยับ แนวนอน {dx:+.0f} แนวตั้ง {dy:+.0f}")
                if abs(dx) >= CAM_MIN_SHIFT and abs(dx) >= 1.5 * abs(dy):
                    rs.append(dx / d)
            pos, neg = [r for r in rs if r > 0], [r for r in rs if r < 0]
            if len(rs) >= 3 and (not pos or not neg):
                self.ratio, self.axis, self.ok = float(np.median(rs)), axis, True
                return True
            if verbose and pitch >= CAM_MIN_SHIFT:
                log(f"[กล้อง] แกน{'ซ้าย-ขวา' if axis == 'x' else 'ขึ้น-ลง'} = เงย/ก้ม (ฉากขยับแนวตั้ง {pitch:.0f}px) ไม่ใช่หมุนซ้ายขวา")
        return False

    def calibrate(self, backends=None):
        """ลองหลายวิธีสั่งลาก (input/sendevent, ช้า/เร็ว) จนกว่ากล้องจะหมุนซ้ายขวา แล้วจำค่าที่ใช้ได้"""
        global CAM_DRAG_MS
        combos = backends or [("input", 380), ("input", 700), ("sendevent", 600), ("input", 200)]
        for be, ms in combos:
            TOUCH_BACKEND["camera"] = be
            CAM_DRAG_MS = ms
            log(f"[กล้อง] ลองวิธี {be} ลาก {ms}ms")
            if self.try_calibrate():
                log(f"[กล้อง] ✅ หมุนซ้ายขวาได้ ({be}, {ms}ms, ลากแนว{'นอน' if self.axis == 'x' else 'ตั้ง'}) ลาก 100px -> ฉากขยับ {self.ratio * 100:+.0f}px")
                self.save()
                return True
        log("[กล้อง] ❌ กล้องไม่หมุนซ้ายขวาทุกวิธี -> บอทจะหามองหินด้วยการเดินแทน")
        TOUCH_BACKEND["camera"] = "none"
        self.ok = False
        return False

    def save(self):
        os.makedirs(DEBUG_DIR, exist_ok=True)
        with open(CAM_FILE, "w") as f:
            json.dump({"v": 2, "backend": TOUCH_BACKEND["camera"], "ms": CAM_DRAG_MS, "ratio": self.ratio, "axis": self.axis}, f)

    def load(self):
        global CAM_DRAG_MS
        try:
            d = json.load(open(CAM_FILE))
            if d.get("v") != 2:
                return False                        # ไฟล์เก่า (ยังไม่รู้เรื่องแกน) -> วัดใหม่
            TOUCH_BACKEND["camera"], CAM_DRAG_MS, self.ratio, self.axis = d["backend"], d["ms"], d["ratio"], d["axis"]
            self.ok = abs(self.ratio) > 1e-3 and TOUCH_BACKEND["camera"] != "none"
            return self.ok
        except Exception:
            return False

    def verify(self):
        """หมุนกล้องซ้าย-ขวาจริงเพื่อเช็กว่ายังหมุนได้ (ทุกครั้งที่เริ่ม) แล้วหมุนกลับที่เดิม -> True/False"""
        rs = []
        for d in (-110, 110):
            dx, dy = self.drag(d)
            ui(f"🎥 หมุนทดสอบ {'ซ้าย' if d < 0 else 'ขวา'} → ฉากขยับ {dx:+.0f}px")
            if abs(dx) >= CAM_MIN_SHIFT and abs(dx) >= 1.5 * abs(dy):
                rs.append(dx / d)
        if len(rs) == 2 and rs[0] * rs[1] > 0:
            self.ratio = float(np.median(rs))
            return True
        return False

    def ensure(self):
        if getattr(self.dev, "dry", False):
            return False
        if self.load():
            log(f"[กล้อง] ใช้ค่าที่วัดไว้ ({TOUCH_BACKEND['camera']}, {CAM_DRAG_MS}ms, แกน {self.axis}, อัตรา {self.ratio:+.2f})")
            if self.verify():
                return True
            ui("🎥 ค่าที่จำไว้ใช้ไม่ได้ → วัดกล้องใหม่")
            self.ok = False
        return self.calibrate()

    def sweep_left(self):
        """หมุนกล้องไปทางซ้าย 1 ก้าว (ฉากเลื่อนไปทางขวา) -> ฉากขยับจริงกี่ px"""
        d = math.copysign(min(CAM_DRAG_PX, CAM_MAX_SHIFT / abs(self.ratio)), self.ratio)
        sh, _ = self.drag(d)
        if abs(sh) < CAM_MIN_SHIFT:
            log("[กล้อง] ลากแล้วฉากไม่หมุน")
        return sh

    def tilt_down(self, px=NAME_TILT_PX):
        """ก้มกล้องลง 1 ระยะ (ลากแกนตั้งฉากกับแกนหมุนซ้ายขวา) ทิศวัดเอง: ก้ม = ฉากเลื่อนขึ้น -> คืนระยะที่ลากจริง (มีเครื่องหมาย) หรือ 0 ถ้าทำไม่ได้"""
        pa = "y" if self.axis == "x" else "x"
        sign = getattr(self, "pitch_sign", None) or 1
        for _ in range(2):
            a = scene_gray(self._snap())
            self._drag_axis(sign * px, pa)
            nap(0.35)
            _, dy, _ = scene_shift(a, scene_gray(self._snap()))
            log(f"[กล้อง] ก้ม ลาก {sign * px:+d}px -> ฉากขยับแนวตั้ง {dy:+.0f}")
            if dy < -CAM_MIN_SHIFT:
                self.pitch_sign = sign
                return sign * px
            self._drag_axis(-sign * px, pa)               # ทิศผิด/ไม่ขยับ: ลากกลับแล้วลองทิศตรงข้าม
            nap(0.3)
            if abs(dy) < CAM_MIN_SHIFT:
                return 0
            sign = -sign
        return 0

    def tilt_restore(self, done):
        """เงยกลับจุดเดิมหลัง tilt_down"""
        if done:
            self._drag_axis(-done, "y" if self.axis == "x" else "x")
            nap(0.3)

    def turn_left_blind(self, shift_px):
        """หมุนกล้องไปทางซ้ายให้ฉากเลื่อนประมาณ shift_px โดยไม่วัดซ้ำ (เร็ว) ลากทีละไม่เกิน CAM_DRAG_PX*1.5"""
        total = shift_px / abs(self.ratio)
        chunk = CAM_DRAG_PX * 1.5
        sgn = 1 if self.ratio > 0 else -1
        while total > 1:
            d = min(chunk, total)
            self._drag_axis(sgn * d, self.axis)
            total -= d
            nap(0.15)

    def turn_by(self, shift_px):
        """หมุนให้ฉากขยับ shift_px (บวก = กล้องหันซ้าย / ลบ = หันขวา) -> ฉากขยับจริงกี่ px"""
        shift_px = max(-CAM_MAX_SHIFT, min(CAM_MAX_SHIFT, shift_px))
        d = shift_px / self.ratio
        d = max(-CAM_DRAG_PX, min(CAM_DRAG_PX, d))
        if abs(d) < 12:
            d = math.copysign(12, d)
        sh, _ = self.drag(d)
        return sh


class Nav:
    """เดินไปตามพิกัด: เรียนรู้ก่อนว่า 'ดันจอยขึ้น/ขวา' พิกัดเปลี่ยนทางไหนเท่าไร (กล้องไม่หมุน ทิศจึงคงที่)"""

    def __init__(self, dev):
        self.dev = dev
        self.last = None
        self.M = None            # คอลัมน์ f (เดินหน้า), r (เดินขวา) หน่วยพิกัดต่อการดัน CAL_MS
        self.speed = None        # หน่วยพิกัดต่อ ms

    def grab(self):
        fr = self.dev.grab()
        if fr is None:
            raise Stop("จับภาพหน้าจอไม่ได้ (เช็คสิทธิ์รูท / su)")
        return fr

    def read(self, tries=4):
        for _ in range(tries):
            c = vote_coords([read_coords_once(self.grab()) for _ in range(3)])
            if c and (self.last is None or math.hypot(c[0] - self.last[0], c[1] - self.last[1]) <= COORD_MAX_JUMP):
                self.last = c
                return c
            nap(0.4)
        return None

    def read_stable(self, timeout=5.0):
        """พิกัดบนจอดีเล: อ่านซ้ำจนสองครั้งติดกันเท่ากัน"""
        t0, prev = time.time(), None
        nap(COORD_SETTLE)
        while time.time() - t0 < timeout:
            c = self.read(tries=2)
            if c and c == prev:
                return c
            prev = c or prev
            nap(0.5)
        return prev

    def calibrate(self):
        p0 = self.read_stable()
        if not p0:
            log("[พิกัด] อ่านพิกัดไม่ได้ (เช็ค ROI['coords'] ด้วย calibrate / ติดตั้ง tesseract)")
            return False
        self.dev.joystick(0, -1, CAL_MS)
        p1 = self.read_stable()
        self.dev.joystick(1, 0, CAL_MS)
        p2 = self.read_stable()
        if not p1 or not p2:
            return False
        f = (p1[0] - p0[0], p1[1] - p0[1])
        r = (p2[0] - p1[0], p2[1] - p1[1])
        nf, nr = math.hypot(*f), math.hypot(*r)
        det = f[0] * r[1] - r[0] * f[1]
        log(f"[พิกัด] เรียนรู้ทิศ: เดินหน้า -> พิกัดเปลี่ยน {f} | เดินขวา -> {r}")
        if nf < 2 or nr < 2 or abs(det) < 0.25 * nf * nr:
            log("[พิกัด] เรียนรู้ทิศไม่สำเร็จ (ติดสิ่งกีดขวาง/ขยับน้อย) จะลองใหม่ที่จุดอื่น")
            return False
        self.M = (f, r)
        self.speed = ((nf + nr) / 2.0) / CAL_MS
        return True

    def goto(self, target, tol, label):
        """เดินไปใกล้พิกัด target -> True เมื่อถึง"""
        log(f"[พิกัด] เดินไป{label} {target}")
        bad = 0
        best = None
        for step in range(GOTO_MAX_STEPS):
            pos = self.read_stable()
            if not pos:
                log("[พิกัด] อ่านพิกัดไม่ได้")
                return False
            d = (target[0] - pos[0], target[1] - pos[1])
            dist = math.hypot(*d)
            log(f"[พิกัด] ก้าว {step + 1}: ตอนนี้ {pos} -> เป้า {target} ห่าง {dist:.1f}")
            if dist <= tol:
                log(f"[พิกัด] ถึง{label}แล้ว {pos}")
                return True
            if self.M is None and not self.calibrate():
                self.dev.joystick(random.uniform(-1, 1), random.uniform(-1, 1), 700)   # ขยับหนีที่ติดแล้วลองใหม่
                continue
            (f, r) = self.M
            det = f[0] * r[1] - r[0] * f[1]
            a = (d[0] * r[1] - r[0] * d[1]) / det           # ส่วนที่ต้องเดินหน้า
            b = (f[0] * d[1] - d[0] * f[1]) / det           # ส่วนที่ต้องเดินขวา
            ms = int(min(1800, max(350, 0.8 * dist / max(self.speed, 1e-6))))
            self.dev.joystick(b, -a, ms)
            if best is not None and dist >= best - 0.5:
                bad += 1
                if bad >= 3:
                    log("[พิกัด] ระยะไม่ลดลง -> เรียนรู้ทิศใหม่ (อาจติดสิ่งกีดขวาง/กล้องขยับ)")
                    self.M = None
                    bad = 0
            else:
                bad = 0
            best = dist if best is None else min(best, dist)
        log(f"[พิกัด] เดินไป{label}ไม่ถึงภายใน {GOTO_MAX_STEPS} ก้าว")
        return False


# =============================================================================
# บันทึกเวลา/โปรไฟล์ เพื่อตรวจว่าอ่านเวลา/เลือกรูปหินถูกช่วงไหม
# =============================================================================
def log_time_row(row):
    os.makedirs(DEBUG_DIR, exist_ok=True)
    new = not os.path.exists(TIME_LOG)
    with open(TIME_LOG, "a", encoding="utf-8") as f:
        if new:
            f.write("เวลาจริง,เวลาในเกม,ช่วง,แหล่งเวลา,โปรไฟล์ที่ใช้,แสงฉากบอกว่า,หินต่อโปรไฟล์(dawn/day/night),พิกัด,เหตุการณ์\n")
        f.write(",".join(str(x) for x in row) + "\n")


# =============================================================================
# ตัวบอทวน
# =============================================================================
def seg_dist(fr, pt):
    """ระยะจากจุด pt ถึงตัวละคร (เส้นตรงจากเท้าถึงหัว) -> ใกล้ = หินชิดข้างตัว/ชิดหน้า (ระยะเท้า->ฐานหินไม่พอ เพราะตัวละครยาวราว 220px)"""
    hx, hy = fr.A("char_head")
    fx, fy = fr.A("char_feet")
    cy = min(max(pt[1], hy), fy)
    return math.hypot(pt[0] - hx, pt[1] - cy)


def draw_track(fr, rocks, tgt, off, aligned, dist, lost=False):
    """วาดผลล็อกเป้าลงภาพ: หินทุกก้อน = กรอบเทา, เป้าที่ล็อก = กรอบ+เส้นจากตัวละคร (แดง = ยังไม่ตรง / เขียว = ล็อกและหันตรงแล้ว)"""
    img = fr.img.copy()
    feet = fr.A("char_feet")
    for r in rocks:
        x, y, w, h = r["bbox"]
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (170, 170, 170), 1)
    cx = feet[0]
    for sgn in (-1, 1):                                                      # เส้นทางตรงหน้าตัวละคร (หินต้องอยู่ระหว่างเส้นนี้)
        lx = int(cx + sgn * LANE_HALF * fr.sx)
        cv2.line(img, (lx, 0), (lx, img.shape[0]), (255, 255, 0), 2)
    if tgt is not None:
        col = (0, 220, 0) if aligned else (0, 0, 255)
        x, y, w, h = tgt["bbox"]
        cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), col, 4)
        cv2.line(img, (int(feet[0]), int(feet[1])), (int(tgt["base"][0]), int(tgt["base"][1])), col, 4)
        cv2.circle(img, (int(tgt["base"][0]), int(tgt["base"][1])), 8, col, -1)
        txt = ("LOCK OK  " if aligned else "TURNING  ") + f"dist={dist:.0f} off={off:+.0f}"
    else:
        col, txt = (0, 0, 255), "NO TARGET" if lost else "SEARCH"
    cv2.rectangle(img, (0, img.shape[0] - 40), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
    cv2.putText(img, txt, (12, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 1.0, col, 2)
    return img


def save_live(img):
    if not VIZ:
        return
    try:
        os.makedirs(DEBUG_DIR, exist_ok=True)
        tmp = os.path.join(DEBUG_DIR, "live_tmp.jpg")
        cv2.imwrite(tmp, img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        os.replace(tmp, os.path.join(DEBUG_DIR, "live.jpg"))
    except Exception:
        pass


def start_viewer(port):
    """เว็บดูสดเล็กๆ: เปิด http://127.0.0.1:PORT ในเบราว์เซอร์ จะเห็น debug/live.jpg รีเฟรชทุกวินาที"""
    import http.server
    os.makedirs(DEBUG_DIR, exist_ok=True)
    with open(os.path.join(DEBUG_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write('<meta charset="utf-8"><title>บอทล็อกเป้า</title><body style="margin:0;background:#000">'
                '<img id="i" style="width:100%"><script>setInterval(()=>{i.src="live.jpg?"+Date.now()},700)</script>')

    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=DEBUG_DIR, **k)

        def log_message(self, *a):
            pass
    try:
        srv = http.server.ThreadingHTTPServer(("0.0.0.0", port), H)
    except OSError as e:
        ui(f"⚠️ เปิดเว็บดูสดไม่ได้ ({e})")
        return
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    ui(f"🖥 ดูสด: เปิด http://127.0.0.1:{port} ในเบราว์เซอร์")


def k_report(bot, rnd, rock):
    bot.report_time(f"มองหาหิน รอบ{rnd + 1} {'เจอ' if rock else 'ไม่เจอ'}")
    return rock is not None


class RockLoop:
    def __init__(self, dev, profiles, period_override=None):
        self.dev, self.profiles = dev, profiles
        self.clock = ClockReader()
        self.clock.override = period_override
        self.nav = Nav(dev)
        self.cam = Camera(dev)
        self.fr = None
        self.names, self.period, self._tstr, self._src = ["day"], "day", "--:--", "none"
        self.black = []
        self.search_pt = SEARCH_COORD
        self.box_pt = BOX_COORD
        self.wander_ang, self.wander_left = 0.0, 0
        self._tkey, self._tui = None, 0.0
        self._nosword = self._nocoord = 0
        self._tcoord = time.time()
        self.stats = {"ตีสำเร็จ": 0, "เก็บ": 0, "กลับบ้านสำเร็จ": 0, "รอบ": 0}

    # ---------------------------------------------------------------- พื้นฐาน
    def frame(self, guard=True, fast=False):
        """fast=True: ไม่อ่านนาฬิกา/ไม่เช็กยาม (ใช้ตอนตี/เก็บของที่ต้องเร็ว)"""
        fr = self.dev.grab()
        if fr is None:
            raise Stop("จับภาพหน้าจอไม่ได้ (เช็คสิทธิ์รูท / su)")
        self.fr = fr
        if fast:
            return fr
        self.names, self.period, self._tstr, self._src = self.clock.current(fr)
        if self.period != getattr(self, "_per_shown", None):
            self._per_shown = self.period
            ui(f"🕐 {self._tstr} → {PERIOD_TH[self.period]}")
        if guard and GUARD:
            self.guard(fr)
        return fr

    def fail_stop(self, reason, fr):
        """หยุดทันที + เซฟภาพที่บอทเห็นไว้ดูว่าพังตรงไหน"""
        try:
            os.makedirs(DEBUG_DIR, exist_ok=True)
            cv2.imwrite(os.path.join(DEBUG_DIR, "fail_frame.png"), fr.img)
            for lay in ("A", "B"):
                for k in ("clock", "coords"):
                    roi = fr.crop(LAYOUT_ROI[lay][k])
                    if roi.size:
                        cv2.imwrite(os.path.join(DEBUG_DIR, f"fail_{k}_{lay}.png"), cv2.resize(roi, None, fx=4, fy=4))
        except Exception:
            pass
        ui(f"⛔ {reason} → หยุด")
        ui("📷 ภาพที่เห็น: debug/fail_*.png")
        raise Stop(reason)

    def guard(self, fr):
        """ยามเฝ้า: ไม่เห็นปุ่มตี หรืออ่านพิกัดไม่ได้ติดกันหลายครั้ง = หยุดทันที (กันบอทกดมั่วตอนออกจากเกม/มีหน้าต่างบัง)"""
        ok, sc = sword_present(fr)
        if ok:
            self._nosword = 0
        elif find_pick_all(fr) or find_close_x(fr) or banner_present(fr):
            pass                                              # มีหน้าต่าง/แถบข้อความบังปุ่มตี: ปกติ ไม่นับ
        else:
            self._nosword += 1
            log(f"[ยาม] ไม่เห็นปุ่มตี {self._nosword}/{GUARD_SWORD_FAILS} (คะแนน {sc:.2f})")
            if self._nosword >= GUARD_SWORD_FAILS:
                self.fail_stop("ไม่เห็นปุ่มตี", fr)
        now = time.time()
        if now - self._tcoord >= GUARD_COORD_EVERY:
            self._tcoord = now
            if read_coords_once(fr):
                self._nocoord = 0
            else:
                self._nocoord += 1
                log(f"[ยาม] อ่านพิกัดไม่ได้ {self._nocoord}/{GUARD_COORD_FAILS}")
                if self._nocoord >= GUARD_COORD_FAILS:
                    self.fail_stop("อ่านพิกัดไม่ได้", fr)

    def startup_check(self):
        """ก่อนเริ่ม: ต้องเห็นปุ่มตีและอ่านพิกัดได้ ไม่งั้นหยุดทันที"""
        if not GUARD:
            return
        if pytesseract is None or not HAS_TESS:
            ui("⛔ ยังไม่ได้ลง pytesseract → รัน: pip install pytesseract")
            raise Stop("ไม่มี pytesseract (อ่านนาฬิกา/พิกัดไม่ได้)")
        why, fr = "", None
        for k in range(4):
            fr = self.frame(guard=False)
            s_ok, _ = sword_present(fr)
            c = vote_coords([read_coords_once(fr)] + [read_coords_once(self.frame(guard=False)) for _ in range(2)]) if s_ok else None
            if s_ok and c:
                ui(f"✅ เห็นปุ่มตี + พิกัด {c[0]},{c[1]}")
                if not COORD_REFS:
                    ui("💡 อ่านเลขพิกัดเพี้ยน? ใส่พิกัดจริง: run --pos X,Y (จำไว้ใช้ครั้งต่อไป)")
                self._tcoord = time.time()
                return
            why = "ไม่เห็นปุ่มตี (ไม่ได้อยู่ในเกม?)" if not s_ok else "อ่านพิกัดไม่ได้"
            nap(1.0)
        self.fail_stop(why, fr)

    def blacklisted(self, rock):
        now = time.time()
        self.black = [(p, e) for p, e in self.black if e > now]
        return any(math.hypot(rock["base"][0] - p[0], rock["base"][1] - p[1]) < 60 * self.fr.sx for p, _ in self.black)

    def report_time(self, event):
        """บันทึกเวลาเกมที่อ่านได้/โปรไฟล์ที่ใช้/หินที่แต่ละโปรไฟล์เห็น ลงไฟล์ + หน้าจอแสดงบรรทัดเดียวเมื่อเปลี่ยน"""
        fr = self.fr
        counts = {p: len(find_rocks(fr, [p], self.profiles)[0]) for p in PERIODS}
        vis = visual_period(fr.img)
        best = max(counts, key=counts.get) if any(counts.values()) else "-"
        msg = (f"เวลาเกม {self._tstr} ช่วง={self.period} (แหล่ง {self._src}) ใช้โปรไฟล์ {self.names} | แสงฉาก={vis} | "
               f"หินที่เห็น dawn={counts['dawn']} day={counts['day']} night={counts['night']}")
        log("[เวลา] " + msg)
        key = (self._tstr, self.period, self._src)
        if key != self._tkey and (time.time() - self._tui > 40 or self._tkey is None or key[1:] != self._tkey[1:]):
            self._tkey, self._tui = key, time.time()
            if self._tstr == "--:--":
                ui(f"⚠️ อ่านเวลาไม่ได้ ใช้แสงฉาก={PERIOD_TH.get(vis, vis)}")
            else:
                ui(f"🕐 {self._tstr} {PERIOD_TH[self.period]}")
        if best != "-" and best not in self.names and counts[best] >= 2 and counts[best] > 2 * max(
                [counts[n] for n in self.names if n in counts] + [0]):
            ui(f"⚠️ รูปหินไม่ตรงเวลา ({PERIOD_TH[best]}เจอมากกว่า)")
        log_time_row([time.strftime("%Y-%m-%d %H:%M:%S"), self._tstr, self.period, self._src, "+".join(self.names), vis,
                      f"{counts['dawn']}/{counts['day']}/{counts['night']}", self.nav.last or "", event])

    # ---------------------------------------------------------------- ไปจุดต่างๆ
    def go_search(self):
        if not self.search_pt:
            return True
        ui("📍 ไปจุดมองหา")
        ok = self.nav.goto(self.search_pt, GOTO_TOL, "จุดมองหา")
        if not ok:
            ui("⚠️ ไปจุดมองหาไม่ถึง (อ่านพิกัดไม่ได้?)")
        return ok

    def go_box(self):
        if not self.box_pt:
            log("[กล่อง] ยังไม่ได้ตั้ง BOX_COORD -> ข้ามขั้นไปกล่อง")
            ui("⏭ ข้ามกล่อง (ยังไม่ตั้ง BOX_COORD)")
            return None
        ui("🏠 ไปกล่อง")
        ok = self.nav.goto(self.box_pt, BOX_TOL, "กล่อง")
        for k in range(6):                     # ใกล้กล่องแล้วรูปมือจะขึ้น ถ้ายังไม่ขึ้นขยับเข้าใกล้อีกนิด
            fr = self.frame()
            seen, score = hand_present(fr)
            log(f"[กล่อง] ตรวจรูปมือ คะแนน {score:.2f} (เกณฑ์ {HAND_THRESHOLD}) {'เห็น' if seen else 'ยังไม่เห็น'}")
            if seen:
                self.stats["กลับบ้านสำเร็จ"] += 1
                ui("🏠 ✅ กลับบ้านสำเร็จ")
                self.report_time("กลับบ้านสำเร็จ")
                return True
            self.dev.joystick(random.uniform(-1, 1), random.uniform(-1, 1), 450)
            nap(0.8)
        ui("⚠️ ถึงกล่องแล้วไม่เห็นรูปมือ")
        self.report_time("กลับบ้านไม่สำเร็จ")
        return False

    # ---------------------------------------------------------------- มองหา / เดินเข้าหา / ตี
    def explore(self):
        if self.wander_left <= 0:
            self.wander_ang = random.uniform(-math.pi * 0.8, math.pi * 0.8)
            self.wander_left = random.randint(2, 3)
        self.wander_left -= 1
        self.dev.joystick(math.sin(self.wander_ang), -math.cos(self.wander_ang), 800)
        nap(0.3)

    def _rocks_now(self):
        fr = self.frame()
        rocks, _ = find_rocks(fr, self.names, self.profiles)
        return [r for r in rocks if not self.blacklisted(r)]

    def scan360(self):
        """หมุนรอบตัวแบ่ง CAM_SCAN_SLICES ช่วง: เช็กภาพ -> หมุนซ้าย -> เช็กภาพ ... เจอหิน (ยืนยันซ้ำ 2 เฟรม) = คืนก้อนที่ใกล้สุด"""
        n = CAM_SCAN_SLICES if self.cam.ok else 1
        for k in range(n):
            rocks = self._rocks_now()
            if rocks:
                again = self._rocks_now()                    # กันเห็นผิดเฟรมเดียว
                hit = [r for r in again if math.hypot(r["base"][0] - rocks[0]["base"][0], r["base"][1] - rocks[0]["base"][1]) < 120]
                if hit:
                    ui(f"🪨 เจอหิน {len(again)} ก้อน (ช่วง {k + 1}/{n})")
                    log(f"[หา] ใกล้สุดฐาน={hit[0]['base']}")
                    return hit[0]
                log("[หา] เห็นหินเฟรมเดียว ไม่ยืนยัน -> ข้าม")
            if k < n - 1:
                ui(f"🔄 ไม่เจอ หมุนซ้าย {k + 1}/{n}  (ช่วงละ {360 // n}°)")
                self.cam.turn_left_blind(CAM_SCAN_SHIFT)
                nap(0.3)
        return None

    def find_target(self):
        """หมุนรอบตัว 360° (แบ่งเป็นช่วง) หาหิน เจอแล้วคืนก้อนที่ใกล้สุด ไม่เจอเดินสุ่มสั้นๆ แล้วหมุนหาใหม่"""
        for rnd in range(ROCKS_PER_VISIT):
            ui("🔎 หมุนมองหาหิน 360°" if self.cam.ok else "🔎 มองหาหิน")
            rock = self.scan360()
            if k_report(self, rnd, rock):
                return rock
            ui(f"🔎 ไม่เจอ เดินสุ่ม ({rnd + 1}/{ROCKS_PER_VISIT})")
            self.explore()
            if self.search_pt and self.nav.last:      # กันเดินไกลจากจุดมองหาเกินไป
                c = read_coords_once(self.dev.grab())
                if c and math.hypot(c[0] - self.search_pt[0], c[1] - self.search_pt[1]) > LEASH:
                    ui("📍 ไกลเกิน กลับจุดมองหา")
                    self.go_search()
        return None

    def near_pixels(self, fr):
        """พื้นที่สีหินรอบตัวละคร (px อ้างอิง) ใช้ดูว่าหินใหญ่จนชิด/บังตัวละครแล้วไหม"""
        m = rock_mask(fr.img, self.names, self.profiles)
        cx1, cy1, cx2, cy2 = fr.box(CHAR_EXCLUDE)             # ผม/ผิว/กระโปรงตัวละครไม่ใช่หิน
        m[max(0, cy1):cy2, max(0, cx1):cx2] = 0
        x1, y1, x2, y2 = fr.box(NEAR_BOX)
        sub = cv2.morphologyEx(m[y1:y2, x1:x2], cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return float((sub > 0).sum()) / (fr.sx * fr.sy)

    def _arrived(self, fr):
        feet = fr.A("char_feet")
        return {"base": feet, "bbox": (feet[0] - 40, feet[1] - 80, 80, 80), "area": 0, "dist": 0.0, "near": True}

    def sweep_near(self):
        """เดินไม่ขยับ: หมุนกล้องไปทางขวาทีละนิด ดูว่ามีหินชิดตัวละครไหม (บางทีหินอยู่ข้างตัว/ถูกบัง) แล้วหมุนกลับจุดเดิม
        คืน frame ที่ชิดหิน หรือ None ถ้าไม่ชิดหินเลย (คือติดสิ่งกีดขวางจริง)"""
        if not self.cam.ok:
            return None
        ui("👀 เดินไม่ขยับ → ส่ายกล้องขวาดูว่าชิดหินไหม")
        total, hit = 0.0, None
        for i in range(SWEEP_STEPS):
            sh = self.cam.turn_by(-SWEEP_STEP_PX)
            total += sh or 0.0
            fr = self.frame()
            feet = fr.A("char_feet")
            rr, _ = find_rocks(fr, self.names, self.profiles)
            close = [r for r in rr if math.hypot(r["base"][0] - feet[0], r["base"][1] - feet[1]) <= NEAR_TRUST_DIST * fr.sy]
            np_ = self.near_pixels(fr)
            log(f"[ส่าย] ก้าว {i + 1}: หินใกล้ {len(close)} พิกเซลหิน {np_:.0f}")
            if close and np_ >= NEAR_PIX:
                hit = fr
                break
        if abs(total) > 8:
            self.cam.turn_by(-total)                            # หมุนกลับที่เดิม
            nap(0.2)
        if hit is not None:
            ui("📍 ชิดหินแล้ว (เจอตอนส่ายกล้อง)")
            return self.frame()
        ui("🧱 ไม่ชิดหิน = มีของขวางจริง")
        return None

    def _turn_back(self, total):
        """หมุนกล้องกลับที่เดิม (ฉากต้องขยับ -total)"""
        rem = -total
        for _ in range(4):
            if abs(rem) < 20:
                break
            got = self.cam.turn_by(max(-CAM_MAX_SHIFT, min(CAM_MAX_SHIFT, rem))) or 0.0
            if abs(got) < 1:
                break
            rem -= got

    def peek_sides(self, fr, rocks, tgt, dist):
        """เดินไปหาเป้าแล้ว ส่องหินซ้าย/ขวา: หันกล้องให้หินข้างๆ มาอยู่ตรงหน้า (ตอนมองตรงจะเห็นระยะจริง) ถ้าใกล้กว่าเป้า = เปลี่ยนเป้า
        ไม่ใกล้กว่า = หมุนกลับที่เดิม -> คืนหินก้อนใหม่ (กล้องหันตรงอยู่แล้ว) หรือ None"""
        feet = fr.A("char_feet")
        band = CAM_CENTER_BAND * fr.sx
        tb = tgt["base"]
        cands = []
        for sgn in (-1, 1):
            c = [r for r in rocks if r is not tgt and not self.blacklisted(r)
                 and sgn * (r["bbox"][0] + r["bbox"][2] / 2.0 - feet[0]) > band * ALIGN_OUT
                 and math.hypot(r["base"][0] - tb[0], r["base"][1] - tb[1]) > 60 * fr.sx
                 and r["dist"] < 2.5 * max(dist, 150 * fr.sy)]
            if c:
                cands.append(min(c, key=lambda r: abs(r["bbox"][0] + r["bbox"][2] / 2.0 - feet[0])))
        for cand in cands:
            side = "ซ้าย" if cand["bbox"][0] + cand["bbox"][2] / 2.0 < feet[0] else "ขวา"
            ui(f"👀 ส่องหินด้าน{side}ว่าใกล้กว่าไหม")
            total, last, found = 0.0, cand["base"], cand
            off = cand["bbox"][0] + cand["bbox"][2] / 2.0 - feet[0]
            for _ in range(4):
                sh = self.cam.turn_by(max(-CAM_MAX_SHIFT, min(CAM_MAX_SHIFT, -off * ALIGN_GAIN))) or 0.0
                total += sh
                last = (last[0] + sh, last[1])
                f2 = self.frame()
                rr, _ = find_rocks(f2, self.names, self.profiles)
                near = sorted((r for r in rr if not self.blacklisted(r) and math.hypot(r["base"][0] - last[0], r["base"][1] - last[1]) < 1.3 * LOCK_RADIUS * f2.sx),
                              key=lambda r: math.hypot(r["base"][0] - last[0], r["base"][1] - last[1]))
                if not near:
                    found = None
                    break
                found, last = near[0], near[0]["base"]
                off = found["bbox"][0] + found["bbox"][2] / 2.0 - f2.A("char_feet")[0]
                if abs(off) <= band * ALIGN_IN:
                    break
            if found is not None:
                fd = math.hypot(found["base"][0] - f2.A("char_feet")[0], found["base"][1] - f2.A("char_feet")[1])
                save_live(draw_track(f2, rr, found, off, abs(off) <= band, fd))
                if fd < PEEK_NEARER * dist and abs(off) <= band * ALIGN_OUT:
                    ui(f"🔄 หินด้าน{side}ใกล้กว่า ({fd:.0f} < {dist:.0f}) เปลี่ยนเป้า")
                    return found
                ui(f"↩ หินด้าน{side}ไม่ใกล้กว่า ({fd:.0f} vs {dist:.0f}) หมุนกลับ")
            self._turn_back(total)
            fr = self.frame()
            rocks, _ = find_rocks(fr, self.names, self.profiles)
        return None

    def unstick(self, n, off):
        """ติดต้นไม้/หิน/พุ่มไม้: เบี่ยงข้างสลับซ้ายขวา เดินอ้อม ถ้ายังติดถอยหลังก่อน ติดเกิน UNSTICK_MAX = ยอมแพ้ก้อนนี้"""
        if n >= UNSTICK_MAX:
            ui("🧱 ติดไม่หลุด ข้ามก้อนนี้")
            return False
        sgn = (1 if off >= 0 else -1) * (1 if n % 2 == 1 else -1)
        ui(f"🧱 ติดสิ่งกีดขวาง เบี่ยง{'ขวา' if sgn > 0 else 'ซ้าย'} {n}/{UNSTICK_MAX}")
        if n >= 3:
            self.dev.joystick(0, 1, 500)                       # ถอยออกมาก่อน
            nap(0.15)
        self.dev.joystick(sgn, 0, 600 + 250 * n)
        nap(0.15)
        self.dev.joystick(0, -1, 700)                          # เดินหน้าผ่านสิ่งกีดขวาง
        nap(0.15)
        return True

    def turn_to_lane(self, fr, tgt):
        """หมุนกล้องให้หินที่ล็อกเข้ามาอยู่ในเส้น (กลางจอ ±LANE_IN) หมุนเต็มระยะที่เยื้อง ไม่หมุนสั้น
        -> (frame, หินก้อนที่ล็อก หรือ None ถ้าเสียเป้า, อยู่ในเส้นแล้วไหม)"""
        for i in range(ALIGN_MAX_TURNS):
            feet = fr.A("char_feet")
            off = tgt["bbox"][0] + tgt["bbox"][2] / 2.0 - feet[0]
            if abs(off) <= LANE_IN * fr.sx:
                return fr, tgt, True
            cmd = max(-ALIGN_STEP_CAP, min(ALIGN_STEP_CAP, -off * ALIGN_GAIN))
            sh = self.cam.turn_by(cmd) or 0.0
            if abs(sh) < 4:                                      # วัดฉากขยับไม่ได้ (พื้นเรียบ) สมมติว่าขยับตามสั่ง
                sh = cmd * 0.8
            exp = (tgt["base"][0] + sh, tgt["base"][1])
            fr = self.frame()
            rr, _ = find_rocks(fr, self.names, self.profiles)
            cand = sorted((r for r in rr if not self.blacklisted(r) and math.hypot(r["base"][0] - exp[0], r["base"][1] - exp[1]) < 2.0 * LOCK_RADIUS * fr.sx),
                          key=lambda r: math.hypot(r["base"][0] - exp[0], r["base"][1] - exp[1]))
            if not cand:
                return fr, None, False
            tgt = cand[0]
            off2 = tgt["bbox"][0] + tgt["bbox"][2] / 2.0 - fr.A("char_feet")[0]
            log(f"[หมุน] ครั้ง {i + 1}: สั่ง {cmd:+.0f} ฉากขยับ {sh:+.0f} เยื้องเหลือ {off2:+.0f}")
            save_live(draw_track(fr, rr, tgt, off2, abs(off2) <= LANE_IN * fr.sx, math.hypot(*(np.array(tgt["base"]) - np.array(fr.A("char_feet"))))))
        feet = fr.A("char_feet")
        off = tgt["bbox"][0] + tgt["bbox"][2] / 2.0 - feet[0]
        return fr, tgt, abs(off) <= LANE_HALF * fr.sx

    def approach(self, rock):
        """เดินเข้าหาหิน: หินซ้าย/ขวา -> หมุนกล้อง, อยู่กลาง -> เดินตรง, ติดของขวาง -> เบี่ยง, ชิดแล้ว (หินใหญ่บังตัว/หายตอนชิด) -> ตี"""
        d, lost, last = self.dev, 0, rock["base"]
        blocked = no_prog = stalls = walks = peeks = 0
        last_dist = last_seg = None
        last_inlane = False
        side_tries = 0
        for step in range(APPROACH_MAX_STEPS):
            fr = self.frame()
            feet = fr.A("char_feet")
            if step >= 1 and last_dist is not None and last_dist <= NEAR_TRUST_DIST * fr.sy and self.near_pixels(fr) >= NEAR_PIX:
                ui("📍 ชิดหินแล้ว")
                return self._arrived(fr)
            rocks, _ = find_rocks(fr, self.names, self.profiles)
            rocks = [r for r in rocks if not self.blacklisted(r)]
            same = sorted((r for r in rocks if math.hypot(r["base"][0] - last[0], r["base"][1] - last[1]) < LOCK_RADIUS * fr.sx),
                          key=lambda r: math.hypot(r["base"][0] - last[0], r["base"][1] - last[1]))
            tgt = same[0] if same else None                      # ล็อกก้อนเดิมเท่านั้น ไม่สลับไปก้อนอื่น
            if tgt is None:
                lost += 1
                save_live(draw_track(fr, rocks, None, 0, False, 0, lost=True))
                ui(f"🔴 เสียเป้า ({lost}/3)")
                if last_seg is not None and last_seg <= 1.8 * ATTACK_TRY_DIST * fr.sy and last_inlane:
                    ui("📍 หินหายตอนชิดตัว (ตัวละครบังหิน) → ลองกดตี")
                    return self._arrived(fr)
                if last_dist is not None and last_dist <= LOST_NEAR_DIST * fr.sy and self.near_pixels(fr) >= LOST_NEAR_PIX:
                    ui("📍 ชิดหินแล้ว (หินบังกล้อง)")
                    return self._arrived(fr)
                if lost >= 3:
                    ui("⚠️ หินหลุดจอ")
                    return None
                d.joystick(0, -1, 400)
                continue
            lost, last = 0, tgt["base"]
            dx, dy = tgt["base"][0] - feet[0], tgt["base"][1] - feet[1]
            dist = math.hypot(dx, dy)
            seg = seg_dist(fr, tgt["base"])
            last_seg = seg
            ms = int(min(STEP_MS_MAX, max(STEP_MS_MIN, dist * (3.0 if dist > 200 * fr.sy else 1.6))))
            if seg < 160 * fr.sy:                                # ใกล้แล้ว เดินสั้นๆ ให้ชิดเท้าพอดี ไม่เลยหิน
                ms = int(min(ms, max(250, seg * WALK_MS_PER_PX * 0.8)))
            cx = tgt["bbox"][0] + tgt["bbox"][2] / 2.0
            off = cx - feet[0]                                   # + = หินอยู่ขวาของกลางจอ
            band = CAM_CENTER_BAND * fr.sx
            side = "ซ้าย" if off < -LANE_HALF * fr.sx else ("ขวา" if off > LANE_HALF * fr.sx else "กลาง")
            inlane = (not self.cam.ok) or abs(off) <= LANE_HALF * fr.sx
            last_inlane = inlane
            if inlane and seg <= ATTACK_TRY_DIST * fr.sy:        # ชิดตัว + อยู่หน้าตัวละคร (ในเส้น) ถึงกดตีได้
                save_live(draw_track(fr, rocks, tgt, off, True, dist))
                ui("📍 ชิดตัวแล้ว อยู่ในเส้น → ลองกดตี")
                return tgt
            if side_tries < 2:                                   # หินข้างๆ ที่ชิดตัวอยู่แล้ว: ลองตีเลย (ตัวละครจะวิ่งเข้าหาเอง)
                sn = [r for r in rocks if r is not tgt and seg_dist(fr, r["base"]) <= SIDE_TRY_DIST * fr.sy]
                if sn:
                    side_tries += 1
                    sn.sort(key=lambda r: seg_dist(fr, r["base"]))
                    save_live(draw_track(fr, rocks, sn[0], 0.0, True, 0.0))
                    ui("📍 เห็นหินชิดข้างตัว → ลองกดตี")
                    return sn[0]
            if self.cam.ok:
                log(f"[เดิน] ก้าว {step + 1}: หินอยู่ด้าน{side} (เยื้อง {off:+.0f}px) ห่าง {dist:.0f}px")
                if not inlane:
                    save_live(draw_track(fr, rocks, tgt, off, False, dist))
                    ui(f"🔴 หินอยู่{side} {abs(off):.0f}px นอกเส้น → หมุนกล้องให้เข้าเส้น")
                    fr, nt, ok = self.turn_to_lane(fr, tgt)
                    if nt is None:
                        lost += 1
                        ui(f"🔴 เสียเป้าตอนหมุน ({lost}/3)")
                        if lost >= 3:
                            return None
                        last_dist = None
                        continue
                    last, last_dist = nt["base"], None
                    continue
                walks += 1
                if peeks < PEEK_MAX and walks % PEEK_EVERY == 1 and seg > ATTACK_TRY_DIST * 1.5 * fr.sy:
                    peeks += 1
                    new = self.peek_sides(fr, rocks, tgt, dist)
                    if new is not None:                          # เจอหินข้างๆ ที่ใกล้กว่า = เปลี่ยนเป้า (หันตรงอยู่แล้ว)
                        last, last_dist = new["base"], None
                        continue
                save_live(draw_track(fr, rocks, tgt, off, True, dist))
                ui(f"🟢 อยู่ในเส้นแล้ว → เดินตรง ห่าง {dist:.0f}")
                dx, dy = 0.0, -1.0
            else:
                ui(f"🚶 เดิน ห่าง {dist:.0f}")
                log(f"[เดิน] dx={dx:+.0f} dy={dy:+.0f} นาน {ms}ms (กล้องไม่หมุน ใช้จอยทิศตรง)")
            before = fr.img
            d.joystick(dx, dy, ms)
            nap(0.15)
            fr2 = d.grab()
            moved = fr2 is not None and frame_diff(before, fr2.img) >= STUCK_FRAME_DIFF
            if last_dist is not None and dist > last_dist - max(2.5, 0.015 * dist) * fr.sy:    # หินไกล: ระยะในภาพลดช้าตามธรรมชาติ
                no_prog += 1
            else:
                no_prog = 0
            last_dist = dist
            stalls = 0 if moved else stalls + 1
            if stalls >= STALL_NEED or no_prog >= 5:
                blocked += 1
                no_prog = stalls = 0
                fr_n = self.sweep_near()
                if fr_n is not None:
                    return self._arrived(fr_n)
                if not self.unstick(blocked, off):
                    return None
                last_dist = None
        return None

    def hit_once(self, rock):
        """ตีแล้วดูแถบชื่อ -> 'ore' (ตีสำเร็จ) | 'unknown' (ตีโดนแต่ชื่อไม่ชัด) | 'bad' (ไม่ใช่แร่) | 'miss'"""
        d = self.dev
        for t in range(ENGAGE_TRIES):
            fr = self.frame()
            d.hold(*fr.A("sword"), 1000)
            nap(0.4)
            fr = self.frame()
            bar = read_target_bar(fr)
            if bar["engaged"]:
                name = read_target_name(fr, bar)
                v = name_verdict(name, self.period)
                if v == "unknown" and NAME_TILT and self.cam.ok:
                    v, name = self.reread_name_tilted(name, v)
                short = (name.split("|")[0].strip() or "?")[:14]
                ui(f"⚔️ ตี: {short.title()} เลือด {bar['hp'] * 100:.0f}%")
                log(f"[ตี] ชื่อที่อ่านได้ '{name}' -> {v}")
                return v
            log(f"[ตี] ยังไม่โดนเป้า ({t + 1}/{ENGAGE_TRIES}) -> ขยับเข้าใกล้อีกนิด")
            rocks, _ = find_rocks(fr, self.names, self.profiles)
            rocks = [r for r in rocks if not self.blacklisted(r)]
            if rocks and self.cam.ok:
                offr = rocks[0]["bbox"][0] + rocks[0]["bbox"][2] / 2.0 - fr.A("char_feet")[0]
                if abs(offr) > LANE_HALF * fr.sx:                # หินไม่อยู่หน้าตัวละคร: หมุนกล้องให้เข้าเส้นก่อนตีใหม่
                    ui(f"🔴 ตียังไม่โดน หินอยู่ข้าง {abs(offr):.0f}px → หมุนกล้องให้เข้าเส้น")
                    self.turn_to_lane(fr, rocks[0])
                    nap(0.2)
                    continue
            if rocks:
                feet = fr.A("char_feet")
                dx, dy = rocks[0]["base"][0] - feet[0], rocks[0]["base"][1] - feet[1]
                dist = math.hypot(dx, dy)
                ui(f"⚔️ ตียังไม่โดน ({t + 1}/{ENGAGE_TRIES}) เข้าไปอีก {dist:.0f}")
                ms = int(min(1500, max(400, seg_dist(fr, rocks[0]["base"]) * WALK_MS_PER_PX * 0.6)))
                d.joystick(dx, dy, ms)
            else:
                ui(f"⚔️ ตียังไม่โดน ({t + 1}/{ENGAGE_TRIES}) ดันตรง")
                d.joystick(0, -1, 350)                       # หินชิด/บังตัว: ดันไปข้างหน้าอีกนิดแล้วตีใหม่
            nap(0.2)
        ui("⚠️ ตีไม่โดนเป้า")
        return "miss"

    def reread_name_tilted(self, name, v):
        """อ่านชื่อไม่ออก: ก้มกล้องลง 1 ระยะ (พื้นหลังหลังแถบชื่อเปลี่ยนจากท้องฟ้าเป็นพื้นหญ้า) อ่านใหม่ แล้วเงยกลับที่เดิม"""
        ui("🎥 อ่านชื่อไม่ออก → ก้มกล้องลงอ่านใหม่")
        done = self.cam.tilt_down()
        if not done:
            ui("🎥 ก้มกล้องไม่ได้ ข้าม")
            return v, name
        try:
            nap(0.3)
            for _ in range(3):
                fr = self.frame(fast=True)
                bar = read_target_bar(fr)
                if not bar["engaged"]:
                    continue
                n2 = read_target_name(fr, bar)
                v2 = name_verdict(n2, self.period)
                log(f"[ชื่อ] หลังก้มกล้อง '{n2}' -> {v2}")
                if v2 != "unknown":
                    ui("🎥 อ่านชื่อได้แล้ว")
                    return v2, n2
        finally:
            self.cam.tilt_restore(done)
            ui("🎥 เงยกล้องกลับที่เดิม")
        return v, name

    def chop_until_empty(self):
        """ตีรัวจนเลือดเป้าหมดหมด -> 'empty' | 'gone' (แถบหาย) | 'stuck' (เลือดไม่ลด) | 'timeout'
        แตะดาบรัว BURST_TAPS ครั้งแล้วค่อยอ่านเลือด (ไม่ใช้นาฬิกา/ยามในลูปนี้ให้เร็ว) เหนื่อย = พักแล้วตีต่อ"""
        d, t0, last_hp, stale, nudges, lost = self.dev, time.time(), None, 0, 0, 0
        slot_done = False
        while time.time() - t0 < HIT_TIMEOUT:
            fr = self.frame(fast=True)
            if banner_present(fr):
                ui(f"😮‍💨 เหนื่อย พัก {TIRED_REST_SEC}วิ")
                nap(TIRED_REST_SEC)
                continue
            bar = read_target_bar(fr)
            if last_hp is not None and last_hp <= SINGLE_TAP_HP + 0.05 and find_hand(fr):
                ui("🖐 เห็นรูปมือ = แร่แตกแล้ว หยุดตี")
                return "empty"
            if bar["engaged"] and last_hp is not None and bar["hp"] >= last_hp + JUMP_UP:
                ui("⛏ เลือดพุ่งกลับ = ก้อนเดิมแตกแล้ว (ระบบล็อกก้อนใหม่) หยุดตี")
                return "empty"
            if not bar["engaged"]:
                if last_hp is not None and last_hp <= 0.20:
                    ui("⛏ แถบเป้าหาย (เลือดใกล้หมด) = หินแตกแล้ว")
                    return "gone"
                lost += 1                                    # เลือดยังเหลือเยอะแต่แถบหาย: น่าจะอ่านพลาด ตีต่อก่อน
                log(f"[ตี] แถบเป้าหายตอนเลือดเหลือ {0 if last_hp is None else last_hp * 100:.0f}% ({lost}/{BAR_LOST_RETRY})")
                if lost >= BAR_LOST_RETRY:
                    ui("⚠️ แถบเป้าหาย ทั้งที่เลือดยังเหลือ")
                    return "lost"
                d.burst(*fr.A("sword"), 1)
                nap(0.2)
                continue
            lost = 0
            if bar["hp"] <= 0.02 and (last_hp is None or last_hp > 0.25):
                fr = self.frame(fast=True)                   # เลือดกระโดดจากเยอะเป็น 0 = อาจอ่านพลาด ดูซ้ำ
                bar = read_target_bar(fr)
                if not bar["engaged"]:
                    continue
            if bar["hp"] <= 0.02:
                ui("⛏ เลือดหมดแล้ว")
                return "empty"
            ui(f"⛏ ตีรัว เลือด {bar['hp'] * 100:.0f}%")
            if last_hp is not None and bar["hp"] >= last_hp - 0.01:
                stale += 1
            else:
                stale = 0
            last_hp = bar["hp"]
            if stale >= STALE_CYCLES:
                nudges += 1
                stale = 0
                if nudges > 3:
                    ui("⚠️ ตีแล้วเลือดไม่ลด")
                    return "stuck"
                if pickaxe_state(fr) == "broken":
                    self.repair_pickaxe()
                    continue
                ui("⛏ เลือดไม่ลด ขยับเข้าไปอีกนิด")
                d.joystick(0, -1, 300)
            if SLOT1_AT_HP and not slot_done and bar["hp"] <= SLOT1_AT_HP:
                slot_done = True                                 # เลือดเหลือน้อย: กดช่อง 1 ครั้งเดียว ช้าๆ แล้วตีปกติต่อ
                ui(f"🔢 เลือดเหลือ ≤{SLOT1_AT_HP * 100:.0f}% → กดช่อง 1 (ช้าๆ 1 ครั้ง)")
                d.hold(*fr.A("pickaxe_slot"), SLOT1_PRESS_MS)
                nap(0.3)
            d.burst(*fr.A("sword"), 1 if bar["hp"] <= SINGLE_TAP_HP else BURST_TAPS)
        ui("⚠️ ตีนานเกินกำหนด")
        return "timeout"

    def collect(self):
        """เห็นรูปมือกดเลย (กดที่รูปมือ ถ้าไม่เห็นกดตำแหน่งปุ่มมือ) -> Pick All (เก็บทุกอย่าง) -> วนจนรูปมือหาย
        เก็บแล้วหน้าต่าง Pick All ไม่ขึ้นก็ไม่เป็นไร: ถ้ารูปมือหายไปแล้วถือว่าเก็บเข้ากระเป๋าแล้ว"""
        d = self.dev
        if COLLECT_DELAY > 0:
            nap(COLLECT_DELAY)
        got = seen = False
        for k in range(PICK_TRIES):
            fr = self.frame(fast=True)
            h = find_hand(fr)
            seen = seen or bool(h)
            if k > 0 and seen and not h and find_pick_all(fr) is None:
                break                                            # รูปมือหายแล้ว = ไม่มีของเหลือ
            d.tap(*(h or fr.A("hand_guess")))
            if k == 0:
                ui("🖐 กดรูปมือ" if h else "🖐 กดตำแหน่งรูปมือ (ตรวจไม่เจอ)")
            nap(PICK_WAIT)
            fr = self.frame(fast=True)
            pa = find_pick_all(fr)
            if pa:
                d.tap(*pa)
                nap(AFTER_PICK_WAIT)
                self.stats["เก็บ"] += 1
                got = True
                ui("🎒 เก็บของแล้ว (Pick All)")
                fr = self.frame(fast=True)
                if bag_state(fr) == "full":
                    raise Stop("กระเป๋าเต็ม")
                if find_hand(fr) and k < PICK_TRIES - 1:
                    continue                                     # ยังมีรูปมือ (ของกองอื่น) กดต่อ
                return True
        if not got:
            fr = self.frame(fast=True)
            if seen and find_hand(fr) is None:
                ui("🎒 เก็บแล้ว (รูปมือหาย ไม่มีหน้าต่าง Pick All)")
                self.stats["เก็บ"] += 1
                return True
            ui("⚠️ ไม่เจอ Pick All (เซฟภาพ debug/nopick.png)")
            try:
                os.makedirs(DEBUG_DIR, exist_ok=True)
                cv2.imwrite(os.path.join(DEBUG_DIR, "nopick.png"), self.fr.img)
            except Exception:
                pass
        return got

    def repair_pickaxe(self):
        """ขวานแดง: กดช่องขวาน -> รอหน้าต่างซ่อมขึ้น -> กด Fix -> รอ -> กด Equip Now (ตำแหน่งปุ่มเดียวกัน) -> ปิดถ้ายังค้าง"""
        d = self.dev
        for attempt in range(REPAIR_MAX_FAILS):
            ui(f"🔧 ขวานแดง ซ่อม ({attempt + 1}/{REPAIR_MAX_FAILS})")
            fr = self.frame(guard=False, fast=True)
            d.tap(*fr.A("pickaxe_slot"))
            for _ in range(6):                                   # รอหน้าต่างขึ้น
                nap(REPAIR_WAIT / 2)
                fr = self.frame(guard=False, fast=True)
                if repair_dialog_open(fr):
                    break
            else:
                ui("⚠️ หน้าต่างซ่อมไม่ขึ้น ลองใหม่")
                continue
            d.tap(*fr.A("repair_btn"))                           # Fix
            nap(REPAIR_WAIT)
            fr = self.frame(guard=False, fast=True)
            if repair_dialog_open(fr):
                d.tap(*fr.A("repair_btn"))                       # Equip Now
                nap(REPAIR_WAIT)
                fr = self.frame(guard=False, fast=True)
            if repair_dialog_open(fr):
                d.tap(*fr.A("repair_close"))                     # ยังค้างอยู่ ปิดหน้าต่าง
                nap(REPAIR_WAIT)
                fr = self.frame(guard=False, fast=True)
            if pickaxe_state(fr) != "broken":
                ui("🔧 ซ่อมขวานแล้ว")
                return True
        raise Stop("ซ่อมขวานไม่สำเร็จ (ไม้ไม่พอ?)")

    def leave(self, rock, why):
        """ไม่ใช่หิน/ตีไม่โดน: ถอยออกจากเป้า เบี่ยงข้าง และจำตำแหน่งไว้ ไม่กลับมาตีอีก"""
        ui(f"🚪 {why} ออกจากเป้า")
        self.dev.joystick(0, 1, 1000)
        nap(0.2)
        self.dev.joystick(random.choice([-1, 1]), 0, 700)
        nap(0.2)
        self.black.append((rock["base"], time.time() + BLACKLIST_SEC))

    def mine_one(self):
        if pickaxe_state(self.frame(guard=False, fast=True)) == "broken":
            self.repair_pickaxe()
        for k in range(ROCK_TRIES_PER_ROUND):
            rock = self.find_target()
            if rock is None:
                return False
            rock = self.approach(rock)
            if rock is None:
                continue
            res = self.hit_once(rock)
            if res in ("ore", "unknown"):
                self.stats["ตีสำเร็จ"] += 1
                ui("✅ ตีโดน ชื่อตรงแร่" if res == "ore" else "✅ ตีโดน (อ่านชื่อไม่ชัด)")
                self.report_time("ตีสำเร็จ" if res == "ore" else "ตีสำเร็จ(ชื่อไม่ยืนยัน)")
                end = self.chop_until_empty()
                if end in ("empty", "gone"):
                    self.collect()
                    return True
                self.leave(rock, "ตีไม่เข้า")
                continue
            self.leave(rock, "ไม่ใช่หิน" if res == "bad" else "ตีไม่โดน")
            self.report_time(f"ตีไม่สำเร็จ:{res}")
        return False

    # ---------------------------------------------------------------- ลูปหลัก
    def setup(self):
        self.frame(guard=False)
        self.startup_check()
        ui("🎥 ตรวจกล้อง...")
        self.cam.ensure()
        ui("🎥 กล้อง: " + (f"ใช้ได้ (ลากแนว{'นอน' if self.cam.axis == 'x' else 'ตั้ง'})" if self.cam.ok else "หมุนไม่ได้ → เดินหาแทน"))
        ui(f"🕐 {self._tstr} {PERIOD_TH[self.period]} ({self._src})")
        if not COORD_REFS:
            self.search_pt = None
            self.box_pt = None
            ui("⏭ ไม่ได้ใส่ --pos → ไม่เดินตามพิกัด (หาหินรอบตัว)")
        elif not self.search_pt:
            c = self.nav.read_stable()
            if c:
                self.search_pt = c
                ui(f"📍 จุดมองหา = ตำแหน่งนี้ {c[0]},{c[1]}")
            else:
                ui("⚠️ อ่านพิกัดไม่ได้ → ไม่เดินตามพิกัด")
        if not self.box_pt:
            log("ไม่ได้ตั้ง BOX_COORD/--pos -> จะข้ามขั้นตอนไปกล่อง")

    def run(self, max_rounds=0):
        self.setup()
        while True:
            self.stats["รอบ"] += 1
            r = self.stats["รอบ"]
            ui(f"━━ รอบ {r} │ ตี {self.stats['ตีสำเร็จ']} │ เก็บ {self.stats['เก็บ']} │ บ้าน {self.stats['กลับบ้านสำเร็จ']} ━━")
            self.go_search()
            ok = self.mine_one()
            if ok:
                self.go_search()
                self.go_box()
            else:
                ui("↩ รอบนี้ยังไม่สำเร็จ เริ่มใหม่")
            self.go_search()
            if max_rounds and r >= max_rounds:
                return


class ImageDevice(Device):
    """ใช้ทดสอบ: ป้อนภาพนิ่งแทนการจับจากเครื่อง และไม่ส่งคำสั่งจริง"""

    def __init__(self, frame):
        super().__init__(dry=True)
        self.fixed = frame
        self.cmds = []

    def sh(self, cmd, binary=False, timeout=25):
        self.cmds.append(cmd)
        return b"" if binary else ""

    def grab(self):
        self.last = self.fixed
        return self.fixed

    def camera_available(self):
        return False


# =============================================================================
# คำสั่งเสริม
# =============================================================================
def countdown(sec, msg):
    log(msg)
    for s in range(sec, 0, -1):
        sys.stdout.write(f"\r{time.strftime('%H:%M:%S')}  เริ่มใน {s} วินาที...   ")
        sys.stdout.flush()
        nap(1)
    sys.stdout.write("\r\n")


def cmd_calibrate(args):
    profiles = load_profiles()
    if args.image:
        raw = cv2.imread(args.image)
        if raw is None:
            sys.exit(f"เปิดไฟล์ภาพไม่ได้: {args.image}")
        fr = Frame(cv2.resize(raw, (REF_W, REF_H)), raw.shape[:2], True) if args.landscape else make_frame(raw)
    else:
        dev = Device(dry=args.dry)
        ok, who = dev.root_ok()
        log(f"รูท: {'OK' if ok else 'ไม่ใช่ -> ' + who}")
        fr = dev.grab()
        if fr is None:
            sys.exit("จับภาพไม่ได้")
    clk = ClockReader()
    clk.update(fr, force=True)
    names, period, tstr, src = clk.current(fr)
    log(f"ภาพดิบ {fr.raw_shape[1]}x{fr.raw_shape[0]} -> ภาพแนวนอน {fr.w}x{fr.h} (หมุน={fr.rotated}) tesseract={'มี' if HAS_TESS else 'ไม่มี'}")
    log(f"เวลาในเกม: {tstr} ช่วง={period} โปรไฟล์ที่ใช้={names} แหล่ง={src} แสงฉาก={visual_period(fr.img)}")
    log(f"พิกัดบนจอ: {read_coords_once(fr)}")
    seen, score = hand_present(fr)
    log(f"รูปมือ: {'เห็น' if seen else 'ไม่เห็น'} คะแนน {score:.2f} (ไอคอนมือที่ใช้ {len(hand_masks())} แบบ)")
    bar = read_target_bar(fr)
    log(f"แถบเป้าหมาย: ขึ้น={bar['engaged']} เลือด={bar['hp']:.2f} ชื่อ='{read_target_name(fr, bar)}'")
    counts = {p: len(find_rocks(fr, [p], profiles)[0]) for p in PERIODS}
    log(f"หินที่แต่ละโปรไฟล์เห็น: {counts}")
    rocks, _ = find_rocks(fr, names, profiles)
    ov = fr.img.copy()
    for name, c in (("joy_center", (0, 255, 0)), ("sword", (0, 255, 0)), ("hand_guess", (255, 0, 255)), ("char_feet", (255, 255, 0))):
        cv2.circle(ov, fr.A(name), 10, c, 2)
        cv2.putText(ov, name, (fr.A(name)[0] + 12, fr.A(name)[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
    for name in ("coords", "clock", "target_bar"):
        x1, y1, x2, y2 = fr.box(name)
        cv2.rectangle(ov, (x1, y1), (x2, y2), (200, 200, 0), 1)
        cv2.putText(ov, name, (x1 + 2, y1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)
    for r in rocks:
        x, y, w, h = r["bbox"]
        cv2.rectangle(ov, (x, y), (x + w, y + h), (0, 0, 255), 2)
    os.makedirs(DEBUG_DIR, exist_ok=True)
    out = os.path.join(DEBUG_DIR, "calibrate.png")
    cv2.imwrite(out, ov)
    log(f"เซฟภาพตรวจ: {out}  (กรอบแดง=หินที่เห็น, วงกลม=ปุ่มที่บอทใช้, กรอบเหลือง=โซนอ่านพิกัด/เวลา/แถบเป้าหมาย)")


def cmd_test_input(args):
    dev = Device(dry=args.dry)
    ok, who = dev.root_ok()
    log(f"รูท: {'OK' if ok else 'ไม่ใช่ -> ' + who}")
    install_stop_handlers()
    countdown(5, "สลับเข้าเกมแล้วรอ (ตัวละครอย่าอยู่ใกล้มอน)")
    a = dev.grab()
    if a is None:
        sys.exit("จับภาพไม่ได้")
    os.makedirs(DEBUG_DIR, exist_ok=True)
    if args.what == "camera":
        global CAM_DRAG_MS
        TOUCH_BACKEND["camera"] = args.backend
        CAM_DRAG_MS = args.ms
        cam = Camera(dev)
        log(f"ทดสอบกล้อง: วิธี {args.backend} ลาก {args.ms}ms (ลากซ้าย-ขวา-ซ้าย แล้ววัดว่าฉากขยับกี่ px)")
        cv2.imwrite(os.path.join(DEBUG_DIR, "camera_before.png"), a.img)
        good = cam.try_calibrate()
        b = dev.grab()
        cv2.imwrite(os.path.join(DEBUG_DIR, "camera_after.png"), b.img)
        if good:
            cam.save()
            log(f"✅ กล้องหมุนได้ (ลาก 100px -> ฉากขยับ {cam.ratio * 100:+.0f}px) บันทึกค่าแล้ว ใช้ `run` ได้เลย")
        else:
            log("❌ กล้องไม่หมุน ลอง: python rock_loop.py camera-cal   (ลองทุกวิธีให้อัตโนมัติ)")
        return
    if args.what == "joystick":
        dev.joystick(0, -1, 1500)
    else:
        dev.hold(*a.A("sword"), 1500)
    nap(1.0)
    b = dev.grab()
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{args.what}_before.png"), a.img)
    cv2.imwrite(os.path.join(DEBUG_DIR, f"{args.what}_after.png"), b.img)
    log(f"ภาพต่างกัน {frame_diff(a.img, b.img):.1f} (เกิน ~6 = ฉากขยับจริง) เซฟภาพก่อน-หลังใน debug/")


def cmd_camera_cal(args):
    dev = Device(dry=args.dry)
    ok, who = dev.root_ok()
    log(f"รูท: {'OK' if ok else 'ไม่ใช่ -> ' + who}")
    install_stop_handlers()
    countdown(5, "สลับเข้าเกมแล้วรอ (ยืนในที่โล่ง ให้เห็นฉากไกล/ท้องฟ้า)")
    cam = Camera(dev)
    if dev.grab() is None:
        sys.exit("จับภาพไม่ได้")
    cam.calibrate()


def cmd_stop(args):
    open(STOP_FILE, "w").close()
    log(f"ส่งคำสั่งหยุดแล้ว ({STOP_FILE}) บอทที่กำลังรันจะหยุดภายในไม่กี่วินาที")


def cmd_doctor(args):
    """ตรวจทุกอย่างทีเดียว: root / tesseract / จับภาพ / อ่านเวลา-พิกัด-ชื่อ -> บอกว่าอะไรพัง"""
    log("--- ตรวจระบบ ---")
    log(f"pytesseract: {'มี' if pytesseract else 'ไม่มี  -> pip install pytesseract'}")
    tb = shutil.which("tesseract")
    log(f"tesseract โปรแกรม: {tb or 'ไม่พบ  -> pkg install tesseract (Termux) / apt install tesseract-ocr'}")
    if tb:
        try:
            v = subprocess.run([tb, "--version"], capture_output=True, timeout=10, stdin=subprocess.DEVNULL)
            log("tesseract รุ่น: " + (v.stdout or v.stderr).decode("utf-8", "ignore").splitlines()[0])
        except Exception as e:
            log(f"tesseract รันไม่ได้: {e}")
    if args.image:
        raw = cv2.imread(args.image)
        if raw is None:
            sys.exit(f"เปิดไฟล์ภาพไม่ได้: {args.image}")
        fr = Frame(cv2.resize(raw, (REF_W, REF_H)), raw.shape[:2], True) if args.landscape else make_frame(raw)
    else:
        dev = Device(dry=args.dry)
        ok, who = dev.root_ok()
        log(f"รูท (su): {'OK' if ok else 'ไม่ใช่ -> ' + who}")
        fr = dev.grab()
        if fr is None:
            sys.exit("จับภาพไม่ได้ (screencap ผ่าน su ใช้ไม่ได้)")
        log(f"จับภาพได้ {fr.raw_shape[1]}x{fr.raw_shape[0]} -> แนวนอน {fr.w}x{fr.h}")
    os.makedirs(DEBUG_DIR, exist_ok=True)
    clk = ClockReader()
    got = clk._ocr(fr)
    log(f"นาฬิกา: อ่านได้ {[fmt_min(m) for m in got][:6]} เลย์เอาต์ที่ใช้ได้={_LAST_LAYOUT[0]}")
    log(f"พิกัด: {read_coords_once(fr)}")
    bar = read_target_bar(fr)
    log(f"แถบเป้าหมาย: ขึ้น={bar['engaged']} ชื่อ='{read_target_name(fr, bar)}'")
    for lay in ("A", "B"):
        for k in ("clock", "coords"):
            roi = fr.crop(LAYOUT_ROI[lay][k])
            if roi.size:
                cv2.imwrite(os.path.join(DEBUG_DIR, f"doctor_{k}_{lay}.png"), cv2.resize(roi, None, fx=4, fy=4))
    log(f"เซฟภาพโซนอ่านเวลา/พิกัดไว้ที่ {DEBUG_DIR}/doctor_*.png (ถ้ากรอบไม่ตรงตัวเลข ใช้ --layout A|B หรือแก้ LAYOUT_ROI)")


def main():
    ap = argparse.ArgumentParser(description="บอทวนทดสอบ: จุดมองหา -> หาหิน -> ตี -> กล่อง -> วน")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--max-rounds", type=int, default=0, help="วนกี่รอบแล้วหยุด (0 = ไม่หยุด)")
    r.add_argument("--period", choices=list(PERIODS), help="บังคับช่วงเวลา (ทดสอบ)")
    r.add_argument("--no-countdown", action="store_true")
    r.add_argument("--pos", nargs="+", help="พิกัดจริงตอนเริ่ม เช่น 6356,14178 (ใช้ซ่อมเลขที่ OCR อ่านเพี้ยน และจำไว้ใน pos_ref.json)")
    r.add_argument("--collect-delay", type=float, help="รอกี่วินาทีหลังเลือดหมดก่อนกดปุ่มมือ (ค่าเริ่ม 0)")
    r.add_argument("--pick-wait", type=float, help="กดปุ่มมือแล้วรอกี่วินาทีก่อนหา Pick All (ค่าเริ่ม 0.4)")
    r.add_argument("--scan-shift", type=int, help="ฉากเลื่อนกี่ px ต่อช่วงหมุน 60° (ค่าเริ่ม 850 มากขึ้น=หมุนมากขึ้น)")
    r.add_argument("--clock-interval", type=float, help="อ่านนาฬิกาในเกมทุกกี่วินาที (ค่าเริ่ม 3 น้อยลง=อ่านถี่ขึ้น แต่ช้าลง)")
    r.add_argument("--no-slot1", action="store_true", help="ตอนเลือดเหลือน้อยไม่ใช้การกดช่อง 1 (ใช้ปุ่มดาบตลอด)")
    r.add_argument("--no-viz", action="store_true", help="ไม่เซฟภาพ debug/live.jpg (เส้นแดง/เขียวแสดงเป้าที่ล็อก)")
    r.add_argument("--view", type=int, nargs="?", const=8765, help="เปิดเว็บดูภาพล็อกเป้าสด (พอร์ตเริ่มต้น 8765)")
    r.add_argument("--no-guard", action="store_true", help="ปิดยามเฝ้า (ไม่หยุดเองเมื่อไม่เห็นปุ่มตี/พิกัด)")
    r.add_argument("--verbose", action="store_true", help="แสดงรายละเอียดเต็มบนจอ (ปกติลงไฟล์ debug/bot.log)")
    r.add_argument("--dry", action="store_true", help="ไม่ส่งคำสั่งจริง")
    r.add_argument("--image", help="ทดสอบกับภาพนิ่ง (ภาพแนวนอนที่หมุนกลับแล้ว) ไม่ต่อเครื่อง")
    c = sub.add_parser("calibrate")
    c.add_argument("--image")
    c.add_argument("--landscape", action="store_true")
    c.add_argument("--dry", action="store_true")
    t = sub.add_parser("test-input")
    t.add_argument("what", choices=["joystick", "hold", "camera"])
    t.add_argument("--backend", choices=["input", "sendevent"], default="input")
    t.add_argument("--ms", type=int, default=380, help="ลากนานกี่ ms (กล้อง)")
    t.add_argument("--dry", action="store_true")
    cc = sub.add_parser("camera-cal")
    cc.add_argument("--dry", action="store_true")
    sub.add_parser("stop")
    dr = sub.add_parser("doctor")
    dr.add_argument("--image")
    dr.add_argument("--landscape", action="store_true")
    dr.add_argument("--dry", action="store_true")
    for sp in (r, c, t, cc, dr):
        sp.add_argument("--layout", choices=["auto", "A", "B"], default=None, help="ตำแหน่ง HUD: A=ซ้ายบน B=ขวาบน/ใต้มินิแมพ (ค่าเริ่มต้น auto)")
    args = ap.parse_args()
    global LAYOUT, QUIET, GUARD, SEARCH_COORD, COLLECT_DELAY, PICK_WAIT, CAM_SCAN_SHIFT, CLOCK_READ_INTERVAL, VIZ, SLOT1_AT_HP
    if getattr(args, "no_guard", False):
        GUARD = False
    if getattr(args, "collect_delay", None) is not None:
        COLLECT_DELAY = max(0.0, args.collect_delay)
    if getattr(args, "pick_wait", None) is not None:
        PICK_WAIT = max(0.05, args.pick_wait)
    if getattr(args, "no_viz", False):
        VIZ = False
    if getattr(args, "no_slot1", False):
        SLOT1_AT_HP = 0.0
    if getattr(args, "clock_interval", None) is not None:
        CLOCK_READ_INTERVAL = max(0.0, args.clock_interval)
    if getattr(args, "scan_shift", None):
        CAM_SCAN_SHIFT = max(100, args.scan_shift)
    if args.cmd != "stop":                       # ไฟล์ STOP เก่าที่ค้างอยู่ ต้องไม่ทำให้รันรอบใหม่หยุดทันที
        try:
            os.remove(STOP_FILE)
        except OSError:
            pass
    if getattr(args, "layout", None):
        LAYOUT = args.layout
    if args.cmd in ("run", "doctor"):
        load_pos_ref()
        if getattr(args, "pos", None):
            try:
                pos = [int(v) for v in "".join(args.pos).replace(" ", "").split(",") if v][:2]
                set_coord_ref(pos)
                save_pos_ref()
                if SEARCH_COORD is None:
                    SEARCH_COORD = tuple(pos)
            except Exception:
                sys.exit("--pos ต้องเป็นรูปแบบ X,Y เช่น --pos 6356,14178")
    if args.cmd == "run":
        QUIET = not args.verbose
    if args.cmd == "stop":
        cmd_stop(args)
        return
    if args.cmd == "doctor":
        cmd_doctor(args)
        return
    if args.cmd == "camera-cal":
        try:
            cmd_camera_cal(args)
        except Stop as e:
            log(f"[หยุด] {e}")
        return
    if args.cmd == "calibrate":
        cmd_calibrate(args)
        return
    if args.cmd == "test-input":
        try:
            cmd_test_input(args)
        except Stop as e:
            log(f"[หยุด] {e}")
        return
    if args.image:
        raw = cv2.imread(args.image)
        if raw is None:
            sys.exit(f"เปิดไฟล์ภาพไม่ได้: {args.image}")
        dev = ImageDevice(Frame(cv2.resize(raw, (REF_W, REF_H)), raw.shape[:2], True))
    else:
        dev = Device(dry=args.dry)
        ok, who = dev.root_ok()
        log(f"รูท: {'OK' if ok else 'ไม่ใช่รูท -> ' + who}")
    if not HAS_TESS:
        log("เตือน: ไม่พบ pytesseract/tesseract -> อ่านชื่อแร่/นาฬิกา/พิกัดไม่ได้ (บอทจะเดินตามพิกัดไม่ได้)")
    profiles = load_profiles()
    bot = RockLoop(dev, profiles, args.period)
    _to_file("", "=" * 20 + " เริ่มรัน " + "=" * 20)
    install_stop_handlers()
    if getattr(args, "view", None):
        start_viewer(args.view)
    try:
        if not (args.no_countdown or args.image):
            countdown(COUNTDOWN_SEC, "สลับเข้าเกม Utopia: Origin ได้เลย!")
        bot.run(args.max_rounds)
    except Stop as e:
        ui(f"⏹ หยุด: {e}")
    except KeyboardInterrupt:
        ui("⏹ หยุด (Ctrl+C)")
    QUIET = False
    save_pos_ref()
    ui(f"📊 ตี {bot.stats['ตีสำเร็จ']} │ เก็บ {bot.stats['เก็บ']} │ บ้าน {bot.stats['กลับบ้านสำเร็จ']} │ รอบ {bot.stats['รอบ']}")
    ui("📄 รายละเอียด: debug/bot.log")


if __name__ == "__main__":
    main()
