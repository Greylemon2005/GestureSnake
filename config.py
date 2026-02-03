# config.py
import cv2

# 摄像头：会自动尝试这些索引
CAM_INDEX_CANDIDATES = [0, 1, 2]

# 摄像头后端：会按顺序尝试
CAP_BACKENDS = [
    ("DSHOW", cv2.CAP_DSHOW),
    ("MSMF", cv2.CAP_MSMF),
    ("DEFAULT", None),
]

CAM_W, CAM_H = 640, 360
MIRROR = True

SHOW_CAMERA = True  # True: 显示摄像头窗口（按 Q 关闭该窗口，不影响识别）

# Gesture thresholds (normalized by hand size)
PINCH_DIST_RATIO = 0.30       # OK 更容易触发：略放宽
OPEN_THUMB_SEP_RATIO = 0.42   # 张开判定：略放宽

# Debounce
DIR_HOLD_FRAMES = 4           # 越小越灵敏，但太小会抖
PINCH_HOLD_FRAMES = 7
OPEN_HOLD_FRAMES = 9
PINCH_COOLDOWN_SEC = 0.45
OPEN_COOLDOWN_SEC = 0.9

# Snake game
CELL = 20
GRID_W, GRID_H = 30, 22       # 600 x 440
WIN_W, WIN_H = GRID_W * CELL, GRID_H * CELL

TICK_BASE = 6                 # 移动速度：调小更慢（建议 5~8）
