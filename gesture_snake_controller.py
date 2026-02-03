import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import pygame
import mediapipe as mp


# =========================
# Config
# =========================
CAM_INDEX_CANDIDATES = [0, 1, 2]

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


# =========================
# Gesture Recognition Utils
# =========================
def lm_xy(lm, w, h):
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)

def dist(a, b) -> float:
    return float(np.linalg.norm(a - b))

@dataclass
class Debounce:
    hold_frames: int
    cooldown_sec: float
    _last_label: str = ""
    _count: int = 0
    _cooldown_until: float = 0.0
    _armed: bool = True  # edge-trigger

    def update(self, label: str, now: float) -> bool:
        """返回 True：本帧触发一次动作（边沿触发+防抖+冷却）"""
        if now < self._cooldown_until:
            self._last_label = label
            self._count = 0
            self._armed = True
            return False

        if label != self._last_label:
            self._last_label = label
            self._count = 1
            self._armed = True
            return False

        self._count += 1
        if self._armed and self._count >= self.hold_frames and label:
            self._armed = False
            self._cooldown_until = now + self.cooldown_sec
            return True
        return False

class DirectionSmoother:
    def __init__(self, hold_frames: int):
        self.hold_frames = hold_frames
        self.last_dir = ""
        self.candidate = ""
        self.count = 0

    def update(self, candidate_dir: str) -> str:
        if not candidate_dir:
            self.candidate = ""
            self.count = 0
            return self.last_dir

        if candidate_dir != self.candidate:
            self.candidate = candidate_dir
            self.count = 1
            return self.last_dir

        self.count += 1
        if self.count >= self.hold_frames:
            self.last_dir = candidate_dir
        return self.last_dir

def fingers_extended(landmarks, w, h):
    """
    判断四指是否伸直（经验规则：tip.y < pip.y）
    """
    lm = mp.solutions.hands.HandLandmark
    pts = {i: lm_xy(landmarks[i], w, h) for i in range(21)}
    ext = {}
    for name, tip, pip in [
        ("index", lm.INDEX_FINGER_TIP, lm.INDEX_FINGER_PIP),
        ("middle", lm.MIDDLE_FINGER_TIP, lm.MIDDLE_FINGER_PIP),
        ("ring", lm.RING_FINGER_TIP, lm.RING_FINGER_PIP),
        ("pinky", lm.PINKY_TIP, lm.PINKY_PIP),
    ]:
        ext[name] = bool(pts[tip][1] < pts[pip][1])  # y越小越靠上
    return ext, pts

def estimate_hand_size(pts):
    lm = mp.solutions.hands.HandLandmark
    return max(1e-6, dist(pts[lm.WRIST], pts[lm.MIDDLE_FINGER_MCP]))

def detect_ok_pinch(pts, hand_size) -> bool:
    lm = mp.solutions.hands.HandLandmark
    d = dist(pts[lm.THUMB_TIP], pts[lm.INDEX_FINGER_TIP]) / hand_size
    return d < PINCH_DIST_RATIO

def detect_open_hand(ext, pts, hand_size) -> bool:
    # 四指伸直
    if not (ext["index"] and ext["middle"] and ext["ring"] and ext["pinky"]):
        return False
    # 拇指与食指分开一定距离（避免四指伸直但并拢误判）
    lm = mp.solutions.hands.HandLandmark
    sep = dist(pts[lm.THUMB_TIP], pts[lm.INDEX_FINGER_TIP]) / hand_size
    return sep > OPEN_THUMB_SEP_RATIO

def detect_pointing(ext) -> bool:
    # 食指伸直，ring/pinky收起；middle允许偶尔伸直一点点会更灵敏
    return ext["index"] and (not ext["ring"]) and (not ext["pinky"])

def detect_v_down(ext) -> bool:
    # V手势：index + middle 伸直，ring + pinky 收起
    return ext["index"] and ext["middle"] and (not ext["ring"]) and (not ext["pinky"])

def pointing_direction_no_down(pts) -> str:
    """
    只输出 LEFT/RIGHT/UP，避免 DOWN 不稳定
    """
    lm = mp.solutions.hands.HandLandmark
    v = pts[lm.INDEX_FINGER_TIP] - pts[lm.WRIST]
    dx, dy = float(v[0]), float(v[1])

    if abs(dx) > abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    else:
        return "UP" if dy < 0 else ""  # dy>0(向下)直接不输出方向


# =========================
# Shared Gesture State
# =========================
@dataclass
class GestureState:
    direction: str = ""        # UP/DOWN/LEFT/RIGHT
    pause_toggle: bool = False
    restart: bool = False
    label: str = "INIT"
    hand_seen: bool = False
    cam_info: str = ""


def try_open_camera() -> Tuple[Optional[cv2.VideoCapture], str]:
    """
    依次尝试不同 index 与 backend，返回 cap 与描述信息
    """
    for idx in CAM_INDEX_CANDIDATES:
        for name, backend in CAP_BACKENDS:
            if backend is None:
                cap = cv2.VideoCapture(idx)
            else:
                cap = cv2.VideoCapture(idx, backend)

            if cap is not None and cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
                info = f"CAM idx={idx}, backend={name}"
                return cap, info

            try:
                cap.release()
            except:
                pass

    return None, "CAMERA_OPEN_FAILED"


class GestureWorker(threading.Thread):
    def __init__(self, state: GestureState):
        super().__init__(daemon=True)
        self.state = state
        self._stop = threading.Event()
        self.lock = threading.Lock()

        self.dir_smoother = DirectionSmoother(DIR_HOLD_FRAMES)
        self.pinch_db = Debounce(PINCH_HOLD_FRAMES, PINCH_COOLDOWN_SEC)
        self.open_db = Debounce(OPEN_HOLD_FRAMES, OPEN_COOLDOWN_SEC)

        self.show_camera = SHOW_CAMERA

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            cap, cam_info = try_open_camera()
            with self.lock:
                self.state.cam_info = cam_info

            if cap is None:
                with self.lock:
                    self.state.label = "CAMERA_OPEN_FAILED"
                    self.state.hand_seen = False
                print("[GestureWorker] CAMERA_OPEN_FAILED. Close apps using camera or try other index.")
                return

            print("[GestureWorker] Opened:", cam_info)

            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            drawer = mp.solutions.drawing_utils

            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    with self.lock:
                        self.state.label = "CAMERA_READ_FAILED"
                        self.state.hand_seen = False
                    time.sleep(0.01)
                    continue

                if MIRROR:
                    frame = cv2.flip(frame, 1)

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                now = time.time()
                pause_event = False
                restart_event = False
                label = "NO_HAND"
                hand_seen = False
                candidate_dir = ""

                if result.multi_hand_landmarks:
                    hand_seen = True
                    hand_landmarks = result.multi_hand_landmarks[0]

                    ext, pts = fingers_extended(hand_landmarks.landmark, w, h)
                    hand_size = estimate_hand_size(pts)

                    is_open = detect_open_hand(ext, pts, hand_size)
                    is_ok = detect_ok_pinch(pts, hand_size)
                    is_v = detect_v_down(ext)
                    is_point = detect_pointing(ext)

                    # Priority: restart(open) > pause(OK) > v_down(DOWN) > pointing(LEFT/RIGHT/UP)
                    if is_open:
                        label = "OPEN_HAND -> RESTART"
                        if self.open_db.update("OPEN", now):
                            restart_event = True
                    elif is_ok:
                        label = "OK_PINCH -> PAUSE"
                        if self.pinch_db.update("PINCH", now):
                            pause_event = True
                    elif is_v:
                        candidate_dir = "DOWN"
                        label = "V_SIGN -> DOWN"
                    elif is_point:
                        d = pointing_direction_no_down(pts)
                        candidate_dir = d
                        label = f"POINT -> {d}" if d else "POINT"
                    else:
                        label = "RELAX"

                    stable_dir = self.dir_smoother.update(candidate_dir)
                else:
                    stable_dir = self.dir_smoother.update("")
                    label = "NO_HAND"

                if self.show_camera:
                    if result.multi_hand_landmarks:
                        drawer.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    cv2.putText(frame, f"{cam_info}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, label, (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.imshow("Camera (press Q to close this window)", frame)
                    k = cv2.waitKey(1) & 0xFF
                    if k in (ord('q'), ord('Q')):
                        cv2.destroyWindow("Camera (press Q to close this window)")
                        self.show_camera = False

                with self.lock:
                    self.state.hand_seen = hand_seen
                    self.state.label = label
                    self.state.direction = stable_dir
                    self.state.pause_toggle = pause_event
                    self.state.restart = restart_event

            cap.release()
            try:
                cv2.destroyAllWindows()
            except:
                pass

        except Exception as e:
            import traceback
            with self.lock:
                self.state.label = "WORKER_EXCEPTION"
                self.state.hand_seen = False
            print("[GestureWorker] Exception:", e)
            traceback.print_exc()
            return


# =========================
# Snake Game (pygame)
# =========================
DIR_V = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}

def rand_food(exclude_set):
    while True:
        p = (np.random.randint(0, GRID_W), np.random.randint(0, GRID_H))
        if p not in exclude_set:
            return p

def reset_game():
    snake = [(GRID_W//2, GRID_H//2), (GRID_W//2 - 1, GRID_H//2), (GRID_W//2 - 2, GRID_H//2)]
    direction = "RIGHT"
    food = rand_food(set(snake))
    score = 0
    alive = True
    paused = False
    return snake, direction, food, score, alive, paused

def draw_cell(screen, x, y, color):
    pygame.draw.rect(screen, color, pygame.Rect(x*CELL, y*CELL, CELL, CELL))

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("GestureSnake (Embedded) - MediaPipe Hands")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 18)

    state = GestureState()
    worker = GestureWorker(state)
    worker.start()
    print("[Main] GestureWorker started:", worker.is_alive())

    snake, direction, food, score, alive, paused = reset_game()

    def apply_dir(new_dir: str):
        nonlocal direction
        if not new_dir:
            return
        if direction and OPPOSITE.get(direction) == new_dir:
            return  # 禁止180°掉头
        direction = new_dir

    # Start screen
    start = True
    while start:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                worker.stop()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    start = False
                if event.key == pygame.K_ESCAPE:
                    worker.stop()
                    return

        screen.fill((15, 15, 18))
        t1 = font.render("GestureSnake (Embedded)", True, (220, 220, 220))
        t2 = font.render("ENTER to start | ESC to quit", True, (200, 200, 200))
        t3 = font.render("Gestures: Point=Left/Right/Up | V=Down | OK=Pause | OpenHand=Restart", True, (180, 180, 180))
        screen.blit(t1, (20, 40))
        screen.blit(t2, (20, 70))
        screen.blit(t3, (20, 100))
        pygame.display.flip()
        clock.tick(30)

    # Game loop
    move_acc = 0.0
    last_time = time.time()

    while True:
        now = time.time()
        dt = now - last_time
        last_time = now
        if alive and (not paused):
            move_acc += dt
        else:
            # 暂停/死亡时不积累，避免恢复瞬间补跑
            move_acc = 0.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                worker.stop()
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    worker.stop()
                    return
                # keyboard fallback
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_r:
                    snake, direction, food, score, alive, paused = reset_game()
                    move_acc = 0.0
                    last_time = time.time()

        # Consume gesture state
        with worker.lock:
            g_dir = state.direction
            g_pause = state.pause_toggle
            g_restart = state.restart
            g_label = state.label
            g_seen = state.hand_seen
            g_cam = state.cam_info

            state.pause_toggle = False
            state.restart = False

        if g_restart:
            snake, direction, food, score, alive, paused = reset_game()
            move_acc = 0.0
            last_time = time.time()

        if g_pause:
            paused = not paused

        if alive and (not paused):
            apply_dir(g_dir)

        # Update movement
        step_time = 1.0 / max(1, TICK_BASE)

        if alive and (not paused) and move_acc >= step_time:
            move_acc -= step_time

            vx, vy = DIR_V[direction]
            head = snake[0]
            new_head = (head[0] + vx, head[1] + vy)

            if new_head[0] < 0 or new_head[0] >= GRID_W or new_head[1] < 0 or new_head[1] >= GRID_H:
                alive = False
            elif new_head in snake:
                alive = False
            else:
                snake.insert(0, new_head)
                if new_head == food:
                    score += 1
                    food = rand_food(set(snake))
                else:
                    snake.pop()

        # Render
        screen.fill((12, 12, 14))
        draw_cell(screen, food[0], food[1], (255, 90, 90))

        for i, (x, y) in enumerate(snake):
            c = (90, 220, 140) if i == 0 else (60, 160, 100)
            draw_cell(screen, x, y, c)

        # HUD
        hud1 = font.render(f"Score: {score}", True, (230, 230, 230))
        hud2 = font.render(f"Gesture: {g_label}", True, (200, 200, 200))
        hud3 = font.render(f"Hand: {'YES' if g_seen else 'NO'} | Dir: {g_dir or '-'}", True, (180, 180, 180))
        hud4 = font.render(f"{g_cam}", True, (120, 120, 120))
        screen.blit(hud1, (8, 6))
        screen.blit(hud2, (8, 28))
        screen.blit(hud3, (8, 50))
        screen.blit(hud4, (8, 72))

        if paused:
            msg = font.render("PAUSED (OK pinch to toggle)", True, (255, 255, 120))
            screen.blit(msg, (8, 96))
        if not alive:
            msg = font.render("GAME OVER (Open hand to restart)", True, (255, 160, 160))
            screen.blit(msg, (8, 118))

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
