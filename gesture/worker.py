# gesture/worker.py
import time
import threading
import cv2
import mediapipe as mp

from config import (
    MIRROR, SHOW_CAMERA,
    PINCH_DIST_RATIO, OPEN_THUMB_SEP_RATIO,
    DIR_HOLD_FRAMES, PINCH_HOLD_FRAMES, OPEN_HOLD_FRAMES,
    PINCH_COOLDOWN_SEC, OPEN_COOLDOWN_SEC,
)
from gesture.types import GestureState
from gesture.camera import try_open_camera
from gesture.utils import Debounce, DirectionSmoother, lm_xy, dist


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
    if not (ext["index"] and ext["middle"] and ext["ring"] and ext["pinky"]):
        return False
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
    只输出 LEFT/RIGHT/UP，避免 DOWN 不稳定（DOWN 用 V 手势触发）
    """
    lm = mp.solutions.hands.HandLandmark
    v = pts[lm.INDEX_FINGER_TIP] - pts[lm.WRIST]
    dx, dy = float(v[0]), float(v[1])

    if abs(dx) > abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    return "UP" if dy < 0 else ""


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
