# gesture/utils.py
from dataclasses import dataclass
import numpy as np

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


def lm_xy(lm, w, h):
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)

def dist(a, b) -> float:
    return float(np.linalg.norm(a - b))
