# gesture/types.py
from dataclasses import dataclass

@dataclass
class GestureState:
    direction: str = ""        # UP/DOWN/LEFT/RIGHT
    pause_toggle: bool = False
    restart: bool = False
    label: str = "INIT"
    hand_seen: bool = False
    cam_info: str = ""
