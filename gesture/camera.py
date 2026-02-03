# gesture/camera.py
from typing import Optional, Tuple
import cv2

from config import CAM_INDEX_CANDIDATES, CAP_BACKENDS, CAM_W, CAM_H

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
