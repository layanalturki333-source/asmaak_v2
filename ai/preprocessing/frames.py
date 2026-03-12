"""
Frame preprocessing for camera or video input.

OpenCV reads BGR; MediaPipe expects RGB. This module provides
simple resize and color conversion helpers.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def rgb_from_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR (OpenCV) to RGB for MediaPipe."""
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def resize_keep_aspect(
    frame: np.ndarray,
    target_size: Tuple[int, int],
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """
    Resize frame so it fits inside target_size (width, height),
    preserving aspect ratio. Optionally pad to exact target_size.
    """
    h, w = frame.shape[:2]
    tw, th = target_size
    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interpolation)
    return resized
