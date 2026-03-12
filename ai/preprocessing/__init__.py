"""
Preprocessing utilities for frames and sequences.

Currently minimal; can be extended for normalization, augmentation, etc.
"""

from .frames import resize_keep_aspect, rgb_from_bgr

__all__ = ["resize_keep_aspect", "rgb_from_bgr"]
