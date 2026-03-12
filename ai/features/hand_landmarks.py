"""
MediaPipe hand landmark extraction.

Extracts 21 3D landmarks per hand from RGB frames.
Output is a flat vector of shape (63,) for one hand (21 * 3) or (126,) for two hands.
Designed for use in a sequence: each frame -> one feature vector.
"""

import numpy as np
from typing import Optional, Tuple

try:
    import mediapipe as mp
except ImportError:
    mp = None

from ai.config import (
    NUM_HAND_LANDMARKS,
    COORDS_PER_LANDMARK,
    MAX_NUM_HANDS,
    FEATURE_DIM,
)


class HandLandmarkExtractor:
    """
    Extracts hand landmarks from RGB frames using MediaPipe Hands.
    Returns a fixed-size vector per frame for use in sequence models.
    """

    def __init__(
        self,
        max_num_hands: int = MAX_NUM_HANDS,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        if mp is None:
            raise ImportError("mediapipe is required. Install with: pip install mediapipe")
        self.max_num_hands = max_num_hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._expected_dim = max_num_hands * NUM_HAND_LANDMARKS * COORDS_PER_LANDMARK

    def extract(self, frame_rgb: np.ndarray) -> np.ndarray:
        """
        Extract hand landmark features from one RGB frame (H, W, 3), uint8.
        Returns a vector of shape (FEATURE_DIM,) with values in [0, 1] or zero-filled
        if no hand is detected.
        """
        features, _ = self.extract_with_results(frame_rgb)
        return features

    def extract_with_results(self, frame_rgb: np.ndarray):
        """
        Process one RGB frame and return (features, results).
        results is the raw MediaPipe results object (for drawing landmarks).
        """
        assert frame_rgb.ndim == 3 and frame_rgb.shape[2] == 3
        results = self.hands.process(frame_rgb)
        features = np.zeros(self._expected_dim, dtype=np.float32)

        if not results.multi_hand_landmarks:
            return features, results

        idx = 0
        for hand_landmarks in results.multi_hand_landmarks[: self.max_num_hands]:
            for lm in hand_landmarks.landmark:
                features[idx] = lm.x
                features[idx + 1] = lm.y
                features[idx + 2] = lm.z
                idx += COORDS_PER_LANDMARK
        return features, results

    def extract_sequence(
        self, frames_rgb: np.ndarray
    ) -> np.ndarray:
        """
        Extract features for a sequence of frames.
        frames_rgb: (T, H, W, 3)
        Returns: (T, FEATURE_DIM)
        """
        T = frames_rgb.shape[0]
        out = np.zeros((T, self._expected_dim), dtype=np.float32)
        for t in range(T):
            out[t] = self.extract(frames_rgb[t])
        return out

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.hands.close()

    def __enter__(self) -> "HandLandmarkExtractor":
        return self

    def __exit__(self, *args) -> None:
        self.close()
