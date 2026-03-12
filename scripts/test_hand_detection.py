#!/usr/bin/env python3
"""
Quick test script: open webcam with OpenCV and run MediaPipe Hands locally.
Use this to verify hand detection works outside the web app.

Run from repo root:
  python3 scripts/test_hand_detection.py

Press 'q' to quit.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2
import mediapipe as mp

# MediaPipe Hands: same settings as the app
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam (camera index 0)")
    sys.exit(1)

print("Webcam opened. Show your hand to the camera. Press 'q' to quit.")
print("MediaPipe settings: max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break
    frame_count += 1
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    hand_detected = results.multi_hand_landmarks is not None and len(results.multi_hand_landmarks) > 0
    if hand_detected:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                rgb,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
            )
        if frame_count % 30 == 1:
            print("Hand detected")
    else:
        if frame_count % 30 == 1:
            print("No hand detected")
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.putText(
        frame,
        "Hand: YES" if hand_detected else "Hand: NO",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0) if hand_detected else (0, 0, 255),
        2,
    )
    cv2.imshow("Hand detection test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print("Done.")
