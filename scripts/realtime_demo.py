"""
Realtime webcam demo for Asmaak ArSL sign recognition.

Captures frames from the camera, extracts hand landmarks, buffers a sequence,
and runs the model to display the predicted Arabic sign label on screen.
Press 'q' to quit.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from ai.config import (
    CAMERA_INDEX,
    MODELS_DIR,
    DEFAULT_MODEL_FILENAME,
    INFERENCE_SEQ_LEN,
    ensure_dirs,
)
from ai.features import HandLandmarkExtractor
from ai.preprocessing import rgb_from_bgr
from ai.inference import SignPredictor


def main():
    parser = argparse.ArgumentParser(description="Realtime ArSL sign recognition demo")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=MODELS_DIR / DEFAULT_MODEL_FILENAME,
        help="Path to model checkpoint",
    )
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="Camera device index")
    parser.add_argument("--seq_len", type=int, default=INFERENCE_SEQ_LEN, help="Frames per prediction")
    args = parser.parse_args()

    ensure_dirs()
    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        print("Train a model first: python scripts/train.py --data_dir data")
        return

    predictor = SignPredictor(args.checkpoint)
    extractor = HandLandmarkExtractor()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    # Buffer of landmark vectors for sequence
    buffer = []
    pred_label = "—"
    last_pred_time = 0
    pred_interval = 0.5  # Minimum seconds between predictions

    print("Realtime ArSL demo. Show a sign to the camera. Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = rgb_from_bgr(frame)
            feats = extractor.extract(frame_rgb)
            buffer.append(feats)
            if len(buffer) > args.seq_len:
                buffer.pop(0)

            # Run model when we have enough frames and enough time passed
            now = time.time()
            if len(buffer) >= args.seq_len and (now - last_pred_time) >= pred_interval:
                seq = np.stack(buffer, axis=0)
                seq_t = torch.from_numpy(seq).float().unsqueeze(0)
                _, pred_label = predictor.predict(seq_t)
                last_pred_time = now

            # Draw predicted label on frame
            h, w = frame.shape[:2]
            cv2.putText(
                frame,
                pred_label,
                (20, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Asmaak - ArSL", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()


if __name__ == "__main__":
    main()
