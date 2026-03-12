"""
Extract hand landmarks from sign language videos using MediaPipe Hands.

Reads videos from dataset/<word>/ and writes sequences to data/
(train_sequences.npy, train_labels.npy, labels.json).
Paths are relative to the repository root.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Add project root for imports
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ai.features.hand_landmarks import HandLandmarkExtractor
from ai.config import FEATURE_DIM, DEFAULT_MAX_SEQ_LEN

DATASET_DIR = REPO_ROOT / "dataset"
DATA_DIR = REPO_ROOT / "data"
VOCAB_PATH = REPO_ROOT / "vocabulary_sheet.csv"


def load_vocabulary() -> list[tuple[str, str]]:
    """Load (word, arabic_label) from vocabulary_sheet.csv."""
    rows = []
    with open(VOCAB_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((row["word"].strip(), row["arabic_label"].strip()))
    return rows


def extract_sequence_from_video(video_path: Path, extractor: HandLandmarkExtractor) -> Optional[np.ndarray]:
    """
    Read video, extract one landmark vector per frame.
    Returns (T, FEATURE_DIM) or None if no frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    frames_rgb = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_rgb.append(rgb)
    cap.release()
    if not frames_rgb:
        return None
    frames_rgb = np.array(frames_rgb)
    sequence = extractor.extract_sequence(frames_rgb)
    return sequence


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATASET_DIR.exists():
        print(f"Dataset directory not found: {DATASET_DIR}")
        print("Create dataset/<word>/ and add videos, then run again.")
        sys.exit(1)

    vocab = load_vocabulary()
    word_to_idx = {word: i for i, (word, _) in enumerate(vocab)}
    label_list = [word for word, _ in vocab]

    sequences: list[np.ndarray] = []
    labels: list[int] = []

    with HandLandmarkExtractor() as extractor:
        for word, _ in vocab:
            word_dir = DATASET_DIR / word
            if not word_dir.is_dir():
                print(f"Skipping (no folder): {word_dir}")
                continue
            for video_path in sorted(word_dir.iterdir()):
                if video_path.suffix.lower() not in (".mp4", ".avi", ".mov", ".mkv"):
                    continue
                seq = extract_sequence_from_video(video_path, extractor)
                if seq is None or len(seq) == 0:
                    print(f"Skipping (no frames): {video_path}")
                    continue
                # Optionally cap length to match training
                if seq.shape[0] > DEFAULT_MAX_SEQ_LEN:
                    seq = seq[-DEFAULT_MAX_SEQ_LEN:]
                sequences.append(seq)
                labels.append(word_to_idx[word])
                print(f"  {video_path.name} -> {seq.shape[0]} frames")

    if not sequences:
        print("No sequences extracted. Add videos under dataset/<word>/ and run again.")
        sys.exit(1)

    # Save as object array for variable-length sequences
    sequences_np = np.empty(len(sequences), dtype=object)
    sequences_np[:] = sequences
    np.save(DATA_DIR / "train_sequences.npy", sequences_np)
    np.save(DATA_DIR / "train_labels.npy", np.array(labels, dtype=np.int64))
    with open(DATA_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump(label_list, f, ensure_ascii=False)

    print(f"Saved {len(sequences)} sequences to {DATA_DIR}")
    print(f"Labels: {label_list}")


if __name__ == "__main__":
    main()
