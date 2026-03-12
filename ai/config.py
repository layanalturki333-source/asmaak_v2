"""
Asmaak AI Module - Configuration.

Central configuration for paths, model hyperparameters, and pipeline settings.
Dataset paths are NOT hardcoded; set via environment or override in your training script.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (use env vars or override; do not hardcode dataset paths)
# ---------------------------------------------------------------------------
# Root directory of the project (parent of 'ai/')
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default directories (can be overridden)
DATA_DIR = Path(os.environ.get("ASMAAK_DATA_DIR", PROJECT_ROOT / "data"))
MODELS_DIR = Path(os.environ.get("ASMAAK_MODELS_DIR", PROJECT_ROOT / "models"))
LOGS_DIR = Path(os.environ.get("ASMAAK_LOGS_DIR", PROJECT_ROOT / "logs"))

# Ensure dirs exist when used (created on first use)
def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# MediaPipe / Hand landmarks
# ---------------------------------------------------------------------------
# Number of landmarks per hand (MediaPipe Hands)
NUM_HAND_LANDMARKS = 21
# Coordinates per landmark (x, y, z)
COORDS_PER_LANDMARK = 3
# Max hands to detect (we use 1 for ArSL isolated signs)
MAX_NUM_HANDS = 1

# Feature size: one hand = 21 * 3 = 63; two hands would be 126
FEATURE_DIM = MAX_NUM_HANDS * NUM_HAND_LANDMARKS * COORDS_PER_LANDMARK  # 63

# ---------------------------------------------------------------------------
# Sequence / Model
# ---------------------------------------------------------------------------
# Frames per sequence (variable length supported via padding; this is default max)
DEFAULT_MAX_SEQ_LEN = 32
# Hidden size for LSTM/GRU
HIDDEN_SIZE = 128
# Number of LSTM layers
NUM_LSTM_LAYERS = 2
# Dropout
DROPOUT = 0.3
# Bidirectional LSTM
BIDIRECTIONAL = True

# ---------------------------------------------------------------------------
# Training defaults
# ---------------------------------------------------------------------------
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
# For sequence padding
PAD_VALUE = 0.0

# ---------------------------------------------------------------------------
# Inference / Realtime
# ---------------------------------------------------------------------------
# Camera index for OpenCV
CAMERA_INDEX = 0
# Frames to collect before running sequence model (sliding window or full sequence)
INFERENCE_SEQ_LEN = 24
# Model filename (saved in MODELS_DIR)
DEFAULT_MODEL_FILENAME = "asl_bilstm_best.pt"
