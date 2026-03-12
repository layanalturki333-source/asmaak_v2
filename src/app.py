"""
FastAPI app with WebSocket for realtime webcam sign recognition.

Serves the web UI and accepts WebSocket frames; runs landmark extraction + BiLSTM
and returns the predicted Arabic label. Paths relative to repo root.
"""

import base64
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ai.config import FEATURE_DIM, DEFAULT_MAX_SEQ_LEN
from ai.features.hand_landmarks import HandLandmarkExtractor
from ai.inference.predictor import load_model

import mediapipe as mp

logger = logging.getLogger(__name__)

app = FastAPI(title="Asmaak Prototype")

MODELS_DIR = REPO_ROOT / "models"
TEMPLATES_DIR = REPO_ROOT / "templates"
STATIC_DIR = REPO_ROOT / "static"

CHECKPOINT_PATH = MODELS_DIR / "sign_bilstm_best.pt"
VOCAB_CSV = REPO_ROOT / "vocabulary_sheet.csv"

# Arabic label lookup: word -> arabic
word_to_arabic: dict[str, str] = {}
if VOCAB_CSV.exists():
    import csv
    with open(VOCAB_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            word_to_arabic[row["word"].strip()] = row["arabic_label"].strip()


def get_arabic_label(word: str) -> str:
    return word_to_arabic.get(word, word)


# Lazy init
_model = None
_label_list = None
_extractor = None
_device = None

# MediaPipe drawing (max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5 in HandLandmarkExtractor)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def get_extractor():
    """Return HandLandmarkExtractor (no model required). Use for hand detection only."""
    global _extractor
    if _extractor is None:
        _extractor = HandLandmarkExtractor(
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        logger.info("HandLandmarkExtractor initialized (max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)")
    return _extractor


def get_predictor():
    """Return (model, label_list, extractor, device). Model may be None if checkpoint missing."""
    global _model, _label_list, _extractor, _device
    extractor = get_extractor()
    if _model is None and CHECKPOINT_PATH.exists():
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model, _label_list = load_model(CHECKPOINT_PATH, _device)
        logger.info("Model loaded from %s", CHECKPOINT_PATH)
    return _model, _label_list or [], extractor, _device


# Buffer of recent landmark vectors for sequence inference
frame_buffer: list[np.ndarray] = []
SEQ_LEN = min(24, DEFAULT_MAX_SEQ_LEN)


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = TEMPLATES_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>templates/index.html not found</h1>", status_code=404)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/health")
async def health():
    """Health check; does not require the model. Use to verify server is up."""
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        model, label_list, extractor, device = get_predictor()
    except Exception as e:
        logger.exception("Failed to init predictor")
        await websocket.send_json({"error": str(e)})
        await websocket.close()
        return

    buffer = []
    frame_count = 0
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            if "frame" not in msg:
                continue
            frame_count += 1
            b64 = msg["frame"]
            raw = base64.b64decode(b64)
            arr = np.frombuffer(raw, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                logger.warning("Frame decode failed (imdecode returned None)")
                continue
            h, w = frame.shape[:2]
            if frame_count % 30 == 1:
                logger.info("Frame received: shape=%s", (h, w))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            features, results = extractor.extract_with_results(rgb)
            hand_detected = results.multi_hand_landmarks is not None and len(results.multi_hand_landmarks) > 0
            if hand_detected:
                if frame_count % 30 == 1:
                    logger.info("Hand detected (landmarks=%d)", len(results.multi_hand_landmarks))
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        rgb,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                    )
            else:
                if frame_count % 30 == 1:
                    logger.info("No hand detected")
            _, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            annotated_b64 = base64.b64encode(buf).decode("utf-8")
            payload = {
                "annotated_frame": annotated_b64,
                "hand_detected": hand_detected,
            }
            if model is not None and device is not None:
                buffer.append(features.copy())
                if len(buffer) > SEQ_LEN:
                    buffer.pop(0)
                label_ar = ""
                label_en = ""
                confidence = 0.0
                if len(buffer) >= SEQ_LEN:
                    seq = np.stack(buffer, axis=0).astype(np.float32)
                    seq_t = torch.from_numpy(seq).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = model(seq_t)
                        probs = torch.softmax(logits, dim=1)
                        pred_idx = logits.argmax(dim=1).item()
                        confidence = float(probs[0, pred_idx].item())
                    label_en = label_list[pred_idx] if pred_idx < len(label_list) else ""
                    label_ar = get_arabic_label(label_en)
                payload["landmarks"] = []
                if np.any(np.abs(features) > 1e-6):
                    payload["landmarks"] = [
                        [float(features[i * 3]), float(features[i * 3 + 1]), float(features[i * 3 + 2])]
                        for i in range(21)
                    ]
                payload["label_ar"] = label_ar
                payload["label_en"] = label_en
                payload["confidence"] = confidence
            else:
                payload["landmarks"] = []
                payload["label_ar"] = ""
                payload["label_en"] = ""
                payload["confidence"] = 0.0
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.exception("WebSocket error")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    # Run from repo root:  python3 src/app.py
    uvicorn.run(app, host="0.0.0.0", port=8000)
