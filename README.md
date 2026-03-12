# Asmaak Prototype

Prototype for realtime Arabic sign language recognition (5 words) using MediaPipe Hands and a BiLSTM model. FastAPI + WebSocket for webcam inference; UI uses Arabic labels.

## Vocabulary

| Word       | Arabic |
|-----------|--------|
| hello     | مرحبا  |
| yes       | نعم    |
| no        | لا     |
| thank_you | شكرا   |
| water     | ماء    |

## Project layout

Paths are relative to the repository root.

- `vocabulary_sheet.csv` — word and Arabic label mapping
- `DATASET_GUIDE.md` — how to prepare video data
- `src/extract_landmarks.py` — extract landmarks from videos (MediaPipe) → `data/`
- `src/train.py` — train BiLSTM on sequences in `data/` → `models/`
- `src/app.py` — FastAPI app + WebSocket for realtime inference
- `templates/index.html` — web UI (Arabic labels)
- `dataset/` — place videos per word: `dataset/hello/`, `dataset/yes/`, etc.
- `data/` — extracted sequences and labels (created by `extract_landmarks.py`)
- `models/` — saved model checkpoint (created by `train.py`)
- `static/` — optional static assets

## Setup

```bash
cd /path/to/asmaak_v2
python3 -m venv venv
source venv/bin/activate   # or: venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## 1. Prepare dataset

Put one sign per video in:

```
dataset/
  hello/
    video_001.mp4
    ...
  yes/
    ...
  no/
    ...
  thank_you/
    ...
  water/
    ...
```

See `DATASET_GUIDE.md` for details.

## 2. Extract landmarks

```bash
python3 src/extract_landmarks.py
```

Reads from `dataset/` and writes `data/train_sequences.npy`, `data/train_labels.npy`, `data/labels.json`.

## 3. Train model

```bash
python3 src/train.py
```

Reads from `data/` and saves the best model to `models/sign_bilstm_best.pt`.

## 4. Run the app

From the **repository root** (not from inside `src/`):

```bash
cd /path/to/asmaak_v2
python3 src/app.py
```

Or with uvicorn (from repo root):

```bash
python3 -m uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

Then open http://localhost:8000 in a browser. You can check the server is up with http://localhost:8000/health. Allow webcam access and sign in front of the camera; the predicted Arabic label is shown in the UI.

## Run locally

All paths are relative to the repo root. No extra config is required for local use.
