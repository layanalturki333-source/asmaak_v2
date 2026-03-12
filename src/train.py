"""
Train BiLSTM model on extracted landmark sequences.

Reads from data/ (train_sequences.npy, train_labels.npy, labels.json)
and saves the best checkpoint to models/. Paths are relative to repo root.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ai.config import (
    FEATURE_DIM,
    HIDDEN_SIZE,
    NUM_LSTM_LAYERS,
    DROPOUT,
    BIDIRECTIONAL,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    DEFAULT_MAX_SEQ_LEN,
)
from ai.models.bilstm import SignBiLSTM
from ai.dataset.sequence_dataset import SequenceDataset

DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    seq_path = DATA_DIR / "train_sequences.npy"
    label_path = DATA_DIR / "train_labels.npy"
    labels_path = DATA_DIR / "labels.json"

    if not seq_path.exists() or not label_path.exists():
        print("Run src/extract_landmarks.py first to create data/train_sequences.npy and data/train_labels.npy")
        sys.exit(1)

    sequences = np.load(seq_path, allow_pickle=True)
    if sequences.ndim == 3:
        sequences = [sequences[i] for i in range(len(sequences))]
    else:
        sequences = list(sequences)
    labels = np.load(label_path).astype(np.int64).tolist()

    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            label_list = json.load(f)
    else:
        label_list = [str(i) for i in range(max(labels) + 1)]

    num_classes = len(label_list)
    dataset = SequenceDataset(sequences, labels, max_len=DEFAULT_MAX_SEQ_LEN)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SignBiLSTM(
        input_size=FEATURE_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LSTM_LAYERS,
        num_classes=num_classes,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL,
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_loss = float("inf")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt = {
                "model_state_dict": model.state_dict(),
                "num_classes": num_classes,
                "label_list": label_list,
            }
            torch.save(ckpt, MODELS_DIR / "sign_bilstm_best.pt")
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} loss={avg_loss:.4f}")

    print(f"Training done. Best model saved to {MODELS_DIR / 'sign_bilstm_best.pt'}")


if __name__ == "__main__":
    main()
