"""
Training script for Asmaak ArSL sign recognition model.

Usage:
    python scripts/train.py --data_dir /path/to/data [--epochs 50]

Data dir should contain: train_sequences.npy, train_labels.npy, labels.json
(see ai/dataset/sequence_dataset.py for expected format).
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ai.config import (
    DATA_DIR,
    MODELS_DIR,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    ensure_dirs,
)
from ai.dataset import load_sequence_dataset
from ai.models import SignBiLSTM
from ai.training import train_model
from ai.utils import save_labels


def main():
    parser = argparse.ArgumentParser(description="Train Asmaak sign recognition model")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing train_sequences.npy, train_labels.npy, labels.json",
    )
    parser.add_argument("--save_dir", type=Path, default=MODELS_DIR, help="Where to save checkpoints")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    args = parser.parse_args()

    ensure_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        train_dataset, label_list = load_sequence_dataset(args.data_dir, split="train")
    except FileNotFoundError as e:
        print(e)
        print("Creating dummy data for MVP demo (no dataset found).")
        # Dummy data so training script runs without real data
        import numpy as np
        from ai.dataset import SequenceDataset
        from ai.config import FEATURE_DIM, DEFAULT_MAX_SEQ_LEN

        n_samples = 80
        num_classes = 5
        label_list = ["كلاس_١", "كلاس_٢", "كلاس_٣", "كلاس_٤", "كلاس_٥"]
        sequences = [
            np.random.randn(np.random.randint(10, 30), FEATURE_DIM).astype(np.float32)
            for _ in range(n_samples)
        ]
        labels = [np.random.randint(0, num_classes) for _ in range(n_samples)]
        train_dataset = SequenceDataset(sequences, labels, max_len=DEFAULT_MAX_SEQ_LEN)
        args.save_dir.mkdir(parents=True, exist_ok=True)
        labels_path = args.data_dir / "labels.json"
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        save_labels(label_list, labels_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    num_classes = len(label_list)
    model = SignBiLSTM(num_classes=num_classes)

    result = train_model(
        model,
        train_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        save_dir=args.save_dir,
        label_list=label_list,
    )
    print(f"Training done. Best loss: {result['best_loss']:.4f}")
    print(f"Model saved to: {result['best_path']}")


if __name__ == "__main__":
    main()
