"""
Evaluation script for Asmaak ArSL sign recognition model.

Computes accuracy on a test/validation set.
Usage:
    python scripts/evaluate.py --checkpoint models/asl_bilstm_best.pt --data_dir data
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ai.config import DATA_DIR, BATCH_SIZE
from ai.dataset import load_sequence_dataset
from ai.inference import load_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate Asmaak sign recognition model")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=DATA_DIR,
        help="Directory with val_sequences.npy / test_sequences.npy and labels",
    )
    parser.add_argument("--split", type=str, default="val", help="Split: val or test")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, label_list = load_model(args.checkpoint, device)

    try:
        dataset, _ = load_sequence_dataset(args.data_dir, split=args.split)
    except FileNotFoundError as e:
        print(e)
        print("No evaluation data found. Exiting.")
        return

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = correct / total if total else 0.0
    print(f"Accuracy ({args.split}): {acc:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
