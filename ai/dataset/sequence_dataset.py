"""
Sequence dataset for ArSL sign recognition.

Design: pluggable dataset. Expects either:
- Pre-extracted landmark sequences (e.g. .npy per sample) + label file, or
- Video paths + labels (we extract landmarks on the fly or via a preprocessing step).

This module provides a PyTorch Dataset that loads sequences of shape (T, FEATURE_DIM)
and integer labels. Compatible with datasets like KArSL when data is prepared
in the expected format (see config for paths; do not hardcode).
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Tuple, Union

from ai.config import FEATURE_DIM, PAD_VALUE, DEFAULT_MAX_SEQ_LEN


class SequenceDataset(torch.utils.data.Dataset):
    """
    Dataset of landmark sequences and integer labels.
    Each item is (sequence, label) where sequence is (T, FEATURE_DIM), padded to max_len.
    """

    def __init__(
        self,
        sequences: List[np.ndarray],
        labels: List[int],
        max_len: Optional[int] = None,
        pad_value: float = PAD_VALUE,
    ):
        """
        sequences: list of arrays each (T_i, FEATURE_DIM)
        labels: list of integer class indices
        max_len: if None, use max T_i in this dataset
        """
        assert len(sequences) == len(labels)
        self.sequences = sequences
        self.labels = labels
        self.pad_value = pad_value
        self.max_len = max_len or max(s.shape[0] for s in sequences)
        self.max_len = min(self.max_len, DEFAULT_MAX_SEQ_LEN)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]  # (T, D)
        T, D = seq.shape
        if T > self.max_len:
            seq = seq[-self.max_len:]
            T = self.max_len
        # Pad to max_len
        padded = np.full(
            (self.max_len, D), fill_value=self.pad_value, dtype=np.float32
        )
        padded[-T:] = seq
        x = torch.from_numpy(padded)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def load_sequence_dataset(
    data_dir: Path,
    split: str = "train",
    max_len: Optional[int] = None,
) -> Tuple[SequenceDataset, List[str]]:
    """
    Load dataset from data_dir. Expected structure (pluggable):
        data_dir/
            train_sequences.npy   (or train/X.npy)  - list or array of (T, FEATURE_DIM)
            train_labels.npy     - (N,) integer labels
            labels.json          - list of label strings (index -> Arabic word)

    For KArSL-style: you can preprocess videos to train_sequences.npy and train_labels.npy
    and place labels.json. This function expects:
        {split}_sequences.npy: np.ndarray of shape (N,) dtype=object, each element (T, D)
        or (N, max_T, D) padded array
        {split}_labels.npy: (N,) int
        labels.json: list of strings

    Returns:
        dataset, label_list
    """
    data_dir = Path(data_dir)
    seq_path = data_dir / f"{split}_sequences.npy"
    label_path = data_dir / f"{split}_labels.npy"
    labels_json = data_dir / "labels.json"

    if not seq_path.exists():
        raise FileNotFoundError(
            f"Sequence file not found: {seq_path}. "
            "Prepare data in format {split}_sequences.npy, {split}_labels.npy, labels.json"
        )
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    sequences = np.load(seq_path, allow_pickle=True)
    if sequences.ndim == 3:
        # (N, T, D)
        sequences = [sequences[i] for i in range(len(sequences))]
    else:
        sequences = list(sequences)

    labels = np.load(label_path).astype(np.int64)

    if labels_json.exists():
        import json
        with open(labels_json, "r", encoding="utf-8") as f:
            label_list = json.load(f)
    else:
        label_list = [str(i) for i in range(int(labels.max()) + 1)]

    dataset = SequenceDataset(sequences, labels.tolist(), max_len=max_len)
    return dataset, label_list
