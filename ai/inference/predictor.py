"""
Model loading and prediction for ArSL sign recognition.

Loads saved checkpoint (model + optional label list), runs forward pass,
returns predicted class index and label string.
"""

import torch
from pathlib import Path
from typing import List, Optional, Tuple, Union

from ai.models import SignBiLSTM
from ai.config import FEATURE_DIM, HIDDEN_SIZE, NUM_LSTM_LAYERS, DROPOUT, BIDIRECTIONAL


def load_model(
    checkpoint_path: Union[str, Path],
    device: Optional[torch.device] = None,
) -> Tuple[SignBiLSTM, List[str]]:
    """
    Load model and label list from checkpoint.
    Checkpoint should contain: model_state_dict, num_classes, and optionally label_list.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    num_classes = ckpt.get("num_classes", 10)
    label_list = ckpt.get("label_list")
    if label_list is None:
        label_list = [str(i) for i in range(num_classes)]

    model = SignBiLSTM(
        input_size=FEATURE_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LSTM_LAYERS,
        num_classes=num_classes,
        dropout=DROPOUT,
        bidirectional=BIDIRECTIONAL,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model, label_list


class SignPredictor:
    """
    Wrapper for loading model once and predicting on landmark sequences.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.label_list = load_model(checkpoint_path, self.device)

    def predict(self, sequence: torch.Tensor) -> Tuple[int, str]:
        """
        sequence: (1, T, FEATURE_DIM) or (T, FEATURE_DIM)
        Returns (class_index, label_string).
        """
        if sequence.dim() == 2:
            sequence = sequence.unsqueeze(0)
        sequence = sequence.to(self.device)
        with torch.no_grad():
            logits = self.model(sequence)
            pred = logits.argmax(dim=1).item()
        label = self.label_list[pred] if pred < len(self.label_list) else str(pred)
        return pred, label
