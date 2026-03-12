"""
BiLSTM sequence classifier for isolated Arabic sign recognition.

Input: (batch, seq_len, feature_dim) landmark sequences.
Output: (batch, num_classes) logits.
"""

import torch
import torch.nn as nn
from typing import Optional

from ai.config import (
    FEATURE_DIM,
    HIDDEN_SIZE,
    NUM_LSTM_LAYERS,
    DROPOUT,
    BIDIRECTIONAL,
)


class SignBiLSTM(nn.Module):
    """
    Bidirectional LSTM for sequence classification.
    Uses the final hidden state (concatenated forward+backward) passed through a classifier head.
    """

    def __init__(
        self,
        input_size: int = FEATURE_DIM,
        hidden_size: int = HIDDEN_SIZE,
        num_layers: int = NUM_LSTM_LAYERS,
        num_classes: int = 10,
        dropout: float = DROPOUT,
        bidirectional: bool = BIDIRECTIONAL,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        hidden_out = hidden_size * self.num_directions
        self.classifier = nn.Sequential(
            nn.Linear(hidden_out, hidden_out),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_out, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        out: (B, num_classes)
        """
        # lstm out: (B, T, hidden*directions), (h_n, c_n)
        out, (h_n, _) = self.lstm(x)
        # Take last time step from last layer: h_n (num_layers*directions, B, hidden)
        last_h = h_n[-1]  # (B, hidden) if unidirectional
        if self.bidirectional:
            # h_n: (2, B, hidden) -> concat -> (B, 2*hidden)
            last_h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        logits = self.classifier(last_h)
        return logits
