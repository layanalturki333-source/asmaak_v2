"""
Training pipeline for the sign recognition model.

Uses cross-entropy loss and Adam. Supports checkpoint saving and basic error handling.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, List, Dict, Any

from ai.config import (
    LEARNING_RATE,
    NUM_EPOCHS,
    BATCH_SIZE,
    MODELS_DIR,
    LOGS_DIR,
    ensure_dirs,
)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one epoch; return average loss."""
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    device: Optional[torch.device] = None,
    save_dir: Optional[Path] = None,
    label_list: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Full training loop. Saves best model (by train loss) and label list.
    Returns dict with loss history and best loss.
    """
    ensure_dirs()
    save_dir = Path(save_dir or MODELS_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model = model.to(device)

    history = []
    best_loss = float("inf")
    best_path = save_dir / "asl_bilstm_best.pt"

    for epoch in range(num_epochs):
        try:
            avg_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        except Exception as e:
            raise RuntimeError(f"Training failed at epoch {epoch}: {e}") from e
        history.append(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "num_classes": model.classifier[-1].out_features,
            }
            if label_list is not None:
                state["label_list"] = label_list
            torch.save(state, best_path)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}  loss={avg_loss:.4f}")

    return {"loss_history": history, "best_loss": best_loss, "best_path": str(best_path)}
