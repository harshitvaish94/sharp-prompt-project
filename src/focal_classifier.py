"""Trainable classification system with frozen MiniLM embeddings and PyTorch MLP head.

Evaluates two loss functions:
  Configuration A: Cross-Entropy Loss
  Configuration B: Focal Loss (gamma = 2.0)

Key features:
1. Keeps sentence-transformers/all-MiniLM-L6-v2 encoder frozen.
2. Caches embeddings to disk to prevent redundant computation.
3. Uses identical train / val / test splits with configured SEED.
4. Uses validation Macro-F1 for model selection (checkpointing best epoch).
5. Evaluates final performance on test data using shared save_metrics interface.
"""
import copy
import json
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from src.metrics import compute_metrics, save_metrics  # noqa: E402


def set_seed(seed: int = C.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MLPClassifier(nn.Module):
    """Small PyTorch MLP classification head over sentence embeddings."""

    def __init__(self, in_features: int = 384, hidden_dim: int = 128, num_classes: int = 6, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FocalLoss(nn.Module):
    """Multi-class Focal Loss with modulating factor (1 - p_t)^gamma."""

    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (N, C), targets: (N,)
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce_loss)  # probability of true class
        focal_weight = (1.0 - p_t) ** self.gamma
        loss = focal_weight * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def get_cached_embeddings() -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Load or compute frozen MiniLM embeddings for train, val, and test splits."""
    cache_path = C.PROCESSED_DIR / "minilm_embeddings.pt"
    if cache_path.exists():
        print(f"Loading cached MiniLM embeddings from {cache_path}...")
        data = torch.load(cache_path, weights_only=False)
        return data

    print(f"Generating MiniLM embeddings using {C.EMB_NAME}...")
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(C.EMB_NAME)
    splits = {
        "train": pd.read_csv(C.TRAIN_CSV),
        "val": pd.read_csv(C.VAL_CSV),
        "test": pd.read_csv(C.TEST_CSV),
    }

    cached = {}
    for name, df in splits.items():
        print(f"Encoding {name} split ({len(df)} samples)...")
        texts = df["text"].tolist()
        labels = torch.tensor(df["label"].values, dtype=torch.long)
        embs = embedder.encode(texts, batch_size=64, show_progress_bar=False, convert_to_tensor=True)
        cached[name] = (embs.cpu(), labels)

    torch.save(cached, cache_path)
    print(f"Saved cached embeddings to {cache_path}")
    return cached


def train_and_eval_head(
    loss_type: str,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    gamma: float = 2.0,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = C.SEED,
) -> Tuple[dict, np.ndarray, dict, np.ndarray]:
    """Train MLP classification head and evaluate on test set.

    Returns:
        (test_metrics, test_preds, val_metrics, val_preds)
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLPClassifier(in_features=train_x.shape[1], hidden_dim=128, num_classes=len(C.LABELS)).to(device)

    if loss_type == "focal":
        criterion = FocalLoss(gamma=gamma)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_ds = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_val_f1 = -1.0
    best_weights = None

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            val_logits = model(val_x.to(device))
            val_preds = val_logits.argmax(dim=-1).cpu().numpy()
            val_metrics = compute_metrics(val_y.numpy(), val_preds, C.LABELS)

            if val_metrics["macro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["macro_f1"]
                best_weights = copy.deepcopy(model.state_dict())

    # Load best checkpoint
    model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        final_val_preds = model(val_x.to(device)).argmax(dim=-1).cpu().numpy()
        val_res = compute_metrics(val_y.numpy(), final_val_preds, C.LABELS)

        test_preds = model(test_x.to(device)).argmax(dim=-1).cpu().numpy()
        test_res = compute_metrics(test_y.numpy(), test_preds, C.LABELS)

    return test_res, test_preds, val_res, final_val_preds


def run_focal_experiment(save_path: Path = C.RESULTS_DIR / "focal_metrics.json") -> dict:
    """Execute complete MiniLM + Cross-Entropy and MiniLM + Focal Loss pipeline."""
    set_seed(C.SEED)
    cached = get_cached_embeddings()

    train_x, train_y = cached["train"]
    val_x, val_y = cached["val"]
    test_x, test_y = cached["test"]

    print("\n--- Training Configuration A: MiniLM + Cross-Entropy ---")
    ce_test_res, ce_test_preds, ce_val_res, _ = train_and_eval_head(
        loss_type="ce",
        train_x=train_x, train_y=train_y,
        val_x=val_x, val_y=val_y,
        test_x=test_x, test_y=test_y,
    )
    print(f"Cross-Entropy -> Val Macro-F1: {ce_val_res['macro_f1']:.4f} | Test Macro-F1: {ce_test_res['macro_f1']:.4f} | Test Acc: {ce_test_res['accuracy']:.4f}")

    print("\n--- Training Configuration B: MiniLM + Focal Loss (gamma=2.0) ---")
    focal_test_res, focal_test_preds, focal_val_res, _ = train_and_eval_head(
        loss_type="focal",
        train_x=train_x, train_y=train_y,
        val_x=val_x, val_y=val_y,
        test_x=test_x, test_y=test_y,
        gamma=2.0,
    )
    print(f"Focal Loss (gamma=2) -> Val Macro-F1: {focal_val_res['macro_f1']:.4f} | Test Macro-F1: {focal_test_res['macro_f1']:.4f} | Test Acc: {focal_test_res['accuracy']:.4f}")

    # Build comprehensive output JSON using standard save_metrics
    # Top-level represents the focal loss model, while also archiving cross-entropy
    out = save_metrics(
        save_path,
        method="MiniLM + Focal Loss (gamma=2)",
        y_true=test_y.numpy(),
        y_pred=focal_test_preds,
        labels=C.LABELS,
        gamma=2.0,
        configurations={
            "cross_entropy": {
                "method": "MiniLM + Cross-Entropy",
                "accuracy": ce_test_res["accuracy"],
                "macro_f1": ce_test_res["macro_f1"],
                "per_class_f1": ce_test_res["per_class_f1"],
                "confusion_matrix": ce_test_res["confusion_matrix"],
                "val_macro_f1": ce_val_res["macro_f1"],
            },
            "focal_loss": {
                "method": "MiniLM + Focal Loss (gamma=2)",
                "gamma": 2.0,
                "accuracy": focal_test_res["accuracy"],
                "macro_f1": focal_test_res["macro_f1"],
                "per_class_f1": focal_test_res["per_class_f1"],
                "confusion_matrix": focal_test_res["confusion_matrix"],
                "val_macro_f1": focal_val_res["macro_f1"],
            },
        },
        val_metrics={
            "cross_entropy": ce_val_res,
            "focal_loss": focal_val_res,
        },
    )

    print(f"\nWrote results to {save_path}")
    return out


if __name__ == "__main__":
    run_focal_experiment()
