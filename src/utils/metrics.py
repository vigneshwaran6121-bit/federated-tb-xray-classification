"""
metrics.py
Evaluation metrics for binary TB classification:
  - Accuracy
  - AUC-ROC
  - F1 Score
  - Confusion Matrix
  - Per-class metrics
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


class MetricTracker:
    """
    Tracks and accumulates metrics across batches during training.
    Reset at the start of each epoch.

    Usage:
        tracker = MetricTracker()
        for batch in loader:
            loss, preds, labels = ...
            tracker.update(loss, preds, labels)
        results = tracker.compute()
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._losses = []
        self._preds = []
        self._probs = []  # softmax probabilities for AUC
        self._labels = []

    def update(
        self,
        loss: float,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ):
        """
        Args:
            loss   : scalar loss value for this batch
            logits : raw model output [batch, num_classes]
            labels : ground truth [batch]
        """
        self._losses.append(loss)

        probs = torch.softmax(logits.detach().cpu(), dim=1)
        preds = torch.argmax(probs, dim=1)

        self._probs.extend(probs[:, 1].numpy())  # prob of TB class
        self._preds.extend(preds.numpy())
        self._labels.extend(labels.detach().cpu().numpy())

    def compute(self) -> dict:
        """Computes all metrics from accumulated batch data."""
        labels = np.array(self._labels)
        preds = np.array(self._preds)
        probs = np.array(self._probs)

        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        cm = confusion_matrix(labels, preds)

        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            auc = 0.0  # only one class present in batch

        avg_loss = float(np.mean(self._losses))

        return {
            "loss": round(avg_loss, 4),
            "accuracy": round(acc * 100, 2),  # percentage
            "auc_roc": round(auc, 4),
            "f1_score": round(f1, 4),
            "confusion_matrix": cm.tolist(),
        }


def evaluate_model(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    criterion: torch.nn.Module = None,
) -> dict:
    """
    Full evaluation pass over a DataLoader.

    Args:
        model     : trained PyTorch model
        loader    : DataLoader (val or test)
        device    : cuda or cpu
        criterion : loss function (optional)

    Returns:
        dict of metrics
    """
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    model.eval()
    tracker = MetricTracker()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            tracker.update(loss.item(), logits.cpu(), labels.cpu())

    return tracker.compute()


def print_metrics(metrics: dict, split: str = "Val", node: int = None):
    """Pretty-prints metrics to console."""
    prefix = f"Node {node} | " if node else ""
    print(
        f"  {prefix}{split:5s} → "
        f"Loss: {metrics['loss']:.4f} | "
        f"Acc: {metrics['accuracy']:.2f}% | "
        f"AUC: {metrics['auc_roc']:.4f} | "
        f"F1: {metrics['f1_score']:.4f}"
    )


def print_classification_report(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    classes: list = None,
):
    """Prints sklearn classification report with per-class metrics."""
    if classes is None:
        classes = ["Normal", "Tuberculosis"]

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print("\n" + classification_report(all_labels, all_preds, target_names=classes))
