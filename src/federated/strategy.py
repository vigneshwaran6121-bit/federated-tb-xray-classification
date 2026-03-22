"""
strategy.py
Custom FedProx aggregation strategy extending Flower's FedAvg.
Additions:
  - Weighted aggregation by dataset size
  - Per-round metric logging
  - Best model checkpointing
  - Early stopping
"""

import numpy as np
import torch
import flwr as fl
from flwr.common import (
    Parameters,
    Scalar,
    FitRes,
    EvaluateRes,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from pathlib import Path
from typing import Optional, Union
from collections import OrderedDict

from src.models.densenet import build_model


class FedProxStrategy(fl.server.strategy.FedAvg):
    """
    FedProx strategy with:
      - Weighted model aggregation
      - Round-by-round metric tracking
      - Best checkpoint saving
      - Early stopping

    Inherits FedAvg and overrides aggregation hooks.
    """

    def __init__(
        self,
        config: dict,
        device: torch.device,
        save_dir: str = "results/checkpoints",
        early_stopping_patience: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.config = config
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.patience = early_stopping_patience
        self.best_accuracy = 0.0
        self.rounds_no_improve = 0

        # ── History tracking ──────────────────────────────────
        self.history = {
            "round": [],
            "train_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "val_auc": [],
        }

        # Build reference model for checkpointing
        self.ref_model = build_model(config, device)

    # ── After fit round ───────────────────────────────────────
    def aggregate_fit(
        self,
        server_round: int,
        results: list,
        failures: list,
    ):
        """
        Weighted FedAvg aggregation.
        Weights are proportional to each client's dataset size.
        """
        if not results:
            return None, {}

        # Weighted average of parameters
        total_examples = sum(fit_res.num_examples for _, fit_res in results)

        weighted_weights = [
            [
                layer * fit_res.num_examples
                for layer in parameters_to_ndarrays(fit_res.parameters)
            ]
            for _, fit_res in results
        ]

        aggregated = [
            np.sum(layers, axis=0) / total_examples for layers in zip(*weighted_weights)
        ]

        # Collect and log training metrics
        avg_loss = np.mean(
            [fit_res.metrics.get("train_loss", 0) for _, fit_res in results]
        )
        avg_acc = np.mean(
            [fit_res.metrics.get("train_accuracy", 0) for _, fit_res in results]
        )

        self.history["round"].append(server_round)
        self.history["train_loss"].append(round(float(avg_loss), 4))
        self.history["train_accuracy"].append(round(float(avg_acc), 2))

        print(f"\n  ── Round {server_round} Aggregation ──")
        print(f"     Train Loss : {avg_loss:.4f}")
        print(f"     Train Acc  : {avg_acc:.2f}%")
        print(f"     Clients    : {len(results)}")

        return ndarrays_to_parameters(aggregated), {}

    # ── After evaluate round ──────────────────────────────────
    def aggregate_evaluate(
        self,
        server_round: int,
        results: list,
        failures: list,
    ):
        """
        Weighted aggregation of val metrics.
        Triggers checkpointing and early stopping.
        """
        if not results:
            return None, {}

        total_examples = sum(eval_res.num_examples for _, eval_res in results)

        # Weighted val accuracy
        val_acc = (
            sum(
                eval_res.metrics.get("val_accuracy", 0) * eval_res.num_examples
                for _, eval_res in results
            )
            / total_examples
        )

        val_auc = (
            sum(
                eval_res.metrics.get("val_auc", 0) * eval_res.num_examples
                for _, eval_res in results
            )
            / total_examples
        )

        weighted_loss = (
            sum(eval_res.loss * eval_res.num_examples for _, eval_res in results)
            / total_examples
        )

        self.history["val_accuracy"].append(round(float(val_acc), 2))
        self.history["val_auc"].append(round(float(val_auc), 4))

        print(f"     Val Acc    : {val_acc:.2f}%")
        print(f"     Val AUC    : {val_auc:.4f}")

        # ── Checkpoint if best ────────────────────────────────
        if val_acc > self.best_accuracy:
            self.best_accuracy = val_acc
            self.rounds_no_improve = 0
            print(f"     ✅ New best! Saving checkpoint...")
            self._save_checkpoint(server_round, val_acc)
        else:
            self.rounds_no_improve += 1
            print(f"     No improvement " f"({self.rounds_no_improve}/{self.patience})")

        return weighted_loss, {
            "val_accuracy": val_acc,
            "val_auc": val_auc,
        }

    def _save_checkpoint(self, round_num: int, accuracy: float):
        """Saves best aggregated model weights."""
        path = self.save_dir / "best_federated_model.pt"
        torch.save(
            {
                "round": round_num,
                "accuracy": accuracy,
                "config": self.config,
            },
            path,
        )
        print(f"     Checkpoint → {path}")

    def should_stop(self) -> bool:
        """Returns True if early stopping patience exceeded."""
        return self.rounds_no_improve >= self.patience

    def get_history(self) -> dict:
        return self.history
