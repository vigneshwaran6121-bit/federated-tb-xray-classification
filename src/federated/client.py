"""
client.py
Flower federated learning client.
Each client represents one hospital node.
Responsibilities:
  - Load its local data
  - Train locally for N epochs
  - Send updated weights to server
  - Evaluate on local val set
"""

import torch
import torch.nn as nn
import yaml
import flwr as fl
import numpy as np
from collections import OrderedDict
from pathlib import Path

from src.data.dataset import create_dataloaders
from src.models.densenet import build_model, get_optimizer, get_scheduler
from src.utils.metrics import MetricTracker, print_metrics
from src.privacy.dp_noise import make_dp_model, get_privacy_spent


# ── Helpers ───────────────────────────────────────────────────
def get_parameters(model: nn.Module) -> list:
    """Extract model weights as list of numpy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: list):
    """Load list of numpy arrays back into model weights."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


# ── Flower Client ─────────────────────────────────────────────
class HospitalClient(fl.client.NumPyClient):
    """
    Flower NumPyClient for one hospital node.

    Each hospital:
      1. Receives global model weights from server
      2. Trains locally for `epochs_per_round` epochs
      3. Returns updated weights + training metrics
      4. Evaluates on local val set when asked

    Args:
        node_id    : integer node identifier (1, 2, 3)
        config     : full yaml config dict
        device     : torch.device
        use_dp     : enable differential privacy
    """

    def __init__(
        self,
        node_id: int,
        config: dict,
        device: torch.device,
        use_dp: bool = True,
    ):
        self.node_id = node_id
        self.config = config
        self.device = device
        self.use_dp = use_dp and config["privacy"]["enabled"]

        train_cfg = config["training"]
        data_cfg = config["data"]
        fed_cfg = config["federated"]

        node_dir = Path(data_cfg["splits_dir"]) / f"node_{node_id}"

        # ── Data ──────────────────────────────────────────────
        self.train_loader, self.val_loader, dist = create_dataloaders(
            node_dir=str(node_dir),
            batch_size=train_cfg["batch_size"],
            img_size=data_cfg["img_size"],
            val_split=data_cfg["val_split"],
            num_workers=data_cfg["num_workers"],
            seed=config["project"]["seed"],
            classes=data_cfg["classes"],
        )

        print(f"\n  Node {node_id} data distribution: {dist}")

        # ── Model ─────────────────────────────────────────────
        self.model = build_model(config, device)
        self.optimizer = get_optimizer(self.model, config)
        self.criterion = nn.CrossEntropyLoss()
        self.epochs = train_cfg["epochs_per_round"]

        # ── FedProx proximal term ─────────────────────────────
        self.mu = fed_cfg.get("fedprox_mu", 0.01)

        # ── Differential Privacy ──────────────────────────────
        if self.use_dp:
            (
                self.model,
                self.optimizer,
                self.train_loader,
                self.privacy_engine,
            ) = make_dp_model(
                self.model,
                self.optimizer,
                self.train_loader,
                config,
            )

        # ── Scheduler ─────────────────────────────────────────
        total_steps = self.epochs * len(self.train_loader)
        self.scheduler = get_scheduler(self.optimizer, config, total_steps)

    # ── Flower API ────────────────────────────────────────────
    def get_parameters(self, config=None) -> list:
        return get_parameters(self.model)

    def fit(self, parameters: list, config: dict):
        """
        Called by server each round.
        1. Load global weights
        2. Train locally (with FedProx regularisation)
        3. Return updated weights + metrics
        """
        # Load global model weights
        set_parameters(self.model, parameters)

        # Keep a copy of global weights for FedProx penalty
        global_params = [p.data.clone() for p in self.model.parameters()]

        # ── Training loop ──────────────────────────────────────
        self.model.train()
        tracker = MetricTracker()

        for epoch in range(self.epochs):
            tracker.reset()

            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(images)

                # Cross-entropy loss
                ce_loss = self.criterion(logits, labels)

                # FedProx proximal term:
                # penalises deviation from global model
                prox_term = 0.0
                for local_p, global_p in zip(self.model.parameters(), global_params):
                    prox_term += ((local_p - global_p.to(self.device)) ** 2).sum()

                loss = ce_loss + (self.mu / 2) * prox_term
                loss.backward()

                self.optimizer.step()
                self.scheduler.step()
                tracker.update(ce_loss.item(), logits, labels)

            metrics = tracker.compute()
            print_metrics(metrics, split="Train", node=self.node_id)

        # ── Privacy budget report ──────────────────────────────
        if self.use_dp:
            eps, delta = get_privacy_spent(self.privacy_engine)
            print(f"  Node {self.node_id} | DP spent → ε={eps:.3f}, δ={delta}")

        final_metrics = tracker.compute()
        num_examples = len(self.train_loader.dataset)

        return (
            get_parameters(self.model),
            num_examples,
            {
                "train_loss": final_metrics["loss"],
                "train_accuracy": final_metrics["accuracy"],
                "node_id": float(self.node_id),
            },
        )

    def evaluate(self, parameters: list, config: dict):
        """
        Called by server to evaluate on local val set.
        Returns loss, num_examples, metrics dict.
        """
        set_parameters(self.model, parameters)

        self.model.eval()
        tracker = MetricTracker()
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images)
                loss = criterion(logits, labels)
                tracker.update(loss.item(), logits, labels)

        metrics = tracker.compute()
        print_metrics(metrics, split="Val", node=self.node_id)

        return (
            metrics["loss"],
            len(self.val_loader.dataset),
            {
                "val_accuracy": metrics["accuracy"],
                "val_auc": metrics["auc_roc"],
                "val_f1": metrics["f1_score"],
            },
        )


def create_client_fn(config: dict, device: torch.device, use_dp: bool = True):
    """
    Factory function for Flower simulation.
    Returns a function that creates a client given a node cid string.
    """

    def client_fn(cid: str) -> HospitalClient:
        node_id = int(cid) + 1  # cid is 0-indexed, nodes are 1-indexed
        return HospitalClient(
            node_id=node_id,
            config=config,
            device=device,
            use_dp=use_dp,
        )

    return client_fn
