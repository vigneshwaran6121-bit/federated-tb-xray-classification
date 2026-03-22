"""
train_centralized.py
Centralised baseline training on ALL data combined.
Used to benchmark against the federated approach.
Run: python train_centralized.py
"""

import os
import yaml
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from src.data.dataset import ChestXRayDataset, get_transforms, create_test_loader
from src.models.densenet import build_model, get_optimizer, get_scheduler
from src.utils.metrics import (
    MetricTracker,
    print_metrics,
    evaluate_model,
    print_classification_report,
)
from src.utils.visualize import plot_training_curves, plot_confusion_matrix


# ── Load config ───────────────────────────────────────────────
def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Build combined dataset from all 3 nodes ───────────────────
def build_combined_loaders(config: dict):
    """
    Merges data from all 3 hospital nodes into one dataset.
    This is the centralised setting — all data in one place.
    """
    data_cfg = config["data"]
    train_cfg = config["training"]

    all_train, all_val = [], []

    for node_id in range(1, config["federated"]["num_clients"] + 1):
        node_dir = Path(data_cfg["splits_dir"]) / f"node_{node_id}"

        full_ds = ChestXRayDataset(
            root_dir=str(node_dir),
            split="train",
            img_size=data_cfg["img_size"],
            classes=data_cfg["classes"],
        )

        n_val = int(len(full_ds) * data_cfg["val_split"])
        n_train = len(full_ds) - n_val

        generator = torch.Generator().manual_seed(config["project"]["seed"])
        from torch.utils.data import random_split

        train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=generator)
        val_ds.dataset.transform = get_transforms("val", data_cfg["img_size"])

        all_train.append(train_ds)
        all_val.append(val_ds)

        dist = full_ds.class_distribution()
        print(f"  Node {node_id}: {dist}")

    combined_train = ConcatDataset(all_train)
    combined_val = ConcatDataset(all_val)

    train_loader = DataLoader(
        combined_train,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        combined_val,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )

    return train_loader, val_loader


# ── One training epoch ────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, criterion, device) -> dict:
    model.train()
    tracker = MetricTracker()

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()
        tracker.update(loss.item(), logits, labels)

    return tracker.compute()


# ── Main ──────────────────────────────────────────────────────
def main():
    config = load_config()
    seed = config["project"]["seed"]
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print("      CENTRALISED BASELINE TRAINING")
    print(f"{'='*55}")
    print(f"  Device : {device}")

    # ── Data ──────────────────────────────────────────────────
    print("\n► Loading combined dataset from all nodes...")
    train_loader, val_loader = build_combined_loaders(config)
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val   batches : {len(val_loader)}")

    # ── Model ─────────────────────────────────────────────────
    model = build_model(config, device)
    optimizer = get_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()

    total_steps = config["centralized"]["epochs"] * len(train_loader)
    scheduler = get_scheduler(optimizer, config, total_steps)

    # ── Training ──────────────────────────────────────────────
    epochs = config["centralized"]["epochs"]
    patience = config["centralized"]["early_stopping_patience"]
    best_val_acc = 0.0
    no_improve = 0
    ckpt_path = Path(config["results"]["checkpoints_dir"]) / "best_centralized_model.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    print(f"\n► Training for {epochs} epochs...\n")

    for epoch in range(1, epochs + 1):
        print(f"  Epoch [{epoch:02d}/{epochs}]")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        print_metrics(train_metrics, split="Train")

        # Validate
        val_metrics = evaluate_model(model, val_loader, device, criterion)
        print_metrics(val_metrics, split="Val  ")

        # Record history
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])

        # Checkpoint
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_accuracy": best_val_acc,
                    "config": config,
                },
                ckpt_path,
            )
            print(f"  ✅ Checkpoint saved (acc: {best_val_acc:.2f}%)")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\n  Early stopping at epoch {epoch}")
                break

        print()

    # ── Test set evaluation ───────────────────────────────────
    print("► Loading best model for test evaluation...")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    test_loader = create_test_loader(
        test_dir=config["data"]["test_dir"],
        batch_size=config["training"]["batch_size"],
        img_size=config["data"]["img_size"],
        num_workers=config["data"]["num_workers"],
        classes=config["data"]["classes"],
    )

    test_metrics = evaluate_model(model, test_loader, device, criterion)
    print(f"\n{'='*55}")
    print("  CENTRALISED TEST RESULTS")
    print(f"{'='*55}")
    print_metrics(test_metrics, split="Test ")
    print_classification_report(model, test_loader, device, config["data"]["classes"])

    # ── Plots ─────────────────────────────────────────────────
    plots_dir = config["results"]["plots_dir"]

    plot_training_curves(
        history={
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
            "train_acc": history["train_acc"],
            "val_acc": history["val_acc"],
        },
        save_path=f"{plots_dir}/centralized_training_curves.png",
        title="Centralised Baseline — Training Curves",
    )

    plot_confusion_matrix(
        cm=test_metrics["confusion_matrix"],
        classes=config["data"]["classes"],
        save_path=f"{plots_dir}/centralized_confusion_matrix.png",
        title="Centralised Baseline — Confusion Matrix",
    )

    print(f"\n✅ Centralised training complete!")
    print(f"   Best Val Accuracy : {best_val_acc:.2f}%")
    print(f"   Test Accuracy     : {test_metrics['accuracy']:.2f}%")
    print(f"   Test AUC-ROC      : {test_metrics['auc_roc']:.4f}\n")


if __name__ == "__main__":
    main()
