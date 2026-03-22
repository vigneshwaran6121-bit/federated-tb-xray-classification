"""
evaluate.py
Standalone evaluation script.
Loads any checkpoint and evaluates on the test set.
Run: python evaluate.py --model centralized
     python evaluate.py --model federated
"""

import torch
import yaml
import argparse
from pathlib import Path

from src.models.densenet import build_model
from src.data.dataset import create_test_loader
from src.utils.metrics import (
    evaluate_model,
    print_metrics,
    print_classification_report,
)
from src.utils.visualize import plot_confusion_matrix


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="centralized",
        choices=["centralized", "federated"],
    )
    args = parser.parse_args()
    config = load_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = Path(config["results"]["checkpoints_dir"])
    ckpt_map = {
        "centralized": ckpt_dir / "best_centralized_model.pt",
        "federated": ckpt_dir / "best_federated_model.pt",
    }

    ckpt_path = ckpt_map[args.model]

    print(f"\n{'='*55}")
    print(f"  EVALUATION — {args.model.upper()} MODEL")
    print(f"{'='*55}")

    if not ckpt_path.exists():
        print(f"  ❌ Checkpoint not found: {ckpt_path}")
        return

    model = build_model(config, device)
    ckpt = torch.load(ckpt_path, map_location=device)

    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
        print(f"  Checkpoint epoch : {ckpt.get('epoch', 'N/A')}")
        print(f"  Val accuracy     : {ckpt.get('val_accuracy', 0):.2f}%")

    test_loader = create_test_loader(
        test_dir=config["data"]["test_dir"],
        batch_size=config["training"]["batch_size"],
        img_size=config["data"]["img_size"],
        num_workers=config["data"]["num_workers"],
        classes=config["data"]["classes"],
    )

    test_metrics = evaluate_model(model, test_loader, device)

    print(f"\n  ── Test Set Results ──")
    print_metrics(test_metrics, split="Test")
    print_classification_report(model, test_loader, device, config["data"]["classes"])

    plot_confusion_matrix(
        cm=test_metrics["confusion_matrix"],
        classes=config["data"]["classes"],
        save_path=(
            f"{config['results']['plots_dir']}/" f"{args.model}_confusion_matrix.png"
        ),
        title=f"{args.model.capitalize()} Model — Confusion Matrix",
    )


if __name__ == "__main__":
    main()
