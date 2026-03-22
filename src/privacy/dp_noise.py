"""
dp_noise.py
Differential Privacy via Opacus.
Wraps a model + optimizer with DP guarantees:
  - Gradient clipping (sensitivity)
  - Gaussian noise injection (privacy)
"""

import torch
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator


def make_dp_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    config: dict,
):
    """
    Attaches Opacus PrivacyEngine to model and optimizer.

    Args:
        model        : TBClassifier
        optimizer    : Adam optimizer
        train_loader : training DataLoader
        config       : full yaml config

    Returns:
        (dp_model, dp_optimizer, dp_loader, privacy_engine)
    """
    privacy_cfg = config["privacy"]

    # Opacus requires BatchNorm → GroupNorm conversion
    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)
        print("  ✓ Model layers converted for DP compatibility")

    privacy_engine = PrivacyEngine()

    dp_model, dp_optimizer, dp_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        epochs=config["training"]["epochs_per_round"],
        target_epsilon=privacy_cfg["epsilon"],
        target_delta=privacy_cfg["delta"],
        max_grad_norm=privacy_cfg["max_grad_norm"],
    )

    print(
        f"  DP enabled → ε={privacy_cfg['epsilon']}, "
        f"δ={privacy_cfg['delta']}, "
        f"max_grad_norm={privacy_cfg['max_grad_norm']}"
    )

    return dp_model, dp_optimizer, dp_loader, privacy_engine


def get_privacy_spent(privacy_engine) -> tuple:
    """Returns current (epsilon, delta) spent so far."""
    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    return epsilon, 1e-5
