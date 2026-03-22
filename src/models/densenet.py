"""
densenet.py
DenseNet121 backbone fine-tuned for binary TB classification.
- ImageNet pretrained weights
- Custom classifier head with dropout
- Supports feature extraction mode (freeze backbone)
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import DenseNet121_Weights


class TBClassifier(nn.Module):
    """
    DenseNet121 fine-tuned for Normal vs Tuberculosis classification.

    Architecture:
        DenseNet121 backbone (pretrained on ImageNet)
            └── Custom head:
                    AdaptiveAvgPool
                    Flatten
                    Dropout(p)
                    Linear(1024 → 256)
                    ReLU
                    Dropout(p)
                    Linear(256 → num_classes)

    Args:
        num_classes     : number of output classes (default: 2)
        dropout         : dropout probability (default: 0.3)
        freeze_backbone : if True, only train the classifier head
        pretrained      : use ImageNet pretrained weights
    """

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
        pretrained: bool = True,
    ):
        super(TBClassifier, self).__init__()

        # ── Load DenseNet121 backbone ────────────────────────
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        densenet = models.densenet121(weights=weights)

        # ── Extract feature layers (everything except classifier) ──
        self.features = densenet.features
        self.num_features = densenet.classifier.in_features  # 1024

        # ── Optionally freeze backbone ───────────────────────
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        # ── Custom classification head ───────────────────────
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(self.num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        features = torch.relu(features)
        out = self.classifier(features)
        return out

    def get_num_params(self) -> dict:
        """Returns total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def unfreeze_backbone(self):
        """Unfreeze all backbone layers for full fine-tuning."""
        for param in self.features.parameters():
            param.requires_grad = True

    def freeze_backbone(self):
        """Freeze backbone — only train the head."""
        for param in self.features.parameters():
            param.requires_grad = False


def build_model(config: dict, device: torch.device) -> TBClassifier:
    """
    Builds and returns the model from config.

    Args:
        config : loaded yaml config dict
        device : torch.device (cuda / cpu)

    Returns:
        model moved to device
    """
    model_cfg = config["model"]

    model = TBClassifier(
        num_classes=model_cfg["num_classes"],
        dropout=model_cfg["dropout"],
        pretrained=model_cfg["pretrained"],
        freeze_backbone=False,
    )

    model = model.to(device)

    params = model.get_num_params()
    print(f"\n  Model   : DenseNet121 + Custom Head")
    print(f"  Total params    : {params['total']:,}")
    print(f"  Trainable params: {params['trainable']:,}")
    print(f"  Device          : {device}\n")

    return model


def get_optimizer(model: TBClassifier, config: dict) -> torch.optim.Optimizer:
    """Adam optimizer with weight decay from config."""
    train_cfg = config["training"]
    return torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )


def get_scheduler(optimizer, config: dict, num_steps: int):
    """Cosine annealing LR scheduler."""
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_steps,
        eta_min=1e-6,
    )
