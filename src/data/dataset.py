"""
dataset.py
PyTorch Dataset class for chest X-ray images.
Handles loading, augmentation, and normalization.
"""

import os
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ── ImageNet normalization stats (used for pretrained DenseNet) ──
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(split: str, img_size: int = 224) -> transforms.Compose:
    """
    Returns appropriate transforms for each split.
    Train: augmentation + normalization
    Val/Test: resize + normalization only
    """
    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize((img_size + 20, img_size + 20)),
                transforms.RandomCrop(img_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.Grayscale(num_output_channels=3),  # X-rays are grayscale
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    else:  # val / test
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )


class ChestXRayDataset(Dataset):
    """
    Loads chest X-ray images from a directory with structure:
        root/
          Normal/       image1.png, image2.png ...
          Tuberculosis/ image1.png, image2.png ...

    Args:
        root_dir  : path to the split folder
        split     : 'train' | 'val' | 'test'
        img_size  : resize target (default 224 for DenseNet121)
        classes   : list of class names (order defines label index)
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        img_size: int = 224,
        classes: list = None,
    ):
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.classes = classes or ["Normal", "Tuberculosis"]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = get_transforms(split, img_size)

        self.samples = self._load_samples()

    def _load_samples(self) -> list:
        """Walks root_dir and collects (image_path, label) pairs."""
        samples = []
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            if not cls_dir.exists():
                print(f"  [WARNING] Folder not found: {cls_dir}")
                continue
            label = self.class_to_idx[cls]
            for ext in ("*.png", "*.jpg", "*.jpeg"):
                for img_path in cls_dir.glob(ext):
                    samples.append((str(img_path), label))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

    def class_distribution(self) -> dict:
        """Returns count per class — useful for checking imbalance."""
        dist = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            cls = self.classes[label]
            dist[cls] += 1
        return dist


def create_dataloaders(
    node_dir: str,
    batch_size: int = 32,
    img_size: int = 224,
    val_split: float = 0.10,
    num_workers: int = 4,
    seed: int = 42,
    classes: list = None,
):
    """
    Creates train and val DataLoaders for a single hospital node.
    Splits node data into train/val internally.

    Returns: (train_loader, val_loader, class_distribution)
    """
    from torch.utils.data import random_split, DataLoader

    full_dataset = ChestXRayDataset(
        root_dir=node_dir,
        split="train",
        img_size=img_size,
        classes=classes,
    )

    n_val = int(len(full_dataset) * val_split)
    n_train = len(full_dataset) - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_dataset, [n_train, n_val], generator=generator)

    # Val set should use test transforms (no augmentation)
    val_ds.dataset.transform = get_transforms("val", img_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, full_dataset.class_distribution()


def create_test_loader(
    test_dir: str,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    classes: list = None,
):
    """Creates DataLoader for the global held-out test set."""
    from torch.utils.data import DataLoader

    test_ds = ChestXRayDataset(
        root_dir=test_dir,
        split="test",
        img_size=img_size,
        classes=classes,
    )
    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
