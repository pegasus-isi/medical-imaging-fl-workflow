#!/usr/bin/env python3
"""Train a centralized baseline model on pooled data from all clients.

This provides the upper-bound accuracy that FL should approach.
Runs in PARALLEL with cross-dataset evaluation (no mutual dependency).
"""

import argparse
import json
import socket
import tarfile
from pathlib import Path

import torch

from resource_monitor import ResourceMonitor
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, DataLoader, Dataset

import yaml


class PooledImageDataset(Dataset):
    """Combined dataset from all client shards."""

    def __init__(self, data_dirs: list, image_size: tuple = (224, 224), augment: bool = False):
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size[0], scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        self.samples = []
        self.labels = []

        for data_dir in data_dirs:
            train_dir = data_dir / "train"
            if train_dir.exists():
                for label_dir in sorted(train_dir.iterdir()):
                    if label_dir.is_dir():
                        label = int(label_dir.name) if label_dir.name.isdigit() else 0
                        for img_path in label_dir.glob("*"):
                            self.samples.append(img_path)
                            self.labels.append(label)

        if not self.samples:
            print("  No real images — using synthetic data")
            self.synthetic = True
            self.num_samples = 500
        else:
            self.synthetic = False

    def __len__(self):
        return self.num_samples if self.synthetic else len(self.samples)

    def __getitem__(self, idx):
        if self.synthetic:
            return torch.randn(3, 224, 224), torch.randint(0, 2, (1,)).item()
        from PIL import Image
        img = Image.open(self.samples[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-arch", default="resnet18")
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--output-metrics", required=True)
    parser.add_argument("--config", required=True)
    args, _ = parser.parse_known_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg["datasets"][args.dataset]
    fl_cfg = cfg["fl"]
    num_clients = ds_cfg["num_clients"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Read new config params with backward-compatible defaults
    freeze_backbone = fl_cfg.get("freeze_backbone", False)
    optimizer_type = fl_cfg.get("optimizer", "sgd")
    class_weighted_loss = fl_cfg.get("class_weighted_loss", False)
    use_augmentation = fl_cfg.get("augmentation", False)
    grad_clip = fl_cfg.get("grad_clip", 0.0)

    print(f"Centralized baseline: {args.dataset}, {args.model_arch}, device={device}")

    # Start resource monitoring
    monitor = ResourceMonitor(interval=5.0)
    monitor.start()

    # Extract all client data
    data_dirs = []
    for i in range(num_clients):
        tar_path = f"{args.dataset}_client_{i}_data.tar.gz"
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(".")
        data_dirs.append(Path(f"client_{i}"))

    # Build pooled dataset
    dataset = PooledImageDataset(data_dirs, augment=use_augmentation)
    dataloader = DataLoader(
        dataset, batch_size=fl_cfg["batch_size"], shuffle=True,
        num_workers=4, pin_memory=(device == "cuda"),
    )

    # Build model
    num_classes = ds_cfg.get("num_classes", 2)
    if args.model_arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif args.model_arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes
        )
    model.to(device)

    # Freeze backbone if configured
    if freeze_backbone:
        for name, param in model.named_parameters():
            if "fc" not in name and "classifier" not in name:
                param.requires_grad = False
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # Train for equivalent total epochs: num_rounds * local_epochs
    total_epochs = fl_cfg["num_rounds"] * fl_cfg["local_epochs"]
    # Cap at a reasonable number for centralized
    total_epochs = min(total_epochs, 100)

    # Class-weighted loss
    import numpy as np
    if class_weighted_loss and hasattr(dataset, 'labels') and dataset.labels:
        class_counts = np.bincount(dataset.labels)
        weights = 1.0 / (class_counts.astype(np.float64) + 1e-6)
        weights = weights / weights.sum() * len(class_counts)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))
        print(f"  Class weights: {weights.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer selection
    if optimizer_type == "adam":
        optimizer = optim.Adam(trainable_params, lr=fl_cfg["learning_rate"], weight_decay=1e-4)
    elif optimizer_type == "adamw":
        optimizer = optim.AdamW(trainable_params, lr=fl_cfg["learning_rate"], weight_decay=1e-2)
    else:
        optimizer = optim.SGD(trainable_params, lr=fl_cfg["learning_rate"], momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    print(f"  Training for {total_epochs} epochs on pooled data ({len(dataset)} samples)")
    best_acc = 0.0
    history = []

    for epoch in range(total_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)

            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)

        scheduler.step()
        acc = correct / max(total, 1)
        avg_loss = epoch_loss / max(total, 1)
        best_acc = max(best_acc, acc)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{total_epochs}: loss={avg_loss:.4f}, acc={acc:.4f}")

        history.append({"epoch": epoch + 1, "loss": avg_loss, "accuracy": acc})

    # Evaluate on test set
    from evaluate import TestImageDataset, compute_metrics
    with tarfile.open(args.test_data, "r:gz") as tar:
        tar.extractall(".")

    test_dataset = TestImageDataset(Path("."))
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(
                batch_y.numpy().tolist() if isinstance(batch_y, torch.Tensor)
                else batch_y
            )

    test_metrics = compute_metrics(all_preds, all_labels, num_classes)

    # Stop resource monitoring
    resource_stats = monitor.stop()

    result = {
        "dataset": args.dataset,
        "model_arch": args.model_arch,
        "type": "centralized_baseline",
        "total_epochs": total_epochs,
        "total_samples": len(dataset),
        "best_train_accuracy": best_acc,
        "test": test_metrics,
        "training_history": history,
        "device": device,
        "hostname": socket.gethostname(),
        "resources": resource_stats,
    }

    with open(args.output_metrics, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Final test accuracy: {test_metrics['accuracy']:.4f}, "
          f"F1: {test_metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
