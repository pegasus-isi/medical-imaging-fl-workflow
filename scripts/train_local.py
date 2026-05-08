#!/usr/bin/env python3
"""Local training for a single FL client in one round.

Uses Flower's client-side training utilities for the FL algorithm logic,
but Pegasus orchestrates the scheduling and data movement (via CondorIO).

Inputs:
  - Global model weights from the server (previous round)
  - Client's local data shard

Outputs:
  - Updated local model weights
  - Training metrics (loss, accuracy, num_samples)
"""

import argparse
import json
import socket
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

from resource_monitor import ResourceMonitor
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

# Flower strategy utilities (used offline, not as a server/client)
try:
    from flwr.common import (
        NDArrays,
        ndarrays_to_parameters,
        parameters_to_ndarrays,
    )
    HAS_FLOWER = True
except ImportError:
    HAS_FLOWER = False
    print("Warning: Flower not installed. Using basic PyTorch training loop.")


class LocalImageDataset(Dataset):
    """Simple image dataset loaded from extracted client shard."""

    def __init__(self, data_dir: Path, image_size: tuple = (224, 224), augment: bool = False):
        self.data_dir = data_dir
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
        # Discover images
        self.samples = []
        self.labels = []
        train_dir = data_dir / "train"
        if train_dir.exists():
            for label_dir in sorted(train_dir.iterdir()):
                if label_dir.is_dir():
                    label = int(label_dir.name) if label_dir.name.isdigit() else 0
                    for img_path in label_dir.glob("*"):
                        self.samples.append(img_path)
                        self.labels.append(label)

        if not self.samples:
            # Fallback: generate synthetic data for development/testing
            print("  No real images found — using synthetic data for development")
            self.synthetic = True
            self.num_samples = 100
        else:
            self.synthetic = False

    def __len__(self):
        return self.num_samples if self.synthetic else len(self.samples)

    def __getitem__(self, idx):
        if self.synthetic:
            x = torch.randn(3, 224, 224)
            y = torch.randint(0, 2, (1,)).item()
            return x, y

        from PIL import Image
        img = Image.open(self.samples[idx]).convert("RGB")
        return self.transform(img), self.labels[idx]


def build_model(model_arch: str, num_classes: int) -> nn.Module:
    """Construct the model architecture."""
    if model_arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features, num_classes
        )
    else:
        raise ValueError(f"Unknown model: {model_arch}")
    return model


def train_one_round(
    model: nn.Module,
    dataloader: DataLoader,
    local_epochs: int,
    learning_rate: float,
    algorithm: str,
    global_params: dict = None,
    fedprox_mu: float = 0.01,
    device: str = "cpu",
    optimizer_type: str = "sgd",
    class_weighted_loss: bool = False,
    grad_clip: float = 0.0,
    lr_scheduler_type: str = "none",
    freeze_backbone: bool = False,
) -> dict:
    """Run local training for E epochs.

    Returns dict with loss, accuracy, and num_samples.
    """
    model.to(device)

    # Freeze backbone: only train fc/classifier layers
    if freeze_backbone:
        for name, param in model.named_parameters():
            if "fc" not in name and "classifier" not in name:
                param.requires_grad = False
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    model.train()

    # Class-weighted loss
    if class_weighted_loss and hasattr(dataloader.dataset, 'labels') and dataloader.dataset.labels:
        labels = dataloader.dataset.labels
        class_counts = np.bincount(labels)
        # Inverse frequency weights, normalized
        weights = 1.0 / (class_counts.astype(np.float64) + 1e-6)
        weights = weights / weights.sum() * len(class_counts)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))
        print(f"  Class weights: {weights.tolist()}")
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer selection
    if optimizer_type == "adam":
        optimizer = optim.Adam(trainable_params, lr=learning_rate, weight_decay=1e-4)
    elif optimizer_type == "adamw":
        optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=1e-2)
    else:
        optimizer = optim.SGD(trainable_params, lr=learning_rate, momentum=0.9, weight_decay=1e-4)

    # LR scheduler
    scheduler = None
    if lr_scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=local_epochs)
    elif lr_scheduler_type == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(local_epochs // 3, 1), gamma=0.1)

    # For FedProx: keep a copy of global parameters (only for trainable params)
    if algorithm == "fedprox" and global_params is not None:
        global_tensors = {
            k: v.clone().to(device) for k, v in global_params.items()
        }

    total_loss = 0.0
    correct = 0
    total = 0

    for epoch in range(local_epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # FedProx proximal term (only for trainable params)
            if algorithm == "fedprox" and global_params is not None:
                prox_term = 0.0
                for name, param in model.named_parameters():
                    if param.requires_grad and name in global_tensors:
                        prox_term += ((param - global_tensors[name]) ** 2).sum()
                loss = loss + (fedprox_mu / 2.0) * prox_term

            loss.backward()

            # Gradient clipping
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)

            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)

        total_loss += epoch_loss
        if scheduler is not None:
            scheduler.step()

    avg_loss = total_loss / max(total * local_epochs, 1)
    accuracy = correct / max(total, 1)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "num_samples": total,
        "local_epochs": local_epochs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--algorithm", default="fedavg", choices=["fedavg", "fedprox"])
    parser.add_argument("--model-arch", default="resnet18")
    parser.add_argument("--global-model", required=True)
    parser.add_argument("--client-data", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--output-metrics", required=True)
    parser.add_argument("--selected-clients", required=True)
    parser.add_argument("--fedprox-mu", type=float, default=0.01)
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze all layers except fc/classifier")
    parser.add_argument("--optimizer", choices=["sgd", "adam", "adamw"], default="sgd")
    parser.add_argument("--class-weighted-loss", action="store_true",
                        help="Weight loss by inverse class frequency")
    parser.add_argument("--augmentation", action="store_true",
                        help="Apply data augmentation (RandomResizedCrop, flip, rotate, jitter)")
    parser.add_argument("--grad-clip", type=float, default=0.0,
                        help="Max gradient norm (0=disabled)")
    parser.add_argument("--lr-scheduler", choices=["none", "cosine", "step"], default="none")
    args = parser.parse_args()

    # Check if this client was selected for this round
    with open(args.selected_clients) as f:
        selection = json.load(f)
    if args.client_id not in selection["selected_client_ids"]:
        print(f"Client {args.client_id} not selected for round {args.round}. Passing through global model.")
        # Copy global model as-is and emit zero-contribution metrics
        import shutil
        shutil.copy2(args.global_model, args.output_model)
        metrics = {
            "client_id": args.client_id,
            "round": args.round,
            "selected": False,
            "loss": 0.0,
            "accuracy": 0.0,
            "num_samples": 0,
        }
        with open(args.output_metrics, "w") as f:
            json.dump(metrics, f, indent=2)
        return

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Client {args.client_id}, Round {args.round}, Device: {device}")

    # Start resource monitoring
    monitor = ResourceMonitor(interval=5.0)
    monitor.start()

    # Extract client data
    data_dir = Path(f"client_{args.client_id}_data")
    with tarfile.open(args.client_data, "r:gz") as tar:
        tar.extractall(".")

    # Build model and load global weights
    # Infer num_classes from the global model checkpoint
    global_state = torch.load(args.global_model, map_location="cpu", weights_only=True)
    if args.model_arch == "resnet18":
        num_classes = global_state.get("fc.weight", torch.zeros(2, 1)).shape[0]
    else:
        # efficientnet
        num_classes = 2
        for k in global_state:
            if "classifier" in k and "weight" in k:
                num_classes = global_state[k].shape[0]
                break

    model = build_model(args.model_arch, num_classes)
    model.load_state_dict(global_state)

    # Build dataset and dataloader
    dataset = LocalImageDataset(data_dir, augment=args.augmentation)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )

    # Train
    metrics = train_one_round(
        model=model,
        dataloader=dataloader,
        local_epochs=args.local_epochs,
        learning_rate=args.learning_rate,
        algorithm=args.algorithm,
        global_params=global_state if args.algorithm == "fedprox" else None,
        fedprox_mu=args.fedprox_mu,
        device=device,
        optimizer_type=args.optimizer,
        class_weighted_loss=args.class_weighted_loss,
        grad_clip=args.grad_clip,
        lr_scheduler_type=args.lr_scheduler,
        freeze_backbone=args.freeze_backbone,
    )
    # Stop resource monitoring and collect stats
    resource_stats = monitor.stop()

    metrics["client_id"] = args.client_id
    metrics["round"] = args.round
    metrics["selected"] = True
    metrics["dataset"] = args.dataset
    metrics["device"] = device
    metrics["hostname"] = socket.gethostname()
    metrics["resources"] = resource_stats

    # Save outputs
    torch.save(model.state_dict(), args.output_model)
    with open(args.output_metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Loss: {metrics['loss']:.4f}, Accuracy: {metrics['accuracy']:.4f}, "
          f"Samples: {metrics['num_samples']}, Wall time: {resource_stats['wall_time_seconds']:.1f}s")


if __name__ == "__main__":
    main()
