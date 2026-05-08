#!/usr/bin/env python3
"""Evaluate a global model on the held-out test set.

Computes accuracy, F1 score, and per-class metrics.
Used both for per-round validation and final evaluation.
"""

import argparse
import json
import socket
import tarfile
from pathlib import Path

import numpy as np
import torch

from resource_monitor import ResourceMonitor
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset


class TestImageDataset(Dataset):
    """Test dataset loaded from extracted tar archive."""

    def __init__(self, data_dir: Path, image_size: tuple = (224, 224)):
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.samples = []
        self.labels = []
        test_dir = data_dir / "test"
        if test_dir.exists():
            for label_dir in sorted(test_dir.iterdir()):
                if label_dir.is_dir():
                    label = int(label_dir.name) if label_dir.name.isdigit() else 0
                    for img_path in label_dir.glob("*"):
                        self.samples.append(img_path)
                        self.labels.append(label)

        if not self.samples:
            print("  No real test images — using synthetic data")
            self.synthetic = True
            self.num_samples = 50
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


def build_model(model_arch: str, num_classes: int) -> nn.Module:
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


def compute_metrics(all_preds: list, all_labels: list, num_classes: int) -> dict:
    """Compute accuracy, F1, and per-class metrics."""
    preds = np.array(all_preds)
    labels = np.array(all_labels)

    accuracy = (preds == labels).mean()

    # Per-class precision, recall, F1
    per_class = {}
    f1_scores = []
    for c in range(num_classes):
        tp = ((preds == c) & (labels == c)).sum()
        fp = ((preds == c) & (labels != c)).sum()
        fn = ((preds != c) & (labels == c)).sum()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        f1_scores.append(f1)

        per_class[str(c)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int((labels == c).sum()),
        }

    macro_f1 = float(np.mean(f1_scores))

    return {
        "accuracy": float(accuracy),
        "macro_f1": macro_f1,
        "num_samples": len(labels),
        "per_class": per_class,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-data", required=True)
    parser.add_argument("--output-metrics", required=True)
    parser.add_argument("--model-arch", default="resnet18")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating round {args.round}, dataset {args.dataset}, device {device}")

    # Start resource monitoring
    monitor = ResourceMonitor(interval=5.0)
    monitor.start()

    # Extract test data
    with tarfile.open(args.test_data, "r:gz") as tar:
        tar.extractall(".")

    # Load model
    state = torch.load(args.model, map_location="cpu", weights_only=True)
    if args.model_arch == "resnet18":
        num_classes = state.get("fc.weight", torch.zeros(2, 1)).shape[0]
    else:
        num_classes = 2
        for k in state:
            if "classifier" in k and "weight" in k:
                num_classes = state[k].shape[0]
                break

    model = build_model(args.model_arch, num_classes)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Build test dataloader
    dataset = TestImageDataset(Path("."))
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

    # Evaluate
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(
                batch_y.numpy().tolist() if isinstance(batch_y, torch.Tensor)
                else batch_y
            )

    # Stop resource monitoring
    resource_stats = monitor.stop()

    metrics = compute_metrics(all_preds, all_labels, num_classes)
    metrics["round"] = args.round
    metrics["dataset"] = args.dataset
    metrics["device"] = device
    metrics["hostname"] = socket.gethostname()
    metrics["resources"] = resource_stats

    with open(args.output_metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Accuracy: {metrics['accuracy']:.4f}, "
          f"Macro-F1: {metrics['macro_f1']:.4f}, "
          f"Samples: {metrics['num_samples']}, Wall time: {resource_stats['wall_time_seconds']:.1f}s")


if __name__ == "__main__":
    main()
