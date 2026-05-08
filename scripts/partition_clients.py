#!/usr/bin/env python3
"""Partition raw data into K client shards for federated learning.

TCIA: Pool all collections, assign binary labels per collection,
      then distribute mixed-label images across K clients so each
      client sees both classes.
NIH:  Partition by patient ID hash into K balanced groups.
      Binary label: "No Finding"=0, any pathology=1.

Output directory structure per client shard:
    client_<i>/train/<label>/*.png

Also creates:
  - A held-out global test set with test/<label>/*.png
  - Initial model weights (pre-trained or random)
"""

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import tarfile
from pathlib import Path

import numpy as np
import torch
import torchvision.models as models
import yaml


def partition_tcia(
    raw_dir: Path,
    num_clients: int,
    collections: list,
    test_split: float,
    collection_labels: dict,
):
    """Partition TCIA data — pool all collections, then redistribute.

    All images from all collections are pooled with their binary labels,
    then distributed round-robin across K clients so each client gets
    images from multiple collections with both labels.
    """
    # Step 1: Pool all images with their labels
    all_images = []  # list of (src_path, label, collection_name)

    for collection in collections:
        collection_dir = raw_dir / collection
        if not collection_dir.exists():
            print(f"  Warning: {collection_dir} not found, skipping")
            continue

        png_files = sorted(collection_dir.glob("*.png"))
        if not png_files:
            print(f"  Warning: no PNG files in {collection_dir}")
            continue

        label = collection_labels.get(collection, 0)
        for f in png_files:
            all_images.append((f, label, collection))

        print(f"  {collection}: {len(png_files)} images, label={label}")

    if not all_images:
        print("  ERROR: no TCIA images found")
        return [], []

    random.shuffle(all_images)
    print(f"  Total pooled: {len(all_images)} images "
          f"(label 0: {sum(1 for _, l, _ in all_images if l == 0)}, "
          f"label 1: {sum(1 for _, l, _ in all_images if l == 1)})")

    # Step 2: Split into train/test
    split_idx = max(1, int(len(all_images) * (1 - test_split)))
    train_images = all_images[:split_idx]
    test_images = all_images[split_idx:]

    # Step 3: Distribute train images round-robin across K clients
    client_images = [[] for _ in range(num_clients)]
    for idx, item in enumerate(train_images):
        client_images[idx % num_clients].append(item)

    # Step 4: Build client directories
    client_dirs = []
    for i in range(num_clients):
        client_dir = Path(f"tcia_client_{i}")
        label_counts = {0: 0, 1: 0}

        for src_path, label, collection in client_images[i]:
            label_dir = client_dir / "train" / str(label)
            label_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, label_dir / src_path.name)
            label_counts[label] = label_counts.get(label, 0) + 1

        client_dirs.append(client_dir)
        print(f"  Client {i}: {sum(label_counts.values())} train "
              f"(label 0: {label_counts.get(0, 0)}, label 1: {label_counts.get(1, 0)})")

    # Collect test files
    all_test_files = [(src, label) for src, label, _ in test_images]
    print(f"  Test set: {len(all_test_files)} images")

    return client_dirs, all_test_files


def partition_nih(
    raw_dir: Path,
    num_clients: int,
    test_split: float,
):
    """Partition NIH data by patient ID hash.

    Reads labels.csv produced by download_data.py.
    Binary label: 0 = "No Finding", 1 = any pathology.
    Creates train/<label>/*.png per client.
    """
    client_dirs = []
    images_dir = raw_dir / "images"
    labels_path = raw_dir / "labels.csv"

    if not labels_path.exists():
        print(f"  ERROR: {labels_path} not found")
        return [], []

    # Read labels CSV: filename,label_indices,label_names,binary_label,patient_id
    records = []
    with open(labels_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    if not records:
        print("  ERROR: labels.csv is empty")
        return [], []

    # Group by patient ID
    patient_files = {}
    for rec in records:
        pid = rec.get("patient_id", "0")
        patient_files.setdefault(pid, []).append(rec)

    patients = sorted(patient_files.keys())
    random.shuffle(patients)

    # Reserve test patients
    test_count = max(1, int(len(patients) * test_split))
    test_patients = set(patients[:test_count])
    train_patients = patients[test_count:]

    # Distribute train patients across clients
    client_patient_lists = [[] for _ in range(num_clients)]
    for idx, pid in enumerate(train_patients):
        client_patient_lists[idx % num_clients].append(pid)

    # Build client directories
    all_test_files = []  # list of (src_path, label)

    for i in range(num_clients):
        client_dir = Path(f"nih_client_{i}")
        train_base = client_dir / "train"
        count = 0

        for pid in client_patient_lists[i]:
            for rec in patient_files[pid]:
                fname = rec["filename"]
                binary_label = int(rec.get("binary_label", 0))
                src = images_dir / fname
                if not src.exists():
                    continue

                label_dir = train_base / str(binary_label)
                label_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, label_dir / fname)
                count += 1

        client_dirs.append(client_dir)
        print(f"  Client {i}: {len(client_patient_lists[i])} patients, {count} images")

    # Build test set
    for pid in test_patients:
        for rec in patient_files[pid]:
            fname = rec["filename"]
            binary_label = int(rec.get("binary_label", 0))
            src = images_dir / fname
            if src.exists():
                all_test_files.append((src, binary_label))

    return client_dirs, all_test_files


def build_test_set(test_files: list, dataset: str) -> Path:
    """Build global test set with test/<label>/*.png structure.

    Args:
        test_files: list of (src_path, label) tuples
        dataset: "tcia" or "nih"

    Returns:
        Path to the test directory
    """
    test_dir = Path(f"{dataset}_test")

    for src_path, label in test_files:
        label_dir = test_dir / "test" / str(label)
        label_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, label_dir / src_path.name)

    print(f"  Test set: {len(test_files)} images across "
          f"{len(set(l for _, l in test_files))} labels")
    return test_dir


def create_initial_model(model_arch: str, num_classes: int, output_path: str):
    """Create initial model with pre-trained weights."""
    if model_arch == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif model_arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = torch.nn.Linear(
            model.classifier[1].in_features, num_classes
        )
    else:
        raise ValueError(f"Unknown model architecture: {model_arch}")

    torch.save(model.state_dict(), output_path)
    print(f"  Initial model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["tcia", "nih"])
    parser.add_argument("--num-clients", type=int, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--raw-data", required=True)
    args, _ = parser.parse_known_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg["datasets"][args.dataset]
    fl_cfg = cfg["fl"]

    # Extract raw data
    raw_dir = Path(f"{args.dataset}_extracted")
    with tarfile.open(args.raw_data, "r:gz") as tar:
        tar.extractall(raw_dir)

    # Partition
    test_split = ds_cfg.get("test_split", 0.2)
    if args.dataset == "tcia":
        collection_labels = ds_cfg.get("collection_labels", {})
        client_dirs, test_files = partition_tcia(
            raw_dir / "tcia",
            args.num_clients,
            ds_cfg["collections"],
            test_split,
            collection_labels,
        )
    else:
        client_dirs, test_files = partition_nih(
            raw_dir / "nih",
            args.num_clients,
            test_split,
        )

    # Package client shards as tar.gz
    for i, cd in enumerate(client_dirs):
        tar_name = f"{args.dataset}_client_{i}_data.tar.gz"
        with tarfile.open(tar_name, "w:gz") as tar:
            tar.add(cd, arcname=f"client_{i}")
        print(f"  Packaged {tar_name}")
        shutil.rmtree(cd)

    # Package test data with test/<label>/*.png structure
    if test_files:
        test_dir = build_test_set(test_files, args.dataset)
    else:
        # Create empty test dir structure
        test_dir = Path(f"{args.dataset}_test")
        (test_dir / "test" / "0").mkdir(parents=True, exist_ok=True)
        print("  Warning: no test files available")

    test_tar = f"{args.dataset}_test_data.tar.gz"
    with tarfile.open(test_tar, "w:gz") as tar:
        tar.add(test_dir, arcname=".")
    shutil.rmtree(test_dir)

    # Create initial model
    num_classes = ds_cfg.get("num_classes", 2)
    create_initial_model(fl_cfg["model_arch"], num_classes, f"{args.dataset}_initial_model.pt")

    # Cleanup
    shutil.rmtree(raw_dir)
    print("Partition complete.")


if __name__ == "__main__":
    main()
