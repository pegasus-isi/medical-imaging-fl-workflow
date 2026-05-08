#!/usr/bin/env python3
"""Cross-dataset evaluation: test TCIA model on NIH data and vice versa.

Measures generalization across modalities (3D CT → 2D X-ray).
Runs in PARALLEL with centralized baselines after both FL branches complete.
"""

import argparse
import json
import tarfile
from pathlib import Path

import torch
import torchvision.models as models

from evaluate import TestImageDataset, build_model, compute_metrics


def evaluate_model_on_data(
    model_path: str, test_tar: str, model_arch: str, device: str
) -> dict:
    """Load a model and evaluate on a test set."""
    state = torch.load(model_path, map_location="cpu", weights_only=True)

    if model_arch == "resnet18":
        num_classes = state.get("fc.weight", torch.zeros(2, 1)).shape[0]
    else:
        num_classes = 2
        for k in state:
            if "classifier" in k and "weight" in k:
                num_classes = state[k].shape[0]
                break

    model = build_model(model_arch, num_classes)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Extract test data to a temp dir
    eval_dir = Path(f"cross_eval_{Path(model_path).stem}")
    eval_dir.mkdir(exist_ok=True)
    with tarfile.open(test_tar, "r:gz") as tar:
        tar.extractall(eval_dir)

    dataset = TestImageDataset(eval_dir)
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(
                batch_y.numpy().tolist() if isinstance(batch_y, torch.Tensor)
                else batch_y
            )

    return compute_metrics(all_preds, all_labels, num_classes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-metrics", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tcia-model", required=True)
    parser.add_argument("--tcia-test", required=True)
    parser.add_argument("--nih-model", required=True)
    parser.add_argument("--nih-test", required=True)
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_arch = cfg["fl"]["model_arch"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Cross-dataset evaluation, device={device}")

    # TCIA model → NIH test (cross-modality)
    print("  Evaluating TCIA model on NIH test data...")
    tcia_on_nih = evaluate_model_on_data(
        args.tcia_model, args.nih_test, model_arch, device
    )

    # NIH model → TCIA test (cross-modality)
    print("  Evaluating NIH model on TCIA test data...")
    nih_on_tcia = evaluate_model_on_data(
        args.nih_model, args.tcia_test, model_arch, device
    )

    # TCIA model → TCIA test (same-domain baseline)
    print("  Evaluating TCIA model on TCIA test data (same-domain)...")
    tcia_on_tcia = evaluate_model_on_data(
        args.tcia_model, args.tcia_test, model_arch, device
    )

    # NIH model → NIH test (same-domain baseline)
    print("  Evaluating NIH model on NIH test data (same-domain)...")
    nih_on_nih = evaluate_model_on_data(
        args.nih_model, args.nih_test, model_arch, device
    )

    result = {
        "type": "cross_dataset_evaluation",
        "same_domain": {
            "tcia_model_on_tcia_test": tcia_on_tcia,
            "nih_model_on_nih_test": nih_on_nih,
        },
        "cross_domain": {
            "tcia_model_on_nih_test": tcia_on_nih,
            "nih_model_on_tcia_test": nih_on_tcia,
        },
        "domain_gap": {
            "tcia_gap": tcia_on_tcia["accuracy"] - tcia_on_nih["accuracy"],
            "nih_gap": nih_on_nih["accuracy"] - nih_on_tcia["accuracy"],
        },
    }

    with open(args.output_metrics, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  Results:")
    print(f"    TCIA→TCIA: {tcia_on_tcia['accuracy']:.4f}  |  TCIA→NIH: {tcia_on_nih['accuracy']:.4f}")
    print(f"    NIH→NIH:   {nih_on_nih['accuracy']:.4f}  |  NIH→TCIA: {nih_on_tcia['accuracy']:.4f}")
    print(f"    Domain gap (TCIA): {result['domain_gap']['tcia_gap']:.4f}")
    print(f"    Domain gap (NIH):  {result['domain_gap']['nih_gap']:.4f}")


if __name__ == "__main__":
    main()
