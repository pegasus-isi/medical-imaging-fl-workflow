#!/usr/bin/env python3
"""Generate all figures for the experiment results.

Reads per-round metrics, centralized baselines, cross-dataset results,
and data statistics. Produces publication-quality figures.

All plot jobs run AFTER all metrics are collected (fan-in from both branches).
"""

import argparse
import json
import tarfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml


def load_round_metrics(dataset: str, num_rounds: int) -> list:
    """Load per-round metrics for a dataset."""
    metrics = []
    for t in range(num_rounds):
        path = f"{dataset}_r{t}_round_metrics.json"
        try:
            with open(path) as f:
                metrics.append(json.load(f))
        except FileNotFoundError:
            pass
    return metrics


def plot_convergence(tcia_metrics, nih_metrics, tcia_cent, nih_cent, output_dir):
    """Figure 1: Convergence curves — accuracy vs round."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # TCIA
    if tcia_metrics:
        rounds = [m["round"] for m in tcia_metrics]
        accs = [m["accuracy"] for m in tcia_metrics]
        ax1.plot(rounds, accs, "b-o", label="FL (FedAvg)", markersize=3)
        if tcia_cent:
            ax1.axhline(y=tcia_cent["test"]["accuracy"], color="r",
                        linestyle="--", label="Centralized")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Test Accuracy")
    ax1.set_title("TCIA Collections")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # NIH
    if nih_metrics:
        rounds = [m["round"] for m in nih_metrics]
        accs = [m["accuracy"] for m in nih_metrics]
        ax2.plot(rounds, accs, "g-o", label="FL (FedAvg)", markersize=3)
        if nih_cent:
            ax2.axhline(y=nih_cent["test"]["accuracy"], color="r",
                        linestyle="--", label="Centralized")
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Test Accuracy")
    ax2.set_title("NIH Chest X-Ray14")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "fig1_convergence.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "fig1_convergence.pdf", bbox_inches="tight")
    plt.close()


def plot_data_distribution(tcia_stats, nih_stats, output_dir):
    """Figure 3: Data distribution across clients."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, stats, title in [(axes[0], tcia_stats, "TCIA"), (axes[1], nih_stats, "NIH")]:
        if not stats:
            ax.set_title(f"{title} (no data)")
            continue

        het = stats.get("heterogeneity", {})
        counts = het.get("sample_counts", [])
        if counts:
            clients = list(range(len(counts)))
            ax.bar(clients, counts, color=sns.color_palette("husl", len(counts)))
            ax.set_xlabel("Client ID")
            ax.set_ylabel("Number of Samples")
            ax.set_title(f"{title} — Sample Distribution\n"
                         f"(Avg KL div: {het.get('kl_divergence_avg', 0):.3f})")
        else:
            ax.set_title(f"{title} (no distribution data)")

    plt.tight_layout()
    plt.savefig(output_dir / "fig3_data_distribution.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "fig3_data_distribution.pdf", bbox_inches="tight")
    plt.close()


def plot_cross_dataset(cross_metrics, output_dir):
    """Figure: Cross-dataset generalization heatmap."""
    if not cross_metrics:
        return

    same = cross_metrics.get("same_domain", {})
    cross = cross_metrics.get("cross_domain", {})

    matrix = np.array([
        [same.get("tcia_model_on_tcia_test", {}).get("accuracy", 0),
         cross.get("tcia_model_on_nih_test", {}).get("accuracy", 0)],
        [cross.get("nih_model_on_tcia_test", {}).get("accuracy", 0),
         same.get("nih_model_on_nih_test", {}).get("accuracy", 0)],
    ])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt=".3f", cmap="YlOrRd",
                xticklabels=["TCIA Test", "NIH Test"],
                yticklabels=["TCIA Model", "NIH Model"],
                ax=ax, vmin=0, vmax=1)
    ax.set_title("Cross-Dataset Generalization")
    plt.tight_layout()
    plt.savefig(output_dir / "fig5_cross_dataset.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "fig5_cross_dataset.pdf", bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    args, _ = parser.parse_known_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    num_rounds = cfg["fl"]["num_rounds"]

    # Load all metrics
    tcia_metrics = load_round_metrics("tcia", num_rounds)
    nih_metrics = load_round_metrics("nih", num_rounds)

    tcia_cent = None
    nih_cent = None
    try:
        with open("tcia_centralized_metrics.json") as f:
            tcia_cent = json.load(f)
    except FileNotFoundError:
        pass
    try:
        with open("nih_centralized_metrics.json") as f:
            nih_cent = json.load(f)
    except FileNotFoundError:
        pass

    tcia_stats = None
    nih_stats = None
    try:
        with open("tcia_data_stats.json") as f:
            tcia_stats = json.load(f)
    except FileNotFoundError:
        pass
    try:
        with open("nih_data_stats.json") as f:
            nih_stats = json.load(f)
    except FileNotFoundError:
        pass

    cross_metrics = None
    try:
        with open("cross_dataset_metrics.json") as f:
            cross_metrics = json.load(f)
    except FileNotFoundError:
        pass

    # Generate figures
    print("Generating figures...")
    plot_convergence(tcia_metrics, nih_metrics, tcia_cent, nih_cent, output_dir)
    print("  fig1_convergence")

    plot_data_distribution(tcia_stats, nih_stats, output_dir)
    print("  fig3_data_distribution")

    plot_cross_dataset(cross_metrics, output_dir)
    print("  fig5_cross_dataset")

    # Package figures
    with tarfile.open(args.output, "w:gz") as tar:
        tar.add(output_dir, arcname="figures")
    print(f"  Packaged to {args.output}")


if __name__ == "__main__":
    main()
