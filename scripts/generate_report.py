#!/usr/bin/env python3
"""Generate a summary report from all experiment results.

Collects metrics from both dataset branches, centralized baselines,
and cross-dataset evaluation into a Markdown report.
"""

import argparse
import json
import tarfile
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    args, _ = parser.parse_known_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    num_rounds = cfg["fl"]["num_rounds"]
    report_lines = ["# Federated Learning Experiment Report\n"]

    # Configuration summary
    report_lines.append("## Configuration\n")
    report_lines.append(f"- **Algorithm**: {cfg['fl']['algorithm']}")
    report_lines.append(f"- **Rounds**: {num_rounds}")
    report_lines.append(f"- **Local Epochs**: {cfg['fl']['local_epochs']}")
    report_lines.append(f"- **Batch Size**: {cfg['fl']['batch_size']}")
    report_lines.append(f"- **Learning Rate**: {cfg['fl']['learning_rate']}")
    report_lines.append(f"- **Client Fraction**: {cfg['fl']['client_fraction']}")
    report_lines.append(f"- **Model**: {cfg['fl']['model_arch']}")
    report_lines.append(f"- **Data Configuration**: condorio\n")

    # Per-dataset results
    for dataset in ("tcia", "nih"):
        if not cfg["datasets"].get(dataset, {}).get("enabled", False):
            continue

        report_lines.append(f"## {dataset.upper()} Results\n")
        num_clients = cfg["datasets"][dataset]["num_clients"]
        report_lines.append(f"- **Clients**: {num_clients}\n")

        # Data statistics
        try:
            with open(f"{dataset}_data_stats.json") as f:
                stats = json.load(f)
            het = stats.get("heterogeneity", {})
            report_lines.append(f"### Data Distribution")
            report_lines.append(f"- Avg KL divergence: {het.get('kl_divergence_avg', 0):.4f}")
            counts = het.get("sample_counts", [])
            if counts:
                report_lines.append(f"- Sample counts per client: {counts}")
            report_lines.append("")
        except FileNotFoundError:
            pass

        # FL training results
        report_lines.append("### FL Training\n")
        report_lines.append("| Round | Accuracy | Macro-F1 |")
        report_lines.append("|-------|----------|----------|")
        final_acc = 0.0
        final_f1 = 0.0
        for t in range(num_rounds):
            try:
                with open(f"{dataset}_r{t}_round_metrics.json") as f:
                    m = json.load(f)
                acc = m.get("accuracy", 0)
                f1 = m.get("macro_f1", 0)
                report_lines.append(f"| {t} | {acc:.4f} | {f1:.4f} |")
                final_acc = acc
                final_f1 = f1
            except FileNotFoundError:
                pass
        report_lines.append(f"\n**Final FL accuracy: {final_acc:.4f}, F1: {final_f1:.4f}**\n")

        # Centralized baseline
        try:
            with open(f"{dataset}_centralized_metrics.json") as f:
                cent = json.load(f)
            cent_acc = cent.get("test", {}).get("accuracy", 0)
            cent_f1 = cent.get("test", {}).get("macro_f1", 0)
            report_lines.append(f"### Centralized Baseline")
            report_lines.append(f"- Test accuracy: {cent_acc:.4f}")
            report_lines.append(f"- Test macro-F1: {cent_f1:.4f}")
            gap = cent_acc - final_acc
            report_lines.append(f"- **FL-vs-Centralized gap: {gap:+.4f}**\n")
        except FileNotFoundError:
            pass

    # Cross-dataset evaluation
    try:
        with open("cross_dataset_metrics.json") as f:
            cross = json.load(f)
        report_lines.append("## Cross-Dataset Evaluation\n")
        report_lines.append("| Model \\ Test | TCIA Test | NIH Test |")
        report_lines.append("|-------------|-----------|----------|")

        same = cross.get("same_domain", {})
        cd = cross.get("cross_domain", {})
        tcia_tcia = same.get("tcia_model_on_tcia_test", {}).get("accuracy", 0)
        tcia_nih = cd.get("tcia_model_on_nih_test", {}).get("accuracy", 0)
        nih_tcia = cd.get("nih_model_on_tcia_test", {}).get("accuracy", 0)
        nih_nih = same.get("nih_model_on_nih_test", {}).get("accuracy", 0)

        report_lines.append(f"| TCIA Model | {tcia_tcia:.4f} | {tcia_nih:.4f} |")
        report_lines.append(f"| NIH Model  | {nih_tcia:.4f} | {nih_nih:.4f} |")
        report_lines.append("")
        dg = cross.get("domain_gap", {})
        report_lines.append(f"- TCIA domain gap: {dg.get('tcia_gap', 0):.4f}")
        report_lines.append(f"- NIH domain gap: {dg.get('nih_gap', 0):.4f}\n")
    except FileNotFoundError:
        pass

    report_lines.append("## Figures\n")
    report_lines.append("See `figures/` directory for all plots.\n")

    with open(args.output, "w") as f:
        f.write("\n".join(report_lines))

    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
