#!/usr/bin/env python3
"""Compute per-client data distribution statistics.

Runs in PARALLEL with the FL training rounds (no dependency on round results).
Produces a JSON file with class balance, sample counts, and heterogeneity metrics.
"""

import argparse
import json
import tarfile
from pathlib import Path


def analyze_client_shard(tar_path: str, client_id: int) -> dict:
    """Extract and analyze one client's data shard."""
    stats = {
        "client_id": client_id,
        "total_samples": 0,
        "class_distribution": {},
        "file_count": 0,
    }

    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            members = tar.getmembers()
            stats["file_count"] = len(members)

            for m in members:
                if m.isfile():
                    stats["total_samples"] += 1
                    # Infer class from directory structure (client_X/train/CLASS/file)
                    parts = Path(m.name).parts
                    if len(parts) >= 3:
                        cls = parts[-2]
                        stats["class_distribution"][cls] = \
                            stats["class_distribution"].get(cls, 0) + 1
    except Exception as e:
        stats["error"] = str(e)

    return stats


def compute_heterogeneity(client_stats: list) -> dict:
    """Compute distribution heterogeneity metrics across clients."""
    # Gather all class labels
    all_classes = set()
    for cs in client_stats:
        all_classes.update(cs.get("class_distribution", {}).keys())
    all_classes = sorted(all_classes)

    if not all_classes or not client_stats:
        return {"num_classes": 0, "kl_divergence_avg": 0.0}

    import math

    # Build distribution matrix
    num_clients = len(client_stats)
    distributions = []
    for cs in client_stats:
        dist = cs.get("class_distribution", {})
        total = sum(dist.values()) or 1
        distributions.append([dist.get(c, 0) / total for c in all_classes])

    # Global distribution
    global_dist = [0.0] * len(all_classes)
    total_global = 0
    for cs in client_stats:
        dist = cs.get("class_distribution", {})
        for i, c in enumerate(all_classes):
            global_dist[i] += dist.get(c, 0)
            total_global += dist.get(c, 0)

    if total_global > 0:
        global_dist = [g / total_global for g in global_dist]

    # KL divergence from each client to global
    kl_divs = []
    for d in distributions:
        kl = 0.0
        for i in range(len(all_classes)):
            p = d[i] + 1e-10
            q = global_dist[i] + 1e-10
            kl += p * math.log(p / q)
        kl_divs.append(kl)

    return {
        "num_classes": len(all_classes),
        "classes": all_classes,
        "global_distribution": {c: global_dist[i] for i, c in enumerate(all_classes)},
        "kl_divergence_per_client": kl_divs,
        "kl_divergence_avg": sum(kl_divs) / max(len(kl_divs), 1),
        "sample_counts": [cs.get("total_samples", 0) for cs in client_stats],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num-clients", type=int, required=True)
    args, _ = parser.parse_known_args()

    print(f"Computing statistics for {args.dataset} with {args.num_clients} clients")

    client_stats = []
    for i in range(args.num_clients):
        tar_path = f"{args.dataset}_client_{i}_data.tar.gz"
        stats = analyze_client_shard(tar_path, i)
        client_stats.append(stats)
        print(f"  Client {i}: {stats['total_samples']} samples, "
              f"{len(stats.get('class_distribution', {}))} classes")

    heterogeneity = compute_heterogeneity(client_stats)

    result = {
        "dataset": args.dataset,
        "num_clients": args.num_clients,
        "per_client": client_stats,
        "heterogeneity": heterogeneity,
    }

    output_path = f"{args.dataset}_data_stats.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Avg KL divergence: {heterogeneity['kl_divergence_avg']:.4f}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
