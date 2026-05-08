#!/usr/bin/env python3
"""Aggregate local model updates into a new global model.

Implements FedAvg and FedProx aggregation using Flower's strategy API
when available, with a pure-PyTorch fallback.

This is the fan-in step: receives K local model files and produces
one global model.
"""

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import torch

# Try to use Flower's aggregation utilities
try:
    from flwr.common import (
        NDArrays,
        ndarrays_to_parameters,
        parameters_to_ndarrays,
    )
    from flwr.server.strategy.aggregate import aggregate as flwr_aggregate
    HAS_FLOWER = True
except ImportError:
    HAS_FLOWER = False


def fedavg_aggregate(model_paths: list, metrics_paths: list) -> OrderedDict:
    """Weighted average of model state dicts.

    Weights are proportional to num_samples per client.
    Clients that were not selected (num_samples=0) are excluded.
    """
    # Load metrics to get sample counts
    sample_counts = []
    for mp in metrics_paths:
        with open(mp) as f:
            m = json.load(f)
        sample_counts.append(m.get("num_samples", 0))

    # Filter to only selected clients (num_samples > 0)
    active = [(i, sc) for i, sc in enumerate(sample_counts) if sc > 0]
    if not active:
        # Fallback: equal weight for all
        active = [(i, 1) for i in range(len(model_paths))]

    total_samples = sum(sc for _, sc in active)

    # Weighted average
    avg_state = None
    for idx, sc in active:
        state = torch.load(model_paths[idx], map_location="cpu", weights_only=True)
        weight = sc / total_samples

        if avg_state is None:
            avg_state = OrderedDict()
            for key in state:
                avg_state[key] = state[key].float() * weight
        else:
            for key in state:
                avg_state[key] += state[key].float() * weight

    return avg_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--algorithm", default="fedavg")
    parser.add_argument("--num-clients", type=int, required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--local-model", action="append", required=True)
    parser.add_argument("--local-metrics", action="append", required=True)
    args = parser.parse_args()

    print(f"Aggregating {len(args.local_model)} local models "
          f"(round {args.round}, {args.dataset}, {args.algorithm})")

    # Both FedAvg and FedProx use the same aggregation (weighted average)
    # FedProx differs only in the local training step (proximal term)
    global_state = fedavg_aggregate(args.local_model, args.local_metrics)

    torch.save(global_state, args.output_model)
    print(f"  Global model saved to {args.output_model}")

    # Log aggregation summary
    total_samples = 0
    for mp in args.local_metrics:
        with open(mp) as f:
            m = json.load(f)
        selected = m.get("selected", True)
        ns = m.get("num_samples", 0)
        total_samples += ns
        print(f"  Client {m.get('client_id', '?')}: "
              f"{'selected' if selected else 'skipped'}, "
              f"{ns} samples, "
              f"loss={m.get('loss', 0):.4f}, acc={m.get('accuracy', 0):.4f}")

    print(f"  Total samples aggregated: {total_samples}")


if __name__ == "__main__":
    main()
