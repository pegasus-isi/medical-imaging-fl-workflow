#!/usr/bin/env python3
"""Select a subset of clients for a given FL round.

Uses deterministic seeding (round number) so the selection is reproducible.
Outputs a JSON file listing the selected client IDs.
"""

import argparse
import json
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-clients", type=int, required=True)
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    num_selected = max(1, int(args.num_clients * args.fraction))

    # Deterministic selection based on round number
    rng = random.Random(args.round * 1000 + 42)
    selected = sorted(rng.sample(range(args.num_clients), num_selected))

    result = {
        "round": args.round,
        "num_clients": args.num_clients,
        "fraction": args.fraction,
        "num_selected": num_selected,
        "selected_client_ids": selected,
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Round {args.round}: selected {num_selected}/{args.num_clients} clients: {selected}")


if __name__ == "__main__":
    main()
