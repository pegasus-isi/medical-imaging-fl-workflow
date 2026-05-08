#!/usr/bin/env python3
"""Generate a single FL round SubWorkflow for Pegasus.

This script produces a YAML workflow file for one federated learning round:
  1. select_clients  (pick C*K clients)
  2. train_client_i  (fan-out: parallel local training)
  3. aggregate        (fan-in: FedAvg / FedProx)
  4. validate_global  (evaluate global model)

Usage:
    python fl_round.py --round 0 --dataset tcia --config configs/default.yml
"""

import argparse
import json
import sys
from pathlib import Path

from Pegasus.api import (
    File,
    Job,
    ReplicaCatalog,
    Workflow,
)


def generate_round_workflow(
    round_num: int,
    dataset: str,
    num_clients: int,
    client_fraction: float,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    fl_algorithm: str,
    model_arch: str,
    freeze_backbone: bool = False,
    optimizer: str = "sgd",
    class_weighted_loss: bool = False,
    augmentation: bool = False,
    grad_clip: float = 0.0,
    lr_scheduler: str = "none",
    fedprox_mu: float = 0.0,
) -> Workflow:
    """Build the DAG for a single FL round.

    Parameters
    ----------
    round_num : int
        Zero-indexed round number.
    dataset : str
        Dataset identifier ("tcia" or "nih").
    num_clients : int
        Total number of FL clients (K).
    client_fraction : float
        Fraction of clients sampled per round (C).
    local_epochs : int
        Number of local training epochs (E).
    batch_size : int
        Training batch size.
    learning_rate : float
        Client learning rate.
    fl_algorithm : str
        "fedavg" or "fedprox".
    model_arch : str
        Model architecture name (e.g. "resnet18").
    freeze_backbone : bool
        Freeze all layers except fc/classifier.
    optimizer : str
        Optimizer type: "sgd", "adam", or "adamw".
    class_weighted_loss : bool
        Weight loss by inverse class frequency.
    augmentation : bool
        Apply data augmentation.
    grad_clip : float
        Max gradient norm (0=disabled).
    lr_scheduler : str
        LR scheduler: "none", "cosine", or "step".
    fedprox_mu : float
        FedProx proximal term coefficient.
    """
    wf = Workflow(f"fl_round_{dataset}_r{round_num}")
    prefix = f"{dataset}_r{round_num}"

    # Helper modules staged alongside executable scripts
    resource_monitor_file = File("resource_monitor.py")
    evaluate_helper_file = File("evaluate.py")

    # -- Input files -------------------------------------------------------
    # Global model from previous round (or initial weights for round 0)
    if round_num == 0:
        global_model_in = File(f"{dataset}_initial_model.pt")
    else:
        global_model_in = File(f"{dataset}_global_model_r{round_num - 1}.pt")

    # Per-client data shards (produced by partition step)
    client_data_files = [
        File(f"{dataset}_client_{i}_data.tar.gz") for i in range(num_clients)
    ]
    test_data = File(f"{dataset}_test_data.tar.gz")

    # -- Output files ------------------------------------------------------
    selected_clients_file = File(f"{prefix}_selected_clients.json")
    local_model_files = [
        File(f"{prefix}_local_model_c{i}.pt") for i in range(num_clients)
    ]
    local_metrics_files = [
        File(f"{prefix}_local_metrics_c{i}.json") for i in range(num_clients)
    ]
    global_model_out = File(f"{dataset}_global_model_r{round_num}.pt")
    round_metrics = File(f"{prefix}_round_metrics.json")

    # =====================================================================
    # Job 1: select_clients
    # =====================================================================
    select_job = (
        Job("select_clients")
        .add_args(
            "--num-clients", str(num_clients),
            "--fraction", str(client_fraction),
            "--round", str(round_num),
            "--output", selected_clients_file,
        )
        .add_outputs(selected_clients_file, stage_out=False)
    )
    wf.add_jobs(select_job)

    # =====================================================================
    # Jobs 2..K+1: train_client_{i}  (parallel fan-out)
    # =====================================================================
    train_jobs = []
    for i in range(num_clients):
        train_job = Job("train_local")
        train_job.add_args(
            "--client-id", str(i),
            "--round", str(round_num),
            "--dataset", dataset,
            "--local-epochs", str(local_epochs),
            "--batch-size", str(batch_size),
            "--learning-rate", str(learning_rate),
            "--algorithm", fl_algorithm,
            "--model-arch", model_arch,
            "--global-model", global_model_in,
            "--client-data", client_data_files[i],
            "--output-model", local_model_files[i],
            "--output-metrics", local_metrics_files[i],
            "--selected-clients", selected_clients_file,
            "--fedprox-mu", str(fedprox_mu),
            "--optimizer", optimizer,
            "--lr-scheduler", lr_scheduler,
        )
        if freeze_backbone:
            train_job.add_args("--freeze-backbone")
        if class_weighted_loss:
            train_job.add_args("--class-weighted-loss")
        if augmentation:
            train_job.add_args("--augmentation")
        if grad_clip > 0:
            train_job.add_args("--grad-clip", str(grad_clip))
        train_job.add_inputs(global_model_in, client_data_files[i], selected_clients_file, resource_monitor_file)
        train_job.add_outputs(local_model_files[i], stage_out=False)
        train_job.add_outputs(local_metrics_files[i], stage_out=False)
        wf.add_jobs(train_job)
        wf.add_dependency(train_job, parents=[select_job])
        train_jobs.append(train_job)

    # =====================================================================
    # Job K+2: aggregate_models  (fan-in)
    # =====================================================================
    agg_job = Job("aggregate")
    agg_job.add_args(
        "--round", str(round_num),
        "--dataset", dataset,
        "--algorithm", fl_algorithm,
        "--num-clients", str(num_clients),
        "--output-model", global_model_out,
    )
    for i in range(num_clients):
        agg_job.add_args("--local-model", local_model_files[i])
        agg_job.add_args("--local-metrics", local_metrics_files[i])
        agg_job.add_inputs(local_model_files[i], local_metrics_files[i])
    agg_job.add_outputs(global_model_out, stage_out=True)
    wf.add_jobs(agg_job)
    for tj in train_jobs:
        wf.add_dependency(agg_job, parents=[tj])

    # =====================================================================
    # Job K+3: validate_global
    # =====================================================================
    validate_job = (
        Job("evaluate")
        .add_args(
            "--round", str(round_num),
            "--dataset", dataset,
            "--model", global_model_out,
            "--test-data", test_data,
            "--output-metrics", round_metrics,
            "--model-arch", model_arch,
        )
        .add_inputs(global_model_out, test_data, resource_monitor_file)
        .add_outputs(round_metrics, stage_out=True)
    )
    wf.add_jobs(validate_job)
    wf.add_dependency(validate_job, parents=[agg_job])

    # Build replica catalog for helper modules that scripts import
    scripts_dir = Path(__file__).resolve().parent / "scripts"
    rc = ReplicaCatalog()
    rc.add_replica("local", "resource_monitor.py", (scripts_dir / "resource_monitor.py").as_uri())
    rc.add_replica("local", "evaluate.py", (scripts_dir / "evaluate.py").as_uri())

    return wf, rc


def main():
    parser = argparse.ArgumentParser(description="Generate an FL round sub-workflow")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--dataset", choices=["tcia", "nih"], required=True)
    parser.add_argument("--config", type=str, default="configs/default.yml")
    parser.add_argument("--output", type=str, help="Output YAML path")
    args = parser.parse_args()

    # Load config
    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg["datasets"][args.dataset]
    fl_cfg = cfg["fl"]

    wf, rc = generate_round_workflow(
        round_num=args.round,
        dataset=args.dataset,
        num_clients=ds_cfg["num_clients"],
        client_fraction=fl_cfg["client_fraction"],
        local_epochs=fl_cfg["local_epochs"],
        batch_size=fl_cfg["batch_size"],
        learning_rate=fl_cfg["learning_rate"],
        fl_algorithm=fl_cfg["algorithm"],
        model_arch=fl_cfg["model_arch"],
        freeze_backbone=fl_cfg.get("freeze_backbone", False),
        optimizer=fl_cfg.get("optimizer", "sgd"),
        class_weighted_loss=fl_cfg.get("class_weighted_loss", False),
        augmentation=fl_cfg.get("augmentation", False),
        grad_clip=fl_cfg.get("grad_clip", 0.0),
        lr_scheduler=fl_cfg.get("lr_scheduler", "none"),
        fedprox_mu=fl_cfg.get("fedprox_mu", 0.0),
    )

    out_path = args.output or f"fl_round_{args.dataset}_r{args.round}.yml"
    wf.write(out_path)
    print(f"Wrote round workflow to {out_path}")


if __name__ == "__main__":
    main()
