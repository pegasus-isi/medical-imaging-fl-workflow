#!/usr/bin/env python3
"""Generate the top-level Federated Learning Pegasus workflow.

Architecture (maximizes parallelism):
  - TCIA branch and NIH branch run FULLY IN PARALLEL
  - Within each branch: data prep → sequential FL rounds (SubWorkflows)
  - Within each round: K client training jobs run in PARALLEL (fan-out/fan-in)
  - Statistics jobs run in parallel with training (no dependency on rounds)
  - Centralized baselines run in parallel with cross-dataset comparison
  - Final plots + report depend on all upstream results

Data staging: CondorIO — HTCondor manages all file transfers.

Usage:
    python fl_main.py --config configs/default.yml [--output fl_main.yml]
"""

import argparse
import sys
from pathlib import Path

import yaml
from Pegasus.api import (
    Container,
    File,
    Job,
    Properties,
    ReplicaCatalog,
    SiteCatalog,
    Site,
    Directory,
    FileServer,
    Operation,
    Arch,
    OS,
    SubWorkflow,
    Transformation,
    TransformationCatalog,
    Workflow,
)

from fl_round import generate_round_workflow


# =========================================================================
# Catalog builders
# =========================================================================

def build_properties(cfg: dict) -> Properties:
    """Configure Pegasus properties for CondorIO execution."""
    props = Properties()
    props["pegasus.data.configuration"] = "condorio"
    # Retry failed jobs up to 3 times
    props["dagman.retry"] = "3"
    # Cleanup intermediate files to save space
    props["pegasus.file.cleanup.scope"] = "inplace"
    # Enable integrity checking
    props["pegasus.integrity.checking"] = "nosymlink"
    # Stage worker package from submit host instead of downloading on workers
    props["pegasus.transfer.worker.package"] = "true"
    return props


def build_site_catalog(cfg: dict, config_name: str = "") -> SiteCatalog:
    """Build site catalog with local + condorpool sites."""
    sc = SiteCatalog()

    # Per-experiment output directory to avoid collisions when running
    # multiple experiments concurrently.
    output_dir = Path.cwd() / "work" / "outputs"
    if config_name:
        output_dir = output_dir / config_name

    # -- Local site --------------------------------------------------------
    local_site = Site("local", arch=Arch.X86_64, os_type=OS.LINUX)
    local_site.add_directories(
        Directory(Directory.SHARED_SCRATCH, Path.cwd() / "work" / "scratch")
            .add_file_servers(
                FileServer(f"file://{Path.cwd()}/work/scratch", Operation.ALL)
            ),
        Directory(Directory.LOCAL_STORAGE, output_dir)
            .add_file_servers(
                FileServer(f"file://{output_dir}", Operation.ALL)
            ),
    )
    sc.add_sites(local_site)

    # -- CondorPool site ---------------------------------------------------
    condorpool = Site("condorpool", arch=Arch.X86_64, os_type=OS.LINUX)
    condorpool.add_condor_profile(universe="vanilla")
    condorpool.add_pegasus_profile(style="condor")
    sc.add_sites(condorpool)

    return sc


def build_transformation_catalog(cfg: dict) -> TransformationCatalog:
    """Register all script transformations with container."""
    tc = TransformationCatalog()

    scripts_dir = Path(__file__).resolve().parent / "scripts"

    container = Container(
        "fl-training",
        Container.DOCKER,
        image=f"docker:///{cfg['execution']['container']}",
        bypass_staging=True,
    )
    container.add_pegasus_profile(container_arguments="--shm-size=1g")
    tc.add_containers(container)

    scripts = [
        "download_data",
        "partition_clients",
        "select_clients",
        "train_local",
        "aggregate",
        "evaluate",
        "compute_statistics",
        "centralized_baseline",
        "cross_dataset_eval",
        "plot_results",
        "generate_report",
    ]
    for script_name in scripts:
        tx = Transformation(
            script_name,
            site="local",
            pfn=(scripts_dir / f"{script_name}.py").as_uri(),
            is_stageable=True,
            container=container,
        )
        # Resource profiles for GPU jobs
        if script_name in ("train_local", "evaluate", "centralized_baseline"):
            tx.add_condor_profile(
                request_gpus=str(cfg["execution"].get("gpu_per_train_job", 1))
            )
            tx.add_condor_profile(
                request_cpus=str(cfg["execution"].get("cpu_per_train_job", 4))
            )
            tx.add_condor_profile(
                request_memory=str(cfg["execution"].get("mem_per_train_job", 16384))
            )
        # Add ClassAd attributes for workflow-level tracking in stampede DB
        from Pegasus.api import Namespace
        tx.add_profiles(Namespace.CONDOR, "+FL_Algorithm", f'"{cfg["fl"]["algorithm"]}"')
        tx.add_profiles(Namespace.CONDOR, "+FL_NumRounds", str(cfg["fl"]["num_rounds"]))
        tx.add_profiles(Namespace.CONDOR, "+FL_JobType", f'"{script_name}"')
        tc.add_transformations(tx)

    return tc


def build_replica_catalog(config_path: str, cfg: dict, sub_workflow_files: list = None) -> ReplicaCatalog:
    """Register initial input files."""
    rc = ReplicaCatalog()
    # Register the experiment config file so Pegasus can stage it to jobs
    from pathlib import Path
    rc.add_replica("local", "experiment_config.yml", Path(config_path).resolve().as_uri())
    # Register helper Python modules that scripts import at runtime
    scripts_dir = Path(__file__).resolve().parent / "scripts"
    rc.add_replica("local", "resource_monitor.py", (scripts_dir / "resource_monitor.py").as_uri())
    rc.add_replica("local", "evaluate.py", (scripts_dir / "evaluate.py").as_uri())
    # Register sub-workflow YAML files so Pegasus can stage them for planning
    for swf in (sub_workflow_files or []):
        rc.add_replica("local", swf, Path(swf).resolve().as_uri())
    # Register pre-staged raw data files if configured
    for dataset in ("tcia", "nih"):
        ds_cfg = cfg.get("datasets", {}).get(dataset, {})
        raw_data_path = ds_cfg.get("raw_data_path")
        if raw_data_path:
            rc.add_replica(
                "local",
                f"{dataset}_raw_data.tar.gz",
                Path(raw_data_path).resolve().as_uri(),
            )
    return rc


# =========================================================================
# Dataset branch builder (TCIA or NIH)
# =========================================================================

def build_dataset_branch(
    wf: Workflow,
    dataset: str,
    cfg: dict,
    config_file: File,
    config_name: str = "",
) -> tuple:
    """Build all jobs for one dataset branch.

    Returns (last_round_subwf, stats_job, all_round_metrics_files)
    so the caller can wire up cross-dataset dependencies.
    """
    ds_cfg = cfg["datasets"][dataset]
    fl_cfg = cfg["fl"]
    num_clients = ds_cfg["num_clients"]
    num_rounds = fl_cfg["num_rounds"]

    # -- Download (skipped if raw_data_path is set) -----------------------
    raw_data = File(f"{dataset}_raw_data.tar.gz")
    raw_data_path = ds_cfg.get("raw_data_path")

    if raw_data_path:
        # Pre-staged data: skip download job, register file in replica catalog
        download_job = None
        print(f"  {dataset}: using pre-staged data from {raw_data_path}")
    else:
        download_job = (
            Job("download_data", node_label=f"download_{dataset}")
            .add_args("--dataset", dataset, "--config", config_file)
            .add_inputs(config_file)
            .add_outputs(raw_data, stage_out=False)
        )
        wf.add_jobs(download_job)

    # -- Partition into client shards --------------------------------------
    client_data_files = [
        File(f"{dataset}_client_{i}_data.tar.gz") for i in range(num_clients)
    ]
    test_data = File(f"{dataset}_test_data.tar.gz")
    initial_model = File(f"{dataset}_initial_model.pt")

    partition_job = (
        Job("partition_clients", node_label=f"partition_{dataset}")
        .add_args(
            "--dataset", dataset,
            "--num-clients", str(num_clients),
            "--config", config_file,
            "--raw-data", raw_data,
        )
        .add_inputs(raw_data, config_file)
        .add_outputs(test_data, stage_out=True)
        .add_outputs(initial_model, stage_out=True)
    )
    for cf in client_data_files:
        partition_job.add_outputs(cf, stage_out=False, register_replica=True)
    wf.add_jobs(partition_job)
    if download_job:
        wf.add_dependency(partition_job, parents=[download_job])

    # -- Compute statistics (parallel with training — no dependency on rounds)
    stats_out = File(f"{dataset}_data_stats.json")
    stats_job = (
        Job("compute_statistics", node_label=f"stats_{dataset}")
        .add_args("--dataset", dataset, "--num-clients", str(num_clients))
        .add_outputs(stats_out, stage_out=True)
    )
    for cf in client_data_files:
        stats_job.add_inputs(cf)
    wf.add_jobs(stats_job)
    wf.add_dependency(stats_job, parents=[partition_job])

    # -- FL Training Rounds (sequential SubWorkflows) ----------------------
    round_metrics_files = []
    round_yml_files = []
    prev_round_subwf = None

    for t in range(num_rounds):
        # Generate the round sub-workflow YAML
        round_wf, round_rc = generate_round_workflow(
            round_num=t,
            dataset=dataset,
            num_clients=num_clients,
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
        prefix = f"{config_name}_" if config_name else ""
        round_yml = f"{prefix}fl_round_{dataset}_r{t}.yml"
        round_wf.write(round_yml)
        round_yml_files.append(round_yml)

        # Write sub-workflow replica catalog and properties
        round_rc_yml = f"{prefix}fl_round_{dataset}_r{t}.rc.yml"
        round_rc.write(round_rc_yml)
        round_yml_files.append(round_rc_yml)

        round_props_file = f"{prefix}fl_round_{dataset}_r{t}.properties"
        base_dir = Path(__file__).resolve().parent
        with open(round_props_file, "w") as pf:
            pf.write(f"pegasus.catalog.replica=YAML\n")
            pf.write(f"pegasus.catalog.replica.file={Path(round_rc_yml).resolve()}\n")
            pf.write(f"pegasus.catalog.transformation=YAML\n")
            pf.write(f"pegasus.catalog.transformation.file={base_dir / 'transformations.yml'}\n")
            pf.write(f"pegasus.catalog.site=YAML\n")
            pf.write(f"pegasus.catalog.site.file={base_dir / 'sites.yml'}\n")
            pf.write(f"pegasus.transfer.worker.package=true\n")
            pf.write(f"pegasus.data.configuration=condorio\n")
        round_yml_files.append(round_props_file)

        # Create SubWorkflow job in the top-level DAG
        round_subwf = SubWorkflow(round_yml, is_planned=False, node_label=f"fl_round_{dataset}_r{t}")
        round_subwf.add_args(
            "--conf", str(Path(round_props_file).resolve()),
            "--output-sites", "local",
            "--cluster", "horizontal",
            "-vvv",
        )

        # Wire inputs: global model from prev round (or initial), client data, test data
        if t == 0:
            round_subwf.add_inputs(initial_model)
        else:
            prev_global_model = File(f"{dataset}_global_model_r{t - 1}.pt")
            round_subwf.add_inputs(prev_global_model)

        for cf in client_data_files:
            round_subwf.add_inputs(cf)
        round_subwf.add_inputs(test_data)

        # Outputs
        global_model_out = File(f"{dataset}_global_model_r{t}.pt")
        round_metrics = File(f"{dataset}_r{t}_round_metrics.json")
        round_subwf.add_outputs(global_model_out, stage_out=True)
        round_subwf.add_outputs(round_metrics, stage_out=True)
        round_metrics_files.append(round_metrics)

        wf.add_jobs(round_subwf)

        # Dependencies: round T depends on partition + round T-1
        if prev_round_subwf is None:
            wf.add_dependency(round_subwf, parents=[partition_job])
        else:
            wf.add_dependency(round_subwf, parents=[prev_round_subwf])

        prev_round_subwf = round_subwf

    # Return final state so caller can wire cross-dataset convergence
    final_global_model = File(f"{dataset}_global_model_r{num_rounds - 1}.pt")
    return prev_round_subwf, stats_job, round_metrics_files, final_global_model, test_data, round_yml_files


# =========================================================================
# Main workflow assembly
# =========================================================================

def build_workflow(cfg: dict, config_name: str = "") -> Workflow:
    """Assemble the full top-level FL workflow with parallel dataset branches."""
    wf = Workflow("fl_main")

    config_file = File("experiment_config.yml")

    # Track per-dataset outputs for the convergence stage
    branch_outputs = {}
    all_sub_workflow_files = []

    # =====================================================================
    # PARALLEL BRANCHES: one per enabled dataset
    # =====================================================================
    for dataset in ("tcia", "nih"):
        if not cfg["datasets"].get(dataset, {}).get("enabled", False):
            continue

        last_round, stats_job, metrics_files, final_model, test_data, round_ymls = \
            build_dataset_branch(wf, dataset, cfg, config_file, config_name=config_name)

        all_sub_workflow_files.extend(round_ymls)

        branch_outputs[dataset] = {
            "last_round": last_round,
            "stats_job": stats_job,
            "metrics_files": metrics_files,
            "final_model": final_model,
            "test_data": test_data,
        }

    # =====================================================================
    # CONVERGENCE STAGE: fan-in from both branches
    # =====================================================================
    all_parents = []
    all_metrics = []
    all_stats = []

    for ds, out in branch_outputs.items():
        all_parents.append(out["last_round"])
        all_parents.append(out["stats_job"])
        all_metrics.extend(out["metrics_files"])
        all_stats.append(File(f"{ds}_data_stats.json"))

    # -- Centralized baselines (parallel, one per dataset) -----------------
    centralized_jobs = []
    for ds, out in branch_outputs.items():
        cent_metrics = File(f"{ds}_centralized_metrics.json")
        cent_job = (
            Job("centralized_baseline", node_label=f"baseline_{ds}")
            .add_args(
                "--dataset", ds,
                "--model-arch", cfg["fl"]["model_arch"],
                "--test-data", out["test_data"],
                "--output-metrics", cent_metrics,
                "--config", config_file,
            )
            .add_inputs(config_file, out["test_data"],
                        File("resource_monitor.py"), File("evaluate.py"))
            .add_outputs(cent_metrics, stage_out=True)
        )
        # Add all client data as inputs (centralized trains on all data)
        num_clients = cfg["datasets"][ds]["num_clients"]
        for i in range(num_clients):
            cf = File(f"{ds}_client_{i}_data.tar.gz")
            cent_job.add_inputs(cf)
        wf.add_jobs(cent_job)
        # Can start once partition is done — don't need to wait for FL rounds
        # But to keep it simple, start after last round so we reuse the data files
        wf.add_dependency(cent_job, parents=[out["last_round"]])
        centralized_jobs.append(cent_job)
        all_metrics.append(cent_metrics)

    # -- Cross-dataset comparison (only if both datasets enabled) ----------
    cross_eval_job = None
    if len(branch_outputs) == 2:
        cross_metrics = File("cross_dataset_metrics.json")
        # cross_dataset_eval imports from evaluate.py which imports resource_monitor.py
        evaluate_file = File("evaluate.py")
        resource_monitor_file = File("resource_monitor.py")
        rc.add_replica("local", "evaluate.py",
                        (scripts_dir / "evaluate.py").as_uri())
        rc.add_replica("local", "resource_monitor.py",
                        (scripts_dir / "resource_monitor.py").as_uri())
        cross_eval_job = (
            Job("cross_dataset_eval", node_label="cross_eval")
            .add_args("--output-metrics", cross_metrics, "--config", config_file)
            .add_inputs(config_file, evaluate_file, resource_monitor_file)
            .add_outputs(cross_metrics, stage_out=True)
        )
        for ds, out in branch_outputs.items():
            cross_eval_job.add_inputs(out["final_model"], out["test_data"])
            cross_eval_job.add_args(f"--{ds}-model", out["final_model"])
            cross_eval_job.add_args(f"--{ds}-test", out["test_data"])
        wf.add_jobs(cross_eval_job)
        for ds, out in branch_outputs.items():
            wf.add_dependency(cross_eval_job, parents=[out["last_round"]])
        all_metrics.append(cross_metrics)

    # -- Generate plots (depends on all metrics + stats) -------------------
    plot_parents = list(centralized_jobs)
    if cross_eval_job:
        plot_parents.append(cross_eval_job)
    for out in branch_outputs.values():
        plot_parents.append(out["stats_job"])
        plot_parents.append(out["last_round"])

    figures_tar = File("figures.tar.gz")
    plot_job = (
        Job("plot_results", node_label="plot_results")
        .add_args("--output", figures_tar, "--config", config_file)
        .add_inputs(config_file)
        .add_outputs(figures_tar, stage_out=True)
    )
    for m in all_metrics:
        plot_job.add_inputs(m)
    for s in all_stats:
        plot_job.add_inputs(s)
    wf.add_jobs(plot_job)
    for p in plot_parents:
        wf.add_dependency(plot_job, parents=[p])

    # -- Generate report (depends on plots) --------------------------------
    report_out = File("experiment_report.md")
    report_job = (
        Job("generate_report", node_label="generate_report")
        .add_args("--figures", figures_tar, "--output", report_out, "--config", config_file)
        .add_inputs(config_file, figures_tar)
        .add_outputs(report_out, stage_out=True)
    )
    for m in all_metrics:
        report_job.add_inputs(m)
    for s in all_stats:
        report_job.add_inputs(s)
    wf.add_jobs(report_job)
    wf.add_dependency(report_job, parents=[plot_job])

    return wf, all_sub_workflow_files


def main():
    parser = argparse.ArgumentParser(
        description="Generate the top-level FL Pegasus workflow"
    )
    parser.add_argument(
        "--config", type=str, default="configs/default.yml",
        help="Experiment configuration YAML",
    )
    parser.add_argument(
        "--output", type=str, default="fl_main.yml",
        help="Output workflow YAML path",
    )
    parser.add_argument(
        "--plan", action="store_true",
        help="Also write catalogs and properties, then run pegasus-plan",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Use config filename as experiment label for namespacing
    config_name = Path(args.config).stem

    # Build workflow — use config name to namespace round YAML files
    wf, sub_workflow_files = build_workflow(cfg, config_name=config_name)

    # Build catalogs
    props = build_properties(cfg)
    sc = build_site_catalog(cfg, config_name=config_name)
    tc = build_transformation_catalog(cfg)
    rc = build_replica_catalog(args.config, cfg, sub_workflow_files)

    if args.plan:
        # Write all catalogs alongside the workflow
        props.write()
        sc.write()
        tc.write()
        rc.write()

    wf.write(args.output)
    print(f"Wrote top-level workflow to {args.output}")
    print(f"  Datasets enabled: {[ds for ds in ('tcia', 'nih') if cfg['datasets'].get(ds, {}).get('enabled')]}")
    print(f"  Rounds per dataset: {cfg['fl']['num_rounds']}")
    print(f"  Data configuration: condorio")

    if args.plan:
        print("\nCatalogs written. Run:")
        print(f"  pegasus-plan --conf pegasus.properties "
              f"--sites condorpool "
              f"--output-sites local "
              f"--dir work/submit "
              f"--cleanup leaf "
              f"--submit {args.output}")


if __name__ == "__main__":
    main()
