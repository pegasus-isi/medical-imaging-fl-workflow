# Workflow-Driven Federated Learning on Pegasus

Reproducible federated learning experiments for cross-institutional medical imaging, orchestrated by [Pegasus WMS](https://pegasus.isi.edu/) and executed on distributed HTCondor pools across multiple sites (MAX, NCSA, TACC, WASH).

> **Note:** This workflow was generated with the assistance of [Claude Code](https://docs.anthropic.com/en/docs/claude-code) (Anthropic) using the [pegasus-ai](https://github.com/pegasus-isi/pegasus-ai) plugin for Pegasus-specific workflow scaffolding, debugging, and catalog generation. All code was reviewed and validated by the authors.

## Overview

This project implements an end-to-end Federated Learning (FL) pipeline as a Pegasus DAG workflow. The workflow automates data preparation, per-client local training, server-side aggregation, evaluation, and multi-round iteration across two medical imaging datasets.

### Key Features

- **Dual-dataset parallel branches** — TCIA (3D CT/MRI) and NIH Chest X-Ray14 (2D) run as fully independent, parallel DAG branches
- **Hierarchical SubWorkflows** — Each FL round is a Pegasus `SubWorkflow` with fan-out/fan-in parallelism across K clients
- **CondorIO data staging** — HTCondor manages all file transfers; no shared filesystem required
- **Flower FL strategies** — FedAvg and FedProx via [Flower](https://flower.ai/), with Pegasus replacing Flower's gRPC communication layer
- **Built-in resource monitoring** — GPU utilization, memory, and wall-clock timing collected per job via `ResourceMonitor`
- **Ensemble Manager sweeps** — Hyperparameter sweeps run as parallel ensembles with throttling
- **Full provenance** — Pegasus tracks every job, file, and metric for reproducibility

### Parallelism Points

| Level | What runs in parallel |
|---|---|
| Dataset branches | TCIA and NIH pipelines run simultaneously |
| Per-round training | K client `train_local` jobs fan-out in parallel |
| Statistics | `compute_statistics` runs alongside training rounds |
| Baselines | Centralized baselines for both datasets in parallel |
| Cross-dataset eval | Runs in parallel with centralized baselines |
| Ensemble sweeps | Independent experiment configs run concurrently |

## Datasets

| Dataset | Role | Modality | Clients | Source | Access |
|---|---|---|---|---|---|
| **TCIA Collections** | Primary — naturally decentralized | 3D (CT/MRI) | 5 collections from different institutions | TCIA REST API via `tcia_utils` | Open |
| **NIH Chest X-Ray14** | Secondary — cross-dataset evaluation | 2D (X-Ray) | 5 patient-hash partitions | HuggingFace (`BahaaEldin0/NIH-Chest-Xray-14`) | Open |

### TCIA Collections

| Collection | API Name | Label | Description |
|---|---|---|---|
| NSCLC-Radiomics | `NSCLC-Radiomics` | 0 | Non-small cell lung cancer CT (Maastricht) |
| TCGA-LUAD | `TCGA-LUAD` | 1 | Lung adenocarcinoma (multi-site TCGA) |
| LIDC-IDRI | `LIDC-IDRI` | 0 | Lung nodule detection (7 institutions) |
| NSCLC-Radiogenomics | `NSCLC Radiogenomics` | 1 | NSCLC with genomic data (Stanford) |
| RIDER-Lung-CT | `RIDER Lung CT` | 0 | Repeat CT scans (Memorial Sloan Kettering) |

DICOM series are downloaded via `tcia_utils.nbia`, the middle slice is extracted and converted to 224×224 RGB PNG. Binary labels (0/1) are assigned per collection for the lung classification task.

### NIH Chest X-Ray14

Images are streamed from HuggingFace and resized to 224×224 RGB PNG. A `labels.csv` is generated with multi-label disease indices and binary labels (0 = No Finding, 1 = any pathology). Partitioning is by patient ID hash.

## Workflow DAG Architecture

```
          TCIA Branch                        NIH Branch
     (runs in parallel)                 (runs in parallel)
              │                                 │
    ┌─────────▼─────────┐             ┌─────────▼─────────┐
    │  download_tcia     │  (optional) │  download_nih      │  (optional)
    │  or pre-staged     │             │  or pre-staged     │
    └─────────┬─────────┘             └─────────┬─────────┘
    ┌─────────▼─────────┐             ┌─────────▼─────────┐
    │  partition_tcia    │             │  partition_nih     │
    └──┬──────────────┬─┘             └──┬──────────────┬─┘
       │              │                  │              │
  ┌────▼────┐   ┌─────▼──────┐    ┌─────▼────┐  ┌─────▼──────┐
  │  stats  │   │  Round 1   │    │  stats   │  │  Round 1   │
  │  (async)│   │  (SubWF)   │    │  (async) │  │  (SubWF)   │
  └─────────┘   └─────┬──────┘    └──────────┘  └─────┬──────┘
                       │  ...                          │  ...
                ┌──────▼──────┐              ┌─────────▼──────┐
                │  Round T    │              │  Round T       │
                │  (SubWF)    │              │  (SubWF)       │
                └──────┬──────┘              └─────────┬──────┘
                       │                               │
         ┌─────────────┴───────────────────────────────┘
         │                CONVERGE
    ┌────▼────┐  ┌───────────────┐  ┌───────────────┐
    │ central │  │ cross_dataset │  │ central       │
    │ _tcia   │  │ _eval         │  │ _nih          │
    └────┬────┘  └───────┬───────┘  └───────┬───────┘
         └───────────────┼──────────────────┘
                   ┌─────▼─────┐
                   │ plot_     │
                   │ results   │
                   └─────┬─────┘
                   ┌─────▼─────┐
                   │ generate_ │
                   │ report    │
                   └───────────┘

  Within each Round SubWorkflow:

    select_clients
         │
    ┌────▼────┬──────────┬──────────┐
    │train_c0 │ train_c1 │...train_cK│  ← parallel fan-out
    └────┬────┴────┬─────┴────┬─────┘
         └─────────┼──────────┘
            ┌──────▼──────┐
            │  aggregate  │              ← fan-in
            └──────┬──────┘
            ┌──────▼──────┐
            │  validate   │
            └─────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- [Pegasus WMS](https://pegasus.isi.edu/downloads/) 5.0+
- [HTCondor](https://htcondor.org/) 10+
- NVIDIA GPU + CUDA 12.2 (for training jobs)

### 1. Install Python Dependencies

```bash
cd medical-imaging-fl-workflow
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Build the Training Container (required before running any workflow)

The Docker container packages PyTorch, Flower, and all training scripts. Every workflow job runs inside this container.

```bash
docker build -f containers/Dockerfile.fl-training -t fl-training:latest .
```

> **Note:** If using a remote registry (e.g., for FABRIC nodes), push after building:
> ```bash
> docker tag fl-training:latest <registry>/fl-training:latest
> docker push <registry>/fl-training:latest
> ```
> Then update `execution.container` in the config YAML to match.

### 3. Prepare Data

The workflow supports two data modes:

**Option A: Automatic download (default)** — The workflow downloads data as part of the DAG via `download_data.py`. TCIA images are fetched via `tcia_utils` and converted from DICOM to PNG. NIH images are streamed from HuggingFace.

**Option B: Pre-staged data (recommended for large datasets)** — Download data once on your local machine, `scp` to the server, and point the config at the pre-staged tar.gz files:

```bash
# Download locally (laptop)
python scripts/download_datav2.py --dataset tcia --config configs/default.yml
python scripts/download_datav2.py --dataset nih  --config configs/default.yml

# Transfer to server
scp tcia_raw_data.tar.gz server:/data/
scp nih_raw_data.tar.gz  server:/data/

# Set in config YAML (uncomment and update path)
# datasets.tcia.raw_data_path: "/data/tcia_raw_data.tar.gz"
# datasets.nih.raw_data_path:  "/data/nih_raw_data.tar.gz"
```

When `raw_data_path` is set, the download job is skipped and `partition_clients` reads directly from the pre-staged archive.

#### Data Directory Structure

After download and partitioning, each client shard contains:
```
client_<i>/
├── train/
│   ├── 0/     # label 0 images (*.png)
│   └── 1/     # label 1 images (*.png)
```

The test set follows the same structure:
```
test/
├── 0/         # label 0 test images (*.png)
└── 1/         # label 1 test images (*.png)
```

### 4. Run a Smoke Test (2 rounds, 2 clients)

```bash
python3 fl_main.py --config configs/default.yml --output fl_main.yml --plan

pegasus-plan \
    --conf pegasus.properties \
    --sites condorpool \
    --output-sites local \
    --dir work/submit \
    --cleanup leaf \
    --submit fl_main.yml
```

### 5. Run the Full Experiment (20 rounds, 5 clients)

```bash
python3 fl_main.py --config configs/exp_full.yml --output fl_main.yml --plan

pegasus-plan \
    --conf pegasus.properties \
    --sites condorpool \
    --output-sites local \
    --dir work/submit \
    --cleanup leaf \
    --submit fl_main.yml
```

### 6. Run a Hyperparameter Sweep

```bash
cd medical-imaging-fl-workflow

# Submit all experiment configs as a parallel ensemble
./run_sweep.sh --max-running 4

# Monitor
pegasus-em workflows fl_sweep_*
```

### 7. Run a Single Experiment Config

```bash
cd medical-imaging-fl-workflow
./plan_fl_workflow.sh --config configs/exp_e2_algorithm.yml
```

## Experiments

| ID | Name | Purpose | Key Parameters |
|---|---|---|---|
| E1 | Baseline | Central vs. FL accuracy gap | K=5, T=50, FedAvg |
| E2 | Algorithm | FedAvg vs. FedProx | K=5, T=50, varying mu |
| E3 | Scalability | Client count impact | K={3,5,10} |
| E4 | Communication | Rounds vs. local epochs trade-off | T={10,25,50}, E={1,3,5} |
| E5 | Cross-Dataset | Cross-modality generalization | TCIA + NIH, domain gap |
| E6 | Improved Training | Fix model collapse from E1-E5 | Frozen backbone, Adam, class-weighted loss, augmentation |

### E6: Improved Training

Experiments E1-E5 exhibited model collapse to majority-class prediction (~46-54% accuracy). Root causes: full-model averaging destroying pretrained ImageNet features, no class imbalance handling, aggressive learning rate (0.01 with SGD), no data augmentation on small client shards, and a bug where `fedprox_mu` was never passed to `train_local.py`.

E6 fixes all of these:

| Parameter | E1-E5 (baseline) | E6 (improved) | Rationale |
|-----------|-------------------|---------------|-----------|
| `learning_rate` | 0.01 | **0.0001** | 100x lower — prevents catastrophic overwriting of pretrained weights |
| `optimizer` | SGD | **Adam** | Adaptive per-parameter LR; gentler updates |
| `freeze_backbone` | false | **true** | Only train classifier head; preserves ImageNet features |
| `class_weighted_loss` | false | **true** | Handles class imbalance; prevents majority-class collapse |
| `augmentation` | false | **true** | Regularizes small client shards (~50-100 images) |
| `grad_clip` | 0.0 | **1.0** | Stabilizes training; prevents gradient explosion |
| `lr_scheduler` | none | **cosine** | Gradual warmdown over local epochs |

Additional training options added in E6:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--freeze-backbone` | flag | off | Freeze all layers except `fc`/`classifier` |
| `--optimizer` | choice | `sgd` | `sgd`, `adam`, or `adamw` |
| `--class-weighted-loss` | flag | off | Weight loss by inverse class frequency |
| `--augmentation` | flag | off | Data augmentation (RandomResizedCrop, HorizontalFlip, Rotation, ColorJitter) |
| `--grad-clip` | float | `0.0` | Max gradient norm (0 = disabled) |
| `--lr-scheduler` | choice | `none` | `none`, `cosine`, or `step` |

All new config keys use `.get()` with defaults matching E1-E5 behavior, so existing configs work without modification.

## Configuration

All experiments are driven by YAML config files in `configs/`.

| Config | Rounds | Epochs | Clients | Fraction | Purpose |
|--------|--------|--------|---------|----------|---------|
| `default.yml` | 2 | 1 | 2 | 1.0 | Smoke test / pipeline validation |
| `exp_full.yml` | 20 | 3 | 5 | 0.6 | Full experiment |
| `exp_e1_baseline.yml` | 20 | 3 | 5 | 1.0 | Central vs. FL accuracy gap |
| `exp_e2_algorithm.yml` | 20 | 3 | 5 | 1.0 | FedAvg vs. FedProx |
| `exp_e3_scalability.yml` | 20 | 3 | 3 | 1.0 | Client count scaling |
| `exp_e4_communication.yml` | 10 | 5 | 5 | 1.0 | Rounds vs. local epochs trade-off |
| `exp_e5_cross_dataset.yml` | 20 | 3 | 5 | 1.0 | Cross-modality generalization |
| `exp_e6_improved.yml` | 50 | 5 | 10 | 1.0 | Improved training (fixes model collapse) |

### Data Configuration Parameters

| Parameter | Section | Default | Description |
|---|---|---|---|
| `max_series_per_collection` | `datasets.tcia` | 50 | Max DICOM series to download per TCIA collection |
| `collection_labels` | `datasets.tcia` | — | Binary label (0/1) mapping per collection |
| `max_samples` | `datasets.nih` | 2000 | Max images to download from NIH dataset |
| `download_source` | `datasets.nih` | `huggingface` | Download source: `huggingface` or `nih_box` |
| `raw_data_path` | `datasets.*` | — | Path to pre-staged tar.gz (skips download job) |

## Data Staging: CondorIO

This workflow uses `pegasus.data.configuration=condorio`:
- HTCondor manages all file transfers via `transfer_input_files` / `transfer_output_files`
- No shared filesystem or separate staging site required
- Files are tar.gz archives to minimize transfer count and overhead

## Execution Environment

### Current Worker Pool

| Worker | GPUs | Memory | Site |
|--------|------|--------|------|
| MAX-gpu-worker-1 | 2x Tesla T4 | 16 GB | MAX |
| MAX-gpu-worker-2 | 2x Tesla T4 | 16 GB | MAX |
| NCSA-gpu-worker-1 | 2x Tesla T4 | 16 GB | NCSA |
| TACC-worker-1 | — | 16 GB | TACC |
| WASH-worker-1 | — | 16 GB | WASH |
| pegasus-submit | — | 16 GB | Local |

GPU training jobs require ~15 GB memory + 1 GPU, which requires a full machine's allocation.

### Pegasus Properties

Key properties set by `fl_main.py`:
- `pegasus.data.configuration = condorio` — HTCondor-managed transfers
- `pegasus.file.cleanup.scope = none` — prevents premature removal of shared data between sub-workflows
- `pegasus.integrity.checking = nosymlink`
- `dagman.retry = 3` — automatic retry on job failure

## Known Issues and Fixes

1. **Empty input cache for sub-workflows** — With `cleanup.scope = deferred`, Pegasus may remove scratch files between sequential sub-workflow rounds. Fixed by setting `cleanup.scope = none`.

2. **Partition outputs not propagating** — Client data files from `partition_clients` must use `stage_out=False, register_replica=True` to remain available in scratch for all downstream sub-workflows.

3. **Memory constraints** — Training jobs request ~15 GB. With dynamic slot partitioning, jobs will only match when a full machine is available (no other jobs consuming its slots).

## Metrics

### Workflow Outputs

| File Pattern | Description |
|---|---|
| `{dataset}_global_model_r{N}.pt` | Global model after round N (~43 MB ResNet-18) |
| `{dataset}_r{N}_round_metrics.json` | Per-round accuracy, macro-F1, per-class precision/recall/F1 |
| `{dataset}_centralized_metrics.json` | Centralized baseline (all data, no FL) |
| `{dataset}_data_stats.json` | Per-client sample counts, class distribution, KL divergence |
| `cross_dataset_metrics.json` | Cross-domain evaluation (each model tested on other dataset) |
| `figures.tar.gz` | Training curves, confusion matrices, plots |
| `experiment_report.md` | Auto-generated summary with tables |

### Metric JSON Structure

**Round metrics** (`{dataset}_r{N}_round_metrics.json`):
```json
{
  "accuracy": 0.50,
  "macro_f1": 0.069,
  "num_samples": 50,
  "per_class": { "0": {"precision": 0.5, "recall": 1.0, "f1": 0.67, "support": 25}, "1": {...} },
  "round": 0,
  "dataset": "tcia"
}
```

**Cross-dataset metrics** (`cross_dataset_metrics.json`):
- `same_domain`: Each model on its own test set
- `cross_domain`: Each model on the other's test set
- `domain_gap`: Difference in accuracy (cross - same)

**Data stats** (`{dataset}_data_stats.json`):
- `per_client[i].total_samples`, `class_distribution`
- `heterogeneity.kl_divergence_per_client`, `kl_divergence_avg`

### Extracting Pegasus Workflow Metrics

```bash
pegasus-statistics -s all <run-dir>        # Summary (makespan, jobs, etc.)
pegasus-statistics -s jb_stats <run-dir>   # Per-job timing breakdown
pegasus-analyzer <run-dir>                 # Failure analysis
pegasus-graphviz <run-dir>                 # DAG visualization
```

**Stampede DB queries** (access via `sqlite3 <run-dir>/workflow.db`):

```sql
-- Per-job compute timing
SELECT j.exec_job_id, ji.site, ji.cluster_start,
       ji.cluster_duration, ji.remote_duration, ji.exitcode
FROM job j JOIN job_instance ji ON j.job_id = ji.job_id
WHERE j.type_desc = 'compute'
ORDER BY ji.cluster_start;

-- Queue wait time (submit to execute)
SELECT j.exec_job_id,
       MIN(CASE WHEN js.state = 'SUBMIT' THEN js.timestamp END) as submit_time,
       MIN(CASE WHEN js.state = 'EXECUTE' THEN js.timestamp END) as execute_time,
       MIN(CASE WHEN js.state = 'EXECUTE' THEN js.timestamp END) -
       MIN(CASE WHEN js.state = 'SUBMIT' THEN js.timestamp END) as wait_seconds
FROM job j
JOIN job_instance ji ON j.job_id = ji.job_id
JOIN jobstate js ON ji.job_instance_id = js.job_instance_id
WHERE j.type_desc = 'compute'
GROUP BY j.exec_job_id
ORDER BY submit_time;
```

## References

- [Pegasus WMS Documentation](https://pegasus.isi.edu/documentation/)
- [Pegasus Ensemble Manager](https://pegasus.isi.edu/documentation/reference-guide/pegasus-service.html)
- [Pegasus SubWorkflow API](https://pegasus.isi.edu/documentation/python/Pegasus.api.html#Pegasus.api.workflow.SubWorkflow)
- [Flower FL Framework](https://flower.ai/)
- McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (FedAvg), AISTATS 2017
- Li et al., "Federated Optimization in Heterogeneous Networks" (FedProx), MLSys 2020
