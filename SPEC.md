# Specification: Federated Learning Workflow on Pegasus

## 1. Project Overview

**Goal:** Build a fully reproducible, end-to-end Federated Learning (FL) pipeline orchestrated by Pegasus WMS. The workflow automates data preparation, per-client local training, server-side aggregation, evaluation, and multi-round iteration — all expressed as a DAG that can run on distributed HPC/cloud resources.

**Key Contributions:**
1. A reusable Pegasus workflow template for Federated Learning experiments
2. Demonstration on naturally heterogeneous medical imaging datasets (TCIA, NIH Chest X-Ray)
3. Reproducibility via workflow provenance, container packaging, and open-access data
4. Performance analysis of FL convergence under real-world data heterogeneity

---

## 2. Datasets

Based on the evaluation in `FL_Dataset_Evaluation.md`:

| Dataset | Role | Modality | Access |
|---|---|---|---|
| **TCIA Collections** (primary) | Naturally decentralized multi-site oncology imaging | 3D (CT/MRI) | Open |
| **NIH Chest X-Ray14** (secondary) | Cross-dataset domain shift evaluation | 2D (X-Ray) | Open |

Each dataset's institutions/sources are treated as FL **clients** (data silos).

---

## 3. Federated Learning Design

### 3.1 FL Algorithm
- **FedAvg** (baseline) — weighted averaging of client model updates
- **FedProx** (heterogeneity-aware) — proximal term to handle non-IID data
- Comparison against **centralized training** baseline

### 3.2 Model Architecture
- ResNet-18 / EfficientNet-B0 (configurable) for image classification
- Transfer learning from ImageNet pre-trained weights

### 3.3 FL Parameters
| Parameter | Default | Sweep Range |
|---|---|---|
| Number of clients (K) | 5 | 3, 5, 10 |
| Rounds (T) | 50 | 10, 25, 50, 100 |
| Local epochs (E) | 5 | 1, 3, 5 |
| Batch size | 32 | 16, 32, 64 |
| Learning rate | 0.01 | 0.0001, 0.001, 0.01 |
| Client fraction per round (C) | 1.0 | 0.5, 0.8, 1.0 |
| Optimizer | sgd | sgd, adam, adamw |
| Freeze backbone | false | true, false |
| Class-weighted loss | false | true, false |
| Data augmentation | false | true, false |
| Gradient clipping | 0.0 | 0.0, 1.0 |
| LR scheduler | none | none, cosine, step |
| FedProx mu | 0.0 | 0.0, 0.01, 0.1 |

---

## 4. Pegasus Workflow Architecture

### 4.1 High-Level DAG Structure

The workflow is a **hierarchical workflow** using Pegasus `SubWorkflow` for modularity. **Both datasets run as fully parallel branches** that converge only at the final cross-dataset evaluation stage.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          TOP-LEVEL WORKFLOW (fl_main)                         │
│                                                                              │
│  ┌─────────────────── TCIA BRANCH (parallel) ──────────────────────────┐    │
│  │                                                                      │    │
│  │  ┌───────────┐   ┌─────────┐   ┌─────────┐       ┌─────────┐      │    │
│  │  │ download  │──▶│partition│──▶│ Round 1 │──...──▶│ Round T │      │    │
│  │  │ tcia *    │   │ tcia    │   │ (SubWF) │       │ (SubWF) │      │    │
│  │  └───────────┘   └────┬────┘   └─────────┘       └────┬────┘      │    │
│  │                       │                                │           │    │
│  │                  ┌────▼────┐                   tcia_global_model   │    │
│  │                  │  stats  │                   tcia_metrics        │    │
│  │                  │  tcia   │                                       │    │
│  │                  └─────────┘                                       │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────── NIH BRANCH (parallel) ───────────────────────────┐    │
│  │                                                                      │    │
│  │  ┌───────────┐   ┌─────────┐   ┌─────────┐       ┌─────────┐      │    │
│  │  │ download  │──▶│partition│──▶│ Round 1 │──...──▶│ Round T │      │    │
│  │  │ nih *     │   │ nih     │   │ (SubWF) │       │ (SubWF) │      │    │
│  │  └───────────┘   └────┬────┘   └─────────┘       └────┬────┘      │    │
│  │                       │                                │           │    │
│  │                  ┌────▼────┐                   nih_global_model    │    │
│  │                  │  stats  │                   nih_metrics         │    │
│  │                  │  nih    │                                       │    │
│  │                  └─────────┘                                       │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌───────────────── CONVERGE (fan-in from both branches) ──────────────┐    │
│  │                                                                      │    │
│  │  ┌──────────────┐  ┌────────────────┐  ┌──────────────┐            │    │
│  │  │ cross_dataset│  │ centralized    │  │ generate     │            │    │
│  │  │ _comparison  │  │ _baseline (x2) │  │ _plots       │            │    │
│  │  └──────────────┘  └────────────────┘  └──────┬───────┘            │    │
│  │                                               │                    │    │
│  │                                        ┌──────▼───────┐            │    │
│  │                                        │  generate    │            │    │
│  │                                        │  _report     │            │    │
│  │                                        └──────────────┘            │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
```

*\* Download jobs are optional — skipped when `raw_data_path` is set (pre-staged data).*

**Parallelism points:**
- TCIA and NIH data preparation run **simultaneously**
- TCIA and NIH FL training loops run **simultaneously**
- Within each round: K client training jobs run **in parallel** (fan-out)
- Centralized baselines for both datasets run **in parallel** with cross-dataset comparison
- Statistics jobs run **in parallel** with the first training round (no dependency)

### 4.2 Stage 1 — Data Preparation (`data_prepare`)

| Job | Description | Inputs | Outputs |
|---|---|---|---|
| `download_data` | Fetch real images: TCIA DICOM→PNG via `tcia_utils`, NIH via HuggingFace streaming. **Skipped** when `raw_data_path` is set (pre-staged data). | config.yml | `{dataset}_raw_data.tar.gz` |
| `partition_clients` | Split data into K client shards with `train/<label>/*.png` structure. TCIA: one collection per client, binary labels from `collection_labels`. NIH: patient-ID hash partition, binary label (No Finding=0, pathology=1). | raw_data.tar.gz, config.yml | `client_{i}_data.tar.gz`, `test_data.tar.gz`, `initial_model.pt` |
| `compute_statistics` | Per-client data distribution stats (class balance, volume counts) | client_*_data/ | data_stats.json |

**Pre-staged data mode:** When `raw_data_path` is set in the dataset config, the `download_data` job is omitted from the DAG and the pre-staged tar.gz is registered in the Pegasus replica catalog. This avoids re-downloading large datasets on every workflow run.

### 4.3 Stage 2 — FL Training Round (`fl_round` SubWorkflow)

Each round is a **SubWorkflow** (`SubWorkflow("fl_round.yml", is_planned=False)`).

```
┌───────────────────────────────────────────────────────────────┐
│                   FL ROUND SubWorkflow                         │
│                                                               │
│  ┌──────────────┐                                             │
│  │ select_clients│  (select C*K clients for this round)       │
│  └──────┬───────┘                                             │
│         │                                                     │
│    ┌────▼────┐  ┌──────────┐  ┌──────────┐                   │
│    │ train_0 │  │ train_1  │  │ train_K  │  (parallel)       │
│    │ (local) │  │ (local)  │  │ (local)  │                   │
│    └────┬────┘  └────┬─────┘  └────┬─────┘                   │
│         │            │             │                           │
│    ┌────▼────────────▼─────────────▼────┐                     │
│    │          aggregate_models           │  (FedAvg / FedProx)│
│    │     (fan-in: weighted averaging)    │                     │
│    └────────────────┬───────────────────┘                     │
│                     │                                         │
│    ┌────────────────▼───────────────────┐                     │
│    │         validate_global             │                     │
│    │   (eval global model on test set)   │                     │
│    └────────────────┬───────────────────┘                     │
│                     │                                         │
│              global_model_round_t.pt                           │
│              round_metrics_t.json                              │
└───────────────────────────────────────────────────────────────┘
```

**Key jobs per round:**

| Job | Parallelism | Description |
|---|---|---|
| `select_clients` | 1 | Randomly sample `C*K` clients; output selected client IDs |
| `train_client_{i}` | K (fan-out) | Local training on client i's data shard; outputs local model weights |
| `aggregate_models` | 1 (fan-in) | Weighted average of local models → new global model |
| `validate_global` | 1 | Evaluate global model on held-out test set; output metrics |

### 4.4 Stage 3 — Evaluation & Visualization (`evaluation`)

| Job | Description | Inputs | Outputs |
|---|---|---|---|
| `aggregate_metrics` | Collect per-round metrics into single timeline | round_metrics_*.json | training_history.json |
| `per_client_eval` | Evaluate final global model on each client's test split | global_model_final.pt, client_*_test/ | per_client_metrics.json |
| `centralized_baseline` | Train centralized model on pooled data for comparison | all client data | centralized_metrics.json |
| `generate_plots` | Convergence curves, accuracy per client, data distribution plots | *.json metrics | figures/*.png |
| `generate_report` | Summary markdown/LaTeX with key results | all metrics + figures | report.md |

### 4.5 Round Orchestration Strategy

Because FL training is inherently iterative (round T depends on round T-1), we have two options:

#### Option A: Pegasus Ensemble Manager (Recommended)
Use `pegasus-em` to manage sequential round submission:
- Create an ensemble `fl_experiment`
- A **trigger script** monitors for the completion of round T and submits round T+1
- Each round is a standalone Pegasus workflow submitted to the ensemble
- Allows throttling, priority control, and fault recovery per round
- Reference: [pegasus-service.html](https://pegasus.isi.edu/documentation/reference-guide/pegasus-service.html)

#### Option B: Hierarchical Workflow with SubWorkflow
Use a single top-level workflow where each round is a `SubWorkflow`:
```python
from Pegasus.api import *

main_wf = Workflow("fl_main")

for t in range(T):
    round_wf = SubWorkflow(f"fl_round_{t}.yml", is_planned=False)
    round_wf.add_args("--round", str(t), "--output-sites", "local", "-vvv")
    main_wf.add_jobs(round_wf)

    if t > 0:
        main_wf.add_dependency(round_wf, parents=[prev_round_wf])
    prev_round_wf = round_wf
```
- Simpler to set up; entire FL experiment is one workflow submission
- Data dependencies between rounds are handled via files (global model checkpoint)
- Reference: [Pegasus.api.workflow.SubWorkflow](https://pegasus.isi.edu/documentation/python/Pegasus.api.html#Pegasus.api.workflow.SubWorkflow)

#### Recommendation
**Use Option B (SubWorkflow)** — it is self-contained, fully reproducible from a single `pegasus-plan` invocation, and the provenance captured by Pegasus covers the entire experiment. Use Option A for production-scale experiments where hundreds of rounds or fault recovery across rounds is critical.

### 4.6 Hyperparameter Sweep via Ensemble Manager

For the parameter sweep experiments (varying K, E, lr, C), use the **Ensemble Manager**:
- Each combination of hyperparameters is a separate FL workflow
- Submit all combinations as an ensemble with throttling (e.g., `max_running=4`)
- This naturally parallelizes independent experiment configurations

```bash
pegasus-em server
pegasus-em create fl_sweep
for config in configs/*.yml; do
    pegasus-em submit fl_sweep.$(basename $config .yml) \
        ./plan_fl_workflow.sh --config $config
done
```

---

## 5. Software Stack & Containers

### 5.1 Training Container
```
Base: nvidia/cuda:12.2-runtime-ubuntu22.04
Python: 3.11
Frameworks:
  - PyTorch 2.2+
  - torchvision
  - Flower (flwr) 1.x — FL strategies (FedAvg, FedProx), client API, and metrics
  - scikit-learn
  - pandas, numpy
  - matplotlib, seaborn (for plots)
  - pydicom, nibabel (for TCIA DICOM/NIfTI handling)
  - tcia_utils — TCIA/NBIA REST API client for DICOM series download
  - pylibjpeg, pylibjpeg-libjpeg — DICOM JPEG transfer syntax decompression
  - datasets, huggingface_hub — HuggingFace dataset streaming (NIH ChestX-ray14)
```

### 5.2 Pegasus Container
```
Base: pegasus/pegasus:latest (or matching version)
Includes: pegasus-plan, pegasus-run, pegasus-em, condor
```

### 5.3 Key Scripts (Transformations)

| Script | Language | Purpose |
|---|---|---|
| `resource_monitor.py` | Python | Background GPU/CPU/RAM monitoring thread (used by training scripts) |
| `download_data.py` | Python | Real dataset download: TCIA DICOM→PNG via `tcia_utils`, NIH via HuggingFace streaming |
| `download_datav2.py` | Python | Standalone test script for data downloads (supports `--no-tar` for local inspection) |
| `partition_clients.py` | Python | Partition into K client shards with `train/<label>/*.png` structure; create initial model |
| `select_clients.py` | Python | Deterministic per-round client selection (seeded by round) |
| `train_local.py` | Python | Single-client local training (Flower + PyTorch); supports backbone freezing, optimizer selection (SGD/Adam/AdamW), class-weighted loss, data augmentation, gradient clipping, LR scheduling (cosine/step), FedProx proximal term; collects resource metrics |
| `aggregate.py` | Python | Server-side model aggregation (FedAvg/FedProx weighted avg) |
| `evaluate.py` | Python | Model evaluation — accuracy, F1, per-class metrics; collects resource metrics |
| `compute_statistics.py` | Python | Per-client data distribution stats + KL divergence heterogeneity |
| `centralized_baseline.py` | Python | Train centralized model on pooled data (upper bound); same training improvements as `train_local.py` for fair comparison; collects resource metrics |
| `cross_dataset_eval.py` | Python | Cross-modality eval: TCIA model↔NIH data and vice versa |
| `plot_results.py` | Python | Generate convergence, distribution, and cross-dataset figures |
| `generate_report.py` | Python | Compile all results into Markdown experiment report |

---

## 6. Catalogs

### 6.1 Replica Catalog (RC)
- Input config files (experiment parameters)
- Pre-trained model weights (ImageNet checkpoint)
- Pre-staged raw data archives (when `raw_data_path` is set in dataset config)
- Sub-workflow YAML files (for Pegasus to plan each round)
- Helper modules (`resource_monitor.py`, `evaluate.py`) registered for staging to remote workers

### 6.2 Transformation Catalog (TC)
- All Python scripts listed in Section 5.3
- Scripts are **stageable** (`is_stageable=True, site="local"`) — staged from the submit host to workers at runtime rather than baked into the container
- Container references for each transformation (`bypass_staging=True` to mount host work directory)

### 6.3 Site Catalog (SC)
Target execution environments:
- **local** — submit host (workflow planning, lightweight jobs)
- **condorpool** — HTCondor pool across MAX, NCSA, TACC, WASH sites

### 6.4 Sub-Workflow Catalog Propagation
Each FL round sub-workflow is planned independently at runtime. To give the inner planner access to replica, transformation, and site catalogs, `fl_main.py` generates a per-sub-workflow `.properties` file containing:
- `pegasus.catalog.replica.file` — per-round RC with helper module registrations
- `pegasus.catalog.transformation.file` — path to shared `transformations.yml`
- `pegasus.catalog.site.file` — path to shared `sites.yml`
- `pegasus.transfer.worker.package=true` — stage Pegasus worker package from submit host
- `pegasus.data.configuration=condorio`

The properties file is passed to each `SubWorkflow` via `--conf`.

---

## 7. Execution Environment

### 7.1 Compute Requirements
| Job Type | CPU | GPU | Memory | Wall Time |
|---|---|---|---|---|
| `download_data` | 1 | 0 | 4 GB | 30 min |
| `partition_clients` | 1 | 0 | 8 GB | 10 min |
| `train_local` (per client) | 4 | 1 (NVIDIA) | 16 GB | 1-4 hrs |
| `aggregate_models` | 2 | 0 | 8 GB | 5 min |
| `validate_global` | 2 | 1 | 8 GB | 15 min |
| `centralized_baseline` | 4 | 1 | 32 GB | 4-8 hrs |
| `generate_plots` | 1 | 0 | 4 GB | 5 min |

### 7.2 Data Staging — CondorIO
- **CondorIO mode** (`pegasus.data.configuration=condorio`) — HTCondor manages all file transfers
- Input/output files are declared in Condor submit files via `transfer_input_files` / `transfer_output_files`
- No shared filesystem or separate staging site required
- Ideal for FABRIC where nodes do not share storage
- Pegasus automatically generates the transfer directives; jobs see files in their working directory

---

## 8. Experiment Plan

### 8.1 Experiments

| Experiment | Purpose | Configuration |
|---|---|---|
| **E1: Baseline** | Central vs. FL accuracy gap | K=5, T=50, FedAvg |
| **E2: Algorithm Comparison** | FedAvg vs. FedProx under heterogeneity | K=5, T=50, varying heterogeneity |
| **E3: Scalability** | Impact of client count | K={3,5,10}, T=50 |
| **E4: Communication Efficiency** | Fewer rounds, more local epochs | T={10,25,50}, E={1,3,5} |
| **E5: Cross-Dataset** | Generalization across modalities | TCIA primary, NIH Chest X-Ray secondary |
| **E6: Improved Training** | Fix model collapse from E1-E5 | Frozen backbone, Adam, lr=0.0001, class-weighted loss, augmentation, grad clipping |

### 8.2 Metrics
- **Accuracy** (global and per-client)
- **F1 Score** (macro-averaged)
- **Convergence Rate** (rounds to reach target accuracy)
- **Communication Cost** (total bytes transferred / number of round-trips)
- **Workflow Makespan** (via `pegasus-statistics`)
- **Per-job wall time and resource usage** (from Pegasus provenance DB)

### 8.3 Expected Figures
1. Convergence curves: global accuracy vs. round (FedAvg vs. FedProx vs. centralized)
2. Per-client accuracy heatmap (clients x rounds)
3. Data distribution visualization (class imbalance across clients)
4. Scalability plot: makespan vs. number of clients
5. Communication efficiency: accuracy vs. total data transferred
6. Workflow DAG visualization (from `pegasus-graphviz`)

---

## 9. Project Directory Structure

```
medical-imaging-fl-workflow/           # Self-contained workflow directory
├── README.md                          # Project overview, quick start, architecture diagram
├── requirements.txt                   # Python dependencies (PyTorch, Flower, Pegasus API, etc.)
├── SPEC.md                            # This file — full technical specification
├── fl_main.py                         # Top-level workflow generator (parallel dual-dataset DAG)
├── fl_round.py                        # Per-round SubWorkflow generator (fan-out/fan-in)
├── plan_fl_workflow.sh                # Plan + submit a single FL workflow via pegasus-plan
├── run_sweep.sh                       # Launch hyperparameter sweep via Ensemble Manager
├── configs/
│   ├── default.yml                    # Smoke-test config (K=2, T=2, FedAvg, both datasets)
│   ├── exp_full.yml           # Full experiment config (K=5, T=20, 3 local epochs, 0.6 fraction)
│   ├── exp_e1_baseline.yml            # E1: Central vs. FL accuracy gap
│   ├── exp_e2_algorithm.yml           # E2: FedAvg vs. FedProx
│   ├── exp_e3_scalability.yml         # E3: Client count scaling (K=3,5,10)
│   ├── exp_e4_communication.yml       # E4: Rounds vs. local epochs trade-off
│   ├── exp_e5_cross_dataset.yml       # E5: Cross-modality generalization
│   └── exp_e6_improved.yml           # E6: Improved training (fixes model collapse)
├── scripts/
│   ├── resource_monitor.py             # GPU/CPU/RAM monitoring (ResourceMonitor class)
│   ├── download_data.py               # Real dataset download (TCIA DICOM→PNG, NIH HuggingFace)
│   ├── download_datav2.py            # Standalone test script for data downloads
│   ├── partition_clients.py           # Data partitioning into train/<label>/*.png + initial model
│   ├── select_clients.py              # Deterministic per-round client selection
│   ├── train_local.py                 # Local client training (Flower + PyTorch)
│   ├── aggregate.py                   # FedAvg / FedProx weighted aggregation
│   ├── evaluate.py                    # Model evaluation (accuracy, F1, per-class)
│   ├── compute_statistics.py          # Data distribution analysis + KL divergence
│   ├── centralized_baseline.py        # Centralized training upper bound
│   ├── cross_dataset_eval.py          # Cross-dataset generalization measurement
│   ├── plot_results.py                # Publication-quality figure generation
│   └── generate_report.py             # Markdown experiment report compilation
└── containers/
    └── Dockerfile.fl-training         # CUDA 12.2 + PyTorch + Flower container
```

---

## 10. References

- Pegasus WMS Documentation: https://pegasus.isi.edu/documentation/
- Pegasus Ensemble Manager: https://pegasus.isi.edu/documentation/reference-guide/pegasus-service.html
- Pegasus SubWorkflow API: https://pegasus.isi.edu/documentation/python/Pegasus.api.html#Pegasus.api.workflow.SubWorkflow
- McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (FedAvg), AISTATS 2017
- Li et al., "Federated Optimization in Heterogeneous Networks" (FedProx), MLSys 2020
- TCIA: https://www.cancerimagingarchive.net/
- NIH Chest X-Ray14: https://nihcc.app.box.com/v/ChestXray-NIHCC
- Flower FL Framework: https://flower.ai/

---

## 11. Design Decisions (Resolved)

1. **Orchestration: SubWorkflow + Ensemble Manager**
   - Individual FL experiment runs use **SubWorkflow** (Option B) — one `pegasus-plan` invocation captures full provenance for an entire FL training run.
   - Hyperparameter sweeps use the **Ensemble Manager** — each config is a separate workflow submitted to an ensemble with throttling (`max_running`).

2. **FL Framework: Flower (flwr)**
   - Use the **Flower** federated learning framework for FL strategy implementations (FedAvg, FedProx).
   - Flower's strategy logic runs inside Pegasus job scripts — Pegasus handles orchestration/scheduling, Flower handles the FL algorithm internals.
   - This avoids reimplementing well-tested aggregation algorithms and gives access to Flower's metrics, logging, and strategy extensibility.
   - Note: We do **not** use Flower's server/client networking architecture. Instead, each Pegasus `train_local` job uses Flower's `Client` API to perform local training, and the `aggregate` job uses Flower's `Strategy` API (e.g., `FedAvg.aggregate_fit()`) to combine model updates. Pegasus file-based data flow replaces Flower's gRPC communication.

3. **GPU Infrastructure: Distributed HTCondor Pool**
   - GPU-accelerated training jobs run on an HTCondor pool spanning multiple sites (MAX, NCSA, TACC, WASH).
   - GPU workers (Tesla T4) at MAX and NCSA handle `train_local` and `evaluate` jobs.
   - Non-GPU workers at TACC and WASH handle lightweight jobs (partitioning, aggregation, statistics).
   - The geographically distributed pool naturally models the federated scenario.

4. **TCIA Client Mapping: 5 Collections**
   - Select **5 TCIA collections** from different contributing institutions to serve as natural FL clients (matching default K=5).
   - Collections should be chosen for: (a) sufficient sample count per client, (b) clear institutional provenance, (c) overlapping classification task (e.g., lung cancer staging/detection).
   - Candidate collections to evaluate:
     - NSCLC-Radiomics (Maastricht)
     - TCGA-LUAD (multi-site TCGA)
     - LIDC-IDRI (multi-site, 7 institutions)
     - NSCLC-Radiogenomics (Stanford/Palo Alto VA)
     - RIDER Lung CT (Memorial Sloan Kettering)
   - Final selection pending data availability survey.

5. **FL + Workflow Co-Design**
   - **Equal emphasis** on FL methodology and workflow orchestration.
   - Key narrative: workflow-aware FL design improves both **reproducibility** and **convergence analysis**.
   - Contributions:
     - (a) A reusable Pegasus workflow template for FL experiments
     - (b) Demonstration that workflow provenance enables deeper FL analysis (per-round resource usage, communication cost tracking, fault recovery)
     - (c) FL results on naturally heterogeneous medical imaging across FABRIC's distributed GPU infrastructure
   - This positions the project at the intersection of scientific workflows and FL.

6. **Data Staging: CondorIO**
   - Use `pegasus.data.configuration=condorio` — HTCondor manages all file transfers.
   - Input/output files declared in Condor submit files via `transfer_input_files` / `transfer_output_files`.
   - No shared filesystem or separate staging site required.
   - Ideal for FABRIC where nodes do not share storage.
   - All data shards and model checkpoints are tar.gz archives to minimize transfer count.

---

## 12. Setup & Dependencies

### 12.1 Python Dependencies

All dependencies are specified in `requirements.txt`:

| Package | Version | Purpose |
|---|---|---|
| `pegasus-wms.api` | >=5.0.6 | Pegasus Workflow API (Workflow, SubWorkflow, catalogs) |
| `torch` | >=2.2.0 | Deep learning framework |
| `torchvision` | >=0.17.0 | Image transforms, pre-trained models (ResNet-18, EfficientNet-B0) |
| `flwr[simulation]` | >=1.7, <2.0 | Flower FL strategies (FedAvg, FedProx), client API |
| `pydicom` | >=2.4.0 | DICOM medical image reading (TCIA) |
| `nibabel` | >=5.2.0 | NIfTI neuroimaging format support |
| `Pillow` | >=10.0.0 | General image loading |
| `numpy` | >=1.26.0 | Numerical computing |
| `pandas` | >=2.1.0 | Data manipulation |
| `scikit-learn` | >=1.4.0 | Metrics, preprocessing |
| `matplotlib` | >=3.8.0 | Plotting |
| `seaborn` | >=0.13.0 | Statistical visualization |
| `pyyaml` | >=6.0.1 | Configuration file parsing |
| `tcia_utils` | >=3.3 | TCIA/NBIA REST API client for DICOM series download |
| `pylibjpeg` | >=2.0 | DICOM pixel data decompression |
| `pylibjpeg-libjpeg` | >=2.0 | JPEG transfer syntax codec for pylibjpeg |
| `datasets` | >=2.14 | HuggingFace dataset loading and streaming |
| `huggingface_hub` | >=0.20 | HuggingFace Hub API client |

### 12.2 System Prerequisites

- Python 3.11+
- [Pegasus WMS](https://pegasus.isi.edu/downloads/) 5.0+
- [HTCondor](https://htcondor.org/) 10+
- Docker (for container build)
- NVIDIA GPU + CUDA 12.2 (for training jobs on FABRIC)

### 12.3 Installation

```bash
cd medical-imaging-fl-workflow
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
docker build -f containers/Dockerfile.fl-training -t fl-training:latest .
```

---

## 13. Implementation Status

| Component | Status | Files |
|---|---|---|
| Specification | Complete | `SPEC.md` |
| Top-level workflow generator | Complete | `fl_main.py` |
| Per-round SubWorkflow generator | Complete | `fl_round.py` |
| All training/eval scripts (12) | Complete | `scripts/*.py` |
| Resource monitoring | Complete | `scripts/resource_monitor.py` |
| Experiment configs (8) | Complete | `configs/*.yml` |
| Ensemble sweep scripts | Complete | `plan_fl_workflow.sh`, `run_sweep.sh` |
| Docker container | Complete | `containers/Dockerfile.fl-training` |
| Dependencies | Complete | `requirements.txt` |
| README | Complete | `README.md` |
| Smoke test (2 rounds, 2 clients) | Passed | `configs/default.yml` |
| E1-E5 experiments (50 rounds, 10 clients) | Complete | `configs/exp_e1_*` through `exp_e5_*` |
| E6 improved training (50 rounds, 10 clients) | Complete | `configs/exp_e6_improved.yml` |

## 14. Key Fixes Applied

1. **`pegasus.file.cleanup.scope`**: Changed from `deferred` to `none`. Deferred cleanup removed scratch files between sequential sub-workflow rounds, leaving empty input caches for downstream planners.

2. **Partition output staging**: Client data files changed from `stage_out=True` to `stage_out=False, register_replica=True`. Intermediate files stay in scratch for sub-workflows while being registered in the JDBCRC.

3. **Condor ClassAd attributes**: Added `+FL_Algorithm`, `+FL_NumRounds`, `+FL_JobType` to all job transformations via `add_profiles(Namespace.CONDOR, ...)` for tracking in the Pegasus stampede DB.

4. **Real data downloads**: Replaced placeholder/metadata-only downloads with actual image acquisition:
   - **TCIA**: `download_data.py` uses `tcia_utils.nbia.downloadSeries()` to fetch DICOM files, extracts middle slice per series, converts to 224×224 RGB PNG via pydicom. Collection names mapped to TCIA API names (some use spaces, e.g., `NSCLC Radiogenomics`).
   - **NIH**: Switched from deprecated `alkzar90/NIH-Chest-X-ray-dataset` (loading script no longer supported) to `BahaaEldin0/NIH-Chest-Xray-14` which provides streaming parquet access.
   - **Partition structure**: `partition_clients.py` now produces `train/<label>/*.png` and `test/<label>/*.png` directories, matching the expected structure in `train_local.py` and `evaluate.py`. This eliminates fallback to synthetic `torch.randn()` data.

5. **Pre-staged data support**: Added `raw_data_path` config parameter. When set, `fl_main.py` skips the download job and registers the pre-staged tar.gz in the Pegasus replica catalog. This avoids re-downloading large datasets (46GB NIH, ~17GB TCIA) on every workflow run.

6. **Data-limiting parameters**: Added `max_series_per_collection` (TCIA) and `max_samples` (NIH) config params to control download size for development and testing.

7. **Stageable scripts**: Changed transformation catalog from container-installed (`is_stageable=False`, PFN inside container at `/opt/scripts/`) to submit-host-staged (`is_stageable=True`, PFN on local filesystem). This avoids rebuilding the Docker container when scripts change. Container uses `bypass_staging=True` to mount the host work directory.

8. **Sub-workflow catalog propagation**: Sub-workflows planned at runtime need their own catalog paths. Added per-sub-workflow `.properties` files with RC, TC, SC paths and `pegasus.transfer.worker.package=true` to stage the Pegasus worker package from the submit host.

9. **E6 model collapse fix**: Experiments E1-E5 collapsed to majority-class prediction. Root causes: full-model averaging destroying pretrained features, no class imbalance handling, aggressive LR (0.01 with SGD), no augmentation, and `fedprox_mu` never passed to `train_local.py`. E6 adds backbone freezing, Adam optimizer, class-weighted loss, data augmentation, gradient clipping, cosine LR scheduler, and fixes the FedProx parameter passthrough.

10. **FedProx bug fix**: `fedprox_mu` was defined in config but never passed from `fl_round.py` to `train_local.py` CLI args, so all FedProx experiments (E2) ran as plain FedAvg. Fixed by always passing `--fedprox-mu` to train jobs.
