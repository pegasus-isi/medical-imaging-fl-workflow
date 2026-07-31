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

The **As run** column gives the value used by the reported experiments E1 through E5 (see §8.1 for the per-experiment deviations). The **Design default** column is the value proposed during specification, kept here because §8.1 and §8.6 refer to it.

| Parameter | Design default | As run (E1-E5) | Sweep Range |
|---|---|---|---|
| Number of clients (K) | 5 | 10 (5 in E3) | 3, 5, 10 |
| Rounds (T) | 50 | 50 (10 in E4) | 10, 25, 50, 100 |
| Local epochs (E) | 5 | 2 (5 in E4) | 1, 3, 5 |
| Batch size | 32 | 32 | 16, 32, 64 |
| Learning rate | 0.01 | 0.001 | 0.0001, 0.001, 0.01 |
| Client fraction per round (C) | 1.0 | 1.0 | 0.5, 0.8, 1.0 |
| Optimizer | sgd | sgd (momentum 0.9, weight decay $10^{-4}$) | sgd, adam, adamw |
| Freeze backbone | false | false | true, false |
| Class-weighted loss | false | false | true, false |
| Data augmentation | false | false | true, false |
| Gradient clipping | 0.0 | 0.0 | 0.0, 1.0 |
| LR scheduler | none | none | none, cosine, step |
| FedProx mu | 0.0 | 0.0 (0.01 in E2) | 0.0, 0.01, 0.1 |

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

These are design-time targets. The values actually requested at runtime come from the `execution` block of the experiment config (`gpu_per_train_job`, `cpu_per_train_job`, `mem_per_train_job`, `max_walltime`) and are translated into HTCondor `request_*` profiles on each transformation, so they can be lowered to fit the available workers without editing the generators. Constraint C4 in §8.4 is the check that they fit.

### 7.2 Data Staging — CondorIO
- **CondorIO mode** (`pegasus.data.configuration=condorio`) — HTCondor manages all file transfers
- Input/output files are declared in Condor submit files via `transfer_input_files` / `transfer_output_files`
- No shared filesystem or separate staging site required
- Ideal for FABRIC where nodes do not share storage
- Pegasus automatically generates the transfer directives; jobs see files in their working directory

---

## 8. Experiment Plan

### 8.1 Experiments

Configurations below are **as run**, matching `configs/exp_e*.yml` and the reported results. Every experiment enables both datasets, uses ResNet-18 with an ImageNet-pretrained backbone, batch size 32, and full client participation (C=1.0). E1 through E5 use SGD with momentum 0.9 and weight decay $10^{-4}$ at lr=0.001. Each configuration varies one factor relative to E1, except where noted.

| Experiment | Purpose | Configuration (as run) |
|---|---|---|
| **E1: Baseline** | Central vs. FL accuracy gap | FedAvg, K=10, T=50, E=2 |
| **E2: Algorithm Comparison** | FedAvg vs. FedProx under heterogeneity | FedProx (mu=0.01), K=10, T=50, E=2 |
| **E3: Scalability** | Impact of client count | FedAvg, K=5, T=50, E=2 |
| **E4: Communication Efficiency** | Fewer rounds, more local epochs | FedAvg, K=10, T=10, E=5 |
| **E5: Cross-Dataset** | Generalization across modalities | FedAvg, K=10, T=50, E=2 (FL rounds not carried through, see §13) |
| **E6: Improved Training** | Address the collapse seen in E1-E5 | FedAvg, K=10, T=50, E=5, Adam at lr=0.0001, frozen backbone, class-weighted loss, augmentation, gradient clipping 1.0, cosine schedule |

**E6 deviates from E1 in more than the optimizer settings.** Its NIH block sets `num_classes: 14` where E1 through E5 set `2`, so the E6 NIH model head has 14 outputs while `partition_clients.py` emits binary labels. The consequences are visible in the released metrics: E6 NIH round 0 accuracy is 0.0032 (chance level across 14 classes) before recovering to 0.4618 from round 1 onward, and its macro F1 of 0.093 is dominated by twelve classes with zero support. The E6 TCIA branch is unaffected (`num_classes: 2`). This value is preserved in `configs/exp_e6_improved.yml` because it is what produced the reported E6 numbers, so treat E6 as a single-factor variation on TCIA only.

### 8.2 Metrics
- **Accuracy** (global and per-client)
- **F1 Score** (macro-averaged)
- **Convergence Rate** (rounds to reach target accuracy)
- **Communication Cost** (total bytes transferred / number of round-trips)
- **Workflow Makespan** (via `pegasus-statistics`)
- **Per-job wall time and resource usage** (from Pegasus provenance DB)

### 8.3 Expected Figures

**Produced by the workflow** (`plot_results.py`, written as both 300 dpi PNG and PDF):

1. `fig1_convergence` — global accuracy and F1 vs. round for each dataset, with the centralized baseline drawn as a reference line
3. `fig3_data_distribution` — per-client class balance for each dataset (heterogeneity evidence)
5. `fig5_cross_dataset` — cross-modality evaluation matrix (TCIA model on NIH data and vice versa)

**Derived outside the workflow**, from the Pegasus provenance database or from several runs, and therefore not acceptance criteria for a single run:

2. Per-client accuracy heatmap (clients x rounds)
4. Scalability plot: makespan vs. number of clients (requires the E3 sweep)
6. Workflow DAG visualization (`pegasus-graphviz` on the planned DAG)

Figure numbering is kept from the original plan so that the identifiers used in code, in output directories, and in the paper stay stable even though only 1, 3, and 5 are generated by the workflow itself.

### 8.4 Constraints

Properties the implementation **must** satisfy. These are the statements the user checks during specification review, before any code is generated, and they are re-checked after every fix applied during debugging. A violation is an implementation defect, not a design choice.

| ID | Constraint | How it is checked |
|---|---|---|
| **C1** | **Round structure.** Exactly `T` sequential `fl_round` sub-workflows per enabled dataset. Round `t` depends only on round `t-1`. Within a round: `select_clients` → `ceil(C*K)` parallel `train_client_*` jobs → a single `aggregate_models` fan-in → a single `validate_global`. | Planned sub-workflow count equals `T x (enabled datasets) + 1`; DAG inspected with `pegasus-graphviz` |
| **C2** | **Inter-round state is a single file.** The only state carried from round `t-1` to round `t` is `{dataset}_global_model_r{t-1}.pt` (or `{dataset}_initial_model.pt` at `t=0`). It must be staged out and registered so the next round's planner can resolve it, and cleanup must not remove it between rounds. | Round `t`'s replica catalog resolves the previous round's model; `pegasus.file.cleanup.scope` is not `deferred` (see §14.1) |
| **C3** | **Dataset branch independence.** The TCIA and NIH branches share no intermediate files and converge only at `cross_dataset_eval` / `plot_results` / `generate_report`. A failure in one branch must not block the other. | No cross-branch edges in the DAG before the converge stage |
| **C4** | **Site resource limits.** GPU jobs (`train_local`, `evaluate`, `centralized_baseline`) request exactly 1 GPU and stay within the per-job CPU and memory limits declared in `execution` (`gpu_per_train_job`, `cpu_per_train_job`, `mem_per_train_job`), which must not exceed what a single-Tesla-T4 worker at MAX or NCSA can provide. Non-GPU jobs must not request a GPU, so they remain schedulable on TACC and WASH. Per-job walltime must stay under `execution.max_walltime`. | HTCondor profiles on each transformation (§7.1); jobs land on the intended sites in the pool rather than sitting idle in the queue |
| **C5** | **No shared filesystem.** All data movement goes through CondorIO. Every job declares all of its inputs and outputs; no script may reach for a submit-host path at runtime. | `pegasus.data.configuration=condorio`; jobs succeed on workers with unrelated scratch filesystems |
| **C6** | **The container image is not staged per round.** Workers pull the training image once from the registry and reuse the local Docker cache; a multi-gigabyte image copied per round sub-workflow exhausts worker scratch. | `bypass_staging=True` on the `Container` definition (§14.11) |
| **C7** | **Deterministic client selection.** Client selection is seeded by the round number, so re-planning or rescuing a round selects the same clients. | Re-running `select_clients.py` for a given round yields an identical `selected_clients.json` |
| **C8** | **Per-job provenance.** Every job carries the `+FL_Algorithm`, `+FL_NumRounds`, and `+FL_JobType` ClassAds so per-round cost, resource usage, and communication volume can be recovered from the Pegasus stampede database after the fact. | Attributes present in the Condor submit files and queryable in the stampede DB |
| **C9** | **No silent synthetic-data fallback.** If the expected `train/<label>/*.png` structure is missing, training must fail visibly rather than substitute random tensors. Silent fallbacks produce plausible-looking metrics for a broken run. | No `torch.randn()` data path in `train_local.py` (§14.4) |
| **C10** | **Every declared output is written, even on failure.** A job that exits without producing its declared output files causes HTCondor to hold the job on stage-out and stalls the DAG. A job that cannot do useful work writes the empty declared output and then exits non-zero. | Failed jobs appear as failures in `pegasus-analyzer`, not as held jobs |
| **C11** | **Configuration-driven experiments.** All FL parameters (`K`, `T`, `E`, batch size, learning rate, `C`, algorithm, `fedprox_mu`, architecture) come from a single config YAML. Adding an experiment must not require editing the workflow generators. | `configs/exp_e*.yml` differ from `configs/default.yml` only in configuration, and `fl_main.py` is unchanged between experiments |

### 8.5 Non-Constraints

Degrees of freedom deliberately left to the implementation. Changes confined to this list do **not** invalidate the reviewed specification and do not require re-approval, which keeps review focused on the decisions that matter.

| ID | Non-constraint |
|---|---|
| **N1** | Names of intermediate files, as long as they are unique per dataset, round, and client (the `{dataset}_r{t}_*` convention is a convenience, not a requirement) |
| **N2** | Internal structure of the wrapper scripts: argument parsing, helper decomposition, module layout, and how much code `train_local.py`, `evaluate.py`, and `centralized_baseline.py` share |
| **N3** | Archive format and granularity of data shards (one `tar.gz` per client is used, but a directory or another archive format is acceptable under CondorIO) |
| **N4** | Which pre-trained checkpoint source is used for the backbone, and the internal details of the model definition, provided the architecture is selectable from config |
| **N5** | Whether aggregation calls Flower's `Strategy` API or computes the weighted average directly, provided FedAvg and FedProx semantics are preserved |
| **N6** | Plot styling, color choices, and file formats beyond the required PNG plus PDF pair, and whether the report is Markdown or LaTeX |
| **N7** | Job-to-site mapping beyond the GPU / non-GPU distinction in C4; all remaining placement is left to HTCondor |
| **N8** | The random number generator used for client selection, provided it is seeded by round (C7) |
| **N9** | How raw data is obtained (live download through `tcia_utils` and HuggingFace, or a pre-staged archive through `raw_data_path`); both paths must exist but their internals are free |
| **N10** | DAGMan tuning values: retry counts, throttles, and `max_running` for ensemble sweeps |
| **N11** | Absolute values of the FL accuracy and F1 results (see §8.6.5) |

### 8.6 Expected Outcomes

Acceptance criteria for the generated code. Together with §8.4 these are what the user reviews before implementation, and what the debugging skill treats as the definition of "working" when it diagnoses a failure.

#### 8.6.1 Artifacts each stage must produce

| Stage | Required artifacts (per enabled dataset) | Acceptance check |
|---|---|---|
| Data preparation | `{dataset}_client_{i}_data.tar.gz` for `i` in `0..K-1`, `{dataset}_test_data.tar.gz`, `{dataset}_initial_model.pt`, `{dataset}_data_stats.json` | Shard count equals `K`; each shard contains `train/<label>/*.png` and `test/<label>/*.png`; stats report a non-uniform class balance across clients |
| FL round `t` | `{dataset}_r{t}_selected_clients.json`, `{dataset}_r{t}_local_model_c{i}.pt` and `{dataset}_r{t}_local_metrics_c{i}.json` for each selected client, `{dataset}_global_model_r{t}.pt`, `{dataset}_r{t}_round_metrics.json` | One round-metrics file per round, `r0` through `r{T-1}`, each containing accuracy and macro F1 on the held-out test set |
| Centralized baseline | `{dataset}_centralized_metrics.json` | Present for every enabled dataset, and reports accuracy above the majority-class rate |
| Cross-dataset evaluation | `cross_dataset_metrics.json` | Contains both directions (TCIA model on NIH data and the reverse) |
| Visualization | `figures/fig1_convergence.{png,pdf}`, `figures/fig3_data_distribution.{png,pdf}`, `figures/fig5_cross_dataset.{png,pdf}`, `figures.tar.gz` | All three figures render with data from the current run, not placeholders |
| Report | `experiment_report.md` | References every metrics file produced by the run |

The per-round timeline is materialized as the set of `{dataset}_r{t}_round_metrics.json` files rather than a single aggregated history file, and per-client evaluation is folded into `validate_global` and the per-client local metrics. Either shape satisfies §4.4.

#### 8.6.2 Smoke-test gate

`configs/default.yml` (both datasets, `K=2`, `T=2`, capped downloads) is the gate that must pass before any full-scale run:

- The workflow plans into 5 sub-workflows (2 datasets x 2 rounds, plus the top level).
- It completes end to end on the HTCondor pool without manual intervention.
- It stages out 8 metrics files (per dataset: 1 stats, 2 round metrics, 1 centralized), plus `cross_dataset_metrics.json`, the 6 figure files, `figures.tar.gz`, and `experiment_report.md`.
- Round 1 consumes the global model produced by round 0, confirming C2 on real infrastructure rather than by inspection.

#### 8.6.3 Full-scale gate

The E1 configuration (`K=10`, `T=50`, both datasets) must plan into 101 sub-workflows (50 rounds x 2 datasets, plus the top level) and roughly 2,400 jobs once Pegasus-generated staging and cleanup jobs are included, and must run to completion on the distributed pool. Job-level retries are permitted by `dagman.retry=3` but should not be needed for infrastructure-stable runs; a DAG-level rescue is acceptable when the debugging skill has applied a fix and resubmitted.

#### 8.6.4 Metrics that must appear in the outputs

Accuracy and macro F1 (global and per client), per-class metrics, a per-round timeline covering all `T` rounds, communication volume derived from model-checkpoint sizes and round-trip counts, workflow makespan from `pegasus-statistics`, and per-job walltime and resource usage from the stampede database and the `resource_monitor.py` samples.

#### 8.6.5 What is explicitly not an acceptance criterion

Federated model quality is not a criterion. The purpose of the experiments is to demonstrate that the generated workflow executes correctly at scale and produces scientifically interpretable output, so a scientifically plausible negative result is a pass, not a defect. Concretely:

- A large federated-versus-centralized accuracy gap is an expected consequence of limited data per client under a natural, non-IID partition.
- Collapse to majority-class prediction after the first aggregation round, as observed in E6 even with backbone freezing, Adam, class-weighted loss, augmentation, gradient clipping, and cosine scheduling, is reported as a finding about weight averaging under small per-client datasets. It does not fail the workflow.
- What *does* fail acceptance: missing or empty artifacts, metrics that are identical across rounds because the global model never propagated (a C2 violation), or metrics computed on synthetic rather than real data (a C9 violation).

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
│   ├── exp_test_quick.yml             # Short pipeline check (K=5, T=5) on pre-staged data
│   ├── exp_full.yml                   # Full experiment config (K=5, T=20, E=3, C=0.6)
│   ├── exp_e1_baseline.yml            # E1: Central vs. FL accuracy gap (K=10, T=50, E=2)
│   ├── exp_e2_algorithm.yml           # E2: FedAvg vs. FedProx (mu=0.01)
│   ├── exp_e3_scalability.yml         # E3: Client count scaling (K=5)
│   ├── exp_e4_communication.yml       # E4: Rounds vs. local epochs (T=10, E=5)
│   ├── exp_e5_cross_dataset.yml       # E5: Cross-modality generalization
│   └── exp_e6_improved.yml            # E6: Training optimizations (Adam, frozen backbone)
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
| E1-E4 experiments | Complete | `configs/exp_e1_*` through `exp_e4_*` |
| E5 cross-dataset study | Partial — data preparation, centralized baselines, and cross-dataset evaluation ran; the FL rounds were not carried through, so E5 is out of scope for the reported evaluation. Numbering is retained so config and output identifiers stay stable. | `configs/exp_e5_cross_dataset.yml` |
| E6 improved training (50 rounds, 10 clients) | Complete — two DAG-level rescues, both diagnosed and fixed by the debugging skill | `configs/exp_e6_improved.yml` |

Executed configurations, as reported in the evaluation: E1 (`K=10`, `T=50`, FedAvg baseline), E2 (FedProx), E3 (`K=5`), E4 (`T=10`, `E=5`), E6 (training optimizations). E1 and E2 each planned into 101 sub-workflows and over 2,400 jobs, satisfying the full-scale gate in §8.6.3.

## 14. Key Fixes Applied

1. **`pegasus.file.cleanup.scope`**: Changed from `deferred` to `inplace`. Deferred cleanup removed scratch files between sequential sub-workflow rounds, leaving empty input caches for downstream planners and violating C2. In-place cleanup reclaims space within a round while keeping the global model checkpoint available to the next round.

2. **Partition output staging**: Client data files changed from `stage_out=True` to `stage_out=False, register_replica=True`. Intermediate files stay in scratch for sub-workflows while being registered in the JDBCRC.

3. **Condor ClassAd attributes**: Added `+FL_Algorithm`, `+FL_NumRounds`, `+FL_JobType` to all job transformations via `add_profiles(Namespace.CONDOR, ...)` for tracking in the Pegasus stampede DB.

4. **Real data downloads**: Replaced placeholder/metadata-only downloads with actual image acquisition:
   - **TCIA**: `download_data.py` uses `tcia_utils.nbia.downloadSeries()` to fetch DICOM files, extracts middle slice per series, converts to 224×224 RGB PNG via pydicom. Collection names mapped to TCIA API names (some use spaces, e.g., `NSCLC Radiogenomics`).
   - **NIH**: Switched from deprecated `alkzar90/NIH-Chest-X-ray-dataset` (loading script no longer supported) to `BahaaEldin0/NIH-Chest-Xray-14` which provides streaming parquet access.
   - **Partition structure**: `partition_clients.py` now produces `train/<label>/*.png` and `test/<label>/*.png` directories, matching the expected structure in `train_local.py` and `evaluate.py`.
   - **This did NOT eliminate the synthetic fallback, and an earlier version of this entry wrongly claimed it did.** The fallback code remained in `train_local.py`, `evaluate.py` and `centralized_baseline.py`, and in `train_local.py` it was the path actually taken on every reported run, because `partition_clients.py` packs each shard with `arcname=f"client_{i}"` while `train_local.py` read `client_{i}_data`. All 3,700 training jobs across E1, E2, E3, E4 and E6 trained on `torch.randn` tensors. See fix 16 and `analysis/CRITICAL_synthetic_training_data.md`.

5. **Pre-staged data support**: Added `raw_data_path` config parameter. When set, `fl_main.py` skips the download job and registers the pre-staged tar.gz in the Pegasus replica catalog. This avoids re-downloading large datasets (46GB NIH, ~17GB TCIA) on every workflow run.

6. **Data-limiting parameters**: Added `max_series_per_collection` (TCIA) and `max_samples` (NIH) config params to control download size for development and testing.

7. **Stageable scripts**: Changed transformation catalog from container-installed (`is_stageable=False`, PFN inside container at `/opt/scripts/`) to submit-host-staged (`is_stageable=True`, PFN on local filesystem). This avoids rebuilding the Docker container when scripts change. Container uses `bypass_staging=True` to mount the host work directory.

8. **Sub-workflow catalog propagation**: Sub-workflows planned at runtime need their own catalog paths. Added per-sub-workflow `.properties` files with RC, TC, SC paths and `pegasus.transfer.worker.package=true` to stage the Pegasus worker package from the submit host.

9. **E6 model collapse fix**: Experiments E1-E5 collapsed to majority-class prediction. Suspected causes: full-model averaging destroying pretrained features, no class imbalance handling, SGD at lr=0.001 on small client shards, and no augmentation. E6 adds backbone freezing, Adam optimizer at lr=0.0001, class-weighted loss, data augmentation, gradient clipping, and a cosine LR scheduler, and it fixes the FedProx parameter passthrough. E6 did not resolve the collapse, which is reported as a finding rather than a defect (§8.6.5).

10. **FedProx parameter passthrough**: `fedprox_mu` was defined in config but never passed from `fl_round.py` to `train_local.py`, so the configured value was ignored and `train_local.py` fell back to its argparse default. Fixed by always passing `--fedprox-mu` to train jobs. **E2's results are unaffected**: its configured `fedprox_mu` of 0.01 happens to equal the argparse default, and the proximal term is applied whenever `--algorithm fedprox` is set, so E2 did run genuine FedProx at mu=0.01. This was verified against the submitted job arguments in `work/submit/.../run0007`, which carry `--algorithm fedprox` with no `--fedprox-mu`. The bug would have silently mattered for any config choosing a different mu.

11. **Worker disk exhaustion (container staging)**: Each round's sub-workflow staged its own copy of the multi-gigabyte training image, so worker scratch usage grew with the number of rounds and training jobs began failing mid-run. Fixed by setting `bypass_staging=True` on the `Container` definition, so workers pull the image once from the registry and reuse the local Docker cache. This is the constraint recorded as C6 in §8.4.

12. **Concurrent experiment collisions**: Running several experiment configs at once let them write into the same output directory and overwrite each other's results. Fixed by namespacing the output directory and the generated round sub-workflow YAML files by config name, so a sweep can run configurations concurrently.

13. **Class-weight / model-head mismatch**: Class weights were computed from the number of labels observed in a client shard, which can be smaller than the model's output dimension, producing a weight tensor whose length did not match the `CrossEntropyLoss` expectation and failing the training job. Fixed by inferring `num_classes` from the model head (and from the global checkpoint when loading) rather than from the observed labels.

14. **Undeclared helper dependencies**: `train_local.py` and `evaluate.py` import `resource_monitor.py`, and the round jobs use `evaluate.py` as a helper module. Because the scripts are staged from the submit host rather than baked into the container, these modules have to be declared as job inputs and registered in the per-round replica catalog; otherwise jobs fail at import time on the worker. This was applied and works in `fl_round.py` for the training and validation jobs.

    **The `cross_dataset_eval` half needed a second attempt.** A first attempt placed `rc.add_replica` calls inside `build_workflow`, where neither `rc` nor `scripts_dir` is in scope, so that version of `fl_main.py` raised `NameError` and could not generate a workflow at all. It was never used for any reported run. Corrected on 2026-07-31 by dropping those calls, since `build_replica_catalog` already registers both modules, and keeping only the input declarations on the job.

    **This is a deliberate deviation from the as-run code.** During the reported runs `cross_dataset_eval` did not declare `evaluate.py` as an input and resolved `from evaluate import ...` from the copy of `scripts/` that `containers/Dockerfile.fl-training` bakes into the image at `/opt/scripts/`. That worked in every reported run, but it is fragile, because every other script is staged from the submit host, so rebuilding the container can leave the imported helper at a different version from the staged one. Declaring the inputs removes that dependency. The change affects only which files are staged to one evaluation job and does not alter any FL computation or reported metric.

16. **Synthetic training data (2026-07-31, invalidates the reported FL metrics)**: `train_local.py` read its shard from `client_{id}_data` while `partition_clients.py` packs it as `client_{i}`, so `LocalImageDataset` discovered no images, took the synthetic fallback, and trained on `torch.randn(3, 224, 224)` with random labels. The job printed one line and exited 0. Counted from the Kickstart stdout of every training job: E1 1000/1000, E2 1000/1000, E3 500/500, E4 200/200, E6 1000/1000. Evaluation, the centralized baselines and the cross-dataset evaluation used real data, so the reported federated numbers are noise-trained models measured on the real test set, which is why they sit at majority-class level.

    Fixed by reading `client_{id}`, matching `centralized_baseline.py`, and by replacing the fallback in all three scripts with a `FileNotFoundError` that reports the directory contents, so constraint C9 is now satisfied literally and this class of error cannot recur silently. Verified in the `kthare10/fl-training:latest` container against a shard packed exactly as `partition_clients.py` packs it: the fixed path discovers the images and the old path raises.

    Found by the specification-conformance review described in `analysis/static_detectability.md`. Ten independent reviewers were given only the specification and the code, and all ten returned FAIL on C9. **The affected experiments must be re-run before any federated result is reported.**

15. **Configs reconciled with the submit host (2026-07-30)**: The configs in this repository had drifted from the versions that produced the reported results, which were recovered from the submit host and from the as-submitted round sub-workflow YAML files under `work/submit/.../run0006` through `run0010`. `exp_e1` through `exp_e5` differed in six fields: `local_epochs` (2, not 5), `learning_rate` (0.001, not 0.01), E3 `num_clients` (5, not 10), NIH `num_classes` (2, not 14), `max_series_per_collection` (2000), and `max_samples` (50000). The as-submitted arguments confirm the corrected values, for example E1 round 0 carries `--local-epochs 2 --batch-size 32 --learning-rate 0.001`. The stale NIH `num_classes: 14` was also corrected in `default.yml` and `exp_full.yml`, which produced no reported results. It is deliberately left at 14 in `exp_e6_improved.yml`, which is as run (see §8.1).
