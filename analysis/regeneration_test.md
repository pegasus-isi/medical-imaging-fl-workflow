# Regeneration test

The paper claims a workflow can be regenerated from its persisted specification.
This is the test of that claim. On 2026-07-31, `SPEC.md` was given to a fresh
Claude Code session with the `pegasus-ai` plugin, running with Opus in a
container with Pegasus 6.0.0-dev, and asked to build the workflow. No file from
the original implementation was available to it.

The specification handed over was byte-identical to the committed one
(md5 `3d01aceb34d3f42782c3a601c0271223`), so it included the constraints in
section 8.4, the non-constraints in section 8.5, the expected outcomes in
section 8.6, and the implementation status and key fixes in sections 13 and 14.
This therefore tests the strong claim, that the persisted living document
suffices, rather than whether a design-only specification would.

## Criteria, fixed before comparison

1. **Structural equivalence.** Same DAG shape for the same configuration.
2. **Constraint conformance.** Satisfies C1 through C11 of section 8.4.
3. **Plans and runs.** Concretizes to an executable DAG.
4. **Defect recurrence.** How many of the 9 defects in
   `static_detectability.md` reappear.

## 1. Structural equivalence: exact match

Both generators were run on the smoke-test configuration (two datasets, K=2,
T=2) with the same Pegasus API.

| | Original | Regenerated |
|---|---|---|
| Top-level jobs | 15 | 15 |
| of which sub-workflows | 4 | 4 |
| `download_data` | 2 | 2 |
| `partition_clients` | 2 | 2 |
| `compute_statistics` | 2 | 2 |
| `centralized_baseline` | 2 | 2 |
| `cross_dataset_eval`, `plot_results`, `generate_report` | 1 each | 1 each |
| Jobs per round | 5 | 5 |
| Round composition | select, 2 train, aggregate, evaluate | select, 2 train, aggregate, evaluate |
| Round dependencies | select to trains to aggregate to evaluate | select to trains to aggregate to evaluate |
| Round r1 consumes r0 global model | yes | yes |

The only differences are the names of generated round files
(`default_fl_round_tcia_r0.yml` against `fl_round_tcia_r0.yml`) and job
identifiers, both of which non-constraint N1 explicitly leaves free.

## 2. Constraint conformance

Satisfied and, in seven cases, cited by constraint ID in the generated source:
C1, C2, C3, C4, C6, C7 and C8 appear as comments next to the code that
implements them. C5 (CondorIO) and C9 (no synthetic fallback) hold. C11
(config-driven) holds, with all nine experiment configurations reproduced.

C4 is implemented in `fl_round.py` as per-job Condor profiles rather than on the
transformations in `fl_main.py`, which is an N2 difference. It also sets
`request_gpus = 0` explicitly on non-GPU jobs, which satisfies the second half
of C4 more directly than the original does.

## 3. Plans successfully

`pegasus-plan` concretized the regenerated workflow with no errors in 6.2
seconds, producing a 42 job executable DAG with four sub-workflow planning jobs.
Full execution was not attempted, so the section 8.6.2 smoke-test gate is
confirmed only as far as planning.

## 4. Defect recurrence: none

None of the 9 defects reappeared. Seven passages in the generated code cite the
originating fix by number, referencing fixes 2, 5, 10, 12, 13 and 14.

| Defect | Recurred | Evidence in regenerated code |
|---|---|---|
| 1, cleanup scope | no | `pegasus.file.cleanup.scope = inplace` in both properties files |
| 2, partition staging | no | `stage_out=False, register_replica=True` on client tars, annotated as fix 2 |
| 4, synthetic fallback | no | no random-tensor path in `train_local.py` |
| 8, catalog propagation | no | `subworkflow.properties` carries replica, transformation and site catalogs plus worker package |
| 10, FedProx passthrough | no | `--fedprox-mu` always passed, annotated as fix 10 |
| 11, container staging | no | `bypass_staging=True` on the container |
| 12, experiment collisions | no | scratch and output directories namespaced by experiment name, annotated as fix 12 |
| 13, class-weight mismatch | no | `num_classes` read from the checkpoint, annotated as fix 13 |
| 14, helper dependencies | no | `resource_monitor.py` and `evaluate.py` declared, annotated as fix 14 |

## An inversion worth recording

While running the comparison, the **original** `fl_main.py` as committed failed
with `NameError: name 'rc' is not defined`, because a block registering the
helper modules had been added inside `build_workflow` where neither `rc` nor
`scripts_dir` is in scope, duplicating registrations that
`build_replica_catalog` already performs. The regenerated version implemented the
same fix correctly.

So on the day of the test the specification-derived workflow ran and the
hand-maintained one did not. The original has since been repaired by dropping
the out-of-scope `rc.add_replica` calls and keeping the input declarations on the
job, a deliberate deviation from the as-run code recorded in SPEC.md section 14,
entry 14.

The same review round found a far more serious defect that the regenerated
workflow also avoided: every reported training job trained on synthetic noise.
See `CRITICAL_synthetic_training_data.md`.

## What this does not show

One regeneration by one model. It does not measure variance across runs, does not
test whether a design-only specification (sections 1 to 12) would suffice, and
does not establish that the regenerated workflow produces the same scientific
results, since it was not executed to completion. Structural and constraint
equivalence is not behavioural equivalence.
