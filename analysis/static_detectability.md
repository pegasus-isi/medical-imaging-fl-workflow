# Could a pre-submission check have caught it?

Every entry in SPEC.md section 14 is a change made during development. This file
classifies each one, first by whether it was actually a defect, and then by
whether it was detectable by static inspection of the workflow generator and
wrapper scripts against the specification's constraints (section 8.4), before
any job was submitted.

The classification is a judgement, not a measurement, so the reasoning for each
entry is given in full and can be disputed. "Detectable" means the evidence was
present in the source, not that any particular tool would have surfaced it.
Whether a checker actually finds them is a separate question.

## Summary

| Category | Count |
|---|---|
| Genuine defects | 9 |
| Of those, fully detectable by static inspection | 7 |
| Of those, visible as an anti-pattern with a data or scale dependent trigger | 2 |
| Not defects (features, instrumentation, design changes, findings) | 5 |
| Provenance drift, detectable only by comparing artifact against run records | 1 |

Two findings matter more than the headline count.

**Three of the nine defects were silent.** Fixes 4, 10, and 12 produced no job
failure. The workflow ran to completion and gave wrong or overwritten results.
A runtime debugging loop cannot find these by construction, because there is no
failure to diagnose. All three were statically detectable.

**The two that static inspection would only partly catch are both threshold
bugs.** In fixes 11 and 13 the anti-pattern is plainly visible in the source but
whether it fires depends on worker disk capacity and on the class distribution of
a particular shard. A checklist can flag the pattern. It cannot predict the
trigger.

## Per-entry classification

| # | Entry | Defect? | Silent? | Statically detectable? | Evidence available before submission |
|---|---|---|---|---|---|
| 1 | Cleanup scope `deferred` | yes | no | **yes** | The property value sits next to `SubWorkflow` jobs that consume prior rounds' outputs. Constraint C2 states the global model must survive between rounds, so the combination is checkable. |
| 2 | Partition output staging | yes | no | **yes** | `stage_out=True` on a file consumed by a later sub-workflow, with no `register_replica`. This is an existing `/pegasus-review` checklist item. |
| 3 | Condor ClassAds added | no | n/a | n/a | Instrumentation for provenance, not a fix. |
| 4 | Placeholder data downloads | yes | **yes** | **yes** | A `torch.randn()` fallback path in `train_local.py` is visible in the source and is exactly what constraint C9 forbids. **Correction (2026-07-31): this was never actually fixed.** The fallback remained and was the live path on every reported run. See `CRITICAL_synthetic_training_data.md`. |
| 5 | Pre-staged data support | no | n/a | n/a | New `raw_data_path` capability. |
| 6 | Data-limiting parameters | no | n/a | n/a | New config knobs for development. |
| 7 | Stageable scripts | no | n/a | n/a | Design change to avoid container rebuilds. |
| 8 | Sub-workflow catalog propagation | yes | no | **yes** | `SubWorkflow` jobs generated without a `--conf` carrying replica, transformation, and site catalog paths. Visible entirely in the generator. |
| 9 | E6 model collapse | no | n/a | n/a | A scientific finding, which section 8.6.5 explicitly excludes from acceptance criteria. |
| 10 | FedProx passthrough | yes | **yes** | **yes** | `fedprox_mu` is read from config and never appears in the argument list. A dataflow check on config keys catches it. |
| 11 | Worker disk exhaustion | yes | no | **partial** | The anti-pattern, a container staged per sub-workflow, is visible. Predicting exhaustion needs image size and worker scratch capacity, which the source does not contain. |
| 12 | Concurrent experiment collisions | yes | **yes** | **yes** | A fixed output directory not parameterized by config name, against a specification (section 4.6) that describes running configurations concurrently under the Ensemble Manager. Needs the spec, not just the code. |
| 13 | Class-weight and model-head mismatch | yes | no | **partial** | The length mismatch between weights derived from observed labels and the model head is visible by inspection, but it only fires for a shard missing a class. |
| 14 | Undeclared helper dependencies | yes | no | **yes** | Imports of `resource_monitor.py` and `evaluate.py` with no matching `add_inputs` or replica registration. The cleanest static signal of the set. |
| 15 | Configs drifted from the run records | not a runtime defect | **yes** | no, but detectable | Nothing in the workflow source is wrong. Detecting it requires comparing the committed configs against the as-submitted job arguments, which is a different kind of check and the one that found it. |

## What this does and does not support

It supports the claim that a static pass over the generated code, checked against
the specification's constraints, could in principle have caught most of the
defects that were instead found by running the workflow, including all three that
never announced themselves as failures.

It does not measure any tool. It says nothing about false positives, which is the
cost side of running such a check, and nothing about whether `/pegasus-review` or
any other checker actually flags these cases. Establishing that requires running
a checker against seeded versions of these defects and reporting both what it
catches and what it invents. That experiment has not been run.
