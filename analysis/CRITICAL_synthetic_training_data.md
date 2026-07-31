# Critical: every reported federated training job trained on synthetic noise

Found 2026-07-31 by a blind specification-conformance review, then confirmed
directly from the run logs on the submit host.

## The finding

`partition_clients.py:274` packs each client shard with `arcname=f"client_{i}"`,
so the archive expands to `client_0/train/<label>/*.png`. But
`train_local.py:293` looks for its data under `Path(f"client_{args.client_id}_data")`,
that is `client_0_data/train/...`, which never exists.

`LocalImageDataset` therefore discovers no images, sets `self.synthetic = True`,
and `__getitem__` returns `torch.randn(3, 224, 224)` with a random label
(`train_local.py:80-95`). The job prints one line, "No real images found", and
exits 0.

`centralized_baseline.py:114` uses the matching `client_{i}` name, which is why
the centralized numbers are unaffected.

## Scope, counted from the Kickstart stdout of every training job

| Experiment | Training jobs on synthetic data |
|---|---|
| E1 | 1000 / 1000 |
| E2 | 1000 / 1000 |
| E3 | 500 / 500 |
| E4 | 200 / 200 |
| E6 | 1000 / 1000 |

Every federated training job in every reported experiment, 3,700 of 3,700.

By contrast, in E1:

| Job type | On synthetic data |
|---|---|
| `evaluate` (per-round validation) | 0 / 100 |
| `centralized_baseline` | 0 / 2 |
| `cross_dataset_eval` | 0 / 1 |

Evaluation and the centralized baselines used the real data. The test archive is
packed with `arcname="."` and expands correctly.

## What the reported numbers therefore mean

Every federated model was trained on random tensors and then evaluated against
the real held-out test set. That yields approximately majority-class accuracy,
which is what was observed and reported: 47.2 percent on TCIA and 46.1 percent on
NIH for E1. The centralized baselines trained on real data and reached 96.0 and
67.0 percent.

So the reported federated-versus-centralized gap is the gap between training on
noise and training on data. It is not a result about federated averaging, about
data heterogeneity, or about per-client sample size. Specifically these
statements are not supported:

- that the gap reflects limited effective data per client,
- that FedProx stagnates earlier than FedAvg,
- that fewer clients raise TCIA accuracy,
- that fewer rounds with more local epochs give the best F1,
- that E6's optimizations fail because of an interaction between weight averaging
  and small per-client datasets.

E6's NIH round 0 accuracy of 0.0032 and its macro F1 of 0.093 are consistent with
random-tensor training through a 14 output head.

## What is unaffected

The workflow and systems results do not depend on the contents of the training
tensors. Jobs were planned, staged, scheduled, executed, retried and recovered
exactly as recorded. Sub-workflow counts, job counts, makespans, the wall-clock
decomposition in `README.md`, the staging measurements, the debugging narrative
and the regeneration test all stand. The centralized baselines and the
cross-dataset evaluation used real data.

## How it was found, and why it was not found earlier

Five independent reviewers, each given only the specification and a copy of the
code, all returned FAIL on C9. Four of the five independently identified the
`arcname` against `client_{id}_data` mismatch as the reason the forbidden path was
not merely present but always taken.

It survived months of running the workflow because nothing failed. The jobs
exited 0, wrote every declared output, produced well-formed metrics, and the
resulting accuracies were low but plausible for federated learning on
heterogeneous medical images. This is the failure mode SPEC.md section 8.6.5
anticipated in writing, "metrics computed on synthetic rather than real data (a
C9 violation)", and the anticipation did not help because nobody checked.

Two spec claims are wrong as a result. Section 14 entry 4 says the partition
restructuring "eliminates fallback to synthetic `torch.randn()` data": the
fallback was never removed, and it was the live path. `static_detectability.md`
counts defect 4 among those fixed, which is incorrect.

## Fix

One line. Either pack with `arcname=f"client_{i}_data"` or look under
`Path(f"client_{args.client_id}")`. The fallback itself should also be replaced
by a hard failure, which is what C9 requires, so that this class of error cannot
recur silently. Re-running the affected experiments is the only way to obtain
real federated results.
