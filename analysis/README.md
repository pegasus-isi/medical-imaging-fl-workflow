# Where the makespan goes

Each FL round is planned as its own Pegasus sub-workflow. This directory
accounts for a round's wall-clock time using only the provenance the reported
runs already left behind, so no re-execution is needed.

The headline result is that these workflows are bound by **data staging**, not by
computation and not by sub-workflow orchestration.

## Reproducing

Run on the submit host, against a run directory:

```sh
python3 analysis/round_overhead.py work/submit/<user>/pegasus/fl_main/run0006
```

Every number below is printed by that script. Only the standard library is used.

## Why two provenance sources are needed

Kickstart `*.out.000` records give each job's **process** start time and
duration. Under CondorIO, HTCondor transfers a job's inputs and outputs
*outside* that window, so Kickstart alone cannot see staging cost and will make
a staging-bound workflow look mysteriously idle.

The Condor event log `fl_round_*.dag.nodes.log` supplies the rest: event 000
(submit), 001 (execute), 005 (terminate), and the "Run Bytes Received/Sent By
Job" counters. Per job that yields:

| Quantity | Definition |
|---|---|
| `queue` | execute minus submit, waiting for a slot to be matched |
| `slot` | terminate minus execute, whole occupancy of the slot |
| `exec` | Kickstart process duration |
| `slot - exec` | slot time not spent running the process |

## Measured results, E1

Medians across 100 rounds. Round wall time is 1179.7 s, of which a process is
running for 115.8 s (union of execution intervals, 9.8 percent of wall), DAGMan
startup and teardown take 13.2 s, and summed process execution is 130.1 s.
Data moved per round is 2194 MB in and 493 MB out.

| Transformation | n | queue (s) | slot (s) | exec (s) | slot-exec (s) | in (MB) | out (MB) |
|---|---|---|---|---|---|---|---|
| `aggregate` | 100 | 19.0 | 159.0 | 2.5 | 156.5 | 466.2 | 44.8 |
| `select_clients` | 100 | 19.5 | 147.5 | 0.0 | 147.5 | 18.3 | 0.0 |
| `evaluate` | 100 | 22.5 | 139.0 | 13.6 | 125.4 | 265.7 | 0.0 |
| `train_local` | 1000 | 227.0 | 132.0 | 7.8 | 124.2 | 140.8 | 44.8 |
| `stage_` (control) | 400 | 0.0 | 4.0 | 4.2 | -0.2 | 0.0 | 0.0 |
| `create_dir` (control) | 100 | 4.0 | 2.0 | 2.2 | -0.2 | 0.0 | 0.0 |
| `register_` (control) | 400 | 0.0 | 2.0 | 2.0 | -0.0 | 0.0 | 0.0 |

At the whole-workflow level the sequential chain of round sub-workflows accounts
for 17.02 of E1's 19.99 hour makespan, 13.66 of E3's 16.62 hours, and 3.72 of
E4's 4.81 hours.

## What the numbers support

**Staging dominates.** A client training job executes for 7.8 seconds and spends
a further 124 seconds of its slot outside process execution while HTCondor moves
141 MB in and 45 MB out. The control rows are what license attributing that gap
to transfer: jobs receiving no large inputs show a gap of essentially zero, so
the gap is not generic slot overhead.

**Orchestration is not the bottleneck.** DAGMan startup and teardown cost 13.2
seconds of a 1180 second round, so sub-workflow machinery is a small fraction of
the cost, in contrast to what the round-level idle time might first suggest.

**Slot contention is real but secondary.** A training job waits a median 227
seconds from submit to execute, which is expected because a round of E1 wants
ten concurrent GPU jobs per dataset branch from a pool of nine.

**The redundancy is structural.** Each round moves about 2.7 GB, so a 50 round
run moves roughly 270 GB, largely because the same client shards and the global
model are re-staged for every round.

## Two ways to get this wrong

1. **Do not divide summed job durations by wall time and call it efficiency.** A
   round fans out to K client training jobs that may run concurrently, so that
   ratio can exceed one and measures nothing. Use the interval union for
   occupancy.
2. **Do not sum per-job queue times across a round.** The waits overlap, so the
   sum readily exceeds the round's wall time.

An earlier version of this analysis made the first mistake and, because
Kickstart cannot see file transfer, concluded that scheduler dispatch latency
dominated. Adding the Condor event log showed the gap was staging.

## What this does not show

These measurements do not compare against a flat DAG, a different data
configuration such as a shared filesystem or `bypass_staging` on the client
shards, a different scheduler configuration, or a larger per-client data volume.
They therefore do not establish that any alternative would be faster. They
establish where the time goes in the configuration that was run. Confirming that
caching client shards on workers would help requires an experiment that has not
been run.
