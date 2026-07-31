#!/usr/bin/env python3
"""Account for the wall-clock time of the FL round sub-workflows.

Two provenance sources are needed, and using only the first is what makes this
analysis easy to get wrong.

Kickstart writes a `*.out.000` record per job holding the *process* start time
and duration. Under CondorIO, however, HTCondor transfers a job's input and
output files outside that window, so Kickstart alone cannot see staging cost.

The Condor event log `fl_round_*.dag.nodes.log` supplies the rest: event 000
(submit), 001 (execute) and 005 (terminate) timestamps, plus the "Run Bytes
Received/Sent By Job" counters. Together they give, per job:

    queue      execute minus submit, time waiting for a slot to be matched
    slot       terminate minus execute, the whole occupancy of the slot
    exec       Kickstart process duration
    unexplained  slot minus exec, dominated by HTCondor file transfer

`unexplained` should only be attributed to transfer if jobs that receive no
large inputs show no comparable gap. This script prints those jobs as a control
so the attribution can be checked rather than assumed.

Two traps this script exists to avoid:

1. Do not divide summed job durations by wall time and call it efficiency. A
   round fans out to K client training jobs that may run concurrently, so the
   ratio can exceed one and measures nothing. Use the interval union.
2. Do not sum per-job queue times across a round. The waits overlap, so the sum
   readily exceeds the round's wall time.

Usage:
    python3 round_overhead.py <run-dir> [--jobs] [--gaps]

where <run-dir> is a Pegasus run directory such as
    work/submit/<user>/pegasus/fl_main/run0006

Stdlib only, so it runs on the submit host without installing anything.

Caveat: `*-0.dag.metrics` describes the most recent DAGMan invocation, so for a
workflow that needed a DAG-level rescue its top-level wall time covers only the
final attempt. Per-round figures are unaffected when each round completed once.
"""

import json
import os
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime

ROUND_RE = re.compile(r"fl_round_(tcia|nih)_r(\d+)")
KS_DUR = re.compile(r"^\s+duration:\s*([0-9.]+)", re.M)
KS_START = re.compile(r"^\s+start:\s*(\S+)", re.M)
KS_TX = re.compile(r"^\s+transformation:\s*\"?([^\"\n]+)\"?", re.M)
EV = re.compile(r"^(\d{3}) \((\d+)\.\d+\.\d+\) (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
EV_NODE = re.compile(r"^\s+DAG Node: (\S+)")
EV_RECV = re.compile(r"^\s*(\d+)\s+-\s+Run Bytes Received By Job")
EV_SENT = re.compile(r"^\s*(\d+)\s+-\s+Run Bytes Sent By Job")

# Jobs that receive no large inputs, used as the control for attributing the
# slot-minus-exec gap to file transfer.
CONTROL = ("create_dir", "register_", "stage_")


def round_key(text):
    m = ROUND_RE.search(text)
    return f"{m.group(1)}_r{int(m.group(2))}" if m else None


def transformation_of(node):
    name = re.sub(r"_ID\d+$", "", node)
    return re.sub(r"^(create_dir|stage_|register_|clean_up).*", r"\1", name)


def busy_time(intervals):
    """Measure of the union of (start, end) intervals."""
    intervals = sorted(intervals)
    total, cur_start, cur_end = 0.0, *intervals[0]
    for start, end in intervals[1:]:
        if start > cur_end:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    return total + cur_end - cur_start


def kickstart(path):
    """First invocation record in a Kickstart file as (tx, start_epoch, duration)."""
    try:
        text = open(path, errors="replace").read()
    except OSError:
        return None
    for record in text.split("- invocation:")[1:]:
        dur, start, tx = KS_DUR.search(record), KS_START.search(record), KS_TX.search(record)
        if dur and start and tx:
            try:
                epoch = datetime.fromisoformat(start.group(1)).timestamp()
            except ValueError:
                return None
            return tx.group(1).strip(), epoch, float(dur.group(1))
    return None


def parse_events(log_path):
    """Per-job Condor events from a DAG nodes log."""
    jobs = defaultdict(dict)
    current = None
    for line in open(log_path, errors="replace"):
        event = EV.match(line)
        if event:
            current = event.group(2)
            stamp = datetime.strptime(event.group(3), "%Y-%m-%d %H:%M:%S").timestamp()
            field = {"000": "submit", "001": "execute", "005": "terminate"}.get(event.group(1))
            if field:
                jobs[current][field] = stamp
            continue
        if current is None:
            continue
        for pattern, field, cast in ((EV_NODE, "node", str), (EV_RECV, "recv", int),
                                     (EV_SENT, "sent", int)):
            found = pattern.match(line)
            if found:
                jobs[current].setdefault(field, cast(found.group(1)))
    return [j for j in jobs.values()
            if all(k in j for k in ("submit", "execute", "terminate", "node"))]


def collect(run_dir):
    rounds = {}
    seen_logs = set()
    for dirpath, _dirs, files in os.walk(run_dir):
        in_round = None
        for part in dirpath.split(os.sep):
            if part.startswith("fl_round_"):
                in_round = round_key(part)
                break

        for name in files:
            path = os.path.join(dirpath, name)
            if name.endswith(".dag.metrics") and round_key(name):
                try:
                    wall = json.load(open(path)).get("duration")
                except (ValueError, OSError):
                    continue
                rounds.setdefault(round_key(name), {})["wall"] = wall
            elif name.endswith(".dag.nodes.log") and in_round:
                real = os.path.realpath(path)
                if real in seen_logs:
                    continue
                seen_logs.add(real)
                entry = rounds.setdefault(in_round, {})
                entry.setdefault("jobs", [])
                for job in parse_events(path):
                    ks = kickstart(os.path.join(dirpath, "00", "00", job["node"] + ".out.000"))
                    job["tx"] = transformation_of(job["node"])
                    job["exec"] = ks[2] if ks else None
                    job["ks_start"] = ks[1] if ks else None
                    entry["jobs"].append(job)
    return {k: v for k, v in rounds.items() if v.get("wall") and v.get("jobs")}


def report(run_dir, show_jobs=False):
    rounds = collect(run_dir)
    if not rounds:
        print(f"{run_dir}: no round sub-workflow provenance found")
        return

    walls, busies, frames, execs, ins, outs = [], [], [], [], [], []
    by_tx = defaultdict(list)
    for data in rounds.values():
        jobs = [j for j in data["jobs"] if j.get("exec") is not None]
        if not jobs:
            continue
        intervals = [(j["ks_start"], j["ks_start"] + j["exec"]) for j in jobs
                     if j.get("ks_start")]
        walls.append(data["wall"])
        if intervals:
            span = max(e for _s, e in intervals) - min(s for s, _e in intervals)
            busies.append(busy_time(intervals))
            frames.append(data["wall"] - span)
        execs.append(sum(j["exec"] for j in jobs))
        ins.append(sum(j.get("recv") or 0 for j in jobs))
        outs.append(sum(j.get("sent") or 0 for j in jobs))
        for j in jobs:
            by_tx[j["tx"]].append(j)

    med = statistics.median
    print(f"\n{run_dir}")
    print(f"  round sub-workflows              : {len(rounds)}")
    print(f"  median round wall                : {med(walls):.1f} s")
    if busies:
        print(f"    a job's process running        : {med(busies):.1f} s "
              f"(union of intervals, {100 * med(busies) / med(walls):.1f}% of wall)")
        print(f"    DAGMan startup and teardown    : {med(frames):.1f} s")
    print(f"    summed process execution       : {med(execs):.1f} s")
    print(f"  median data moved per round      : {med(ins) / 1e6:.0f} MB in, "
          f"{med(outs) / 1e6:.0f} MB out")

    print(f"\n  {'transformation':16s} {'n':>5s} {'queue':>7s} {'slot':>7s} "
          f"{'exec':>7s} {'slot-exec':>10s} {'in_MB':>7s} {'out_MB':>7s}")
    for tx, jobs in sorted(by_tx.items(), key=lambda x: -med([j["slot"] if "slot" in j
                                                              else j["terminate"] - j["execute"]
                                                              for j in x[1]])):
        q = med([j["execute"] - j["submit"] for j in jobs])
        s = med([j["terminate"] - j["execute"] for j in jobs])
        e = med([j["exec"] for j in jobs])
        note = "  <- control" if tx.startswith(CONTROL) else ""
        print(f"  {tx:16s} {len(jobs):5d} {q:7.1f} {s:7.1f} {e:7.1f} {s - e:10.1f} "
              f"{med([j.get('recv') or 0 for j in jobs]) / 1e6:7.1f} "
              f"{med([j.get('sent') or 0 for j in jobs]) / 1e6:7.1f}{note}")
    print("\n  Attribute slot-exec to file transfer only if the control rows,")
    print("  which receive no large inputs, show no comparable gap.")


if __name__ == "__main__":
    targets = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not targets:
        sys.exit(__doc__)
    for target in targets:
        report(target, show_jobs="--jobs" in sys.argv)
