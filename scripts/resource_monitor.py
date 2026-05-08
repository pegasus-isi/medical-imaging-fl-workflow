#!/usr/bin/env python3
"""Lightweight resource monitoring for FL training jobs.

Collects GPU utilization, GPU memory, CPU/RAM usage, and wall-clock timing.
Designed to run as a background thread during training and produce a summary
dict suitable for inclusion in metrics JSON files.

Usage:
    from resource_monitor import ResourceMonitor

    monitor = ResourceMonitor(interval=5.0)
    monitor.start()
    # ... do training ...
    stats = monitor.stop()  # returns dict with resource metrics
"""

import os
import subprocess
import threading
import time
from dataclasses import dataclass, field


@dataclass
class ResourceMonitor:
    """Background thread that samples GPU and system resource metrics."""

    interval: float = 5.0  # sampling interval in seconds
    _samples: list = field(default_factory=list, repr=False)
    _thread: threading.Thread = field(default=None, repr=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _start_time: float = field(default=0.0, repr=False)
    _end_time: float = field(default=0.0, repr=False)

    def start(self):
        """Begin background sampling."""
        self._start_time = time.time()
        self._stop_event.clear()
        self._samples = []
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        """Stop sampling and return aggregated statistics."""
        self._stop_event.set()
        self._end_time = time.time()
        if self._thread is not None:
            self._thread.join(timeout=10)
        return self._aggregate()

    def _sample_loop(self):
        """Periodically sample GPU and system metrics."""
        while not self._stop_event.is_set():
            sample = self._take_sample()
            if sample:
                self._samples.append(sample)
            self._stop_event.wait(timeout=self.interval)

    def _take_sample(self) -> dict:
        """Take a single resource measurement."""
        sample = {"timestamp": time.time()}

        # GPU metrics via nvidia-smi
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                gpus = []
                for line in result.stdout.strip().split("\n"):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 6:
                        gpus.append({
                            "index": int(parts[0]),
                            "utilization_pct": float(parts[1]),
                            "memory_used_mb": float(parts[2]),
                            "memory_total_mb": float(parts[3]),
                            "temperature_c": float(parts[4]),
                            "power_draw_w": float(parts[5]) if parts[5] != "[N/A]" else 0.0,
                        })
                sample["gpus"] = gpus
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # CPU and memory via psutil or /proc
        try:
            import psutil
            sample["cpu_percent"] = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            sample["ram_used_mb"] = mem.used / (1024 * 1024)
            sample["ram_total_mb"] = mem.total / (1024 * 1024)
            sample["ram_percent"] = mem.percent
        except ImportError:
            try:
                with open("/proc/meminfo") as f:
                    meminfo = {}
                    for line in f:
                        parts = line.split()
                        meminfo[parts[0].rstrip(":")] = int(parts[1])
                sample["ram_total_mb"] = meminfo.get("MemTotal", 0) / 1024
                sample["ram_used_mb"] = (
                    meminfo.get("MemTotal", 0)
                    - meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
                ) / 1024
            except (FileNotFoundError, KeyError):
                pass

        return sample

    def _aggregate(self) -> dict:
        """Aggregate samples into summary statistics."""
        wall_time = self._end_time - self._start_time
        stats = {
            "wall_time_seconds": round(wall_time, 2),
            "num_samples": len(self._samples),
            "sample_interval_seconds": self.interval,
        }

        if not self._samples:
            return stats

        # GPU aggregation
        gpu_utils = []
        gpu_mem_used = []
        gpu_mem_total = []
        gpu_temps = []
        gpu_power = []

        for s in self._samples:
            for gpu in s.get("gpus", []):
                gpu_utils.append(gpu["utilization_pct"])
                gpu_mem_used.append(gpu["memory_used_mb"])
                gpu_mem_total.append(gpu["memory_total_mb"])
                gpu_temps.append(gpu["temperature_c"])
                gpu_power.append(gpu["power_draw_w"])

        if gpu_utils:
            stats["gpu"] = {
                "utilization_pct_mean": round(sum(gpu_utils) / len(gpu_utils), 1),
                "utilization_pct_max": round(max(gpu_utils), 1),
                "memory_used_mb_mean": round(sum(gpu_mem_used) / len(gpu_mem_used), 0),
                "memory_used_mb_max": round(max(gpu_mem_used), 0),
                "memory_total_mb": round(gpu_mem_total[0], 0) if gpu_mem_total else 0,
                "temperature_c_max": round(max(gpu_temps), 1),
                "power_draw_w_mean": round(sum(gpu_power) / len(gpu_power), 1),
            }

        # RAM aggregation
        ram_used = [s.get("ram_used_mb", 0) for s in self._samples if "ram_used_mb" in s]
        if ram_used:
            stats["ram"] = {
                "used_mb_mean": round(sum(ram_used) / len(ram_used), 0),
                "used_mb_max": round(max(ram_used), 0),
                "total_mb": round(self._samples[0].get("ram_total_mb", 0), 0),
            }

        # CPU aggregation
        cpu_pcts = [s.get("cpu_percent", 0) for s in self._samples if "cpu_percent" in s]
        if cpu_pcts:
            stats["cpu"] = {
                "percent_mean": round(sum(cpu_pcts) / len(cpu_pcts), 1),
                "percent_max": round(max(cpu_pcts), 1),
            }

        return stats
