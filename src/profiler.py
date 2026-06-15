"""
Lightweight background resource profiler.

Logs CPU, RAM, and GPU stats to a CSV file at a configurable interval.
Uses line-buffered I/O so data survives unexpected kills.
"""
import csv
import os
import subprocess
import threading
from datetime import datetime

import psutil
import torch


def _gpu_stats():
    """Return GPU stats dict. Uses nvidia-smi for utilization, torch.cuda for memory."""
    stats = {}

    if torch.cuda.is_available():
        stats["gpu_mem_alloc_MB"] = torch.cuda.memory_allocated() / 1e6
        stats["gpu_mem_reserved_MB"] = torch.cuda.memory_reserved() / 1e6
        stats["gpu_mem_total_MB"] = torch.cuda.get_device_properties(0).total_memory / 1e6
    else:
        stats["gpu_mem_alloc_MB"] = 0
        stats["gpu_mem_reserved_MB"] = 0
        stats["gpu_mem_total_MB"] = 0

    # nvidia-smi for utilization % and temperature
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        ).strip().split("\n", maxsplit=1)[0]
        util, temp = out.split(",")
        stats["gpu_util_%"] = float(util.strip())
        stats["gpu_temp_C"] = float(temp.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, OSError):
        stats["gpu_util_%"] = -1
        stats["gpu_temp_C"] = -1

    return stats


def _system_stats():
    """Return CPU and RAM stats dict."""
    mem = psutil.virtual_memory()
    return {
        "cpu_%": psutil.cpu_percent(interval=None),
        "ram_used_MB": mem.used / 1e6,
        "ram_total_MB": mem.total / 1e6,
        "ram_%": mem.percent,
    }


class ResourceProfiler:
    """Background daemon that periodically writes resource stats to a CSV."""

    def __init__(self, out_dir, interval=30):
        self.interval = interval
        os.makedirs(out_dir, exist_ok=True)
        self.csv_path = os.path.join(out_dir, "profiling.csv")
        self._stop_event = threading.Event()
        self._thread = None
        self._file = None
        self._writer = None

    def start(self):
        """Start the profiler thread and open the CSV file."""
        fieldnames = [
            "timestamp",
            "cpu_%", "ram_used_MB", "ram_total_MB", "ram_%",
            "gpu_mem_alloc_MB", "gpu_mem_reserved_MB", "gpu_mem_total_MB",
            "gpu_util_%", "gpu_temp_C",
            "epoch", "note",
        ]
        self._file = open(self.csv_path, "w", buffering=1, newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self._writer.writeheader()
        self._epoch = -1
        self._note = ""

        # Warm up psutil so the first cpu_percent isn't 0
        psutil.cpu_percent(interval=None)

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(f"Profiler started → {self.csv_path}")

    def set_epoch(self, epoch):
        """Set the current epoch number to log."""
        self._epoch = epoch

    def set_note(self, note):
        """Set a note to be logged with the next sample."""
        self._note = note

    def _sample(self):
        """Collect stats and write a row to the CSV."""
        row = {"timestamp": datetime.now().isoformat()}
        row.update(_system_stats())
        row.update(_gpu_stats())
        row["epoch"] = self._epoch
        row["note"] = self._note
        self._note = ""
        self._writer.writerow(row)

    def _loop(self):
        """Background loop that samples at the specified interval."""
        while not self._stop_event.is_set():
            try:
                self._sample()
            except (OSError, ValueError, RuntimeError) as e:
                print(f"[profiler] error: {e}")
            self._stop_event.wait(self.interval)

    def stop(self):
        """Stop the profiler thread and close the CSV file."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        # Final sample
        try:
            self._sample()
        except (OSError, ValueError, RuntimeError):
            pass
        if self._file is not None:
            self._file.close()
        print(f"Profiler stopped. Data saved to {self.csv_path}")
