"""GPU readings over time, for the live view's GPU charts.

A background thread asks nvidia-smi for one reading every few seconds and keeps the last
hour in a ring buffer. One short-lived process per reading (rather than nvidia-smi's own
loop mode) means nothing can outlive the server if it is killed. Without nvidia-smi (no
NVIDIA GPU, or CI) the buffer simply stays empty.
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections import deque
from typing import Any

QUERY = "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
COMMAND = ["nvidia-smi", "--id=0", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"]


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None  # nvidia-smi prints "[N/A]" for fields a GPU doesn't report


def parse_line(line: str, t: float) -> dict[str, Any] | None:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 6:
        return None
    name, util, used, total, temp, power = parts
    return {
        "t": t,
        "name": name,
        "util": _number(util),
        "mem_used_mb": _number(used),
        "mem_total_mb": _number(total),
        "temp_c": _number(temp),
        "power_w": _number(power),
    }


def read_gpu() -> dict[str, Any] | None:
    """One reading of GPU 0, or None if nvidia-smi isn't available or fails."""
    try:
        out = subprocess.run(COMMAND, capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_line(out.strip(), time.time())


class GpuSampler:
    def __init__(self, interval: int = 2, keep_seconds: int = 3600):
        self.interval = interval
        self._samples: deque[dict[str, Any]] = deque(maxlen=keep_seconds // interval)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="gpu-sampler", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def since(self, t: float) -> list[dict[str, Any]]:
        with self._lock:
            return [s for s in self._samples if s["t"] > t]

    def latest(self, max_age: float = 10.0) -> dict[str, Any] | None:
        with self._lock:
            if self._samples and time.time() - self._samples[-1]["t"] <= max_age:
                return self._samples[-1]
        return None

    def _run(self) -> None:
        failures = 0
        while not self._stop.is_set():
            started = time.time()
            sample = read_gpu()
            if sample:
                failures = 0
                with self._lock:
                    self._samples.append(sample)
            else:
                failures += 1
                if failures >= 3 and not self._samples:
                    return  # no usable GPU here; stop asking
            self._stop.wait(max(0.0, self.interval - (time.time() - started)))
