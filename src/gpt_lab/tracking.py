"""Metric logging to TensorBoard, a JSON-lines file and the console, plus the run status file.

metrics.jsonl has one JSON object per line: {"step", "time", "<tag>": value, ...}, using the
same tags as TensorBoard. status.json describes the run's lifecycle (see train.py).
Both are meant for tools that follow a run, such as a UI, without parsing TensorBoard files.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any


class Tracker:
    def __init__(
        self,
        log_dir: str | Path,
        enabled: bool = True,
        metrics_path: str | Path | None = None,
        start_step: int = 0,
    ):
        self.writer = None
        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=str(log_dir))
            except ImportError:
                print("[tracking] tensorboard not installed; logging to console only")
        self._metrics = None
        if metrics_path is not None:
            metrics_path = Path(metrics_path)
            _drop_rows_after(metrics_path, start_step)
            self._metrics = open(metrics_path, "a", encoding="utf-8", buffering=1)
        self._start = time.time()

    def scalars(self, metrics: dict[str, float], step: int) -> None:
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)
        self._append(step, metrics)

    def text(self, tag: str, text: str, step: int) -> None:
        if self.writer is not None:
            # TensorBoard renders text as markdown; indent to keep it as a literal block.
            self.writer.add_text(tag, "    " + text.replace("\n", "\n    "), step)
        self._append(step, {tag: text})

    def print(self, message: str) -> None:
        elapsed = time.time() - self._start
        print(f"[{elapsed / 3600:6.2f}h] {message}", flush=True)

    def flush(self) -> None:
        if self.writer is not None:
            self.writer.flush()
        if self._metrics is not None:
            self._metrics.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
        if self._metrics is not None:
            self._metrics.close()
            self._metrics = None

    def _append(self, step: int, values: dict[str, Any]) -> None:
        if self._metrics is not None:
            row = {"step": step, "time": time.time()}
            row.update({k: _json_value(v) for k, v in values.items()})
            self._metrics.write(json.dumps(row) + "\n")


def _json_value(value: Any) -> Any:
    # NaN and infinity are not valid JSON (a browser's JSON.parse rejects them).
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _drop_rows_after(path: Path, step: int) -> None:
    """Remove rows logged after `step`, which a resumed run is about to log again.

    A fresh run (step 0) starts with an empty file. Unreadable lines, such as one cut off
    by a crash, are dropped too.
    """
    if not path.exists():
        return
    keep = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("step", 0) <= step:
            keep.append(line + "\n")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(keep), encoding="utf-8")
    os.replace(tmp, path)


def write_status(path: str | Path, status: dict[str, Any]) -> None:
    """Atomically replace the run's status file. Never raises: it must not stop training."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except OSError as e:
        print(f"[tracking] could not write {path}: {e!r}", flush=True)
