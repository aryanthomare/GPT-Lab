"""Metric logging to TensorBoard and the console."""

from __future__ import annotations

import time
from pathlib import Path


class Tracker:
    def __init__(self, log_dir: str | Path, enabled: bool = True):
        self.writer = None
        if enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=str(log_dir))
            except ImportError:
                print("[tracking] tensorboard not installed; logging to console only")
        self._start = time.time()

    def scalars(self, metrics: dict[str, float], step: int) -> None:
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)

    def text(self, tag: str, text: str, step: int) -> None:
        if self.writer is not None:
            # TensorBoard renders text as markdown; indent to keep it as a literal block.
            self.writer.add_text(tag, "    " + text.replace("\n", "\n    "), step)

    def print(self, message: str) -> None:
        elapsed = time.time() - self._start
        print(f"[{elapsed / 3600:6.2f}h] {message}", flush=True)

    def flush(self) -> None:
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
