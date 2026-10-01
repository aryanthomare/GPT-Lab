"""What's in data/: token shards per dataset, free disk space, and the download cache."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path
from typing import Any

from gpt_lab.ui.configs import shard_tokens

# The datasets the UI knows how to prepare, keyed by the name the API uses.
PREPARABLE = {
    "tiny": {"title": "Tiny Shakespeare", "dir": "data/tiny"},
    "fineweb": {"title": "FineWeb-Edu 10B", "dir": "data/fineweb_edu_10B"},
}
HF_CACHE = Path.home() / ".cache" / "huggingface"

_size_cache: dict[str, tuple[float, int]] = {}


def dir_bytes(path: Path, max_age: float = 60.0) -> int:
    """Total size of the files under `path`, recomputed at most once a minute."""
    cached = _size_cache.get(str(path))
    if cached and time.time() - cached[0] < max_age:
        return cached[1]
    total = 0
    for dirpath, _, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_size
            except OSError:
                pass
    _size_cache[str(path)] = (time.time(), total)
    return total


def describe(path: Path) -> dict[str, Any]:
    """Shard counts, tokens and bytes of one dataset directory."""
    out = {"train_shards": 0, "val_shards": 0, "train_tokens": 0, "val_tokens": 0, "bytes": 0}
    for f in sorted(path.glob("*.npy")):
        split = "val" if f.name.startswith("val_") else "train"
        out[f"{split}_shards"] += 1
        out[f"{split}_tokens"] += shard_tokens(f)
        out["bytes"] += f.stat().st_size
    return out


def summary(root: Path) -> dict[str, Any]:
    data = root / "data"
    known_dirs = {v["dir"] for v in PREPARABLE.values()}
    datasets = []
    for key, info in PREPARABLE.items():
        path = root / info["dir"]
        datasets.append({"key": key, **info, "present": path.is_dir(), **describe(path)})
    if data.is_dir():
        for path in sorted(p for p in data.iterdir() if p.is_dir()):
            rel = f"data/{path.name}"
            if rel not in known_dirs and any(path.glob("*.npy")):
                datasets.append(
                    {"key": None, "title": path.name, "dir": rel, "present": True, **describe(path)}
                )
    disk = shutil.disk_usage(root)
    return {
        "datasets": datasets,
        "disk": {"free": disk.free, "total": disk.total},
        "hf_cache": {
            "path": str(HF_CACHE),
            "bytes": dir_bytes(HF_CACHE) if HF_CACHE.exists() else 0,
        },
    }
