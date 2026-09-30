"""Saving, loading and pruning training checkpoints."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import torch

from gpt_lab.config import Config, ModelConfig
from gpt_lab.model import Transformer

_CKPT_RE = re.compile(r"step_(\d+)\.pt$")


def checkpoint_path(ckpt_dir: str | Path, step: int) -> Path:
    return Path(ckpt_dir) / f"step_{step:06d}.pt"


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    config: Config,
    loader_state: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write atomically (temporary file, then rename) so a crash never leaves a broken file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "config": config.to_dict(),
        "loader": loader_state,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "extra": extra or {},
    }
    tmp = path.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)


def list_checkpoints(ckpt_dir: str | Path) -> list[tuple[int, Path]]:
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return []
    found = []
    for p in ckpt_dir.iterdir():
        m = _CKPT_RE.search(p.name)
        if m:
            found.append((int(m.group(1)), p))
    return sorted(found)


def latest_checkpoint(ckpt_dir: str | Path) -> Path | None:
    ckpts = list_checkpoints(ckpt_dir)
    return ckpts[-1][1] if ckpts else None


def prune_checkpoints(ckpt_dir: str | Path, keep_last: int, keep_every: int = 0) -> None:
    """Delete old checkpoints, keeping the newest `keep_last` and multiples of `keep_every`."""
    ckpts = list_checkpoints(ckpt_dir)
    if keep_last <= 0:
        return
    for step, path in ckpts[:-keep_last]:
        if keep_every and step % keep_every == 0:
            continue
        path.unlink(missing_ok=True)


def resolve_checkpoint(path: str | Path) -> Path:
    """Accept a checkpoint file, a checkpoints/ directory, or a run directory."""
    path = Path(path)
    if path.is_file():
        return path
    for candidate in (path, path / "checkpoints"):
        latest = latest_checkpoint(candidate)
        if latest is not None:
            return latest
    raise FileNotFoundError(f"No checkpoint found at '{path}'")


def load_model_from_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[Transformer, Config, dict[str, Any]]:
    ckpt = load_checkpoint(resolve_checkpoint(path), map_location="cpu")
    cfg = Config.from_dict(ckpt["config"])
    model = Transformer(ModelConfig(**ckpt["config"]["model"]))
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg, ckpt
