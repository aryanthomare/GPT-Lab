"""Config presets, the new-run form's field list, and checks on a run before it starts."""

from __future__ import annotations

import dataclasses
import functools
import glob
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gpt_lab.config import Config, DataConfig, ModelConfig, TrainConfig, load_config
from gpt_lab.model import Transformer
from gpt_lab.ui.runs import RESERVED, RUN_NAME

CHOICES = {
    "train.device": ["auto", "cuda", "cpu"],
    "train.dtype": ["bfloat16", "float32"],
    "train.schedule": ["wsd", "cosine"],
}
SET_BY_FORM = {"train.run_name", "train.out_dir"}  # the form sets these itself
SECTIONS = (("model", ModelConfig), ("data", DataConfig), ("train", TrainConfig))


def presets(root: Path) -> list[dict[str, Any]]:
    """Each configs/*.yaml, resolved, with the first comment line as its description."""
    out = []
    for path in sorted((root / "configs").glob("*.yaml")):
        text = path.read_text()
        comment = next(
            (ln.lstrip("# ").strip() for ln in text.splitlines() if ln.startswith("#")), ""
        )
        try:
            config = load_config(path).to_dict()
        except (KeyError, ValueError, TypeError) as e:
            out.append({"name": path.stem, "description": comment, "error": describe_error(e)})
            continue
        out.append({"name": path.stem, "description": comment, "config": config})
    return out


def schema() -> dict[str, list[dict[str, Any]]]:
    """The form's fields, section by section, from the config dataclasses."""
    out = {}
    for section, cls in SECTIONS:
        defaults = cls()
        fields = []
        for f in dataclasses.fields(cls):
            key = f"{section}.{f.name}"
            if key in SET_BY_FORM:
                continue
            type_text = str(f.type)
            base = type_text.replace(" | None", "").replace("None | ", "")
            fields.append(
                {
                    "key": key,
                    "type": "list" if base.startswith("list") else base,
                    "optional": "None" in type_text,
                    "default": getattr(defaults, f.name),
                    "choices": CHOICES.get(key),
                }
            )
        out[section] = fields
    return out


def build(root: Path, preset: str, overrides: dict[str, Any], run_name: str) -> Config:
    """A preset with the form's overrides applied, the same way the training CLI does it."""
    path = root / "configs" / f"{preset}.yaml"
    if not RUN_NAME.match(preset) or not path.is_file():
        raise KeyError(f"There's no preset named {preset}.")
    items = [f"{k}={json.dumps(v)}" for k, v in overrides.items() if k not in SET_BY_FORM]
    cfg = load_config(path, items)
    cfg.train.run_name = run_name
    cfg.train.out_dir = "runs"
    _check_types(cfg)
    return cfg


_TYPE_NAMES = {"int": "a whole number", "float": "a number", "bool": "true or false", "str": "text"}


def _check_types(cfg: Config) -> None:
    """The config dataclasses don't check types, so a form value like "abc" for an int
    would otherwise reach the training code."""
    for field in (f for fields in schema().values() for f in fields):
        section, name = field["key"].split(".")
        value = getattr(getattr(cfg, section), name)
        if value is None and field["optional"]:
            continue
        kind = field["type"]
        ok = {
            "int": isinstance(value, int) and not isinstance(value, bool),
            "float": isinstance(value, (int, float)) and not isinstance(value, bool),
            "bool": isinstance(value, bool),
            "str": isinstance(value, str),
            "list": isinstance(value, list) and all(isinstance(v, str) for v in value),
        }.get(kind, True)
        if not ok:
            expected = _TYPE_NAMES.get(kind, "a list of text lines")
            raise ValueError(f"{field['key']} must be {expected}, not {value!r}.")
        if field["choices"] and value not in field["choices"]:
            raise ValueError(f"{field['key']} must be one of {', '.join(field['choices'])}.")


def describe_error(e: Exception) -> str:
    return str(e.args[0]) if isinstance(e, KeyError) and e.args else str(e)


def shards(root: Path, pattern: str) -> list[str]:
    full = pattern if Path(pattern).is_absolute() else str(root / pattern)
    return sorted(glob.glob(full))


@functools.lru_cache(maxsize=4096)
def _shard_tokens(path: str, mtime_ns: int, size: int) -> int:
    return int(np.load(path, mmap_mode="r").shape[0])


def shard_tokens(path: str | Path) -> int:
    st = Path(path).stat()
    return _shard_tokens(str(path), st.st_mtime_ns, st.st_size)


@functools.lru_cache(maxsize=64)
def _model_stats(fields: tuple) -> dict[str, float]:
    with torch.device("meta"):  # shapes only: no memory, no initialization cost
        model = Transformer(ModelConfig(**dict(fields)))
    return {
        "params": model.num_params(),
        "params_non_embedding": model.num_params(non_embedding=True),
        "flops_per_token": model.flops_per_token(),
    }


def model_stats(mc: ModelConfig) -> dict[str, float]:
    return _model_stats(tuple(sorted(dataclasses.asdict(mc).items())))


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def check(root: Path, cfg: Config, run_name: str, *, new: bool) -> dict[str, Any]:
    """Problems that would stop the run (errors) or that are worth a second look
    (warnings), plus the numbers the form shows next to the settings."""
    mc, dc, tc = cfg.model, cfg.data, cfg.train
    errors: list[str] = []
    warnings: list[str] = []

    if not RUN_NAME.match(run_name) or run_name in RESERVED:
        errors.append(
            "Run names start with a letter or digit and use only letters, digits, dots, "
            "dashes and underscores (64 characters at most)."
        )
    elif new and (root / "runs" / run_name).exists():
        errors.append(
            f"A run named {run_name} already exists. Pick another name, or resume it from its page."
        )

    stats = None
    try:
        stats = model_stats(mc)
    except (AssertionError, ValueError, RuntimeError, TypeError) as e:
        errors.append(f"The model shape doesn't work: {e}")

    tokens_per_micro = tc.micro_batch_size * mc.seq_len
    grad_accum = None
    if tokens_per_micro <= 0 or tc.total_batch_tokens % tokens_per_micro:
        errors.append(
            f"Tokens per step ({tc.total_batch_tokens:,}) must be a multiple of micro batch "
            f"size × sequence length ({tokens_per_micro:,})."
        )
    else:
        grad_accum = tc.total_batch_tokens // tokens_per_micro
    if tc.max_steps <= 0:
        errors.append("Max steps must be at least 1.")
    elif tc.warmup_steps >= tc.max_steps:
        warnings.append("Warmup lasts the whole run, so the learning rate never reaches its peak.")

    device = resolve_device(tc.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        errors.append("This computer has no CUDA GPU available to PyTorch. Set device to cpu.")
    if device == "cpu" and stats and stats["params"] > 20e6:
        warnings.append("Training a model this size on the CPU will be very slow.")

    train_files = shards(root, dc.train_pattern)
    val_files = shards(root, dc.val_pattern)
    if not train_files:
        errors.append(f"No training shards match {dc.train_pattern}. Prepare a dataset first.")
    if tc.eval_every and not val_files:
        errors.append(
            f"No validation shards match {dc.val_pattern}. Prepare the dataset, or set eval_every to 0."
        )
    train_tokens = sum(shard_tokens(f) for f in train_files)
    total_tokens = tc.max_steps * tc.total_batch_tokens
    if train_tokens and total_tokens > train_tokens:
        ratio = total_tokens / train_tokens
        warnings.append(
            f"The run reads {100 * (ratio - 1):.0f}% more tokens than are prepared, so some examples repeat."
            if ratio < 1.1
            else f"The run reads {ratio:.1f}× the training data, so it repeats examples."
        )

    return {
        "errors": errors,
        "warnings": warnings,
        "derived": {
            "params": stats["params"] if stats else None,
            "params_non_embedding": stats["params_non_embedding"] if stats else None,
            "flops_per_token": stats["flops_per_token"] if stats else None,
            "tokens_per_step": tc.total_batch_tokens,
            "grad_accum": grad_accum,
            "total_tokens": total_tokens,
            "tokens_per_param": total_tokens / stats["params"] if stats else None,
            "train_shards": len(train_files),
            "train_tokens": train_tokens,
            "val_shards": len(val_files),
            "device": device,
        },
    }
