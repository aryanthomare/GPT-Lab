"""Experiment configuration: dataclasses loaded from YAML with dotted command-line overrides.

Example:
    cfg = load_config("configs/modern_124m_fineweb10b.yaml", ["train.lr=1e-3"])
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 50304  # GPT-2's 50,257 tokens padded to a multiple of 128
    seq_len: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_kv_head: int | None = None  # None -> same as n_head (no grouped-query attention)
    d_model: int = 768
    ffn_hidden: int | None = None  # None -> 8/3 * d_model rounded up to ffn_multiple_of
    ffn_multiple_of: int = 256
    rope_theta: float = 10000.0
    qk_norm: bool = True
    tie_embeddings: bool = True
    norm_eps: float = 1e-6
    init_std: float = 0.02
    activation_checkpointing: bool = False

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    @property
    def kv_heads(self) -> int:
        return self.n_kv_head or self.n_head

    @property
    def ffn_dim(self) -> int:
        if self.ffn_hidden is not None:
            return self.ffn_hidden
        hidden = int(8 * self.d_model / 3)
        m = self.ffn_multiple_of
        return m * ((hidden + m - 1) // m)


@dataclass
class DataConfig:
    train_pattern: str = "data/fineweb_edu_10B/train_*.npy"
    val_pattern: str = "data/fineweb_edu_10B/val_*.npy"
    hellaswag_path: str = "data/hellaswag/hellaswag_val.jsonl"
    seed: int = 1337


@dataclass
class TrainConfig:
    run_name: str = "modern_124m"
    out_dir: str = "runs"
    device: str = "auto"  # "auto" | "cuda" | "cpu" | "mps"
    dtype: str = "bfloat16"  # autocast dtype on GPU: "bfloat16" | "float32"
    compile: bool = True
    seed: int = 1337

    # Batch: total_batch_tokens = micro_batch_size * seq_len * grad_accum_steps
    micro_batch_size: int = 16
    total_batch_tokens: int = 524_288
    max_steps: int = 19_073  # 10B tokens / 524,288 tokens per step

    # Optimizer
    lr: float = 1e-3
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0

    # Learning-rate schedule
    schedule: str = "wsd"  # "wsd" (warmup-stable-decay) | "cosine"
    warmup_steps: int = 700
    decay_frac: float = 0.2  # wsd: fraction of steps spent in the final linear decay
    min_lr_ratio: float = 0.0

    # Logging / evaluation / checkpoints (0 disables)
    log_every: int = 10
    eval_every: int = 250
    val_tokens: int = 10_485_760
    hellaswag_every: int = 1000
    hellaswag_limit: int | None = None  # None -> all 10,042 examples
    sample_every: int = 1000
    sample_prompts: list[str] = field(
        default_factory=lambda: ["Hello, I'm a language model,", "The theory of relativity"]
    )
    sample_max_tokens: int = 64
    ckpt_every: int = 500
    keep_last: int = 2
    keep_every: int = 5000  # checkpoints at multiples of this are never pruned
    peak_tflops: float | None = None  # GPU peak bfloat16 TFLOPS, for utilization logging

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.run_name


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Config:
        cfg = cls()
        for section, values in (d or {}).items():
            _update_section(cfg, section, values or {})
        return cfg


def _update_section(cfg: Config, section: str, values: dict[str, Any]) -> None:
    if not hasattr(cfg, section):
        raise KeyError(f"Unknown config section '{section}'")
    obj = getattr(cfg, section)
    valid = {f.name for f in dataclasses.fields(obj)}
    for key, value in values.items():
        if key not in valid:
            raise KeyError(f"Unknown config key '{section}.{key}'")
        setattr(obj, key, value)


def apply_overrides(cfg: Config, overrides: list[str]) -> Config:
    """Apply overrides of the form 'section.key=value' (a leading '--' is allowed).

    Values are parsed as YAML, so '1e-3' -> float, 'false' -> bool, 'null' -> None.
    """
    for item in overrides:
        item = item.removeprefix("--")
        if "=" not in item:
            raise ValueError(f"Override must look like section.key=value, got '{item}'")
        dotted, raw = item.split("=", 1)
        if "." not in dotted:
            raise ValueError(f"Override key must be section.key, got '{dotted}'")
        section, key = dotted.split(".", 1)
        value = yaml.safe_load(raw)
        # YAML parses '1e-3' as a string; coerce numbers where the default is numeric.
        current = getattr(getattr(cfg, section, None), key, None)
        if isinstance(current, float) and isinstance(value, (int, str)):
            value = float(value)
        _update_section(cfg, section, {key: value})
    return cfg


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    data: dict[str, Any] = {}
    if path is not None:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    cfg = Config.from_dict(data)
    _coerce_floats(cfg)
    return apply_overrides(cfg, overrides or [])


def _coerce_floats(cfg: Config) -> None:
    """YAML reads '6e-4' (no dot) as a string; convert values whose default is a float."""
    for section in (cfg.model, cfg.data, cfg.train):
        defaults = type(section)()
        for f in dataclasses.fields(section):
            default = getattr(defaults, f.name)
            value = getattr(section, f.name)
            if isinstance(value, str) and (isinstance(default, float) or f.name == "peak_tflops"):
                setattr(section, f.name, float(value))


def save_config(cfg: Config, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
