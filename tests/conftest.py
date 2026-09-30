from __future__ import annotations

import numpy as np
import pytest
import torch

from gpt_lab.config import Config, ModelConfig
from gpt_lab.data import write_shard

VOCAB = 64


def tiny_model_config(**overrides) -> ModelConfig:
    base = dict(vocab_size=VOCAB, seq_len=32, n_layer=2, n_head=4, d_model=32, ffn_multiple_of=16)
    base.update(overrides)
    return ModelConfig(**base)


@pytest.fixture
def model_cfg() -> ModelConfig:
    return tiny_model_config()


@pytest.fixture
def shard_dir(tmp_path):
    """Three small train shards and one val shard of learnable (periodic) token streams."""
    rng = np.random.default_rng(0)
    d = tmp_path / "data"
    pattern = np.arange(VOCAB)
    for i, n in enumerate([2000, 1500, 1800]):
        tokens = np.tile(pattern, n // VOCAB + 1)[:n]
        tokens[rng.integers(0, n, size=n // 50)] = rng.integers(0, VOCAB, size=n // 50)
        write_shard(d / f"train_{i:06d}.npy", tokens)
    write_shard(d / "val_000000.npy", np.tile(pattern, 20))
    return d


@pytest.fixture
def tiny_config(tmp_path, shard_dir) -> Config:
    cfg = Config()
    cfg.model = tiny_model_config()
    cfg.data.train_pattern = str(shard_dir / "train_*.npy")
    cfg.data.val_pattern = str(shard_dir / "val_*.npy")
    t = cfg.train
    t.run_name = "test"
    t.out_dir = str(tmp_path / "runs")
    t.device = "cpu"
    t.dtype = "float32"
    t.compile = False
    t.micro_batch_size = 4
    t.total_batch_tokens = 4 * 32 * 2
    t.max_steps = 6
    t.lr = 3e-3
    t.warmup_steps = 2
    t.log_every = 1
    t.eval_every = 3
    t.val_tokens = 4 * 32 * 2
    t.hellaswag_every = 0
    t.sample_every = 0
    t.ckpt_every = 3
    return cfg


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
