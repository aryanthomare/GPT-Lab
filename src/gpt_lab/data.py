"""Token shards on disk and a resumable, shuffled loader over them.

A shard is a 1-D uint16 `.npy` array of GPT-2 token ids. Documents are concatenated,
each starting with the <|endoftext|> token. The loader cuts shards into non-overlapping
chunks of seq_len + 1 tokens (inputs plus next-token targets), shuffles shard order and
chunk order with a seed, and records its position so training can resume exactly.
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import numpy as np
import torch


def write_shard(path: str | Path, tokens: np.ndarray) -> None:
    tokens = np.asarray(tokens)
    if tokens.size and tokens.max() >= 2**16:
        raise ValueError("Token ids must fit in uint16")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, tokens.astype(np.uint16))


def read_shard(path: str | Path) -> np.ndarray:
    return np.load(path, mmap_mode="r")


def list_shards(pattern: str) -> list[str]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No shards match '{pattern}'")
    return files


class ShardedLoader:
    def __init__(
        self,
        pattern: str,
        batch_size: int,
        seq_len: int,
        seed: int = 0,
        shuffle: bool = True,
    ):
        self.files = list_shards(pattern)
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.shard_pos = 0  # index into this epoch's shard order
        self.chunk_pos = 0  # index into the current shard's chunk order
        self._shard_order: np.ndarray | None = None
        self._loaded: tuple[int, int] | None = None  # (epoch, shard_pos) currently loaded
        self._tokens: np.ndarray | None = None
        self._chunk_order: np.ndarray | None = None

    # -- ordering ---------------------------------------------------------------------
    def _num_chunks(self, tokens: np.ndarray) -> int:
        return max(0, (len(tokens) - 1) // self.seq_len)

    def _ensure_loaded(self) -> None:
        if self._loaded == (self.epoch, self.shard_pos):
            return
        if self._shard_order is None or self._loaded is None or self._loaded[0] != self.epoch:
            if self.shuffle:
                rng = np.random.default_rng([self.seed, self.epoch])
                self._shard_order = rng.permutation(len(self.files))
            else:
                self._shard_order = np.arange(len(self.files))
        shard_idx = int(self._shard_order[self.shard_pos])
        self._tokens = read_shard(self.files[shard_idx])
        n = self._num_chunks(self._tokens)
        if self.shuffle:
            rng = np.random.default_rng([self.seed, self.epoch, shard_idx])
            self._chunk_order = rng.permutation(n)
        else:
            self._chunk_order = np.arange(n)
        self._loaded = (self.epoch, self.shard_pos)

    def _advance_shard(self) -> None:
        self.chunk_pos = 0
        self.shard_pos += 1
        if self.shard_pos >= len(self.files):
            self.shard_pos = 0
            self.epoch += 1

    def _next_chunk(self) -> np.ndarray:
        for _ in range(len(self.files) + 1):
            self._ensure_loaded()
            if self.chunk_pos < len(self._chunk_order):
                start = int(self._chunk_order[self.chunk_pos]) * self.seq_len
                self.chunk_pos += 1
                return np.asarray(self._tokens[start : start + self.seq_len + 1])
            self._advance_shard()
        raise ValueError(f"Shards are shorter than seq_len + 1 = {self.seq_len + 1} tokens")

    # -- public API -------------------------------------------------------------------
    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (inputs, targets), each an int64 CPU tensor of shape (batch, seq_len)."""
        chunks = np.stack([self._next_chunk() for _ in range(self.batch_size)])
        chunks = torch.from_numpy(chunks.astype(np.int64))
        return chunks[:, :-1], chunks[:, 1:]

    def total_chunks(self) -> int:
        return sum(self._num_chunks(read_shard(f)) for f in self.files)

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "shard_pos": self.shard_pos,
            "chunk_pos": self.chunk_pos,
            "seed": self.seed,
            "num_files": len(self.files),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("num_files", len(self.files)) != len(self.files):
            raise ValueError("Shard count changed since the checkpoint was saved")
        self.seed = state["seed"]
        self.epoch = state["epoch"]
        self.shard_pos = state["shard_pos"]
        self.chunk_pos = state["chunk_pos"]
        self._loaded = None
        self._shard_order = None
