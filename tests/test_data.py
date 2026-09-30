import numpy as np
import pytest
import torch

from gpt_lab.data import ShardedLoader, list_shards, read_shard, write_shard


def test_shard_roundtrip(tmp_path):
    tokens = np.array([0, 5, 50256, 65535])
    write_shard(tmp_path / "a.npy", tokens)
    out = read_shard(tmp_path / "a.npy")
    assert out.dtype == np.uint16
    np.testing.assert_array_equal(out, tokens)
    with pytest.raises(ValueError):
        write_shard(tmp_path / "b.npy", np.array([70000]))


def test_batches_are_shifted_targets(shard_dir):
    loader = ShardedLoader(str(shard_dir / "train_*.npy"), batch_size=3, seq_len=32, seed=0)
    x, y = loader.next_batch()
    assert x.shape == y.shape == (3, 32)
    assert x.dtype == torch.int64
    torch.testing.assert_close(x[:, 1:], y[:, :-1])


def test_deterministic_given_seed(shard_dir):
    pattern = str(shard_dir / "train_*.npy")
    a = ShardedLoader(pattern, 4, 32, seed=7)
    b = ShardedLoader(pattern, 4, 32, seed=7)
    c = ShardedLoader(pattern, 4, 32, seed=8)
    xa, xb, xc = a.next_batch()[0], b.next_batch()[0], c.next_batch()[0]
    torch.testing.assert_close(xa, xb)
    assert not torch.equal(xa, xc)


def test_epoch_covers_every_chunk_once(shard_dir):
    pattern = str(shard_dir / "train_*.npy")
    loader = ShardedLoader(pattern, 1, 32, seed=0)
    total = loader.total_chunks()
    assert total == sum((len(read_shard(f)) - 1) // 32 for f in list_shards(pattern))
    seen = set()
    for _ in range(total):
        loader.next_batch()
        shard = int(loader._shard_order[loader.shard_pos])
        seen.add((shard, int(loader._chunk_order[loader.chunk_pos - 1])))
    assert len(seen) == total
    assert loader.epoch == 0
    loader.next_batch()
    assert loader.epoch == 1


def test_resume_matches_uninterrupted(shard_dir):
    pattern = str(shard_dir / "train_*.npy")
    ref = ShardedLoader(pattern, 5, 32, seed=3)
    expected = [ref.next_batch()[0] for _ in range(40)]  # crosses shards and epochs

    first = ShardedLoader(pattern, 5, 32, seed=3)
    got = [first.next_batch()[0] for _ in range(17)]
    state = first.state_dict()
    second = ShardedLoader(pattern, 5, 32, seed=0)
    second.load_state_dict(state)
    got += [second.next_batch()[0] for _ in range(23)]
    for e, g in zip(expected, got, strict=True):
        torch.testing.assert_close(e, g)


def test_sequential_mode_reads_in_order(shard_dir):
    loader = ShardedLoader(str(shard_dir / "val_*.npy"), 2, 32, shuffle=False)
    x, _ = loader.next_batch()
    tokens = read_shard(shard_dir / "val_000000.npy")
    np.testing.assert_array_equal(x[0].numpy(), tokens[:32])
    np.testing.assert_array_equal(x[1].numpy(), tokens[32:64])
