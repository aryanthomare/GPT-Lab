"""Download FineWeb-Edu (sample-10BT) and tokenize it into uint16 .npy shards.

    python scripts/prepare_fineweb_edu.py                       # full ~10B tokens, ~20 GB
    python scripts/prepare_fineweb_edu.py --max-tokens 300_000_000 --out-dir data/fineweb_small

Output: <out-dir>/val_000000.npy (the first shard, held out) and train_000001.npy, ...
Each document is prefixed with <|endoftext|>. Needs `pip install -e ".[data]"`.

On WSL2 keep --out-dir on the Linux filesystem (e.g. under ~/), not /mnt/c.
The Hugging Face download cache (~/.cache/huggingface) can be deleted afterwards.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpt_lab.data import write_shard  # noqa: E402
from gpt_lab.tokenizer import EOT_TOKEN, get_tokenizer  # noqa: E402


def tokenize(doc: dict) -> np.ndarray:
    tokens = [EOT_TOKEN] + get_tokenizer().encode_ordinary(doc["text"])
    return np.asarray(tokens, dtype=np.uint16)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--out-dir", default="data/fineweb_edu_10B")
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--subset", default="sample-10BT")
    p.add_argument("--shard-size", type=int, default=100_000_000, help="tokens per shard")
    p.add_argument("--max-tokens", type=int, default=None, help="stop after this many tokens")
    p.add_argument("--num-proc", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    p.add_argument("--streaming", action="store_true", help="stream instead of full download")
    args = p.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.dataset, name=args.subset, split="train", streaming=args.streaming)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    buf = np.empty(args.shard_size, dtype=np.uint16)
    filled, shard_idx, total = 0, 0, 0
    bar = tqdm(total=args.max_tokens, unit="tok", unit_scale=True)

    def flush(n: int) -> None:
        nonlocal shard_idx
        split = "val" if shard_idx == 0 else "train"
        write_shard(out_dir / f"{split}_{shard_idx:06d}.npy", buf[:n])
        shard_idx += 1

    with mp.Pool(args.num_proc) as pool:
        for tokens in pool.imap(tokenize, ds, chunksize=16):
            if args.max_tokens is not None and total >= args.max_tokens:
                break
            pos = 0
            while pos < len(tokens):
                take = min(len(tokens) - pos, args.shard_size - filled)
                buf[filled : filled + take] = tokens[pos : pos + take]
                filled += take
                pos += take
                if filled == args.shard_size:
                    flush(filled)
                    filled = 0
            total += len(tokens)
            bar.update(len(tokens))
    if filled:
        flush(filled)
    bar.close()
    print(f"wrote {shard_idx} shards ({total:,} tokens) to {out_dir}")
    if shard_idx < 2:
        print("warning: only one shard written, so there is no train split; lower --shard-size")


if __name__ == "__main__":
    main()
