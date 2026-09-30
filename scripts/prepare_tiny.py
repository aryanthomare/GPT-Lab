"""Tokenize a small text file into train/val shards for quick smoke runs.

    python scripts/prepare_tiny.py                    # downloads Tiny Shakespeare (~1 MB)
    python scripts/prepare_tiny.py --input my.txt     # any local text file

Output: data/tiny/train_000000.npy and data/tiny/val_000000.npy (last 10% held out).
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpt_lab.data import write_shard  # noqa: E402
from gpt_lab.tokenizer import EOT_TOKEN, encode  # noqa: E402

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--input", default=None, help="text file (default: download Tiny Shakespeare)")
    p.add_argument("--out-dir", default="data/tiny")
    p.add_argument("--val-frac", type=float, default=0.1)
    args = p.parse_args()

    if args.input:
        text = Path(args.input).read_text(encoding="utf-8")
    else:
        print(f"downloading {SHAKESPEARE_URL}")
        with urllib.request.urlopen(SHAKESPEARE_URL) as r:
            text = r.read().decode("utf-8")

    tokens = np.asarray([EOT_TOKEN] + encode(text), dtype=np.uint16)
    n_val = int(len(tokens) * args.val_frac)
    out = Path(args.out_dir)
    write_shard(out / "train_000000.npy", tokens[:-n_val])
    write_shard(out / "val_000000.npy", tokens[-n_val:])
    print(f"{len(tokens):,} tokens -> {out} (val {n_val:,})")


if __name__ == "__main__":
    main()
