"""HellaSwag commonsense sentence completion (validation split, 10,042 examples).

Each example has a context and four candidate endings. The model scores every ending by
its average per-token loss given the context; the lowest loss is the prediction
("acc_norm", length-normalized). "acc" uses the summed loss instead. Zero-shot.

Reference numbers with this completion-style scoring (moderate confidence):
OpenAI GPT-2 124M ~0.295 acc_norm, GPT-2 350M ~0.37.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

# The author's copy on Hugging Face. The original GitHub copy (rowanz/hellaswag) is no
# longer available (HTTP 451 for the repository, 404 for its raw files).
HELLASWAG_VAL_URL = (
    "https://huggingface.co/datasets/Rowan/hellaswag/resolve/main/data/"
    "validation-00000-of-00001.parquet"
)


def parquet_to_jsonl(src: str | Path, dst: str | Path) -> int:
    """Write the fields the scorer uses as JSON lines; returns the number of examples.

    Needs pyarrow, which the [data] and [eval] extras install (through `datasets`).
    """
    import pyarrow.parquet as pq

    rows = pq.read_table(src).to_pylist()
    with open(dst, "w") as f:
        for r in rows:
            out = {"ind": r.get("ind"), "ctx": r["ctx"], "endings": list(r["endings"])}
            out["label"] = int(r["label"])  # stored as "0".."3" on Hugging Face
            f.write(json.dumps(out) + "\n")
    return len(rows)


def download_hellaswag(path: str | Path) -> Path:
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[hellaswag] downloading {HELLASWAG_VAL_URL} -> {path}")
        parquet = path.with_suffix(".parquet.tmp")
        tmp = path.with_suffix(".tmp")
        try:
            urllib.request.urlretrieve(HELLASWAG_VAL_URL, parquet)
            n = parquet_to_jsonl(parquet, tmp)
            tmp.replace(path)
            print(f"[hellaswag] wrote {n:,} examples")
        finally:
            parquet.unlink(missing_ok=True)
            tmp.unlink(missing_ok=True)
    return path


def iterate_examples(path: str | Path, limit: int | None = None) -> Iterator[dict]:
    with open(path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            yield json.loads(line)


def render_example(
    example: dict, encode: Callable[[str], list[int]]
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return (tokens (4, T), mask (4, T), label). mask is 1 on ending tokens."""
    ctx_tokens = encode(example["ctx"])
    rows, masks = [], []
    for ending in example["endings"]:
        end_tokens = encode(" " + ending)  # GPT-2 tokens carry a leading space
        rows.append(ctx_tokens + end_tokens)
        masks.append([0] * len(ctx_tokens) + [1] * len(end_tokens))
    max_len = max(len(r) for r in rows)
    tokens = torch.zeros((len(rows), max_len), dtype=torch.long)
    mask = torch.zeros((len(rows), max_len), dtype=torch.long)
    for i, (r, m) in enumerate(zip(rows, masks, strict=True)):
        tokens[i, : len(r)] = torch.tensor(r)
        mask[i, : len(m)] = torch.tensor(m)
    return tokens, mask, int(example["label"])


@torch.no_grad()
def score_endings(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    mask: torch.Tensor,
    vocab_limit: int | None = None,
    autocast_ctx=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (sum_loss, mean_loss) per ending, each of shape (num_endings,)."""
    with autocast_ctx or nullcontext():
        logits, _ = model(tokens)
    logits = logits[:, :-1, :vocab_limit].float()
    targets = tokens[:, 1:]
    losses = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none"
    ).view(targets.shape)
    m = mask[:, 1:].to(losses.dtype)
    sum_loss = (losses * m).sum(dim=1)
    mean_loss = sum_loss / m.sum(dim=1).clamp(min=1)
    return sum_loss, mean_loss


def evaluate_hellaswag(
    model: torch.nn.Module,
    path: str | Path,
    *,
    encode: Callable[[str], list[int]] | None = None,
    device: str | torch.device = "cpu",
    limit: int | None = None,
    max_len: int | None = None,
    vocab_limit: int | None = None,
    autocast_ctx=None,
) -> dict[str, float]:
    if encode is None:
        from gpt_lab.tokenizer import encode as gpt2_encode

        encode = gpt2_encode
    was_training = model.training
    model.eval()
    n = correct = correct_norm = 0
    for example in iterate_examples(path, limit):
        tokens, mask, label = render_example(example, encode)
        if max_len is not None and tokens.size(1) > max_len:  # keep the end, where endings are
            tokens, mask = tokens[:, -max_len:], mask[:, -max_len:]
        sum_loss, mean_loss = score_endings(
            model, tokens.to(device), mask.to(device), vocab_limit, autocast_ctx
        )
        n += 1
        correct += int(sum_loss.argmin().item() == label)
        correct_norm += int(mean_loss.argmin().item() == label)
    model.train(was_training)
    return {
        "hellaswag_acc": correct / max(n, 1),
        "hellaswag_acc_norm": correct_norm / max(n, 1),
        "hellaswag_n": n,
    }
