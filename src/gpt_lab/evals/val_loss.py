"""Loss and perplexity on held-out token shards."""

from __future__ import annotations

import math
from contextlib import nullcontext

import torch

from gpt_lab.data import ShardedLoader


@torch.no_grad()
def evaluate_val_loss(
    model: torch.nn.Module,
    pattern: str,
    *,
    batch_size: int,
    seq_len: int,
    max_tokens: int,
    device: str | torch.device,
    autocast_ctx=None,
) -> dict[str, float]:
    """Mean next-token loss over the first `max_tokens` tokens of the validation shards.

    Always reads the same tokens in the same order, so numbers are comparable across
    checkpoints and across models that share the GPT-2 tokenizer.
    """
    loader = ShardedLoader(pattern, batch_size, seq_len, shuffle=False)
    available = loader.total_chunks() // batch_size
    num_batches = max(1, min(max_tokens // (batch_size * seq_len), available))
    ctx = autocast_ctx or nullcontext()
    was_training = model.training
    model.eval()
    total = 0.0
    for _ in range(num_batches):
        x, y = loader.next_batch()
        x, y = x.to(device), y.to(device)
        with ctx:
            _, loss = model(x, y)
        total += loss.item()
    model.train(was_training)
    loss = total / num_batches
    return {
        "val_loss": loss,
        "val_ppl": math.exp(min(loss, 50.0)),
        "val_tokens": num_batches * batch_size * seq_len,
    }
