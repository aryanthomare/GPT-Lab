"""Learning-rate schedules."""

from __future__ import annotations

import math


def lr_at(
    step: int,
    *,
    max_lr: float,
    total_steps: int,
    warmup_steps: int,
    schedule: str = "wsd",
    decay_frac: float = 0.2,
    min_lr_ratio: float = 0.0,
) -> float:
    """Learning rate for a 0-indexed optimizer step.

    - warmup: linear from max_lr / warmup_steps up to max_lr
    - "wsd" (warmup-stable-decay): flat at max_lr, then linear to min_lr over the
      last `decay_frac` of training
    - "cosine": cosine from max_lr down to min_lr after warmup
    """
    min_lr = max_lr * min_lr_ratio
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if schedule == "cosine":
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (max_lr - min_lr)
    if schedule == "wsd":
        decay_steps = max(1, int(round(decay_frac * total_steps)))
        decay_start = total_steps - decay_steps
        if step < decay_start:
            return max_lr
        progress = min((step - decay_start + 1) / decay_steps, 1.0)
        return max_lr - (max_lr - min_lr) * progress
    raise ValueError(f"Unknown schedule '{schedule}'")
