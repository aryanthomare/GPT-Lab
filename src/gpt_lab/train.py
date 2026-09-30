"""Training loop.

    python -m gpt_lab.train --config configs/modern_124m_fineweb10b.yaml
    python -m gpt_lab.train --config configs/modern_124m_fineweb10b.yaml --resume
    python -m gpt_lab.train --config configs/tiny_cpu.yaml --train.max_steps=20

Any config value can be overridden with --section.key=value.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from gpt_lab.checkpoint import (
    checkpoint_path,
    latest_checkpoint,
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)
from gpt_lab.config import Config, load_config, save_config
from gpt_lab.data import ShardedLoader
from gpt_lab.evals.val_loss import evaluate_val_loss
from gpt_lab.model import Transformer, configure_optimizer
from gpt_lab.schedule import lr_at
from gpt_lab.tracking import Tracker


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def autocast_context(device: str, dtype: str):
    if dtype == "bfloat16" and device.startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def train(cfg: Config, resume: bool = False, stop_at: int | None = None) -> dict[str, Any]:
    """Run training. Returns {"losses": [...per step...], "final_step": int, ...metrics}.

    stop_at: stop after this many completed steps without changing the schedule
    (the learning-rate schedule still uses train.max_steps). Useful for tests and
    for splitting a long run into sessions.
    """
    tc, mc, dc = cfg.train, cfg.model, cfg.data
    device = resolve_device(tc.device)
    is_cuda = device.startswith("cuda")
    torch.manual_seed(tc.seed)
    if is_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    ctx = autocast_context(device, tc.dtype)

    run_dir = tc.run_dir
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)

    tokens_per_micro = tc.micro_batch_size * mc.seq_len
    if tc.total_batch_tokens % tokens_per_micro != 0:
        raise ValueError(
            f"total_batch_tokens ({tc.total_batch_tokens}) must be a multiple of "
            f"micro_batch_size * seq_len ({tokens_per_micro})"
        )
    grad_accum = tc.total_batch_tokens // tokens_per_micro

    model = Transformer(mc).to(device)
    optimizer = configure_optimizer(
        model, tc.lr, tc.weight_decay, (tc.beta1, tc.beta2), tc.eps, fused=is_cuda
    )
    loader = ShardedLoader(dc.train_pattern, tc.micro_batch_size, mc.seq_len, seed=dc.seed)

    start_step = 0
    if resume:
        path = latest_checkpoint(ckpt_dir)
        if path is None:
            print(f"[train] --resume given but no checkpoint in {ckpt_dir}; starting fresh")
        else:
            ckpt = load_checkpoint(path, map_location=device)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            loader.load_state_dict(ckpt["loader"])
            torch.set_rng_state(ckpt["torch_rng"].cpu())
            if is_cuda and ckpt.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["cuda_rng"]])
            start_step = ckpt["step"]
            print(f"[train] resumed from {path} at step {start_step}")

    save_config(cfg, run_dir / "config.yaml")
    (run_dir / "git_commit.txt").write_text(_git_commit() + "\n")

    fwd_model = torch.compile(model) if (tc.compile and is_cuda) else model
    tracker = Tracker(run_dir / "tb")
    flops_per_token = model.flops_per_token()
    tracker.print(
        f"model params {model.num_params() / 1e6:.1f}M | device {device} | "
        f"grad_accum {grad_accum} | tokens/step {tc.total_batch_tokens:,} | "
        f"steps {start_step}->{tc.max_steps}"
    )

    tokenizer_ok = True
    hellaswag_path = Path(dc.hellaswag_path)

    def save(step: int, loader_state: dict[str, Any]) -> None:
        path = save_checkpoint(
            checkpoint_path(ckpt_dir, step),
            model=model,
            optimizer=optimizer,
            step=step,
            config=cfg,
            loader_state=loader_state,
        )
        prune_checkpoints(ckpt_dir, tc.keep_last, tc.keep_every)
        tracker.print(f"saved checkpoint {path}")

    def periodic(completed: int, final: bool) -> dict[str, float]:
        nonlocal tokenizer_ok
        metrics: dict[str, float] = {}
        if tc.eval_every and (completed % tc.eval_every == 0 or final):
            metrics.update(
                evaluate_val_loss(
                    fwd_model,
                    dc.val_pattern,
                    batch_size=tc.micro_batch_size,
                    seq_len=mc.seq_len,
                    max_tokens=tc.val_tokens,
                    device=device,
                    autocast_ctx=ctx,
                )
            )
        if tc.hellaswag_every and (completed % tc.hellaswag_every == 0 or final):
            try:
                from gpt_lab.evals.hellaswag import download_hellaswag, evaluate_hellaswag

                download_hellaswag(hellaswag_path)
                metrics.update(
                    evaluate_hellaswag(
                        model,  # uncompiled: variable shapes would trigger recompiles
                        hellaswag_path,
                        device=device,
                        limit=tc.hellaswag_limit,
                        max_len=mc.seq_len,
                        vocab_limit=50257,
                        autocast_ctx=ctx,
                    )
                )
            except Exception as e:  # never kill a long run over an eval problem
                tracker.print(f"hellaswag eval failed ({e!r}); disabling")
                tc.hellaswag_every = 0
        if tc.sample_every and tokenizer_ok and (completed % tc.sample_every == 0 or final):
            try:
                from gpt_lab.generate import sample_text

                for i, prompt in enumerate(tc.sample_prompts):
                    with ctx:
                        text = sample_text(
                            model,
                            prompt,
                            seq_len=mc.seq_len,
                            max_new_tokens=tc.sample_max_tokens,
                            device=device,
                            seed=i,
                        )
                    tracker.text(f"samples/{i}", text, completed)
                    tracker.print(f"sample {i}: {text!r}")
            except Exception as e:
                tracker.print(f"sampling failed ({e!r}); disabling")
                tokenizer_ok = False
        if metrics:
            tracker.scalars({f"eval/{k}": v for k, v in metrics.items()}, completed)
            tracker.print(
                f"eval step {completed} | " + " | ".join(f"{k} {v:.4f}" for k, v in metrics.items())
            )
        return metrics

    losses: list[float] = []
    last_metrics: dict[str, float] = {}
    end_step = tc.max_steps if stop_at is None else min(stop_at, tc.max_steps)
    step = start_step
    # Last consistent (step, loader position) pair, used if training is interrupted.
    safe_step, safe_loader_state = start_step, loader.state_dict()
    t_last = time.time()
    tokens_since = 0
    try:
        for step in range(start_step, end_step):
            if is_cuda:
                torch.cuda.reset_peak_memory_stats()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss_accum = torch.zeros((), device=device)
            for _ in range(grad_accum):
                x, y = loader.next_batch()
                if is_cuda:
                    x = x.pin_memory().to(device, non_blocking=True)
                    y = y.pin_memory().to(device, non_blocking=True)
                else:
                    x, y = x.to(device), y.to(device)
                with ctx:
                    _, loss = fwd_model(x, y)
                loss = loss / grad_accum
                loss_accum += loss.detach()
                loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
            lr = lr_at(
                step,
                max_lr=tc.lr,
                total_steps=tc.max_steps,
                warmup_steps=tc.warmup_steps,
                schedule=tc.schedule,
                decay_frac=tc.decay_frac,
                min_lr_ratio=tc.min_lr_ratio,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            safe_step, safe_loader_state = step + 1, loader.state_dict()

            loss_val = loss_accum.item()  # synchronizes with the GPU
            losses.append(loss_val)
            if not math.isfinite(loss_val):
                raise FloatingPointError(f"loss is {loss_val} at step {step}")
            completed = step + 1
            tokens_since += tc.total_batch_tokens

            if tc.log_every and (completed % tc.log_every == 0 or completed == end_step):
                dt = time.time() - t_last
                tok_s = tokens_since / dt
                tflops = tok_s * flops_per_token / 1e12
                metrics = {
                    "train/loss": loss_val,
                    "train/lr": lr,
                    "train/grad_norm": float(grad_norm),
                    "perf/tokens_per_sec": tok_s,
                    "perf/tflops": tflops,
                    "progress/tokens": completed * tc.total_batch_tokens,
                }
                if tc.peak_tflops:
                    metrics["perf/mfu"] = tflops / tc.peak_tflops
                if is_cuda:
                    metrics["perf/peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
                tracker.scalars(metrics, completed)
                remaining_h = (tc.max_steps - completed) * tc.total_batch_tokens / tok_s / 3600
                tracker.print(
                    f"step {completed:6d}/{tc.max_steps} | loss {loss_val:.4f} | lr {lr:.2e} | "
                    f"norm {float(grad_norm):.3f} | {tok_s / 1e3:.1f}k tok/s | "
                    f"{tflops:.1f} TFLOPS | eta {remaining_h:.1f}h"
                )
                t_last, tokens_since = time.time(), 0

            final = completed == tc.max_steps
            metrics = periodic(completed, final)
            if metrics:
                last_metrics = metrics
                t_last, tokens_since = time.time(), 0  # don't count eval time as training
            if tc.ckpt_every and (completed % tc.ckpt_every == 0 or completed == end_step):
                save(completed, loader.state_dict())
    except KeyboardInterrupt:
        tracker.print(f"interrupted; saving checkpoint at step {safe_step}")
        save(safe_step, safe_loader_state)
    finally:
        tracker.close()

    return {"losses": losses, "final_step": start_step + len(losses), **last_metrics}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", required=True, help="YAML config file")
    p.add_argument("--resume", action="store_true", help="continue from the latest checkpoint")
    p.add_argument("--stop-at", type=int, default=None, help="stop after this many steps")
    args, overrides = p.parse_known_args(argv)
    cfg = load_config(args.config, overrides)
    train(cfg, resume=args.resume, stop_at=args.stop_at)


if __name__ == "__main__":
    main()
