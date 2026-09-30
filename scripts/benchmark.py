"""Measure training throughput and memory for several micro-batch sizes on the GPU.

    python scripts/benchmark.py --config configs/modern_124m_fineweb10b.yaml
    python scripts/benchmark.py --config configs/modern_124m_fineweb10b.yaml --micro-batch-sizes 8,16,32 --no-compile

Uses random tokens (no dataset needed). Runs forward + backward + optimizer step, which is
what training does. Pick the largest micro-batch that fits with some memory headroom, and
set train.micro_batch_size in the config. `--peak-tflops` enables a model FLOPs
utilization column (look up the bfloat16 dense tensor figure for your GPU).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gpt_lab.config import load_config  # noqa: E402
from gpt_lab.model import Transformer, configure_optimizer  # noqa: E402
from gpt_lab.train import autocast_context, resolve_device  # noqa: E402


def bench_one(cfg, micro_bs: int, steps: int, warmup: int, compile_: bool, device: str) -> dict:
    mc = cfg.model
    model = Transformer(mc).to(device)
    opt = configure_optimizer(model, 1e-4, 0.1, fused=device.startswith("cuda"))
    fwd = torch.compile(model) if compile_ else model
    ctx = autocast_context(device, cfg.train.dtype)
    x = torch.randint(0, 50257, (micro_bs, mc.seq_len), device=device)
    y = torch.randint(0, 50257, (micro_bs, mc.seq_len), device=device)
    torch.cuda.reset_peak_memory_stats()
    t0 = None
    for i in range(warmup + steps):
        if i == warmup:
            torch.cuda.synchronize()
            t0 = time.time()
        with ctx:
            _, loss = fwd(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = time.time() - t0
    tok_s = steps * micro_bs * mc.seq_len / dt
    return {
        "tok_s": tok_s,
        "tflops": tok_s * model.flops_per_token() / 1e12,
        "mem_gb": torch.cuda.max_memory_allocated() / 1e9,
        "params": model.num_params(),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--micro-batch-sizes", default="4,8,16,32")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3, help="untimed steps (includes compilation)")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--peak-tflops", type=float, default=None)
    args, overrides = p.parse_known_args()

    cfg = load_config(args.config, overrides)
    device = resolve_device(cfg.train.device)
    if not device.startswith("cuda"):
        sys.exit("benchmark.py needs a CUDA GPU")
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"GPU: {torch.cuda.get_device_name()} | torch {torch.__version__}")
    print(f"{'micro_bs':>8} {'tok/s':>10} {'TFLOPS':>8} {'MFU':>6} {'peak GB':>8}")
    for mbs in [int(s) for s in args.micro_batch_sizes.split(",")]:
        try:
            r = bench_one(cfg, mbs, args.steps, args.warmup, not args.no_compile, device)
            mfu = f"{r['tflops'] / args.peak_tflops:6.1%}" if args.peak_tflops else "     -"
            print(f"{mbs:>8} {r['tok_s']:>10,.0f} {r['tflops']:>8.1f} {mfu} {r['mem_gb']:>8.2f}")
            hours = cfg.train.max_steps * cfg.train.total_batch_tokens / r["tok_s"] / 3600
            print(f"{'':>8} -> full run ({cfg.train.max_steps} steps) would take ~{hours:.1f} h")
        except torch.cuda.OutOfMemoryError:
            print(f"{mbs:>8} out of memory")
        finally:
            torch.cuda.empty_cache()
            torch._dynamo.reset()


if __name__ == "__main__":
    main()
