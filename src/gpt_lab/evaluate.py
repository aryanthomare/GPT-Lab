"""Evaluate a checkpoint or a Hugging Face GPT-2 reference model with identical code.

    python -m gpt_lab.evaluate --checkpoint runs/modern_124m
    python -m gpt_lab.evaluate --hf gpt2
    python -m gpt_lab.evaluate --hf gpt2-medium --lm-eval arc_easy,piqa,winogrande
    python -m gpt_lab.compare runs/*/eval.json runs/reference/*.json   # table

Writes a JSON file of results (default: <run_dir>/eval.json or runs/reference/<name>.json).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gpt_lab.evals.hellaswag import download_hellaswag, evaluate_hellaswag
from gpt_lab.evals.val_loss import evaluate_val_loss
from gpt_lab.tokenizer import GPT2_VOCAB_SIZE
from gpt_lab.train import autocast_context, resolve_device


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="checkpoint file or run directory")
    src.add_argument("--hf", help="Hugging Face model name, e.g. gpt2 or gpt2-medium")
    p.add_argument("--val-pattern", default="data/fineweb_edu_10B/val_*.npy")
    p.add_argument("--val-tokens", type=int, default=10_485_760)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--hellaswag", default="data/hellaswag/hellaswag_val.jsonl")
    p.add_argument("--hellaswag-limit", type=int, default=None)
    p.add_argument("--skip-val", action="store_true")
    p.add_argument("--skip-hellaswag", action="store_true")
    p.add_argument("--lm-eval", default="", help="comma-separated lm-evaluation-harness tasks")
    p.add_argument("--lm-eval-limit", type=int, default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--out", default=None, help="output JSON path")
    args = p.parse_args(argv)

    device = resolve_device(args.device)
    ctx = autocast_context(device, args.dtype)

    if args.checkpoint:
        from gpt_lab.checkpoint import load_model_from_checkpoint, resolve_checkpoint

        ckpt_path = resolve_checkpoint(args.checkpoint)
        model, cfg, ckpt = load_model_from_checkpoint(ckpt_path, device)
        seq_len = cfg.model.seq_len
        name = f"{cfg.train.run_name}@step{ckpt['step']}"
        default_out = ckpt_path.parent.parent / "eval.json"
        info = {"source": str(ckpt_path), "step": ckpt["step"], "params": model.num_params()}
    else:
        from gpt_lab.hf_reference import HFCausalLM

        model = HFCausalLM(args.hf).to(device).eval()
        seq_len = model.seq_len
        name = f"hf:{args.hf}"
        default_out = Path("runs/reference") / f"{args.hf.replace('/', '_')}.json"
        info = {"source": args.hf, "params": sum(p.numel() for p in model.parameters())}

    results: dict = {"name": name, **info}

    if not args.skip_val:
        results.update(
            evaluate_val_loss(
                model,
                args.val_pattern,
                batch_size=args.batch_size,
                seq_len=min(seq_len, 1024),
                max_tokens=args.val_tokens,
                device=device,
                autocast_ctx=ctx,
            )
        )
        print(f"val_loss {results['val_loss']:.4f}")

    if not args.skip_hellaswag:
        download_hellaswag(args.hellaswag)
        results.update(
            evaluate_hellaswag(
                model,
                args.hellaswag,
                device=device,
                limit=args.hellaswag_limit,
                max_len=seq_len,
                vocab_limit=GPT2_VOCAB_SIZE,
                autocast_ctx=ctx,
            )
        )
        print(f"hellaswag acc_norm {results['hellaswag_acc_norm']:.4f}")

    if args.lm_eval:
        from gpt_lab.evals.lm_eval_adapter import run_lm_eval

        tasks = [t.strip() for t in args.lm_eval.split(",") if t.strip()]
        lm_results = run_lm_eval(
            model, tasks, seq_len=seq_len, device=device, autocast_ctx=ctx, limit=args.lm_eval_limit
        )
        results["lm_eval"] = lm_results
        for task, vals in lm_results.items():
            print(task, {k: v for k, v in vals.items() if isinstance(v, float)})

    out = Path(args.out) if args.out else default_out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {out}")
    return results


if __name__ == "__main__":
    main()
