"""Print a markdown table comparing evaluation JSON files written by gpt_lab.evaluate.

python -m gpt_lab.compare runs/*/eval.json runs/reference/*.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _lm_eval_score(vals: dict) -> float | None:
    for key in ("acc_norm,none", "acc,none"):
        if key in vals:
            return vals[key]
    return None


def build_table(results: list[dict]) -> str:
    tasks = sorted({t for r in results for t in r.get("lm_eval", {})})
    header = ["model", "params (M)", "val loss", "HellaSwag acc_norm", *tasks]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in results:
        row = [
            r.get("name", "?"),
            f"{r['params'] / 1e6:.1f}" if "params" in r else "",
            f"{r['val_loss']:.4f}" if "val_loss" in r else "",
            f"{r['hellaswag_acc_norm']:.4f}" if "hellaswag_acc_norm" in r else "",
        ]
        for t in tasks:
            score = _lm_eval_score(r.get("lm_eval", {}).get(t, {}))
            row.append(f"{score:.4f}" if score is not None else "")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("files", nargs="+")
    args = p.parse_args(argv)
    results = [json.loads(Path(f).read_text()) for f in args.files]
    print(build_table(results))


if __name__ == "__main__":
    main()
