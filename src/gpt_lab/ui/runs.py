"""Reading run directories (runs/<name>/): status, metrics, logs, checkpoints and results."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

from gpt_lab.checkpoint import list_checkpoints
from gpt_lab.compare import lm_eval_score
from gpt_lab.ui.jobs import process_running, read_json

RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
LIVE = {"starting", "running"}
RESERVED = {"reference"}  # runs/reference/ holds evaluations of OpenAI's GPT-2

_summary_cache: dict[str, tuple[tuple, dict[str, Any]]] = {}


def run_dir(runs_dir: Path, name: str) -> Path:
    """The directory of an existing run. Raises KeyError for unknown or malformed names."""
    path = runs_dir / name
    if not RUN_NAME.match(name) or name in RESERVED or not path.is_dir():
        raise KeyError(name)
    return path


def run_names(runs_dir: Path) -> list[str]:
    if not runs_dir.is_dir():
        return []
    return sorted(
        p.name
        for p in runs_dir.iterdir()
        if p.is_dir()
        and RUN_NAME.match(p.name)
        and p.name not in RESERVED
        and ((p / "config.yaml").exists() or (p / "status.json").exists())
    )


def read_config(path: Path) -> dict[str, Any] | None:
    try:
        cfg = yaml.safe_load((path / "config.yaml").read_text())
    except (OSError, yaml.YAMLError):
        return None
    return cfg if isinstance(cfg, dict) else None


def read_status(path: Path) -> dict[str, Any] | None:
    """status.json, with "state" set to "crashed" if the process died without saying so."""
    status = read_json(path / "status.json")
    if not isinstance(status, dict):
        return None
    if status.get("state") in LIVE and not process_running(status.get("pid"), "gpt_lab.train"):
        status = {**status, "state": "crashed"}
    return status


# -- metrics.jsonl ------------------------------------------------------------------------
def read_metrics(path: Path, cursor: str | None = None) -> dict[str, Any]:
    """Rows of metrics.jsonl added since `cursor` (returned by the previous call).

    A resumed run rewrites the file, which gives it a new inode. The cursor notices, and
    every row is returned again with "reset": true.
    """
    try:
        f = open(path, "rb")
    except FileNotFoundError:
        return {"rows": [], "cursor": None, "reset": cursor is not None}
    with f:
        st = os.fstat(f.fileno())
        inode, offset = _parse_cursor(cursor)
        reset = inode is not None and (inode != st.st_ino or offset > st.st_size)
        if reset or inode is None:
            offset = 0
        f.seek(offset)
        data = f.read()
    end = data.rfind(b"\n") + 1  # whole lines only; the last one may still be being written
    rows = []
    for line in data[:end].splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return {"rows": rows, "cursor": f"{st.st_ino}:{offset + end}", "reset": reset}


def _parse_cursor(cursor: str | None) -> tuple[int | None, int]:
    try:
        inode, offset = cursor.split(":")  # type: ignore[union-attr]
        return int(inode), int(offset)
    except (AttributeError, ValueError):
        return None, 0


def metrics_summary(path: Path) -> dict[str, Any]:
    """The latest training numbers, evaluation results and text samples in metrics.jsonl."""
    try:
        st = path.stat()
    except FileNotFoundError:
        return {}
    key = (st.st_ino, st.st_size, st.st_mtime_ns)
    cached = _summary_cache.get(str(path))
    if cached and cached[0] == key:
        return cached[1]

    train: dict[str, Any] = {}
    evals: dict[str, Any] = {}
    samples: dict[str, dict[str, Any]] = {}
    best_val = None
    for row in read_metrics(path)["rows"]:
        if "train/loss" in row:
            train = row
        for k, v in row.items():
            if k.startswith("eval/") and v is not None:
                evals[k] = v
            elif k.startswith("samples/"):
                samples[k] = {"tag": k, "step": row.get("step"), "text": v}
        val = row.get("eval/val_loss")
        if val is not None and (best_val is None or val < best_val):
            best_val = val
    summary = {
        "logged_step": train.get("step"),
        "loss": train.get("train/loss"),
        "lr": train.get("train/lr"),
        "tokens_per_sec": train.get("perf/tokens_per_sec"),
        "peak_mem_gb": train.get("perf/peak_mem_gb"),
        "val_loss": evals.get("eval/val_loss"),
        "best_val_loss": best_val,
        "hellaswag": evals.get("eval/hellaswag_acc_norm"),
        "samples": sorted(samples.values(), key=lambda s: s["tag"]),
    }
    _summary_cache[str(path)] = (key, summary)
    return summary


# -- logs and checkpoints -----------------------------------------------------------------
def tail_text(path: Path, lines: int = 200, max_bytes: int = 256 * 1024) -> str:
    """The last `lines` lines of a log file.

    Progress bars redraw themselves with carriage returns; only the latest redraw of
    each line is kept.
    """
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read()
    except OSError:
        return ""
    out = [
        line.rstrip("\r").rsplit("\r", 1)[-1] for line in data.decode(errors="replace").split("\n")
    ]
    if size > max_bytes:
        out = out[1:]  # the first line was probably cut in half
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out[-lines:])


def checkpoints(path: Path) -> list[dict[str, Any]]:
    out = []
    for step, file in list_checkpoints(path / "checkpoints"):
        try:
            st = file.stat()
        except OSError:
            continue
        out.append({"step": step, "file": file.name, "bytes": st.st_size, "mtime": st.st_mtime})
    return out


# -- whole runs ---------------------------------------------------------------------------
def summarize(path: Path, active_job: dict[str, Any] | None = None) -> dict[str, Any]:
    """What the runs list shows for one run."""
    status = read_status(path)
    cfg = read_config(path) or {}
    tc, mc = cfg.get("train") or {}, cfg.get("model") or {}
    m = metrics_summary(path / "metrics.jsonl")

    if status:
        state = status["state"]
    else:  # a run started by the UI whose process hasn't written status.json yet
        state = "starting" if active_job else "unknown"
    step = (status or {}).get("step") or m.get("logged_step") or 0
    max_steps = (status or {}).get("max_steps") or tc.get("max_steps")
    end_step = (status or {}).get("end_step") or max_steps
    tokens_per_step = tc.get("total_batch_tokens")
    eta = None
    if state in LIVE and m.get("tokens_per_sec") and end_step and tokens_per_step:
        eta = max(0, end_step - step) * tokens_per_step / m["tokens_per_sec"]
    return {
        "name": path.name,
        "state": state,
        "step": step,
        "max_steps": max_steps,
        "end_step": end_step,
        "tokens": step * tokens_per_step if tokens_per_step else None,
        "total_tokens": max_steps * tokens_per_step if tokens_per_step and max_steps else None,
        "tokens_per_step": tokens_per_step,
        "eta_seconds": eta,
        "started_at": (status or {}).get("started_at"),
        "updated_at": (status or {}).get("updated_at") or path.stat().st_mtime,
        "last_checkpoint": (status or {}).get("last_checkpoint"),
        "error": (status or {}).get("error"),
        "pid": (status or {}).get("pid"),
        "device": tc.get("device"),
        "model": {k: mc.get(k) for k in ("n_layer", "n_head", "d_model", "seq_len", "vocab_size")},
        "job": active_job["id"] if active_job else None,
        "has_eval": (path / "eval.json").exists(),
        **{
            k: m.get(k)
            for k in ("loss", "val_loss", "best_val_loss", "hellaswag", "tokens_per_sec")
        },
    }


def detail(path: Path, active_job: dict[str, Any] | None = None) -> dict[str, Any]:
    """Everything the run page shows apart from the metric rows and the log."""
    commit = ""
    try:
        commit = (path / "git_commit.txt").read_text().strip()
    except OSError:
        pass
    m = metrics_summary(path / "metrics.jsonl")
    return {
        **summarize(path, active_job),
        "config": read_config(path),
        "checkpoints": checkpoints(path),
        "eval": read_json(path / "eval.json"),
        "commit": commit,
        "samples": m.get("samples", []),
        "lr": m.get("lr"),
        "peak_mem_gb": m.get("peak_mem_gb"),
        "has_log": (path / "train.log").exists(),
    }


def compare_rows(runs_dir: Path) -> dict[str, Any]:
    """Every evaluation result: runs/<name>/eval.json and runs/reference/*.json."""
    files = sorted(runs_dir.glob("*/eval.json")) + sorted((runs_dir / "reference").glob("*.json"))
    rows = []
    for f in files:
        r = read_json(f)
        if not isinstance(r, dict):
            continue
        reference = f.parent.name == "reference"
        lm = r.get("lm_eval") or {}
        rows.append(
            {
                "name": r.get("name") or f.parent.name,
                "run": None if reference else f.parent.name,
                "reference": reference,
                "params": r.get("params"),
                "step": r.get("step"),
                "val_loss": r.get("val_loss"),
                "hellaswag": r.get("hellaswag_acc_norm"),
                "hellaswag_n": r.get("hellaswag_n"),
                "lm_eval": {
                    task: lm_eval_score(v) for task, v in lm.items() if isinstance(v, dict)
                },
                "mtime": f.stat().st_mtime,
            }
        )
    tasks = sorted({t for r in rows for t in r["lm_eval"]})
    return {"rows": rows, "tasks": tasks}
