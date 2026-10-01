"""HTTP API and pages for the GPT-Lab UI.

Everything the UI shows is read from the GPT-Lab checkout (configs/, data/, runs/), and
everything it starts runs as a job (see jobs.py). The server keeps no state of its own,
so it can be restarted at any time without affecting running jobs.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Literal, NoReturn

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from gpt_lab.config import save_config
from gpt_lab.ui import configs, datasets, runs
from gpt_lab.ui.jobs import JobError, JobManager, process_running
from gpt_lab.ui.sampling import Sampler

STATIC = Path(__file__).parent / "static"
HF_MODELS = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
LM_EVAL_TASKS = ["arc_easy", "arc_challenge", "piqa", "winogrande", "lambada_openai"]


class RunRequest(BaseModel):
    preset: str
    run_name: str
    overrides: dict[str, Any] = {}


class PrepareRequest(BaseModel):
    dataset: Literal["tiny", "fineweb"]
    max_tokens: int | None = Field(None, gt=0)
    overwrite: bool = False


class EvaluateRequest(BaseModel):
    run: str | None = None
    hf: str | None = None
    val_loss: bool = True
    hellaswag: bool = True
    hellaswag_limit: int | None = Field(None, gt=0)
    lm_eval: list[str] = []
    batch_size: int = Field(16, gt=0, le=256)


class GenerateRequest(BaseModel):
    run: str
    checkpoint: str | None = None
    prompt: str = Field(..., min_length=1, max_length=4000)
    num_samples: int = Field(1, ge=1, le=5)
    max_new_tokens: int = Field(96, ge=1, le=512)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: int | None = Field(50, ge=1)
    top_p: float | None = Field(None, gt=0.0, le=1.0)
    seed: int = 0


class ShutdownRequest(BaseModel):
    force: bool = False


def fail(status: int, message: str) -> NoReturn:
    raise HTTPException(status, message)


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None  # nvidia-smi prints "[N/A]" for unsupported fields


def create_app(root: Path) -> FastAPI:
    root = Path(root).resolve()
    runs_dir = root / "runs"
    python = sys.executable
    cuda = torch.cuda.is_available()

    def uses_gpu(cfg: dict[str, Any] | None) -> bool:
        device = ((cfg or {}).get("train") or {}).get("device", "auto")
        return str(device).startswith("cuda") or (device == "auto" and cuda)

    def external_gpu_user() -> str | None:
        """A training run on the GPU that the UI didn't start (from a terminal, say)."""
        for name in runs.run_names(runs_dir):
            path = runs_dir / name
            status = runs.read_status(path)
            if status and status["state"] in runs.LIVE and uses_gpu(runs.read_config(path)):
                return f"training run {name}"
        return None

    jobs = JobManager(root, external_gpu_user)
    sampler = Sampler()
    app = FastAPI(title="GPT-Lab", docs_url=None, redoc_url=None, openapi_url=None)
    # Only answer requests addressed to this machine (protects against DNS rebinding).
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])
    app.state.server = None  # set by __main__, so /api/shutdown can stop uvicorn
    app.state.jobs = jobs

    def get_run(name: str) -> Path:
        try:
            return runs.run_dir(runs_dir, name)
        except KeyError:
            fail(404, f"There's no run named {name}.")

    def start_job(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return jobs.start(*args, **kwargs)
        except JobError as e:
            fail(409, str(e))

    def train_cmd(path: Path, resume: bool) -> list[str]:
        cmd = [python, "-m", "gpt_lab.train", "--config", str(path / "config.yaml")]
        return cmd + ["--resume"] if resume else cmd

    # -- server and machine ---------------------------------------------------------------
    gpu_cache: dict[str, Any] = {"at": 0.0, "value": None}

    def gpu_info() -> dict[str, Any] | None:
        if time.time() - gpu_cache["at"] < 2.0:
            return gpu_cache["value"]
        value = None
        query = "name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
        try:
            out = subprocess.run(
                ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.splitlines()[0]
            name, util, used, total, temp, power = (s.strip() for s in out.split(","))
            value = {
                "name": name,
                "util": _number(util),
                "mem_used_mb": _number(used),
                "mem_total_mb": _number(total),
                "temp_c": _number(temp),
                "power_w": _number(power),
            }
        except (OSError, subprocess.SubprocessError, IndexError, ValueError):
            pass
        gpu_cache.update(at=time.time(), value=value)
        return value

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "root": str(root), "cuda": cuda}

    @app.get("/api/system")
    def system() -> dict[str, Any]:
        mem = {}
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, value = line.split(":", 1)
                if key in ("MemTotal", "MemAvailable"):
                    mem[key] = int(value.split()[0]) * 1024
        except (OSError, ValueError):
            pass
        disk = shutil.disk_usage(root)
        return {
            "gpu": gpu_info(),
            "gpu_user": jobs.gpu_user(),
            "memory": {"total": mem.get("MemTotal"), "available": mem.get("MemAvailable")},
            "load": os.getloadavg()[0],
            "cpus": os.cpu_count(),
            "disk": {"free": disk.free, "total": disk.total},
            "running_jobs": len(jobs.running()),
        }

    @app.post("/api/shutdown")
    def shutdown(req: ShutdownRequest) -> dict[str, Any]:
        running = jobs.running()
        if running and not req.force:
            names = ", ".join(j["title"] for j in running)
            fail(
                409,
                f"Still running: {names}. Unless an Ubuntu terminal is open, WSL shuts down "
                "within seconds of the server stopping, and that stops these too. Stop them "
                "first (training saves a checkpoint), or shut down anyway.",
            )
        if app.state.server is None:
            fail(
                503,
                "This server wasn't started with python -m gpt_lab.ui, so it can't stop itself.",
            )
        app.state.server.should_exit = True
        return {"stopping": True}

    # -- presets and new runs -------------------------------------------------------------
    @app.get("/api/presets")
    def presets() -> list[dict[str, Any]]:
        return configs.presets(root)

    @app.get("/api/schema")
    def schema() -> dict[str, Any]:
        return configs.schema()

    def check_request(req: RunRequest, new: bool) -> tuple[Any, dict[str, Any]]:
        try:
            cfg = configs.build(root, req.preset, req.overrides, req.run_name)
        except (KeyError, ValueError, TypeError) as e:
            report = {"errors": [configs.describe_error(e)], "warnings": [], "derived": None}
            return None, {**report, "config": None}
        return cfg, {**configs.check(root, cfg, req.run_name, new=new), "config": cfg.to_dict()}

    @app.post("/api/runs/check")
    def check_run(req: RunRequest) -> dict[str, Any]:
        return check_request(req, new=True)[1]

    @app.post("/api/runs")
    def start_run(req: RunRequest) -> dict[str, Any]:
        cfg, report = check_request(req, new=True)
        if report["errors"]:
            fail(400, report["errors"][0])
        path = runs_dir / req.run_name
        path.mkdir(parents=True)
        save_config(cfg, path / "config.yaml")
        try:
            job = jobs.start(
                "train",
                f"Training {req.run_name}",
                train_cmd(path, resume=False),
                gpu=report["derived"]["device"].startswith("cuda"),
                log=path / "train.log",
                run=req.run_name,
                exclusive=f"run:{req.run_name}",
            )
        except JobError as e:
            shutil.rmtree(path)  # only holds the config written above; frees the name again
            fail(409, str(e))
        return {"run": req.run_name, "job": job}

    # -- runs -----------------------------------------------------------------------------
    @app.get("/api/runs")
    def list_runs() -> list[dict[str, Any]]:
        active = jobs.active_by_run()
        out = [runs.summarize(runs_dir / n, active.get(n)) for n in runs.run_names(runs_dir)]
        return sorted(out, key=lambda r: r["updated_at"] or 0, reverse=True)

    @app.get("/api/runs/{name}")
    def get_run_detail(name: str) -> dict[str, Any]:
        path = get_run(name)
        out = runs.detail(path, jobs.active_by_run().get(name))
        cfg = out["config"]
        if cfg and cfg.get("model"):
            try:
                out["model_stats"] = configs.model_stats(configs.ModelConfig(**cfg["model"]))
            except (AssertionError, TypeError, ValueError, RuntimeError):
                out["model_stats"] = None
        return out

    @app.get("/api/runs/{name}/metrics")
    def run_metrics(name: str, cursor: str | None = None) -> dict[str, Any]:
        return runs.read_metrics(get_run(name) / "metrics.jsonl", cursor)

    @app.get("/api/runs/{name}/log")
    def run_log(name: str, lines: int = 300) -> dict[str, Any]:
        return {"text": runs.tail_text(get_run(name) / "train.log", min(max(lines, 1), 2000))}

    @app.post("/api/runs/{name}/resume")
    def resume_run(name: str) -> dict[str, Any]:
        path = get_run(name)
        summary = runs.summarize(path, jobs.active_by_run().get(name))
        if summary["state"] in runs.LIVE:
            fail(409, f"{name} is already running.")
        if summary["state"] == "finished":
            fail(409, f"{name} has already finished all {summary['max_steps']:,} steps.")
        if not runs.checkpoints(path):
            fail(409, f"{name} has no checkpoint to resume from.")
        job = start_job(
            "train",
            f"Training {name}",
            train_cmd(path, resume=True),
            gpu=uses_gpu(runs.read_config(path)),
            log=path / "train.log",
            run=name,
            exclusive=f"run:{name}",
        )
        return {"run": name, "job": job}

    @app.post("/api/runs/{name}/stop")
    def stop_run(name: str) -> dict[str, Any]:
        path = get_run(name)
        job = jobs.active_by_run().get(name)
        if job:
            try:
                jobs.stop(job["id"])
            except JobError as e:
                fail(409, str(e))
            return {"stopping": True}
        status = runs.read_status(path)
        if (
            status
            and status["state"] in runs.LIVE
            and process_running(status["pid"], "gpt_lab.train")
        ):
            os.kill(status["pid"], signal.SIGTERM)  # a run started outside the UI
            return {"stopping": True}
        fail(409, f"{name} isn't running.")

    @app.get("/api/checkpoints")
    def all_checkpoints() -> list[dict[str, Any]]:
        out = []
        for name in runs.run_names(runs_dir):
            cks = runs.checkpoints(runs_dir / name)
            if cks:
                out.append({"run": name, "checkpoints": cks})
        return out

    # -- jobs -----------------------------------------------------------------------------
    def get_job(job_id: str) -> dict[str, Any]:
        try:
            return jobs.get(job_id)
        except KeyError:
            fail(404, f"There's no job {job_id}.")

    @app.get("/api/jobs")
    def list_jobs(kind: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        out = [j for j in jobs.list() if kind is None or j["kind"] == kind]
        return out[: max(1, min(limit, 500))]

    @app.get("/api/jobs/{job_id}")
    def job_detail(job_id: str) -> dict[str, Any]:
        return get_job(job_id)

    @app.get("/api/jobs/{job_id}/log")
    def job_log(job_id: str, lines: int = 300) -> dict[str, Any]:
        job = get_job(job_id)
        return {"text": runs.tail_text(Path(job["log"]), min(max(lines, 1), 2000))}

    @app.post("/api/jobs/{job_id}/stop")
    def stop_job(job_id: str) -> dict[str, Any]:
        get_job(job_id)
        try:
            return jobs.stop(job_id)
        except JobError as e:
            fail(409, str(e))

    @app.post("/api/jobs/{job_id}/kill")
    def kill_job(job_id: str) -> dict[str, Any]:
        get_job(job_id)
        try:
            return jobs.kill(job_id)
        except JobError as e:
            fail(409, str(e))

    # -- datasets -------------------------------------------------------------------------
    @app.get("/api/data")
    def data() -> dict[str, Any]:
        return datasets.summary(root)

    @app.post("/api/data/prepare")
    def prepare(req: PrepareRequest) -> dict[str, Any]:
        info = datasets.PREPARABLE[req.dataset]
        out_dir = info["dir"]
        if req.dataset == "tiny":
            cmd = [python, "scripts/prepare_tiny.py", "--out-dir", out_dir]
        else:
            cmd = [python, "scripts/prepare_fineweb_edu.py", "--out-dir", out_dir]
            if req.max_tokens:
                cmd += ["--max-tokens", str(req.max_tokens)]
        for name in runs.run_names(runs_dir):
            status = runs.read_status(runs_dir / name)
            data_cfg = (runs.read_config(runs_dir / name) or {}).get("data") or {}
            patterns = (data_cfg.get("train_pattern", ""), data_cfg.get("val_pattern", ""))
            if (
                status
                and status["state"] in runs.LIVE
                and any(p.startswith(out_dir) for p in patterns)
            ):
                fail(
                    409,
                    f"{name} is training on {out_dir}. Stop it before preparing the data again.",
                )
        existing = sorted((root / out_dir).glob("*.npy"))
        if existing and not req.overwrite:
            fail(
                409,
                f"{out_dir} already has {len(existing)} shards. Tick “Replace the existing "
                "shards” to prepare it again.",
            )
        if any(j.get("exclusive") == "prepare" for j in jobs.running()):
            fail(409, "A dataset is already being prepared.")
        for f in existing:  # a smaller new dataset must not leave old shards behind
            f.unlink()
        return start_job("prepare", f"Preparing {info['title']}", cmd, exclusive="prepare")

    # -- evaluation -----------------------------------------------------------------------
    @app.post("/api/evaluate")
    def evaluate(req: EvaluateRequest) -> dict[str, Any]:
        if (req.run is None) == (req.hf is None):
            fail(400, "Choose one model to evaluate.")
        if not (req.val_loss or req.hellaswag or req.lm_eval):
            fail(400, "Choose at least one benchmark.")
        unknown = [t for t in req.lm_eval if t not in LM_EVAL_TASKS]
        if unknown:
            fail(400, f"Unknown benchmark: {', '.join(unknown)}.")
        cmd = [python, "-m", "gpt_lab.evaluate", "--batch-size", str(req.batch_size)]
        val_pattern = "data/fineweb_edu_10B/val_*.npy"
        if req.run:
            path = get_run(req.run)
            if not runs.checkpoints(path):
                fail(409, f"{req.run} has no checkpoint to evaluate yet.")
            val_pattern = ((runs.read_config(path) or {}).get("data") or {}).get(
                "val_pattern", val_pattern
            )
            cmd += ["--checkpoint", str(path)]
            title = f"Evaluating {req.run}"
        else:
            if req.hf not in HF_MODELS:
                fail(400, f"Choose one of {', '.join(HF_MODELS)}.")
            cmd += ["--hf", req.hf]
            title = f"Evaluating OpenAI {req.hf}"
        if req.val_loss:
            if not configs.shards(root, val_pattern):
                fail(
                    409,
                    f"No validation shards match {val_pattern}. Prepare the dataset first, "
                    "or leave out validation loss.",
                )
            cmd += ["--val-pattern", val_pattern]
        else:
            cmd.append("--skip-val")
        if not req.hellaswag:
            cmd.append("--skip-hellaswag")
        elif req.hellaswag_limit:
            cmd += ["--hellaswag-limit", str(req.hellaswag_limit)]
        if req.lm_eval:
            cmd += ["--lm-eval", ",".join(req.lm_eval)]
        return start_job("evaluate", title, cmd, gpu=cuda)

    @app.get("/api/compare")
    def compare() -> dict[str, Any]:
        return runs.compare_rows(runs_dir)

    # -- text generation ------------------------------------------------------------------
    @app.post("/api/generate")
    def generate(req: GenerateRequest) -> dict[str, Any]:
        path = get_run(req.run)
        files = [c["file"] for c in runs.checkpoints(path)]
        if not files:
            fail(409, f"{req.run} has no checkpoint yet.")
        file = req.checkpoint or files[-1]
        if file not in files:
            fail(404, f"{req.run} has no checkpoint {file}.")
        try:
            result = sampler.sample(
                path / "checkpoints" / file,
                req.prompt,
                num_samples=req.num_samples,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_k=req.top_k,
                top_p=req.top_p,
                seed=req.seed,
                use_gpu=cuda and jobs.gpu_user() is None,
            )
        except ValueError as e:
            fail(400, str(e))
        return {**result, "run": req.run, "checkpoint": file}

    # -- pages ----------------------------------------------------------------------------
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> HTMLResponse:
        # Version the asset URLs so the browser picks up edited files.
        version = max(int(p.stat().st_mtime) for p in STATIC.iterdir())
        html = (STATIC / "index.html").read_text().replace("{{v}}", str(version))
        return HTMLResponse(html, headers={"Cache-Control": "no-cache"})

    return app
