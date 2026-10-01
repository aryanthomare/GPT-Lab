"""Background jobs started by the UI: training runs, dataset prep and evaluations.

Every job runs under gpt_lab.ui.runner in a session of its own, so it keeps running when
the web server restarts. A job's files live in runs/.jobs/<id>/: job.json (what was
started), result.json (how it ended, written by the runner) and log.txt, unless the job
logs somewhere else. A job's state is always worked out again from those files and the
process table, so the server keeps nothing in memory that it can't rebuild.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

RUNNING, SUCCEEDED, FAILED, STOPPED, LOST = "running", "succeeded", "failed", "stopped", "lost"
_JOB_ID = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{4}$")


class JobError(Exception):
    """A job can't be started or changed. The message says why, for the person using the UI."""


def process_running(pid: int | None, needle: str) -> bool:
    """True if process `pid` is alive and its command line contains `needle`.

    Checking the command line guards against a recycled PID. A zombie has an empty
    command line, so it counts as not running.
    """
    if not pid:
        return False
    try:
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    return needle.encode() in cmdline


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


class JobManager:
    def __init__(self, root: Path, external_gpu_user: Callable[[], str | None] | None = None):
        """root: the GPT-Lab checkout. external_gpu_user: describes a GPU user that isn't
        a UI job (such as a training run started from a terminal), or returns None."""
        self.root = root
        self.dir = root / "runs" / ".jobs"
        self._external_gpu_user = external_gpu_user
        self._lock = threading.Lock()
        self._runners: list[subprocess.Popen] = []

    # -- starting -------------------------------------------------------------------------
    def start(
        self,
        kind: str,
        title: str,
        cmd: list[str],
        *,
        gpu: bool = False,
        log: Path | None = None,
        run: str | None = None,
        exclusive: str | None = None,
    ) -> dict[str, Any]:
        """Start `cmd`, run from the GPT-Lab root, as a job.

        gpu: refuse to start while another job or run is using the GPU.
        exclusive: refuse to start while a running job has the same key.
        """
        with self._lock:
            running = self.running()
            if gpu:
                user = self.gpu_user(running)
                if user:
                    raise JobError(
                        f"The GPU is busy with {user}. Stop it or wait for it to finish."
                    )
            if exclusive:
                clash = next((j for j in running if j.get("exclusive") == exclusive), None)
                if clash:
                    raise JobError(f"{clash['title']} is already running.")

            job_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"
            job_dir = self.dir / job_id
            job_dir.mkdir(parents=True)
            log = log or job_dir / "log.txt"
            job = {
                "id": job_id,
                "kind": kind,
                "title": title,
                "cmd": cmd,
                "gpu": gpu,
                "run": run,
                "exclusive": exclusive,
                "log": str(log),
                "started_at": time.time(),
                "pid": None,
                "stop_requested_at": None,
            }
            runner = subprocess.Popen(
                [sys.executable, "-m", "gpt_lab.ui.runner", str(job_dir), str(log), "--", *cmd],
                cwd=self.root,
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._runners.append(runner)
            job["pid"] = runner.pid
            write_json(job_dir / "job.json", job)
            return self._with_state(job, job_dir)

    def gpu_user(self, running: list[dict[str, Any]] | None = None) -> str | None:
        """Describe what is using the GPU, or return None if it's free."""
        running = self.running() if running is None else running
        job = next((j for j in running if j["gpu"]), None)
        if job:
            return job["title"]
        return self._external_gpu_user() if self._external_gpu_user else None

    # -- reading --------------------------------------------------------------------------
    def list(self) -> list[dict[str, Any]]:
        """Every job, newest first."""
        self._runners = [r for r in self._runners if r.poll() is None]  # reap finished runners
        jobs = []
        if self.dir.is_dir():
            for job_dir in sorted(self.dir.iterdir(), reverse=True):
                job = read_json(job_dir / "job.json")
                if isinstance(job, dict):
                    jobs.append(self._with_state(job, job_dir))
        return jobs

    def running(self) -> list[dict[str, Any]]:
        return [j for j in self.list() if j["state"] == RUNNING]

    def get(self, job_id: str) -> dict[str, Any]:
        job_dir = self._dir_for(job_id)
        job = read_json(job_dir / "job.json")
        if not isinstance(job, dict):
            raise KeyError(job_id)
        return self._with_state(job, job_dir)

    def active_by_run(self) -> dict[str, dict[str, Any]]:
        """Running jobs keyed by the training run they belong to."""
        return {j["run"]: j for j in self.running() if j["run"]}

    # -- stopping -------------------------------------------------------------------------
    def stop(self, job_id: str) -> dict[str, Any]:
        """Ask a job to stop with SIGTERM. The runner passes it on, and training saves a
        checkpoint before it exits."""
        job = self._mark_stop_requested(job_id)
        try:
            os.kill(job["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
        return self.get(job_id)

    def kill(self, job_id: str) -> dict[str, Any]:
        """Stop a job and every process it started, immediately. Nothing is saved."""
        job = self._mark_stop_requested(job_id)
        try:
            os.killpg(job["pid"], signal.SIGKILL)  # the runner leads its own process group
        except ProcessLookupError:
            pass
        return self.get(job_id)

    def _mark_stop_requested(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self.get(job_id)
            if job["state"] != RUNNING:
                raise JobError(f"{job['title']} isn't running.")
            path = self._dir_for(job_id) / "job.json"
            raw = read_json(path)
            raw["stop_requested_at"] = raw.get("stop_requested_at") or time.time()
            write_json(path, raw)
            return job

    # -- helpers --------------------------------------------------------------------------
    def _dir_for(self, job_id: str) -> Path:
        if not _JOB_ID.match(job_id):
            raise KeyError(job_id)
        return self.dir / job_id

    @staticmethod
    def _with_state(job: dict[str, Any], job_dir: Path) -> dict[str, Any]:
        result = read_json(job_dir / "result.json")
        stop_requested = bool(job.get("stop_requested_at"))
        if isinstance(result, dict):
            code = result.get("exit_code")
            state = STOPPED if stop_requested else SUCCEEDED if code == 0 else FAILED
            return {
                **job,
                "state": state,
                "exit_code": code,
                "finished_at": result.get("finished_at"),
            }
        if process_running(job.get("pid"), str(job_dir)):
            return {**job, "state": RUNNING, "exit_code": None, "finished_at": None}
        # The runner died without recording a result: killed, or WSL shut down under it.
        state = STOPPED if stop_requested else LOST
        return {**job, "state": state, "exit_code": None, "finished_at": None}
