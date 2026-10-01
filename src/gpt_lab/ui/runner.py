"""Run one job command, record how it ended, and pass stop signals on to it.

    python -m gpt_lab.ui.runner <job_dir> <log_file> -- <command ...>

The UI starts every job through this runner in a session of its own, so jobs keep running
when the web server restarts. When the command exits, the runner writes
<job_dir>/result.json with its exit code.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) < 4 or argv[2] != "--":
        print(__doc__, file=sys.stderr)
        return 2
    job_dir, log_path, cmd = Path(argv[0]), Path(argv[1]), argv[3:]

    child: subprocess.Popen | None = None

    def forward(signum, frame) -> None:
        if child is not None:
            try:
                child.send_signal(signum)
            except ProcessLookupError:
                pass

    # Install before starting the command, so an early stop request is never lost.
    signal.signal(signal.SIGTERM, forward)
    signal.signal(signal.SIGINT, forward)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as log:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log.write(f"\n=== {stamp}  $ {shlex.join(cmd)}\n".encode())
        log.flush()
        child = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    code = child.wait()

    tmp = job_dir / "result.json.tmp"
    tmp.write_text(json.dumps({"exit_code": code, "finished_at": time.time()}) + "\n")
    os.replace(tmp, job_dir / "result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
