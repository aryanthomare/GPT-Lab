#!/usr/bin/env bash
# Start the GPT-Lab UI with this checkout's virtual environment, logging to runs/.ui-server.log.
# The Windows launcher (scripts/windows/Start-GPTLabUI.ps1) runs this through wsl.exe.
cd "$(dirname "$0")/.." || exit 1
mkdir -p runs
exec .venv/bin/python -m gpt_lab.ui "$@" >> runs/.ui-server.log 2>&1
