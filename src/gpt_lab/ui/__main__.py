"""Serve the GPT-Lab UI.

    python -m gpt_lab.ui [--port 8000] [--root .]

Then open http://localhost:8000. WSL forwards localhost, so the Windows browser reaches
it too. scripts/windows/Start-GPTLabUI.ps1 starts it from Windows and keeps WSL running.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--host", default="127.0.0.1", help="address to listen on (keep it local)")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--root", default=".", help="the GPT-Lab checkout (configs/, data/, runs/)")
    args = p.parse_args(argv)

    import uvicorn

    from gpt_lab.ui.app import create_app

    app = create_app(Path(args.root))
    server = uvicorn.Server(
        uvicorn.Config(app, host=args.host, port=args.port, log_level="warning")
    )
    app.state.server = server
    print(
        f"GPT-Lab UI on http://localhost:{args.port}  (root {Path(args.root).resolve()})",
        flush=True,
    )
    server.run()


if __name__ == "__main__":
    main()
