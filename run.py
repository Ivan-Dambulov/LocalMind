#!/usr/bin/env python3
"""One-command launcher: python run.py"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> int:
    os.chdir(ROOT)
    app = ROOT / "src" / "ui" / "app.py"
    if not app.exists():
        print(f"UI not found: {app}")
        return 1

    print("========================================")
    print("  LocalMind")
    print("========================================")
    print("Starting Streamlit… (Ctrl+C to stop)\n")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.headless",
        "false",
    ]
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        print("Streamlit not installed. Run:  pip install streamlit")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
