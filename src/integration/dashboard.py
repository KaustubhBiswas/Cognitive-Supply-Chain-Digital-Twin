"""CLI entrypoint for launching the Streamlit dashboard."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    """Launch Streamlit with the project dashboard app."""
    project_root = Path(__file__).resolve().parents[2]
    app_path = project_root / "app.py"

    if not app_path.exists():
        print(f"Dashboard app not found at {app_path}", file=sys.stderr)
        return 1

    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
