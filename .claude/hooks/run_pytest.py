#!/usr/bin/env python3
"""Stop hook: run the pytest suite when the turn touched Python source.

Runs only if there are uncommitted changes to .py files under vision_spectra/ or
tests/. On failure it blocks the stop once (exit 2) so the agent sees and addresses
the failures; to avoid loops it does NOT re-block if it was itself the cause of the
previous stop (stop_hook_active). Uses `pytest -x` for fast first-failure feedback.

Disable by removing the "Stop" entry from .claude/settings.json.
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gitutil import status_paths  # noqa: E402


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        data = {}

    # Avoid loops: if the previous stop was already triggered by this hook, let it stop.
    if data.get("stop_hook_active"):
        return 0

    project_dir = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()

    touched = [
        p
        for p in status_paths(project_dir)
        if p.endswith(".py") and (p.startswith("vision_spectra/") or p.startswith("tests/"))
    ]
    if not touched:
        return 0

    pytest = os.path.join(project_dir, ".venv", "bin", "pytest")
    pytest = pytest if os.path.exists(pytest) else "pytest"
    proc = subprocess.run(
        [pytest, "-q", "-x", "--no-header", "-p", "no:cacheprovider"],
        cwd=project_dir,
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        print("[run-pytest] suite passed.")
        return 0

    tail = (proc.stdout + "\n" + proc.stderr).strip().splitlines()
    sys.stderr.write("⛔ pytest is failing after your changes (auto-run on stop):\n")
    sys.stderr.write("\n".join(tail[-25:]) + "\n")
    sys.stderr.write(
        '\nFix the failing test(s) before finishing, or remove the "Stop" hook in '
        ".claude/settings.json to disable this gate.\n"
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
