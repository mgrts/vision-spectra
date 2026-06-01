#!/usr/bin/env python3
"""PostToolUse hook (Edit|Write|MultiEdit): ruff-fix + ruff-format an edited .py file.

Keeps the working tree consistent with the repo's ruff config (line-length 99, the lint
rule set in pyproject.toml) so diffs stay clean and `git commit` never trips the ruff
pre-commit/CI hooks. Always exits 0 — formatting is best-effort and never blocks the edit.
"""

import json
import os
import subprocess
import sys


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0

    file_path = (data.get("tool_input") or {}).get("file_path", "")
    if not file_path or not file_path.endswith(".py"):
        return 0

    project_dir = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    abs_file = os.path.abspath(file_path)
    abs_proj = os.path.abspath(project_dir)
    try:
        inside = os.path.commonpath([abs_file, abs_proj]) == abs_proj
    except ValueError:
        inside = False
    if not inside or not os.path.isfile(abs_file):
        return 0

    ruff = os.path.join(project_dir, ".venv", "bin", "ruff")
    ruff = ruff if os.path.exists(ruff) else "ruff"

    applied = []
    for label, args in (
        ("check --fix", ["check", "--fix", "--quiet"]),
        ("format", ["format"]),
    ):
        try:
            subprocess.run(
                [ruff, *args, abs_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            applied.append(label)
        except FileNotFoundError:
            pass

    if applied:
        print(
            f"[auto-format] ruff {' + ruff '.join(applied)} applied to "
            f"{os.path.relpath(abs_file, abs_proj)}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
