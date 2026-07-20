from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def git_provenance(project_root: Path) -> dict[str, Any]:
    """Return explicit Git state, including repositories without an initial commit."""

    def run(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(project_root), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )

    revision_result = run("rev-parse", "HEAD")
    revision = revision_result.stdout.strip() if revision_result.returncode == 0 else "UNBORN"
    status_result = run("status", "--porcelain", "--untracked-files=all")
    if status_result.returncode != 0:
        return {"revision": revision, "dirty": None, "status_available": False}
    return {
        "revision": revision,
        "dirty": bool(status_result.stdout.strip()),
        "status_available": True,
    }
