"""Safe launcher that calls existing workflow scripts without modifying them."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]


def launch_integrated_workflow(config_path: str | Path, *, force_recompute: bool = False) -> dict[str, Any]:
    """Start the existing integrated workflow as a subprocess."""
    cfg = Path(config_path)
    if not cfg.is_file():
        raise FileNotFoundError(f"Config not found: {cfg}")
    cmd = [sys.executable, str(ROOT / "scripts" / "run_integrated_tf_eager_workflow.py"), "--config", str(cfg)]
    if force_recompute:
        cmd.append("--force-recompute")
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return {"pid": proc.pid, "cmd": cmd, "process": proc}


def launch_logged(
    cmd: list[str],
    log_path: str | Path,
    *,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Start a command and append stdout/stderr to a log file."""
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("a", encoding="utf-8")
    handle.write("\n$ " + " ".join(cmd) + "\n")
    handle.flush()
    merged = os.environ.copy()
    if env:
        merged.update({key: value for key, value in env.items() if value})
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=merged,
    )
    return {"pid": proc.pid, "cmd": cmd, "log": str(log)}
