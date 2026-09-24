"""Read existing workflow_runtime.json without modifying the integrated workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_workflow_runtime(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def summarize_stages(runtime: dict[str, Any]) -> list[dict[str, Any]]:
    stages = runtime.get("stages") or {}
    if not isinstance(stages, dict):
        return []
    rows: list[dict[str, Any]] = []
    for name, info in stages.items():
        if not isinstance(info, dict):
            continue
        outputs = info.get("outputs") or {}
        rows.append(
            {
                "stage": name,
                "status": info.get("status", "unknown"),
                "started_at": info.get("started_at"),
                "recorded_at": info.get("recorded_at"),
                "elapsed_seconds": info.get("elapsed_seconds"),
                "reuse_path": info.get("reuse_path"),
                "outputs": outputs if isinstance(outputs, dict) else {},
                "error": info.get("error") or info.get("message"),
            }
        )
    # Preserve chronological order when timestamps exist.
    rows.sort(key=lambda r: str(r.get("started_at") or r.get("recorded_at") or r["stage"]))
    return rows


def derive_handoffs(runtime: dict[str, Any]) -> list[dict[str, str]]:
    """Infer stage handoffs from recorded output paths in the existing runtime manifest."""
    handoffs: list[dict[str, str]] = []
    for row in summarize_stages(runtime):
        stage = str(row["stage"])
        status = str(row["status"])
        outputs = row.get("outputs") or {}
        if status == "skipped_reuse" and row.get("reuse_path"):
            handoffs.append(
                {
                    "stage": stage,
                    "event": "reuse",
                    "detail": f"Reused existing artifact: {row['reuse_path']}",
                }
            )
        for key, value in outputs.items():
            handoffs.append(
                {
                    "stage": stage,
                    "event": "handoff",
                    "detail": f"Produced {key} -> {value}",
                }
            )
        if status == "failed" and row.get("error"):
            handoffs.append(
                {
                    "stage": stage,
                    "event": "failure",
                    "detail": str(row["error"]),
                }
            )
    return handoffs
