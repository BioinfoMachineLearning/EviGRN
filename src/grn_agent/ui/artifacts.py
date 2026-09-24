"""Artifact discovery helpers for the Streamlit GUI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from grn_agent.ui.interpret import discover_checkpoint


def load_json(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return raw


def discover_workflow_dirs(artifact_root: str | Path, *, limit: int = 80) -> list[Path]:
    root = Path(artifact_root)
    if not root.is_dir():
        return []
    found: list[Path] = []
    for runtime in sorted(root.rglob("workflow_runtime.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        found.append(runtime.parent)
        if len(found) >= limit:
            break
    return found


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def resolve_workflow_artifacts(workflow_dir: str | Path) -> dict[str, Path | None]:
    """Locate common artifacts under a workflow directory without hardcoding datasets."""
    wf = Path(workflow_dir)
    tf_dir = wf / "tf_eager"
    acq_dir = wf / "acquisition"
    eval_dir = wf / "evaluation"

    scored_candidates = [
        tf_dir / "test_scored_edges.csv",
        wf / "test_scored_edges.csv",
        eval_dir / "test_scored_edges.csv",
    ]
    network_candidates = [
        tf_dir / "test_network.csv",
        wf / "test_network.csv",
    ]
    windows_candidates = [
        tf_dir / "test_windows.jsonl",
        wf / "test_windows.jsonl",
    ]
    evidence_candidates = [
        tf_dir / "test_flat_evidence.jsonl",
        wf / "test_flat_evidence.jsonl",
    ]
    manifest_candidates = [
        acq_dir / "multimodal_manifest.json",
        wf / "multimodal_manifest.json",
    ]
    lit_candidates = [
        wf / "literature_validated.csv",
        tf_dir / "literature_validated.csv",
    ]
    report_candidates = [
        eval_dir / "eval_test_by_ratio.json",
        wf / "eval_test_by_ratio.json",
    ]

    # Prefer exact names; also accept common ablation suffixes via glob.
    scored = _first_existing(scored_candidates)
    if scored is None and tf_dir.is_dir():
        matches = sorted(tf_dir.glob("test_scored_edges*.csv"))
        scored = matches[0] if matches else None

    network = _first_existing(network_candidates)
    if network is None and tf_dir.is_dir():
        matches = sorted(tf_dir.glob("test_network*.csv"))
        network = matches[0] if matches else None

    windows = _first_existing(windows_candidates)
    if windows is None and tf_dir.is_dir():
        matches = sorted(tf_dir.glob("test_windows*.jsonl"))
        windows = matches[0] if matches else None

    return {
        "workflow_dir": wf,
        "runtime": _first_existing([wf / "workflow_runtime.json"]),
        "scored_edges": scored,
        "network": network,
        "windows": windows,
        "flat_evidence": _first_existing(evidence_candidates),
        "multimodal_manifest": _first_existing(manifest_candidates),
        "literature_validated": _first_existing(lit_candidates),
        "literature_dir": (wf / "literature_classifications")
        if (wf / "literature_classifications").is_dir()
        else None,
        "eval_report": _first_existing(report_candidates),
        "pr_curve": _first_existing(
            [
                eval_dir / "pr_curve_by_ratio.png",
                wf / "pr_curve_by_ratio.png",
            ]
        ),
        "roc_curve": _first_existing(
            [
                eval_dir / "roc_curve_by_ratio.png",
                eval_dir / "roc_curve.png",
                wf / "roc_curve_by_ratio.png",
                wf / "roc_curve.png",
            ]
        ),
        "checkpoint": discover_checkpoint(wf),
    }
