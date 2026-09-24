#!/usr/bin/env python3
"""Run acquisition -> tf-eager windows -> training -> inference -> evaluation."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from grn_agent.pipeline.config import load_yaml_config, save_yaml_config


def _run_cmd(cmd: list[str], stage: str) -> None:
    print(f"[integrated-tf-eager] stage={stage} command={' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_runtime_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _record_stage(
    runtime_manifest: dict[str, Any],
    runtime_path: Path,
    *,
    stage: str,
    status: str,
    outputs: dict[str, Any] | None = None,
    reuse_path: str | None = None,
    started_at: str | None = None,
    elapsed_seconds: float | None = None,
) -> None:
    stages = runtime_manifest.setdefault("stages", {})
    stage_info: dict[str, Any] = {
        "status": status,
        "recorded_at": _iso_now(),
    }
    if started_at is not None:
        stage_info["started_at"] = started_at
    if elapsed_seconds is not None:
        stage_info["elapsed_seconds"] = round(float(elapsed_seconds), 3)
    if reuse_path:
        stage_info["reuse_path"] = reuse_path
    if outputs:
        stage_info["outputs"] = outputs
    stages[stage] = stage_info
    _write_runtime_manifest(runtime_path, runtime_manifest)


def _run_timed_stage(
    runtime_manifest: dict[str, Any],
    runtime_path: Path,
    *,
    stage: str,
    cmd: list[str],
    outputs: dict[str, Any] | None = None,
) -> None:
    started_at = _iso_now()
    t0 = time.perf_counter()
    _run_cmd(cmd, stage)
    elapsed = time.perf_counter() - t0
    print(f"[integrated-tf-eager] stage={stage} elapsed={elapsed:.2f}s", flush=True)
    _record_stage(
        runtime_manifest,
        runtime_path,
        stage=stage,
        status="ran",
        outputs=outputs,
        started_at=started_at,
        elapsed_seconds=elapsed,
    )


def _bool(v: Any, default: bool = True) -> bool:
    if v is None:
        return default
    return bool(v)


def _workflow_paths(cfg: dict[str, Any]) -> tuple[str, Path]:
    wf = cfg.get("workflow", {}) if isinstance(cfg.get("workflow", {}), dict) else {}
    wf_id = str(wf.get("id", "integrated_tf_eager_workflow")).strip() or "integrated_tf_eager_workflow"
    artifact_root = Path(str(wf.get("artifact_root", "artifacts"))).expanduser()
    workflow_dir = artifact_root / wf_id
    workflow_dir.mkdir(parents=True, exist_ok=True)
    return wf_id, workflow_dir


def _seed(cfg: dict[str, Any]) -> int:
    wf = cfg.get("workflow", {}) if isinstance(cfg.get("workflow", {}), dict) else {}
    return int(wf.get("seed", cfg.get("seed", 42)))


def _dict_to_cli_args(section: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for k, v in section.items():
        if v is None:
            continue
        flag = f"--{k.replace('_', '-')}"
        if isinstance(v, bool):
            if v:
                out.append(flag)
            continue
        out.extend([flag, str(v)])
    return out


def _merge_dicts(*items: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        for k, v in item.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = _merge_dicts(out[k], v)
            else:
                out[k] = copy.deepcopy(v)
    return out


def _strip_keys(section: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {k: v for k, v in section.items() if k not in keys}


def _tf_eager_train_cmd(py: str, cfg_path: Path, train_cfg: dict[str, Any]) -> list[str]:
    distributed_cfg = (
        train_cfg.get("distributed", {}) if isinstance(train_cfg.get("distributed", {}), dict) else {}
    )
    nproc_per_node = int(distributed_cfg.get("nproc_per_node", 1) or 1)
    if nproc_per_node <= 1:
        return [py, "scripts/train_tf_eager.py", "--config", str(cfg_path)]
    cmd = [py, "-m", "torch.distributed.run"]
    if _bool(distributed_cfg.get("standalone"), default=True):
        cmd.append("--standalone")
    if distributed_cfg.get("master_port") not in (None, ""):
        cmd.extend(["--master_port", str(distributed_cfg["master_port"])])
    cmd.extend(["--nproc_per_node", str(nproc_per_node), "scripts/train_tf_eager.py", "--config", str(cfg_path)])
    if distributed_cfg.get("backend") not in (None, ""):
        cmd.extend(["--ddp-backend", str(distributed_cfg["backend"])])
    return cmd


def _load_base_tf_config(master: dict[str, Any]) -> dict[str, Any]:
    tf = master.get("tf_eager", {}) if isinstance(master.get("tf_eager", {}), dict) else {}
    base_path = tf.get("config") or master.get("tf_eager_config")
    base = load_yaml_config(str(base_path)) if base_path else {}
    for key in (
        "dataset",
        "cell_context",
        "candidates",
        "scoring",
        "eval_track",
        "disable_priors",
        "use_ortholog_lookup",
        "multimodal_manifest",
        "build_device",
        "prior_device",
        "tf_workers",
    ):
        if key in master:
            base[key] = copy.deepcopy(master[key])
    return base


def _stage_enabled(master: dict[str, Any], section: dict[str, Any], default: bool = True) -> bool:
    wf = master.get("workflow", {}) if isinstance(master.get("workflow", {}), dict) else {}
    if _bool(wf.get("inference_evaluation_only"), default=False):
        return False
    return _bool(section.get("enabled"), default=default)


def _resolve_window_cfg(
    master: dict[str, Any],
    *,
    subset: str,
    windows_jsonl: str,
    checkpoint: str,
    split_manifest: str | None,
    acq_manifest: str | None,
    stage_cfg: dict[str, Any],
) -> dict[str, Any]:
    base = _load_base_tf_config(master)
    tf_common = {
        k: v
        for k, v in (master.get("tf_eager", {}) if isinstance(master.get("tf_eager", {}), dict) else {}).items()
        if k not in {"config", "train", "infer"}
    }
    split = master.get("split", {}) if isinstance(master.get("split", {}), dict) else {}
    tf_defaults = {
        "gold_edges": split.get("gold_edges") or master.get("gold_edges"),
        "split_manifest": split_manifest or split.get("out"),
        "strategy": stage_cfg.get("strategy", tf_common.get("strategy", "leave_one_tf_out")),
        "fold_id": stage_cfg.get("fold_id", tf_common.get("fold_id", split.get("fold_id", ""))),
        "subset": subset,
        "windows_jsonl": windows_jsonl,
        "checkpoint": checkpoint,
    }
    tf_defaults = {k: v for k, v in tf_defaults.items() if v not in (None, "")}
    cfg = _merge_dicts(base, {"tf_eager": tf_common}, {"tf_eager": tf_defaults}, {"tf_eager": stage_cfg})
    cfg["tf_eager"]["subset"] = subset
    cfg["tf_eager"]["windows_jsonl"] = windows_jsonl
    cfg["tf_eager"]["checkpoint"] = checkpoint
    if acq_manifest:
        cfg["multimodal_manifest"] = acq_manifest
    cfg.setdefault("seed", _seed(master))
    return cfg


def _resolve_blind_window_cfg(
    master: dict[str, Any],
    *,
    windows_jsonl: str,
    checkpoint: str,
    acq_manifest: str | None,
    stage_cfg: dict[str, Any],
) -> dict[str, Any]:
    cfg = _resolve_window_cfg(
        master,
        subset="test",
        windows_jsonl=windows_jsonl,
        checkpoint=checkpoint,
        split_manifest=None,
        acq_manifest=acq_manifest,
        stage_cfg=stage_cfg,
    )
    cfg["blind"] = True
    cfg["tf_eager"]["windows_jsonl"] = windows_jsonl
    cfg["tf_eager"]["checkpoint"] = checkpoint
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Master tf-eager workflow YAML config")
    ap.add_argument("--force-recompute", action="store_true")
    args = ap.parse_args()

    master = load_yaml_config(args.config)
    wf_id, wf_dir = _workflow_paths(master)
    wf = master.get("workflow", {}) if isinstance(master.get("workflow", {}), dict) else {}
    single_artifact_dir = _bool(wf.get("single_artifact_dir"), default=True)
    inference_evaluation_only = _bool(wf.get("inference_evaluation_only"), default=False)
    seed = _seed(master)
    py = sys.executable
    runtime_path = wf_dir / "workflow_runtime.json"
    workflow_started_at = _iso_now()
    workflow_t0 = time.perf_counter()
    runtime_manifest: dict[str, Any] = {
        "workflow_id": wf_id,
        "config_path": str(Path(args.config).resolve()),
        "workspace": str(wf_dir.resolve()),
        "force_recompute": bool(args.force_recompute),
        "started_at": workflow_started_at,
        "status": "running",
        "stages": {},
    }
    _write_runtime_manifest(runtime_path, runtime_manifest)

    acq_manifest: str | None = None
    split_manifest: str | None = None
    train_windows: str
    test_windows: str
    checkpoint: str
    scored_csv: str
    network_csv: str
    flat_evidence_jsonl: str

    # 1) Acquisition
    acq = master.get("acquisition", {})
    if acq and not isinstance(acq, dict):
        raise SystemExit("acquisition section must be a mapping")
    if isinstance(acq, dict) and _stage_enabled(master, acq, default=False):
        acq_cfg = dict(acq)
        acq_cfg.pop("enabled", None)
        reuse_acq = _bool(acq_cfg.pop("reuse_if_exists", True), default=True) and not args.force_recompute
        # Allow users to specify the desired acquisition manifest path using either
        # acquisition.out_manifest or acquisition.multimodal_manifest.
        if "out_manifest" not in acq_cfg and acq_cfg.get("multimodal_manifest"):
            acq_cfg["out_manifest"] = acq_cfg["multimodal_manifest"]
        acq_cfg.pop("multimodal_manifest", None)
        dataset_cfg = master.get("dataset", {}) if isinstance(master.get("dataset", {}), dict) else {}
        if dataset_cfg.get("tf_file") and "tf_file" not in acq_cfg:
            acq_cfg["tf_file"] = dataset_cfg["tf_file"]
        acq_cfg.pop("gold_network", None)
        if single_artifact_dir and "out_manifest" not in acq_cfg:
            acq_cfg["out_manifest"] = str(wf_dir / "acquisition" / "multimodal_manifest.json")
        if "out_manifest" not in acq_cfg:
            raise SystemExit("acquisition.out_manifest is required when acquisition is enabled")
        acq_manifest = str(acq_cfg["out_manifest"])
        if reuse_acq and Path(acq_manifest).is_file():
            print(f"[integrated-tf-eager] stage=acquisition skip=reuse path={acq_manifest}", flush=True)
            _record_stage(
                runtime_manifest,
                runtime_path,
                stage="acquisition",
                status="skipped_reuse",
                outputs={"out_manifest": acq_manifest},
                reuse_path=acq_manifest,
            )
        else:
            _run_timed_stage(
                runtime_manifest,
                runtime_path,
                stage="acquisition",
                cmd=[py, "scripts/acquire_multimodal_data.py", *_dict_to_cli_args(acq_cfg)],
                outputs={"out_manifest": acq_manifest},
            )
    else:
        acq_candidate = wf_dir / "acquisition" / "multimodal_manifest.json"
        if single_artifact_dir and acq_candidate.is_file():
            acq_manifest = str(acq_candidate)
        elif isinstance(acq, dict) and acq.get("out_manifest") and Path(str(acq["out_manifest"])).is_file():
            acq_manifest = str(acq["out_manifest"])
        elif master.get("multimodal_manifest"):
            acq_manifest = str(master["multimodal_manifest"])

    # 2) Split manifest
    split = master.get("split", {})
    if not isinstance(split, dict):
        raise SystemExit("split section must be a mapping")
    if _stage_enabled(master, split, default=False):
        split_cfg = dict(split)
        split_cfg.pop("enabled", None)
        split_cfg.setdefault("seed", seed)
        dataset_cfg = master.get("dataset", {}) if isinstance(master.get("dataset", {}), dict) else {}
        if dataset_cfg.get("expression_path") and "expression_path" not in split_cfg:
            split_cfg["expression_path"] = dataset_cfg["expression_path"]
        if dataset_cfg.get("tf_file") and "tf_file" not in split_cfg:
            split_cfg["tf_file"] = dataset_cfg["tf_file"]
        if single_artifact_dir:
            split_cfg["out"] = str(wf_dir / "splits" / "split_manifest.csv")
        if "out" not in split_cfg:
            raise SystemExit("split.out is required")
        split_cfg_path = wf_dir / "split.resolved.yml"
        save_yaml_config(split_cfg_path, split_cfg)
        _run_timed_stage(
            runtime_manifest,
            runtime_path,
            stage="split",
            cmd=[py, "scripts/make_tf_holdout_split_manifest.py", "--config", str(split_cfg_path)],
            outputs={"out": str(split_cfg["out"])},
        )
        split_manifest = str(split_cfg["out"])
    else:
        split_candidate = wf_dir / "splits" / "split_manifest.csv"
        if single_artifact_dir and split_candidate.is_file():
            split_manifest = str(split_candidate)
        elif split.get("out") and Path(str(split["out"])).is_file():
            split_manifest = str(split["out"])

    tf_common = master.get("tf_eager", {}) if isinstance(master.get("tf_eager", {}), dict) else {}
    train_stage = master.get("build_train_windows", {}) if isinstance(master.get("build_train_windows", {}), dict) else {}
    test_stage = master.get("build_test_windows", {}) if isinstance(master.get("build_test_windows", {}), dict) else {}
    train_cfg = master.get("train_tf_eager", {}) if isinstance(master.get("train_tf_eager", {}), dict) else {}
    infer_stage = master.get("infer_tf_eager", {}) if isinstance(master.get("infer_tf_eager", {}), dict) else {}
    train_enabled = _stage_enabled(master, train_cfg, default=False)
    build_train_enabled = _stage_enabled(master, train_stage, default=train_enabled)
    build_test_enabled = _stage_enabled(master, test_stage, default=True)
    has_split_manifest = bool(split_manifest)
    split_required_stages_enabled = build_train_enabled or train_enabled
    if not has_split_manifest and split_required_stages_enabled:
        raise SystemExit(
            "split manifest is required for training/split-aware tf-eager workflow; "
            "enable split/train stages explicitly only when you want training; "
            "otherwise the workflow will use blind single-dataset inference"
        )

    if single_artifact_dir:
        checkpoint = str(train_cfg.get("out") or tf_common.get("checkpoint") or (wf_dir / "tf_eager" / "tf_eager.pt"))
        train_windows = str(
            train_stage.get("windows_jsonl") or tf_common.get("train_windows_jsonl") or (wf_dir / "tf_eager" / "train_windows.jsonl")
        )
        test_windows = str(
            test_stage.get("windows_jsonl") or tf_common.get("test_windows_jsonl") or (wf_dir / "tf_eager" / "test_windows.jsonl")
        )
        scored_csv = str(
            infer_stage.get("scored_csv") or infer_stage.get("out_scored_csv") or (wf_dir / "tf_eager" / "test_scored_edges.csv")
        )
        network_csv = str(
            infer_stage.get("network_csv") or infer_stage.get("out_network_csv") or (wf_dir / "tf_eager" / "test_network.csv")
        )
        flat_evidence_jsonl = str(
            infer_stage.get("evidence_jsonl")
            or infer_stage.get("out_evidence_jsonl")
            or (wf_dir / "tf_eager" / "test_flat_evidence.jsonl")
        )
    else:
        checkpoint = str(train_cfg.get("out") or tf_common.get("checkpoint") or "artifacts/tf_eager/tf_eager_bootstrap_compacttoken_reduced_gene_emb.pt")
        train_windows = str(
            train_stage.get("windows_jsonl") or tf_common.get("train_windows_jsonl") or "artifacts/tf_eager/train_windows.jsonl"
        )
        test_windows = str(
            test_stage.get("windows_jsonl") or tf_common.get("test_windows_jsonl") or "artifacts/tf_eager/test_windows.jsonl"
        )
        scored_csv = str(
            infer_stage.get("scored_csv") or infer_stage.get("out_scored_csv") or "artifacts/tf_eager/test_scored_edges.csv"
        )
        network_csv = str(
            infer_stage.get("network_csv") or infer_stage.get("out_network_csv") or "artifacts/tf_eager/test_network.csv"
        )
        flat_evidence_jsonl = str(
            infer_stage.get("evidence_jsonl") or infer_stage.get("out_evidence_jsonl") or "artifacts/tf_eager/test_flat_evidence.jsonl"
        )

    # 3) Build train windows
    if build_train_enabled:
        cfg = _resolve_window_cfg(
            master,
            subset=str(train_stage.get("subset", "train")),
            windows_jsonl=train_windows,
            checkpoint=checkpoint,
            split_manifest=split_manifest,
            acq_manifest=acq_manifest,
            stage_cfg=_strip_keys(train_stage, "enabled", "reuse_if_exists"),
        )
        cfg_path = wf_dir / "tf_eager_build_train.resolved.yml"
        save_yaml_config(cfg_path, cfg)
        if _bool(train_stage.get("reuse_if_exists"), default=True) and Path(train_windows).is_file() and not args.force_recompute:
            print(f"[integrated-tf-eager] stage=build_train_windows skip=reuse path={train_windows}", flush=True)
            _record_stage(
                runtime_manifest,
                runtime_path,
                stage="build_train_windows",
                status="skipped_reuse",
                outputs={"windows_jsonl": train_windows},
                reuse_path=train_windows,
            )
        else:
            _run_timed_stage(
                runtime_manifest,
                runtime_path,
                stage="build_train_windows",
                cmd=[py, "scripts/build_tf_eager_windows.py", "--config", str(cfg_path)],
                outputs={"windows_jsonl": train_windows},
            )

    # 4) Train tf-eager
    if train_enabled:
        cfg = _resolve_window_cfg(
            master,
            subset=str(train_stage.get("subset", "train")),
            windows_jsonl=train_windows,
            checkpoint=checkpoint,
            split_manifest=split_manifest,
            acq_manifest=acq_manifest,
            stage_cfg={},
        )
        cfg["tf_eager"]["train"] = {k: v for k, v in train_cfg.items() if k not in {"enabled", "out", "reuse_if_exists"}}
        cfg_path = wf_dir / "tf_eager_train.resolved.yml"
        save_yaml_config(cfg_path, cfg)
        if _bool(train_cfg.get("reuse_if_exists"), default=False) and Path(checkpoint).is_file() and not args.force_recompute:
            print(f"[integrated-tf-eager] stage=train_tf_eager skip=reuse path={checkpoint}", flush=True)
            _record_stage(
                runtime_manifest,
                runtime_path,
                stage="train_tf_eager",
                status="skipped_reuse",
                outputs={"checkpoint": checkpoint},
                reuse_path=checkpoint,
            )
        else:
            _run_timed_stage(
                runtime_manifest,
                runtime_path,
                stage="train_tf_eager",
                cmd=_tf_eager_train_cmd(py, cfg_path, train_cfg),
                outputs={"checkpoint": checkpoint},
            )

    # 5) Build test windows
    if build_test_enabled or inference_evaluation_only:
        cfg = _resolve_window_cfg(
            master,
            subset=str(test_stage.get("subset", "test")),
            windows_jsonl=test_windows,
            checkpoint=checkpoint,
            split_manifest=split_manifest,
            acq_manifest=acq_manifest,
            stage_cfg=_strip_keys(test_stage, "enabled", "reuse_if_exists"),
        )
        if not has_split_manifest:
            cfg = _resolve_blind_window_cfg(
                master,
                windows_jsonl=test_windows,
                checkpoint=checkpoint,
                acq_manifest=acq_manifest,
                stage_cfg=_strip_keys(test_stage, "enabled", "reuse_if_exists"),
            )
        cfg_path = wf_dir / "tf_eager_build_test.resolved.yml"
        save_yaml_config(cfg_path, cfg)
        if _bool(test_stage.get("reuse_if_exists"), default=True) and Path(test_windows).is_file() and not args.force_recompute:
            print(f"[integrated-tf-eager] stage=build_test_windows skip=reuse path={test_windows}", flush=True)
            _record_stage(
                runtime_manifest,
                runtime_path,
                stage="build_test_windows",
                status="skipped_reuse",
                outputs={"windows_jsonl": test_windows},
                reuse_path=test_windows,
            )
        else:
            _run_timed_stage(
                runtime_manifest,
                runtime_path,
                stage="build_test_windows",
                cmd=[py, "scripts/build_tf_eager_windows.py", "--config", str(cfg_path)],
                outputs={"windows_jsonl": test_windows},
            )

    # 6) Infer and decode
    if _bool(infer_stage.get("enabled"), default=True) or inference_evaluation_only:
        if has_split_manifest:
            cfg = _resolve_window_cfg(
                master,
                subset=str(test_stage.get("subset", "test")),
                windows_jsonl=test_windows,
                checkpoint=checkpoint,
                split_manifest=split_manifest,
                acq_manifest=acq_manifest,
                stage_cfg={},
            )
        else:
            cfg = _resolve_blind_window_cfg(
                master,
                windows_jsonl=test_windows,
                checkpoint=checkpoint,
                acq_manifest=acq_manifest,
                stage_cfg={},
            )
        cfg["tf_eager"]["infer"] = {
            k: v
            for k, v in infer_stage.items()
            if k
            not in {
                "enabled",
                "reuse_if_exists",
                "scored_csv",
                "network_csv",
                "evidence_jsonl",
                "out_scored_csv",
                "out_network_csv",
                "out_evidence_jsonl",
            }
        }
        cfg["tf_eager"]["infer"].update(
            {
                "windows_jsonl": test_windows,
                "checkpoint": checkpoint,
                "scored_csv": scored_csv,
                "network_csv": network_csv,
                "evidence_jsonl": flat_evidence_jsonl,
            }
        )
        cfg_path = wf_dir / "tf_eager_infer.resolved.yml"
        save_yaml_config(cfg_path, cfg)
        if (
            _bool(infer_stage.get("reuse_if_exists"), default=False)
            and Path(scored_csv).is_file()
            and Path(flat_evidence_jsonl).is_file()
            and not args.force_recompute
        ):
            print(f"[integrated-tf-eager] stage=infer_tf_eager skip=reuse path={scored_csv}", flush=True)
            _record_stage(
                runtime_manifest,
                runtime_path,
                stage="infer_tf_eager",
                status="skipped_reuse",
                outputs={
                    "scored_csv": scored_csv,
                    "network_csv": network_csv,
                    "evidence_jsonl": flat_evidence_jsonl,
                },
                reuse_path=scored_csv,
            )
        else:
            _run_timed_stage(
                runtime_manifest,
                runtime_path,
                stage="infer_tf_eager",
                cmd=[py, "scripts/infer_tf_eager.py", "--config", str(cfg_path)],
                outputs={
                    "scored_csv": scored_csv,
                    "network_csv": network_csv,
                    "evidence_jsonl": flat_evidence_jsonl,
                },
            )

    # 7) Evaluation
    ev = master.get("evaluation", {})
    if not isinstance(ev, dict):
        raise SystemExit("evaluation section must be a mapping")
    if _bool(ev.get("enabled"), default=True) or inference_evaluation_only:
        ev_cfg = dict(ev)
        ev_cfg.pop("enabled", None)
        reuse_eval = _bool(ev_cfg.pop("reuse_if_exists", False), default=False)
        evaluation_report_path = ""
        ev_cfg.setdefault("seed", seed)
        ev_cfg.setdefault("scored_csv", scored_csv)
        ev_cfg.setdefault("evidence_jsonl", flat_evidence_jsonl)
        ev_cfg.setdefault("network_csv", network_csv)
        gold_edges = str(ev_cfg.get("gold_edges") or split.get("gold_edges") or master.get("gold_edges") or "").strip()
        if gold_edges:
            ev_cfg["gold_edges"] = gold_edges
        if has_split_manifest:
            ev_cfg.setdefault("split_manifest", split_manifest)
            ev_cfg.setdefault("strategy", tf_common.get("strategy", "leave_one_tf_out"))
            ev_cfg.setdefault("fold_id", tf_common.get("fold_id", split.get("fold_id", "")))
            ev_cfg.setdefault("subset", str(test_stage.get("subset", "test")))
        if single_artifact_dir:
            ev_cfg["out_report"] = str(wf_dir / "evaluation" / "eval_test_by_ratio.json")
        pr_curve_out = str(wf_dir / "evaluation" / "pr_curve_by_ratio.png") if single_artifact_dir else ""
        if not has_split_manifest and not gold_edges:
            print("[integrated-tf-eager] stage=evaluation skip=no_gold_edges", flush=True)
            _record_stage(
                runtime_manifest,
                runtime_path,
                stage="evaluation",
                status="skipped_no_gold_edges",
            )
        else:
            ev_cfg_path = wf_dir / "tf_eager_evaluation.resolved.yml"
            save_yaml_config(ev_cfg_path, ev_cfg)
            evaluation_report_path = str(ev_cfg.get("out_report", "")).strip()
            if reuse_eval and Path(str(ev_cfg.get("out_report", ""))).is_file() and not args.force_recompute:
                print(
                    f"[integrated-tf-eager] stage=evaluation skip=reuse path={ev_cfg['out_report']}",
                    flush=True,
                )
                _record_stage(
                    runtime_manifest,
                    runtime_path,
                    stage="evaluation",
                    status="skipped_reuse",
                    outputs={"out_report": str(ev_cfg["out_report"])},
                    reuse_path=str(ev_cfg["out_report"]),
                )
            else:
                _run_timed_stage(
                    runtime_manifest,
                    runtime_path,
                    stage="evaluation",
                    cmd=[py, "scripts/eval_grn_agent.py", "--config", str(ev_cfg_path)],
                    outputs={"out_report": str(ev_cfg["out_report"])},
                )
            if evaluation_report_path and Path(evaluation_report_path).is_file():
                pr_curve_target = pr_curve_out or str(Path(evaluation_report_path).with_name("pr_curve_by_ratio.png"))
                _run_timed_stage(
                    runtime_manifest,
                    runtime_path,
                    stage="plot_pr_curve",
                    cmd=[
                        py,
                        "scripts/plot_eval_pr_curve.py",
                        "--input",
                        evaluation_report_path,
                        "--out",
                        pr_curve_target,
                        "--formats",
                        "png,pdf",
                        "--allow-missing-curves",
                    ],
                    outputs={"pr_curve_figure": pr_curve_target},
                )

    # 8) Literature Validation (post-inference)
    lit_cfg = master.get("literature_validation", {})
    if not isinstance(lit_cfg, dict):
        lit_cfg = {}
    if _bool(lit_cfg.get("enabled"), default=False):
        lit_cell_type = str(lit_cfg.get("cell_type", "")).strip() or None
        lit_min_p = float(lit_cfg.get("min_p", 0.5))
        lit_limit = lit_cfg.get("limit")
        lit_output = str(lit_cfg.get("output", "")).strip()
        if not lit_output:
            lit_output = str(wf_dir / "literature_validated.csv")
        cmd = [
            py, "scripts/run_literature_validation.py",
            "--input", scored_csv,
            "--output", lit_output,
            "--min-p", str(lit_min_p),
        ]
        if lit_cell_type:
            cmd.extend(["--cell-type", lit_cell_type])
        if lit_limit:
            cmd.extend(["--limit", str(int(lit_limit))])
        _run_timed_stage(
            runtime_manifest,
            runtime_path,
            stage="literature_validation",
            cmd=cmd,
            outputs={"output": lit_output},
        )

    total_elapsed = time.perf_counter() - workflow_t0
    runtime_manifest["completed_at"] = _iso_now()
    runtime_manifest["total_elapsed_seconds"] = round(total_elapsed, 3)
    runtime_manifest["status"] = "completed"
    _write_runtime_manifest(runtime_path, runtime_manifest)
    print(f"[integrated-tf-eager] total_elapsed={total_elapsed:.2f}s runtime_manifest={runtime_path}", flush=True)
    print(f"[integrated-tf-eager] completed workflow_id={wf_id} workspace={wf_dir}", flush=True)


if __name__ == "__main__":
    main()
