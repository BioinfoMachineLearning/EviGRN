"""Build a GUI inference run from the existing integrated workflow template."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

from grn_agent.pipeline.config import load_yaml_config, save_yaml_config


def _existing_file(root: Path, value: object) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = root / path
    return path if path.is_file() else None


def _rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def training_checkpoint(repo_root: str | Path, filename: str = "tf_eager_bootstrap_v2.pt") -> Path | None:
    """Checkpoint stored under trainings/, not a multicontext artifact path."""
    root = Path(repo_root) / "trainings"
    if not root.is_dir():
        return None
    matches = [path for path in root.rglob(filename) if path.is_file()]
    if not matches:
        return None
    return sorted(matches, key=lambda path: len(path.as_posix()))[0]


def _gold_in_directory(directory: Path) -> Path | None:
    preferred = (
        "mHSC-ChIP-seq-network.csv",
        "refNetwork.csv",
    )
    for name in preferred:
        path = directory / name
        if path.is_file():
            return path
    networks = sorted(directory.glob("*network*.csv")) + sorted(directory.glob("*gold*.csv"))
    return networks[0] if networks else None


def _mhsc_l_example(root: Path) -> dict[str, str] | None:
    """mHSC-L sample. E. coli is not used as the GUI example."""
    candidates = [path for path in (root / "Data").rglob("mHSC-L") if path.is_dir()] if (root / "Data").is_dir() else []
    for directory in sorted(candidates, key=lambda path: len(path.as_posix())):
        expression = directory / "ExpressionData.csv"
        tf_file = directory / "TFs.csv"
        if not expression.is_file() or not tf_file.is_file():
            continue
        gold = _gold_in_directory(directory)
        checkpoint = training_checkpoint(root)
        return {
            "expression": _rel(root, expression),
            "tf_file": _rel(root, tf_file),
            "gold": _rel(root, gold) if gold is not None else "",
            "species": "mouse",
            "cell_type": "mHSC-L",
            "cell_line": "",
            "cell_context": "",
            "dataset_id": "mHSC-L",
            "checkpoint": _rel(root, checkpoint) if checkpoint is not None else "",
            "config": _rel(root, directory),
        }
    return None


def discover_example_inputs(repo_root: str | Path) -> dict[str, str] | None:
    """Pick a config whose expression matrix and TF list exist on disk.

    Prefers the mHSC-L sample. Does not select an E. coli bundle.
    The checkpoint path is the training copy of tf_eager_bootstrap_v2.pt when that file exists.
    """
    root = Path(repo_root)
    preferred = _mhsc_l_example(root)
    if preferred is not None:
        return preferred
    conf_dir = root / "conf"
    if not conf_dir.is_dir():
        return None
    best: tuple[tuple[int, int, int, str], dict[str, str]] | None = None
    for config_path in sorted(conf_dir.rglob("*.yml")):
        try:
            cfg = load_yaml_config(config_path)
        except (OSError, ValueError):
            continue
        acquisition = cfg.get("acquisition") if isinstance(cfg.get("acquisition"), dict) else {}
        dataset = cfg.get("dataset") if isinstance(cfg.get("dataset"), dict) else {}
        context = cfg.get("cell_context") if isinstance(cfg.get("cell_context"), dict) else {}
        tf_eager = cfg.get("tf_eager") if isinstance(cfg.get("tf_eager"), dict) else {}
        split = cfg.get("split") if isinstance(cfg.get("split"), dict) else {}
        evaluation = cfg.get("evaluation") if isinstance(cfg.get("evaluation"), dict) else {}
        expression = _existing_file(root, dataset.get("expression_path") or acquisition.get("expr"))
        tf_file = _existing_file(root, dataset.get("tf_file"))
        if expression is None or tf_file is None:
            continue
        species = str(dataset.get("species") or acquisition.get("species") or "").strip()
        blob = " ".join(
            [
                species.lower(),
                expression.as_posix().lower(),
                str(dataset.get("dataset_id") or "").lower(),
            ]
        )
        if "coli" in blob or "ecoli" in blob:
            continue
        gold = _existing_file(
            root,
            acquisition.get("gold_network") or split.get("gold_edges") or evaluation.get("gold_edges"),
        )
        checkpoint = training_checkpoint(root) or _existing_file(root, tf_eager.get("checkpoint"))
        if checkpoint is not None and "multicontext" in checkpoint.as_posix():
            checkpoint = training_checkpoint(root)
        example = {
            "expression": _rel(root, expression),
            "tf_file": _rel(root, tf_file),
            "gold": _rel(root, gold) if gold is not None else "",
            "species": str(dataset.get("species") or acquisition.get("species") or "").strip(),
            "cell_type": str(context.get("cell_type") or acquisition.get("cell_type") or "").strip(),
            "cell_line": str(acquisition.get("cell_line") or "").strip(),
            "cell_context": str(acquisition.get("cell_context") or "").strip(),
            "dataset_id": str(
                dataset.get("dataset_id")
                or (cfg.get("workflow") or {}).get("id")
                or expression.parent.name
            ).strip(),
            "checkpoint": _rel(root, checkpoint) if checkpoint is not None else "",
            "config": _rel(root, config_path),
        }
        rank = (
            0 if gold is not None else 1,
            0 if checkpoint is not None else 1,
            len(example["expression"]),
            example["config"],
        )
        if best is None or rank < best[0]:
            best = (rank, example)
    return None if best is None else best[1]


def safe_dataset_id(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip()).strip("._")
    return cleaned or "evigrn_run"


def ranked_candidate_rows(payload: list[Any] | dict[str, Any]) -> list[dict[str, Any]]:
    raw = payload.get("candidates") if isinstance(payload, dict) else payload
    if not isinstance(raw, list):
        return []
    rows = []
    for item in raw:
        if isinstance(item, dict) and item.get("accession"):
            rows.append(item)
    return rows


def load_ranked_candidates(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.is_file():
        return []
    payload = json.loads(file_path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return ranked_candidate_rows(payload)
    if isinstance(payload, dict):
        return ranked_candidate_rows(payload)
    return []


def prepare_inference_config(
    template: dict[str, Any],
    *,
    run_dir: str | Path,
    dataset_id: str,
    expression: str | Path,
    tf_file: str | Path,
    species: str,
    checkpoint: str | Path,
    cell_type: str = "",
    cell_line: str = "",
    cell_context: str = "",
    gold: str | Path | None = None,
    manifest: str | Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Inference-only config. Training stays off. Gold evaluation is on only when a gold file is set."""
    cfg = copy.deepcopy(template)
    run = Path(run_dir)
    dataset = safe_dataset_id(dataset_id)
    gold_path = str(gold).strip() if gold else ""
    manifest_path = str(manifest or (run / "acquisition" / "multimodal_manifest.json"))

    workflow = cfg.setdefault("workflow", {})
    workflow["id"] = dataset
    workflow["artifact_root"] = str(run.parent)
    workflow["single_artifact_dir"] = True

    acquisition = cfg.setdefault("acquisition", {})
    acquisition["enabled"] = False
    acquisition["expr"] = str(expression)
    acquisition["species"] = species
    acquisition["cell_type"] = cell_type
    acquisition["cell_line"] = cell_line
    acquisition["cell_context"] = cell_context
    acquisition["dataset_id"] = dataset
    acquisition["out_manifest"] = manifest_path
    acquisition["gold_network"] = gold_path

    dataset_cfg = cfg.setdefault("dataset", {})
    dataset_cfg["dataset_id"] = dataset
    dataset_cfg["species"] = species
    dataset_cfg["expression_path"] = str(expression)
    dataset_cfg["tf_file"] = str(tf_file)

    context = cfg.setdefault("cell_context", {})
    context["cell_type"] = cell_type

    cfg["disable_priors"] = True

    split = cfg.setdefault("split", {})
    split["enabled"] = False
    split["gold_edges"] = gold_path

    tf_eager = cfg.setdefault("tf_eager", {})
    tf_eager["checkpoint"] = str(checkpoint)

    train = cfg.setdefault("train_tf_eager", {})
    train["enabled"] = False
    train["out"] = str(checkpoint)
    train["device"] = device

    cfg.setdefault("build_train_windows", {})["enabled"] = False
    cfg.setdefault("build_test_windows", {})["enabled"] = True

    infer = cfg.setdefault("infer_tf_eager", {})
    infer["enabled"] = True
    infer["device"] = device
    infer["reuse_if_exists"] = False

    evaluation = cfg.setdefault("evaluation", {})
    evaluation["enabled"] = bool(gold_path)
    evaluation["gold_edges"] = gold_path
    evaluation["reuse_if_exists"] = False

    literature = cfg.setdefault("literature_validation", {})
    literature["enabled"] = False
    literature["cell_type"] = cell_type
    return cfg


def write_inference_config(template_path: str | Path, out_path: str | Path, **kwargs: Any) -> Path:
    cfg = prepare_inference_config(load_yaml_config(template_path), **kwargs)
    destination = Path(out_path)
    save_yaml_config(destination, cfg)
    return destination


def acquire_rank_command(
    *,
    python: str,
    script: str | Path,
    expression: str | Path,
    species: str,
    tf_file: str | Path,
    out_manifest: str | Path,
    ranked_json: str | Path,
    dataset_id: str,
    cell_type: str = "",
    cell_line: str = "",
    cell_context: str = "",
    gold: str | Path | None = None,
    cache_dir: str | Path,
    max_candidates: int = 5,
) -> list[str]:
    cmd = [
        python,
        str(script),
        "--expr",
        str(expression),
        "--species",
        species,
        "--tf-file",
        str(tf_file),
        "--out-manifest",
        str(out_manifest),
        "--dataset-id",
        safe_dataset_id(dataset_id),
        "--max-atac-candidates",
        str(int(max_candidates)),
        "--ranked-candidates-json",
        str(ranked_json),
        "--rank-only",
        "--skip-motif",
        "--cache-dir",
        str(cache_dir),
    ]
    if cell_type.strip():
        cmd.extend(["--cell-type", cell_type.strip()])
    if cell_line.strip():
        cmd.extend(["--cell-line", cell_line.strip()])
    if cell_context.strip():
        cmd.extend(["--cell-context", cell_context.strip()])
    if gold and str(gold).strip():
        cmd.extend(["--gold-network", str(gold)])
    return cmd


def acquire_harmonize_command(
    *,
    python: str,
    script: str | Path,
    expression: str | Path,
    species: str,
    tf_file: str | Path,
    out_manifest: str | Path,
    dataset_id: str,
    cell_type: str = "",
    cell_line: str = "",
    cell_context: str = "",
    gold: str | Path | None = None,
    cache_dir: str | Path,
    atac_accession: str = "",
    skip_motif: bool = False,
) -> list[str]:
    cmd = [
        python,
        str(script),
        "--expr",
        str(expression),
        "--species",
        species,
        "--tf-file",
        str(tf_file),
        "--out-manifest",
        str(out_manifest),
        "--dataset-id",
        safe_dataset_id(dataset_id),
        "--cache-dir",
        str(cache_dir),
    ]
    if cell_type.strip():
        cmd.extend(["--cell-type", cell_type.strip()])
    if cell_line.strip():
        cmd.extend(["--cell-line", cell_line.strip()])
    if cell_context.strip():
        cmd.extend(["--cell-context", cell_context.strip()])
    if gold and str(gold).strip():
        cmd.extend(["--gold-network", str(gold)])
    accession = atac_accession.strip()
    if accession:
        cmd.extend(["--atac-accession", accession])
    if skip_motif:
        cmd.append("--skip-motif")
    return cmd


def literature_command(
    *,
    python: str,
    script: str | Path,
    scored_csv: str | Path,
    output: str | Path,
    cell_type: str = "",
    min_p: float = 0.5,
    limit: int = 20,
) -> list[str]:
    cmd = [
        python,
        str(script),
        "--input",
        str(scored_csv),
        "--output",
        str(output),
        "--min-p",
        str(min_p),
        "--limit",
        str(int(limit)),
    ]
    if cell_type.strip():
        cmd.extend(["--cell-type", cell_type.strip()])
    return cmd
