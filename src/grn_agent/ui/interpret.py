"""On-demand attention and PCA views for edges selected in the GUI.

Loads an existing TF-EAGER checkpoint when one is present. Does not train,
write checkpoints, or change model code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from grn_agent.ui.viz import load_tf_window

_EVIDENCE_FEATURES = (
    "correlation",
    "accessibility",
    "motif_present",
    "window_vote_fraction",
    "p_present_std",
)


def discover_checkpoint(workflow_dir: str | Path | None) -> Path | None:
    """Find a TF-EAGER checkpoint next to the selected workflow."""
    if workflow_dir is None:
        return None
    wf = Path(workflow_dir)
    preferred = [
        wf / "tf_eager" / "tf_eager.pt",
        wf / "tf_eager.pt",
    ]
    for path in preferred:
        if path.is_file():
            return path
    roots = [wf / "tf_eager", wf, wf.parent / "tf_eager"]
    matches: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*.pt")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            matches.append(path)
    if not matches:
        return None
    named = [path for path in matches if "tf_eager" in path.name]
    return named[0] if named else matches[0]


def edge_key(source_tf: str, target_gene: str) -> tuple[str, str]:
    return (str(source_tf).strip().upper(), str(target_gene).strip().upper())


def gene_position(record: dict[str, Any], target_gene: str) -> int | None:
    gene = str(target_gene).strip().upper()
    for index, item in enumerate(record.get("genes") or []):
        if str(item.get("target_gene", "")).strip().upper() == gene:
            return index
    return None


def attention_frame(summaries: list[dict[str, Any]]) -> pd.DataFrame:
    """Long table of channel-group attention mass for one or more edges."""
    rows: list[dict[str, Any]] = []
    for item in summaries:
        groups = item.get("by_stage_channel_group") or {}
        if not isinstance(groups, dict):
            continue
        for stage, channels in groups.items():
            if not isinstance(channels, dict):
                continue
            for channel, mass in channels.items():
                try:
                    value = float(mass)
                except (TypeError, ValueError):
                    continue
                rows.append(
                    {
                        "edge": item.get("edge"),
                        "source_tf": item.get("source_tf"),
                        "target_gene": item.get("target_gene"),
                        "stage": str(stage),
                        "channel": str(channel),
                        "attention_mass": value,
                    }
                )
    return pd.DataFrame(rows)


def local_attention_frame(summaries: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in summaries:
        local = item.get("by_stage_local_vs_global") or {}
        if not isinstance(local, dict):
            continue
        for stage, parts in local.items():
            if not isinstance(parts, dict):
                continue
            for name, mass in parts.items():
                try:
                    value = float(mass)
                except (TypeError, ValueError):
                    continue
                rows.append(
                    {
                        "edge": item.get("edge"),
                        "source_tf": item.get("source_tf"),
                        "target_gene": item.get("target_gene"),
                        "stage": str(stage),
                        "region": str(name),
                        "attention_mass": value,
                    }
                )
    return pd.DataFrame(rows)


def pca_scores(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2-component PCA via SVD. Returns scores (n, 2) and variance fractions."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 1:
        raise ValueError("PCA needs at least 3 rows and 1 numeric column")
    centered = values - np.nanmean(values, axis=0, keepdims=True)
    centered = np.nan_to_num(centered, nan=0.0, posinf=0.0, neginf=0.0)
    _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    n_comp = min(2, singular.shape[0], vt.shape[0])
    scores = centered @ vt[:n_comp].T
    if n_comp == 1:
        scores = np.column_stack([scores[:, 0], np.zeros(scores.shape[0])])
    variance = (singular ** 2) / max(float((singular ** 2).sum()), 1e-12)
    fractions = np.zeros(2, dtype=np.float64)
    fractions[:n_comp] = variance[:n_comp]
    return scores, fractions


def _motif_to_float(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    number = pd.to_numeric(value, errors="coerce")
    if pd.notna(number):
        return float(number)
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y"}:
        return 1.0
    if text in {"false", "f", "no", "n"}:
        return 0.0
    return float("nan")


def evidence_pca_frame(
    scored: pd.DataFrame,
    selected: list[tuple[str, str]] | None = None,
    *,
    max_rows: int = 2000,
) -> dict[str, Any]:
    """PCA of numeric evidence columns on scored edges. Highlights a selection."""
    if scored.empty:
        raise ValueError("No scored edges available for PCA")
    present = [col for col in _EVIDENCE_FEATURES if col in scored.columns]
    if len(present) < 2:
        raise ValueError("Scored edges do not include enough numeric evidence columns for PCA")
    work = scored.copy()
    for col in present:
        if col == "motif_present":
            work[col] = work[col].map(_motif_to_float)
        else:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=present)
    if len(work) < 3:
        raise ValueError("Fewer than 3 scored edges have complete evidence features")
    if len(work) > max_rows:
        work = work.sort_values("p_present", ascending=False).head(max_rows) if "p_present" in work.columns else work.head(max_rows)
    selected_keys = {edge_key(tf, gene) for tf, gene in (selected or [])}
    matrix = work[present].to_numpy(dtype=np.float64)
    scales = matrix.std(axis=0)
    scales[scales < 1e-8] = 1.0
    scores, variance = pca_scores(matrix / scales)
    frame = pd.DataFrame(
        {
            "source_tf": work["source_tf"].astype(str).str.upper().to_numpy(),
            "target_gene": work["target_gene"].astype(str).str.upper().to_numpy(),
            "pc1": scores[:, 0],
            "pc2": scores[:, 1],
            "p_present": pd.to_numeric(work["p_present"], errors="coerce").to_numpy() if "p_present" in work.columns else np.nan,
        }
    )
    frame["selected"] = [
        edge_key(tf, gene) in selected_keys for tf, gene in zip(frame["source_tf"], frame["target_gene"])
    ]
    frame["edge"] = frame["source_tf"] + " → " + frame["target_gene"]
    return {
        "frame": frame,
        "variance": (float(variance[0]), float(variance[1])),
        "kind": "evidence_features",
        "features": present,
        "note": "PCA of scored-edge evidence features. Selected edges are marked.",
    }


def _load_model(checkpoint: Path, device: str):
    import dataclasses

    import torch

    from grn_agent.models.tf_eager import TfEagerConfig, TfEagerWindowModel

    try:
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint, map_location=device)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(f"Checkpoint {checkpoint} has no model_state")
    raw = payload.get("config") or {}
    allowed = {field.name for field in dataclasses.fields(TfEagerConfig)}
    cfg = TfEagerConfig(**{key: value for key, value in raw.items() if key in allowed})
    model = TfEagerWindowModel(cfg)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model


def _batch_for_window(model, record: dict[str, Any]):
    from grn_agent.models.tf_eager import window_record_to_batch

    cfg = model.cfg
    return window_record_to_batch(
        record,
        token_layout=str(cfg.token_layout),
        drop_token_kinds=list(cfg.drop_token_kinds or []),
        tf_vocab=int(cfg.tf_vocab),
        gene_vocab=int(cfg.gene_vocab),
        context_vocab=int(cfg.context_vocab),
    )


def _to_device(batch, device: str):
    import dataclasses

    import torch

    from grn_agent.models.tf_eager.window_batch import TfEagerWindowBatch

    moved = {
        field.name: getattr(batch, field.name).to(device) if isinstance(getattr(batch, field.name), torch.Tensor) else getattr(batch, field.name)
        for field in dataclasses.fields(TfEagerWindowBatch)
    }
    return TfEagerWindowBatch(**moved)


def attention_for_edges(
    *,
    checkpoint: str | Path,
    windows_path: str | Path,
    edges: list[tuple[str, str]],
    device: str = "cpu",
    max_edges: int = 12,
) -> dict[str, Any]:
    """Run staged cross-attention for the selected TF→gene edges."""
    import torch

    from grn_agent.explain.attention import forward_with_attention, summarize_attention_for_gene

    chosen = [edge_key(tf, gene) for tf, gene in edges][: max(1, int(max_edges))]
    if not chosen:
        raise ValueError("Select at least one edge")
    model = _load_model(Path(checkpoint), device)
    by_tf: dict[str, list[str]] = {}
    for tf, gene in chosen:
        by_tf.setdefault(tf, []).append(gene)

    summaries: list[dict[str, Any]] = []
    missing: list[str] = []
    with torch.no_grad():
        for tf, genes in by_tf.items():
            record = load_tf_window(windows_path, tf)
            if record is None:
                missing.extend(f"{tf} → {gene}" for gene in genes)
                continue
            batch = _to_device(_batch_for_window(model, record), device)
            _logits, attn = forward_with_attention(model, batch)
            for gene in genes:
                position = gene_position(record, gene)
                if position is None:
                    missing.append(f"{tf} → {gene}")
                    continue
                summary = summarize_attention_for_gene(attn, batch, position)
                summary["source_tf"] = tf
                summary["target_gene"] = gene
                summary["edge"] = f"{tf} → {gene}"
                summaries.append(summary)
    if not summaries:
        detail = ", ".join(missing) if missing else "no windows matched"
        raise ValueError(f"Could not compute attention ({detail})")
    note = "Staged cross-attention mass for the selected edges."
    if missing:
        note += " Missing from scanned windows: " + ", ".join(missing)
    return {
        "channels": attention_frame(summaries),
        "local": local_attention_frame(summaries),
        "n_edges": len(summaries),
        "note": note,
    }


def latent_pca_for_edges(
    *,
    checkpoint: str | Path,
    windows_path: str | Path,
    edges: list[tuple[str, str]],
    scored: pd.DataFrame | None = None,
    device: str = "cpu",
    max_tfs: int = 6,
) -> dict[str, Any]:
    """PCA of final decoder states for genes in the selected TFs' windows."""
    import torch

    chosen = [edge_key(tf, gene) for tf, gene in edges]
    if not chosen:
        raise ValueError("Select at least one edge")
    tfs = []
    for tf, _gene in chosen:
        if tf not in tfs:
            tfs.append(tf)
        if len(tfs) >= max(1, int(max_tfs)):
            break
    selected_keys = set(chosen)
    model = _load_model(Path(checkpoint), device)
    vectors: list[np.ndarray] = []
    labels: list[dict[str, Any]] = []
    score_lookup: dict[tuple[str, str], float] = {}
    if scored is not None and not scored.empty and {"source_tf", "target_gene", "p_present"}.issubset(scored.columns):
        for _, row in scored.iterrows():
            try:
                score_lookup[edge_key(row["source_tf"], row["target_gene"])] = float(row["p_present"])
            except (TypeError, ValueError):
                continue

    captured: dict[str, Any] = {}

    def _hook(_module, inputs, _output):
        if inputs and inputs[0] is not None:
            captured["z"] = inputs[0].detach()

    handle = model.head.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            for tf in tfs:
                record = load_tf_window(windows_path, tf)
                if record is None:
                    continue
                batch = _to_device(_batch_for_window(model, record), device)
                captured.clear()
                model(batch)
                latent = captured.get("z")
                if latent is None:
                    continue
                latent_np = latent[0].detach().cpu().numpy()
                genes = list(record.get("genes") or [])
                for index, gene_rec in enumerate(genes):
                    if index >= latent_np.shape[0]:
                        break
                    gene = str(gene_rec.get("target_gene", "")).strip().upper()
                    if not gene:
                        continue
                    vectors.append(latent_np[index])
                    key = edge_key(tf, gene)
                    labels.append(
                        {
                            "source_tf": tf,
                            "target_gene": gene,
                            "selected": key in selected_keys,
                            "p_present": score_lookup.get(key, np.nan),
                        }
                    )
    finally:
        handle.remove()

    if len(vectors) < 3:
        raise ValueError("Model latent PCA needs at least 3 genes in the selected TF windows")
    matrix = np.vstack(vectors)
    scales = matrix.std(axis=0)
    scales[scales < 1e-8] = 1.0
    scores, variance = pca_scores(matrix / scales)
    frame = pd.DataFrame(labels)
    frame["pc1"] = scores[:, 0]
    frame["pc2"] = scores[:, 1]
    frame["edge"] = frame["source_tf"] + " → " + frame["target_gene"]
    return {
        "frame": frame,
        "variance": (float(variance[0]), float(variance[1])),
        "kind": "model_latent",
        "features": ["final_decoder_state"],
        "note": "PCA of the final decoder state for genes in the selected TF windows. Selected edges are marked.",
    }
