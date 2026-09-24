#!/usr/bin/env python3
"""Run inference for tf-eager windows and export scored edge tables."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import torch

from grn_agent.models.tf_eager import TfEagerConfig, TfEagerWindowModel, window_record_to_batch
from grn_agent.models.tf_eager.window_batch import TfEagerWindowBatch
from grn_agent.pipeline.config import load_yaml_config


def _load_windows(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _to_device(batch: TfEagerWindowBatch, device: str | torch.device) -> TfEagerWindowBatch:
    return TfEagerWindowBatch(
        token_kind=batch.token_kind.to(device),
        x_value=batch.x_value.to(device),
        conf=batch.conf.to(device),
        token_target_pos=batch.token_target_pos.to(device),
        token_mask=batch.token_mask.to(device),
        modality=batch.modality.to(device),
        mech_mask=batch.mech_mask.to(device),
        func_mask=batch.func_mask.to(device),
        context_idx=batch.context_idx.to(device),
        tf_idx=batch.tf_idx.to(device),
        gene_idx=batch.gene_idx.to(device),
        gene_pos=batch.gene_pos.to(device),
        gene_mask=batch.gene_mask.to(device),
        labels=batch.labels.to(device),
        sample_weight=batch.sample_weight.to(device),
    )


def _config_from_checkpoint(payload: dict[str, Any]) -> TfEagerConfig:
    raw = payload.get("config", {})
    if not isinstance(raw, dict):
        return TfEagerConfig()
    allowed = {f.name for f in dataclasses.fields(TfEagerConfig)}
    return TfEagerConfig(**{k: v for k, v in raw.items() if k in allowed})


def load_tf_eager_checkpoint(path: str | Path, device: str | torch.device) -> TfEagerWindowModel:
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(f"Invalid tf-eager checkpoint: {path}")
    model = TfEagerWindowModel(_config_from_checkpoint(payload))
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model


def _edge_reasoning(row: dict[str, Any]) -> str:
    """Deterministic scientific rationale from evidence fields (no model change)."""
    from grn_agent.explain.reasoning import build_edge_explanation

    return str(build_edge_explanation(row)["mechanism_reasoning"])


def score_windows(
    windows: list[dict[str, Any]],
    model: TfEagerWindowModel,
    *,
    device: str | torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for rec in windows:
            batch = _to_device(
                window_record_to_batch(
                    rec,
                    token_layout=model.cfg.token_layout,
                    drop_token_kinds=model.cfg.drop_token_kinds,
                    tf_vocab=model.cfg.tf_vocab,
                    gene_vocab=model.cfg.gene_vocab,
                    context_vocab=model.cfg.context_vocab,
                ),
                device,
            )
            probs = torch.sigmoid(model(batch))[0].detach().cpu().numpy()
            source_tf = str(rec["source_tf"]).strip().upper()
            context = rec.get("context") or {}
            context_id = str(context.get("context_id", "context"))
            cell_type = context.get("cell_type")
            species = context.get("species")
            window_index = int(rec.get("window_index", 0))
            for i, gene_rec in enumerate(rec.get("genes", [])):
                target_gene = str(gene_rec.get("target_gene", "")).strip().upper()
                if not target_gene:
                    continue
                evidence = gene_rec.get("evidence") or {}
                row = {
                    "source_tf": source_tf,
                    "target_gene": target_gene,
                    "context_id": context_id,
                    "cell_type": cell_type,
                    "species": species,
                    "window_index": window_index,
                    "window_position": i,
                    "candidate_bucket": gene_rec.get("candidate_bucket"),
                    "p_present": float(probs[i]),
                    "logit": float(torch.logit(torch.tensor(float(probs[i])), eps=1e-7).item()),
                    "confidence_score": float(probs[i]),
                    "label": gene_rec.get("label"),
                    "sample_weight": gene_rec.get("sample_weight", 1.0),
                    "correlation": evidence.get("correlation"),
                    "motif_present": evidence.get("motif_present"),
                    "accessibility": evidence.get("accessibility"),
                    "ensemble_prior": evidence.get("ensemble_prior"),
                }
                row["mechanism_reasoning"] = _edge_reasoning(row)
                rows.append(row)
    return rows


def _first_non_null(values: pd.Series) -> Any:
    for value in values:
        if pd.notna(value):
            return value
    return None


def aggregate_duplicate_scores(
    rows: list[dict[str, Any]],
    *,
    vote_threshold: float = 0.3,
) -> list[dict[str, Any]]:
    """Collapse repeated window scores for the same TF-target-context pair.

    The exported score remains continuous: the mean model probability across
    duplicate window scores.  Vote counts above ``vote_threshold`` are kept as
    reproducibility diagnostics.
    """
    if not rows:
        return []
    df = pd.DataFrame(rows)
    group_cols = ["source_tf", "target_gene", "context_id"]
    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot aggregate scored rows; missing columns: {missing}")

    out_rows: list[dict[str, Any]] = []
    for key, group in df.groupby(group_cols, sort=False, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        probs = group["p_present"].astype(float)
        logits = group["logit"].astype(float) if "logit" in group.columns else None
        n_votes = int(len(group))
        vote_count = int((probs > float(vote_threshold)).sum())
        vote_fraction = float(vote_count / max(n_votes, 1))
        mean_p = float(probs.mean())
        row = group.iloc[0].to_dict()
        row.update(
            {
                "source_tf": key[0],
                "target_gene": key[1],
                "context_id": key[2],
                "p_present": mean_p,
                "confidence_score": mean_p,
                "logit": float(torch.logit(torch.tensor(mean_p), eps=1e-7).item()),
                "p_present_mean": mean_p,
                "p_present_max": float(probs.max()),
                "p_present_min": float(probs.min()),
                "p_present_std": float(probs.std(ddof=0)) if n_votes > 1 else 0.0,
                "window_vote_count": vote_count,
                "window_vote_total": n_votes,
                "window_vote_fraction": vote_fraction,
                "aggregation_vote_threshold": float(vote_threshold),
            }
        )
        if logits is not None:
            row["logit_mean"] = float(logits.mean())
        for col in (
            "cell_type",
            "species",
            "candidate_bucket",
            "label",
            "sample_weight",
            "correlation",
            "motif_present",
            "accessibility",
            "ensemble_prior",
        ):
            if col in group.columns:
                row[col] = _first_non_null(group[col])
        row["mechanism_reasoning"] = _edge_reasoning(row)
        out_rows.append(row)
    return out_rows


def select_network_rows(rows: list[dict[str, Any]], *, threshold: float, topk_per_tf: int) -> list[dict[str, Any]]:
    df = pd.DataFrame(rows)
    if df.empty:
        return []
    selected = df[df["p_present"] >= float(threshold)].copy()
    if topk_per_tf > 0:
        ranked = df.sort_values(["source_tf", "p_present"], ascending=[True, False])
        topk = ranked.groupby("source_tf", as_index=False, group_keys=False).head(int(topk_per_tf))
        selected = pd.concat([selected, topk], ignore_index=True).drop_duplicates(
            subset=["source_tf", "target_gene", "context_id"]
        )
    selected = selected.sort_values(["p_present", "source_tf", "target_gene"], ascending=[False, True, True])
    return selected.to_dict(orient="records")


def write_scores_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(p, index=False)


def write_flat_evidence_jsonl(windows: list[dict[str, Any]], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fp:
        for rec in windows:
            source_tf = str(rec["source_tf"]).strip().upper()
            context = rec.get("context") or {}
            for gene_rec in rec.get("genes", []):
                target_gene = str(gene_rec.get("target_gene", "")).strip().upper()
                if not target_gene:
                    continue
                fp.write(
                    json.dumps(
                        {
                            "schema": "tf_eager_flat_edge_v1",
                            "edge": {"source_tf": source_tf, "target_gene": target_gene},
                            "context": {
                                "context_id": context.get("context_id"),
                                "cell_type": context.get("cell_type"),
                                "metadata": {
                                    "species": context.get("species"),
                                    "dataset_id": context.get("dataset_id"),
                                },
                            },
                            "evidence": gene_rec.get("evidence") or {},
                            "label": gene_rec.get("label"),
                            "sample_weight": gene_rec.get("sample_weight", 1.0),
                            "candidate_bucket": gene_rec.get("candidate_bucket"),
                        }
                    )
                    + "\n"
                )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="")
    ap.add_argument("--windows-jsonl", default="")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--out-scored-csv", default="")
    ap.add_argument("--out-network-csv", default="")
    ap.add_argument("--out-evidence-jsonl", default="")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--topk-per-tf", type=int, default=None)
    ap.add_argument("--aggregation-vote-threshold", type=float, default=None)
    ap.add_argument("--no-aggregate-duplicates", action="store_true")
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    cfg = load_yaml_config(args.config) if args.config.strip() else {}
    tf_cfg = cfg.get("tf_eager", {}) if isinstance(cfg.get("tf_eager", {}), dict) else {}
    infer_cfg = tf_cfg.get("infer", {}) if isinstance(tf_cfg.get("infer", {}), dict) else {}
    scoring_cfg = cfg.get("scoring", {}) if isinstance(cfg.get("scoring", {}), dict) else {}

    def _cfg(key: str, default: Any) -> Any:
        cli_value = getattr(args, key, None)
        if cli_value not in (None, ""):
            return cli_value
        for section in (infer_cfg, tf_cfg, cfg, scoring_cfg):
            if key in section:
                return section[key]
            dashed = key.replace("_", "-")
            if dashed in section:
                return section[dashed]
        return default

    windows_jsonl = str(_cfg("windows_jsonl", ""))
    checkpoint = str(_cfg("checkpoint", ""))
    out_scored_csv = str(_cfg("out_scored_csv", _cfg("scored_csv", "")))
    out_network_csv = str(_cfg("out_network_csv", _cfg("network_csv", "")))
    out_evidence_jsonl = str(_cfg("out_evidence_jsonl", _cfg("evidence_jsonl", "")))
    threshold = float(_cfg("threshold", 0.5))
    topk_per_tf = int(_cfg("topk_per_tf", 0))
    aggregation_vote_threshold = float(_cfg("aggregation_vote_threshold", 0.3))
    aggregate_duplicates = not bool(args.no_aggregate_duplicates or _cfg("no_aggregate_duplicates", False))
    device = str(_cfg("device", "")) or ("cuda" if torch.cuda.is_available() else "cpu")

    if not windows_jsonl or not checkpoint or not out_scored_csv:
        raise SystemExit("Missing required args: --windows-jsonl, --checkpoint, and --out-scored-csv")

    windows = _load_windows(windows_jsonl)
    model = load_tf_eager_checkpoint(checkpoint, device)
    raw_rows = score_windows(windows, model, device=device)
    rows = (
        aggregate_duplicate_scores(raw_rows, vote_threshold=aggregation_vote_threshold)
        if aggregate_duplicates
        else raw_rows
    )
    write_scores_csv(rows, out_scored_csv)
    print(
        f"Wrote {out_scored_csv} scored_edges={len(rows)} raw_scored_edges={len(raw_rows)} "
        f"aggregate_duplicates={aggregate_duplicates} aggregation_vote_threshold={aggregation_vote_threshold}"
    )

    if out_network_csv:
        network_rows = select_network_rows(rows, threshold=threshold, topk_per_tf=topk_per_tf)
        write_scores_csv(network_rows, out_network_csv)
        print(f"Wrote {out_network_csv} network_edges={len(network_rows)} threshold={threshold} topk_per_tf={topk_per_tf}")

    if out_evidence_jsonl:
        write_flat_evidence_jsonl(windows, out_evidence_jsonl)
        print(f"Wrote {out_evidence_jsonl}")


if __name__ == "__main__":
    main()
