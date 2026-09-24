"""Cross-attention attribution helpers for TF-EAGER (post-hoc only).

Does not modify the TF-EAGER model or scoring path. Attention weights are
captured by a temporary runtime monkeypatch of ``_attend`` during analysis.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn

from grn_agent.models.tf_eager.window_batch import (
    TOKEN_LAYOUT_EDGE_COMPACT,
    TOKEN_LAYOUT_EVIDENCE,
    TfEagerTokenKind,
    TfEagerWindowBatch,
)

TOKEN_KIND_NAMES: dict[int, str] = {
    TfEagerTokenKind.TF: "TF",
    TfEagerTokenKind.CTX: "CTX",
    TfEagerTokenKind.GENE: "GENE",
    TfEagerTokenKind.EXPR: "EXPR",
    TfEagerTokenKind.NETWORK: "NETWORK",
    TfEagerTokenKind.MOTIF: "MOTIF",
    TfEagerTokenKind.ACC: "ACC",
    TfEagerTokenKind.LINK: "LINK",
    TfEagerTokenKind.PRIOR: "PRIOR",
    TfEagerTokenKind.ORTHO: "ORTHO",
    TfEagerTokenKind.LIT: "LIT",
    TfEagerTokenKind.PAD: "PAD",
}

CHANNEL_GROUPS: dict[str, tuple[int, ...]] = {
    "mechanistic": (TfEagerTokenKind.MOTIF, TfEagerTokenKind.ACC, TfEagerTokenKind.LINK),
    "functional": (TfEagerTokenKind.EXPR, TfEagerTokenKind.NETWORK),
    "identity": (TfEagerTokenKind.TF, TfEagerTokenKind.CTX, TfEagerTokenKind.GENE),
    "prior_lit": (TfEagerTokenKind.PRIOR, TfEagerTokenKind.ORTHO, TfEagerTokenKind.LIT),
}

COMPACT_FEATURE_SLICES: dict[str, tuple[int, int]] = {
    "expression": (0, 7),
    "network": (7, 12),
    "motif": (12, 15),
    "accessibility": (15, 17),
    "linkage": (17, 20),
    "prior": (20, 26),
    "orthology": (26, 30),
    "literature": (30, 32),
}

COMPACT_OCCLUSION_PLANS: dict[str, tuple[str, ...]] = {
    "drop_motif": ("motif",),
    "drop_accessibility": ("accessibility", "linkage"),
    "drop_expression_network": ("expression", "network"),
    "drop_mechanistic": ("motif", "accessibility", "linkage"),
    "drop_functional": ("expression", "network"),
    "drop_prior_lit": ("prior", "orthology", "literature"),
}


@contextmanager
def capture_cross_attention(model: nn.Module) -> Iterator[dict[str, torch.Tensor | None]]:
    """Temporarily record staged cross-attention weights without changing model code.

    Restores the original ``_attend`` on exit. Logits match the unmodified forward
    path (same MHA call, only ``need_weights=True`` for recording).
    """
    stored: dict[str, torch.Tensor | None] = {
        "stage1": None,
        "stage2": None,
        "stage3": None,
    }
    original = model._attend

    def _attend_capture(
        z: torch.Tensor,
        h: torch.Tensor,
        allow_mask: torch.Tensor,
        valid_mask: torch.Tensor,
        layer: nn.MultiheadAttention,
        *,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        allow = (allow_mask > 0.5) & (valid_mask > 0.5)
        fallback = valid_mask > 0.5
        no_allow = allow.sum(dim=1, keepdim=True) <= 0
        allow = torch.where(no_allow, fallback, allow)
        # Always request weights for capture; still honor need_weights in return.
        out, weights = layer(
            z,
            h,
            h,
            key_padding_mask=~allow,
            need_weights=True,
            average_attn_weights=True,
        )
        if layer is model.stage1:
            stored["stage1"] = weights
        elif layer is model.stage2:
            stored["stage2"] = weights
        else:
            stored["stage3"] = weights
        return out, (weights if need_weights else None)

    model._attend = _attend_capture  # type: ignore[method-assign]
    try:
        yield stored
    finally:
        model._attend = original  # type: ignore[method-assign]


def forward_with_attention(
    model: nn.Module,
    batch: TfEagerWindowBatch,
) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
    """Run ``forward`` while capturing cross-attention maps.

    Prefers the native ``return_attention=True`` path when available; falls back
    to a temporary ``_attend`` monkeypatch for older checkpoints/code.
    """
    forward_fn = getattr(model, "forward", None)
    if forward_fn is not None:
        try:
            out = model(batch, return_attention=True)
            if isinstance(out, tuple) and len(out) == 2:
                logits, attn = out
                return logits, attn
        except TypeError:
            pass
    with capture_cross_attention(model) as attn:
        logits = model(batch)
    return logits, attn


def infer_token_layout(batch: TfEagerWindowBatch, *, batch_index: int = 0) -> str:
    """Detect evidence_tokens vs edge_compact from which token kinds are present."""
    kinds = batch.token_kind[batch_index]
    mask = batch.token_mask[batch_index] > 0.5
    present = set(int(x) for x in kinds[mask].detach().cpu().tolist())
    evidence_kinds = {
        TfEagerTokenKind.EXPR,
        TfEagerTokenKind.NETWORK,
        TfEagerTokenKind.MOTIF,
        TfEagerTokenKind.ACC,
        TfEagerTokenKind.LINK,
        TfEagerTokenKind.PRIOR,
    }
    if present & evidence_kinds:
        return TOKEN_LAYOUT_EVIDENCE
    return TOKEN_LAYOUT_EDGE_COMPACT


def summarize_attention_for_gene(
    attn: dict[str, torch.Tensor | None],
    batch: TfEagerWindowBatch,
    gene_position: int,
    *,
    batch_index: int = 0,
) -> dict[str, Any]:
    """Aggregate stage cross-attention mass by token kind / locality for one query gene."""
    kinds = batch.token_kind[batch_index].detach().cpu().numpy().astype(int)
    mask = batch.token_mask[batch_index].detach().cpu().numpy() > 0.5
    target_pos = batch.token_target_pos[batch_index].detach().cpu().numpy().astype(int)
    layout = infer_token_layout(batch, batch_index=batch_index)

    stages: dict[str, dict[str, float]] = {}
    local_vs_global: dict[str, dict[str, float]] = {}
    for stage_name, weights in attn.items():
        if weights is None:
            continue
        row = weights[batch_index, int(gene_position)].detach().cpu().numpy().astype(np.float64)
        row = np.where(mask, row, 0.0)
        total = float(row.sum())
        if total <= 0:
            continue
        by_kind: dict[str, float] = {name: 0.0 for name in TOKEN_KIND_NAMES.values() if name != "PAD"}
        for idx, mass in enumerate(row):
            if not mask[idx]:
                continue
            name = TOKEN_KIND_NAMES.get(int(kinds[idx]), "UNK")
            if name == "PAD":
                continue
            by_kind[name] = by_kind.get(name, 0.0) + float(mass / total)
        stages[stage_name] = by_kind

        local_mask = (target_pos == int(gene_position)) & mask
        local = float(row[local_mask].sum() / total) if local_mask.any() else 0.0
        gene_tokens = (kinds == TfEagerTokenKind.GENE) & mask
        other_genes = gene_tokens & (target_pos != int(gene_position))
        local_vs_global[stage_name] = {
            "local_target_tokens": local,
            "other_gene_tokens": float(row[other_genes].sum() / total) if other_genes.any() else 0.0,
            "tf_ctx_tokens": float(
                row[((kinds == TfEagerTokenKind.TF) | (kinds == TfEagerTokenKind.CTX)) & mask].sum() / total
            ),
            "other_window_tokens": float(1.0 - local),
        }

    group_mass: dict[str, dict[str, float]] = {}
    for stage_name, by_kind in stages.items():
        group_mass[stage_name] = {}
        for group, kind_ids in CHANNEL_GROUPS.items():
            group_mass[stage_name][group] = float(
                sum(by_kind.get(TOKEN_KIND_NAMES[k], 0.0) for k in kind_ids)
            )

    return {
        "gene_position": int(gene_position),
        "token_layout": layout,
        "by_stage_token_kind": stages,
        "by_stage_channel_group": group_mass,
        "by_stage_local_vs_global": local_vs_global,
    }


def zero_token_kinds(batch: TfEagerWindowBatch, kind_ids: set[int]) -> TfEagerWindowBatch:
    """Clone batch and zero values/confidence for selected token kinds (keep masks)."""
    if not kind_ids:
        return batch
    kind = batch.token_kind
    drop = torch.zeros_like(batch.token_mask, dtype=torch.bool)
    for kid in kind_ids:
        drop = drop | (kind == int(kid))
    x_value = batch.x_value.clone()
    conf = batch.conf.clone()
    x_value = x_value.masked_fill(drop.unsqueeze(-1), 0.0)
    conf = conf.masked_fill(drop, 0.0)
    return TfEagerWindowBatch(
        token_kind=kind,
        x_value=x_value,
        conf=conf,
        token_target_pos=batch.token_target_pos,
        token_mask=batch.token_mask,
        modality=batch.modality,
        mech_mask=batch.mech_mask,
        func_mask=batch.func_mask,
        context_idx=batch.context_idx,
        tf_idx=batch.tf_idx,
        gene_idx=batch.gene_idx,
        gene_pos=batch.gene_pos,
        gene_mask=batch.gene_mask,
        labels=batch.labels,
        sample_weight=batch.sample_weight,
    )


def zero_compact_feature_groups(
    batch: TfEagerWindowBatch,
    groups: tuple[str, ...] | list[str],
    *,
    gene_position: int | None = None,
) -> TfEagerWindowBatch:
    """Zero selected feature slices inside edge_compact GENE tokens.

    If ``gene_position`` is set, only that candidate's GENE token is ablated
    (local evidence occlusion). Otherwise all GENE tokens in the window are.
    """
    x_value = batch.x_value.clone()
    gene_rows = batch.token_kind == TfEagerTokenKind.GENE
    if gene_position is not None:
        gene_rows = gene_rows & (batch.token_target_pos == int(gene_position))
    for group in groups:
        if group not in COMPACT_FEATURE_SLICES:
            raise ValueError(f"unknown compact feature group: {group!r}")
        lo, hi = COMPACT_FEATURE_SLICES[group]
        # broadcast over batch and matching GENE rows
        x_value[:, :, lo:hi] = torch.where(
            gene_rows.unsqueeze(-1),
            torch.zeros_like(x_value[:, :, lo:hi]),
            x_value[:, :, lo:hi],
        )
    return TfEagerWindowBatch(
        token_kind=batch.token_kind,
        x_value=x_value,
        conf=batch.conf,
        token_target_pos=batch.token_target_pos,
        token_mask=batch.token_mask,
        modality=batch.modality,
        mech_mask=batch.mech_mask,
        func_mask=batch.func_mask,
        context_idx=batch.context_idx,
        tf_idx=batch.tf_idx,
        gene_idx=batch.gene_idx,
        gene_pos=batch.gene_pos,
        gene_mask=batch.gene_mask,
        labels=batch.labels,
        sample_weight=batch.sample_weight,
    )


def occlusion_deltas(
    model: torch.nn.Module,
    batch: TfEagerWindowBatch,
    gene_position: int,
    *,
    batch_index: int = 0,
) -> dict[str, float]:
    """Δp under biologically meaningful occlusions (layout-aware)."""
    model.eval()
    layout = infer_token_layout(batch, batch_index=batch_index)
    with torch.no_grad():
        base = float(torch.sigmoid(model(batch))[batch_index, gene_position].item())
        out: dict[str, float] = {"p_base": base, "token_layout": layout}  # type: ignore[dict-item]

        if layout == TOKEN_LAYOUT_EDGE_COMPACT:
            plans = {
                **COMPACT_OCCLUSION_PLANS,
                "drop_local_evidence": tuple(COMPACT_FEATURE_SLICES.keys()),
            }
            for name, groups in plans.items():
                # local candidate evidence only
                occluded = zero_compact_feature_groups(batch, groups, gene_position=gene_position)
                p = float(torch.sigmoid(model(occluded))[batch_index, gene_position].item())
                out[f"p_{name}"] = p
                out[f"delta_{name}"] = base - p
            # neighborhood: zero all OTHER genes' compact features
            other = zero_compact_feature_groups(batch, tuple(COMPACT_FEATURE_SLICES.keys()), gene_position=None)
            # restore local gene features from original
            local_rows = (batch.token_kind == TfEagerTokenKind.GENE) & (batch.token_target_pos == int(gene_position))
            x_restored = other.x_value.clone()
            x_restored = torch.where(local_rows.unsqueeze(-1), batch.x_value, x_restored)
            other_only = TfEagerWindowBatch(
                token_kind=batch.token_kind,
                x_value=x_restored,
                conf=batch.conf,
                token_target_pos=batch.token_target_pos,
                token_mask=batch.token_mask,
                modality=batch.modality,
                mech_mask=batch.mech_mask,
                func_mask=batch.func_mask,
                context_idx=batch.context_idx,
                tf_idx=batch.tf_idx,
                gene_idx=batch.gene_idx,
                gene_pos=batch.gene_pos,
                gene_mask=batch.gene_mask,
                labels=batch.labels,
                sample_weight=batch.sample_weight,
            )
            p_nb = float(torch.sigmoid(model(other_only))[batch_index, gene_position].item())
            out["p_drop_neighborhood"] = p_nb
            out["delta_drop_neighborhood"] = base - p_nb
            # identity embeddings via zeroing gene/tf query indices (hash buckets -> 0)
            id_batch = TfEagerWindowBatch(
                token_kind=batch.token_kind,
                x_value=batch.x_value,
                conf=batch.conf,
                token_target_pos=batch.token_target_pos,
                token_mask=batch.token_mask,
                modality=batch.modality,
                mech_mask=batch.mech_mask,
                func_mask=batch.func_mask,
                context_idx=torch.zeros_like(batch.context_idx),
                tf_idx=torch.zeros_like(batch.tf_idx),
                gene_idx=torch.zeros_like(batch.gene_idx),
                gene_pos=batch.gene_pos,
                gene_mask=batch.gene_mask,
                labels=batch.labels,
                sample_weight=batch.sample_weight,
            )
            p_id = float(torch.sigmoid(model(id_batch))[batch_index, gene_position].item())
            out["p_drop_identity"] = p_id
            out["delta_drop_identity"] = base - p_id
            return out

        # evidence_tokens layout: zero whole token kinds
        plans = {
            "drop_motif": {TfEagerTokenKind.MOTIF},
            "drop_accessibility": {TfEagerTokenKind.ACC, TfEagerTokenKind.LINK},
            "drop_expression_network": {TfEagerTokenKind.EXPR, TfEagerTokenKind.NETWORK},
            "drop_mechanistic": {
                TfEagerTokenKind.MOTIF,
                TfEagerTokenKind.ACC,
                TfEagerTokenKind.LINK,
            },
            "drop_identity": {TfEagerTokenKind.TF, TfEagerTokenKind.CTX, TfEagerTokenKind.GENE},
        }
        for name, kinds in plans.items():
            occluded = zero_token_kinds(batch, kinds)
            p = float(torch.sigmoid(model(occluded))[batch_index, gene_position].item())
            out[f"p_{name}"] = p
            out[f"delta_{name}"] = base - p
    return out
