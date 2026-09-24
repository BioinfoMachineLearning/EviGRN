"""Deterministic scientific explanation helpers for GRNAgent (no LLM)."""

from __future__ import annotations

from typing import Any


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out:  # NaN
        return None
    return out


def _as_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"", "nan", "none", "null"}:
        return None
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return None


def strength_label(p_present: float) -> str:
    if p_present >= 0.80:
        return "strong"
    if p_present >= 0.55:
        return "moderate"
    return "weak"


def build_edge_explanation(row: dict[str, Any]) -> dict[str, Any]:
    """Build quantitative factors + plain scientific rationale from scored-edge fields."""
    tf = str(row.get("source_tf", "")).strip()
    gene = str(row.get("target_gene", "")).strip()
    cell_type = str(row.get("cell_type") or "").strip() or None
    p = _as_float(row.get("p_present"))
    if p is None:
        p = 0.0
    corr = _as_float(row.get("correlation"))
    motif = _as_bool(row.get("motif_present"))
    acc = _as_float(row.get("accessibility"))
    vote_total = _as_float(row.get("window_vote_total"))
    vote_frac = _as_float(row.get("window_vote_fraction"))
    p_std = _as_float(row.get("p_present_std"))

    factors: list[dict[str, Any]] = [
        {"name": "Model probability", "value": round(p, 4), "detail": strength_label(p)}
    ]
    missing: list[str] = []

    if corr is None:
        missing.append("coexpression")
    else:
        if corr > 0.1:
            direction = "activation-leaning"
        elif corr < -0.1:
            direction = "repression-leaning"
        else:
            direction = "weak / undirected"
        factors.append(
            {
                "name": "Coexpression",
                "value": round(corr, 4),
                "detail": direction,
            }
        )

    if motif is None:
        missing.append("motif")
    else:
        factors.append(
            {
                "name": "Motif evidence",
                "value": bool(motif),
                "detail": "present" if motif else "not detected",
            }
        )

    if acc is None:
        missing.append("accessibility")
    else:
        factors.append(
            {
                "name": "Accessibility / linkage",
                "value": round(acc, 4),
                "detail": "supportive" if acc > 0 else "absent or zero",
            }
        )

    if vote_total is not None and vote_total > 1:
        factors.append(
            {
                "name": "Window stability",
                "value": {
                    "vote_fraction": None if vote_frac is None else round(vote_frac, 4),
                    "p_std": None if p_std is None else round(p_std, 4),
                    "n_windows": int(vote_total),
                },
                "detail": "aggregated across TF-centered subgraph samples",
            }
        )

    rationale = explanation_to_mechanism_text(
        tf=tf,
        gene=gene,
        p_present=p,
        cell_type=cell_type,
        corr=corr,
        motif=motif,
        accessibility=acc,
        vote_total=vote_total,
        vote_frac=vote_frac,
        p_std=p_std,
        missing=missing,
    )
    return {
        "source_tf": tf,
        "target_gene": gene,
        "cell_type": cell_type,
        "p_present": p,
        "strength": strength_label(p),
        "factors": factors,
        "missing_modalities": missing,
        "mechanism_reasoning": rationale,
        "source": "deterministic_evidence_template",
    }


def explanation_to_mechanism_text(
    *,
    tf: str,
    gene: str,
    p_present: float,
    cell_type: str | None,
    corr: float | None,
    motif: bool | None,
    accessibility: float | None,
    vote_total: float | None = None,
    vote_frac: float | None = None,
    p_std: float | None = None,
    missing: list[str] | None = None,
) -> str:
    """Plain scientific prose explaining a TF-EAGER edge score."""
    strength = strength_label(p_present)
    context_bit = f" in this {cell_type} context" if cell_type else ""
    sentences = [
        (
            f"TF-EAGER assigns a {strength} probability that {tf} regulates {gene}"
            f"{context_bit} (p_present={p_present:.3f})."
        )
    ]

    support: list[str] = []
    if corr is not None:
        if corr > 0.1:
            support.append(f"positive coexpression (r={corr:.3f}, activation-leaning)")
        elif corr < -0.1:
            support.append(f"negative coexpression (r={corr:.3f}, repression-leaning)")
        else:
            support.append(f"weak coexpression (r={corr:.3f})")
    if motif is True:
        support.append("a motif hit consistent with TF binding potential near the target")
    elif motif is False:
        support.append("no motif hit detected in the available scan")
    if accessibility is not None:
        if accessibility > 0:
            support.append(f"accessible chromatin / linkage support ({accessibility:.3f})")
        else:
            support.append("no positive accessibility support in the local evidence window")

    supportive = 0
    if corr is not None and abs(corr) > 0.1:
        supportive += 1
    if motif is True:
        supportive += 1
    if accessibility is not None and accessibility > 0:
        supportive += 1

    if support:
        sentences.append("Local evidence contributing to this score includes " + "; ".join(support) + ".")
    else:
        sentences.append("Limited structured evidence fields were available for this pair.")

    if vote_total is not None and vote_total > 1:
        stab = f"The score was aggregated across {int(vote_total)} TF-centered subgraph samples"
        if vote_frac is not None:
            stab += f" (fraction above vote threshold={vote_frac:.2f}"
            if p_std is not None:
                stab += f", std={p_std:.3f}"
            stab += ")"
        stab += ", which helps assess reproducibility of the local neighborhood draw."
        sentences.append(stab)

    if missing:
        sentences.append(
            "Unavailable or unreported modalities for this edge: "
            + ", ".join(missing)
            + ". Missing channels should be interpreted as absent evidence rather than negative evidence."
        )

    if strength == "weak":
        sentences.append(
            "Overall, the prediction remains weak and should be treated as exploratory pending additional evidence."
        )
    elif strength == "strong" and supportive >= 2:
        sentences.append(
            "Overall, multiple local evidence channels are consistent with prioritizing this interaction for follow-up."
        )
    elif strength == "strong" and supportive == 1:
        sentences.append(
            "Overall, the high score is supported by a single primary evidence channel; treat as hypothesis-generating "
            "unless additional modalities agree."
        )
    elif strength == "strong":
        sentences.append(
            "Overall, the high score is not explained by motif, accessibility, or coexpression alone; "
            "it likely reflects TF/gene identity and TF-centered neighborhood context, so interpret with caution "
            "and inspect cross-attention / occlusion attribution before biological claims."
        )
    else:
        sentences.append(
            "Overall, evidence is mixed; use the probability together with modality coverage rather than the score alone."
        )

    return " ".join(sentences)


def enrich_scored_row(row: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of a scored-edge row with upgraded mechanism_reasoning."""
    out = dict(row)
    explanation = build_edge_explanation(out)
    out["mechanism_reasoning"] = explanation["mechanism_reasoning"]
    out["explanation_strength"] = explanation["strength"]
    out["explanation_missing_modalities"] = ",".join(explanation["missing_modalities"])
    return out


def format_acquisition_narrative(manifest: dict[str, Any]) -> str:
    """Plain-language acquisition / QC narrative from multimodal_manifest.json."""
    species = manifest.get("species", "unknown")
    cell_type = manifest.get("cell_type", "unknown")
    acc = manifest.get("accessibility") or {}
    motifs = manifest.get("motifs") or {}
    qc = manifest.get("qc") or manifest.get("qc_report") or {}

    accession = acc.get("accession") or "N/A"
    assay = acc.get("assay") or "none"
    source = acc.get("source") or "none"
    coverage = acc.get("promoter_coverage_of_rnaseq_genes")
    motif_count = motifs.get("tf_motif_count")
    motif_overlap = motifs.get("tf_overlap_with_rnaseq")
    motif_status = motifs.get("status")
    passed = bool(qc.get("pass"))
    reasons = qc.get("rejection_reasons") or []
    warnings = qc.get("warnings") or []
    acc_qc = qc.get("accessibility_qc") or {}
    decision = acc_qc.get("decision")
    score = acc_qc.get("score")

    parts = [
        (
            f"Acquisition assembled a multimodal package for {species} / {cell_type}. "
            f"Accessibility source={source}, assay={assay}, accession={accession}."
        )
    ]
    if coverage is not None:
        try:
            parts.append(f"Promoter coverage of RNA-expressed genes was {float(coverage):.2%}.")
        except (TypeError, ValueError):
            pass
    if motif_count is not None:
        motif_line = f"Motif database reported {motif_count} TF motifs"
        if motif_overlap is not None:
            try:
                motif_line += f" with RNA overlap {float(motif_overlap):.2%}"
            except (TypeError, ValueError):
                pass
        if motif_status:
            motif_line += f" (status={motif_status})"
        parts.append(motif_line + ".")

    if score is not None or decision:
        qc_bits = []
        if decision:
            qc_bits.append(f"decision={decision}")
        if score is not None:
            try:
                qc_bits.append(f"score={float(score):.3f}")
            except (TypeError, ValueError):
                pass
        parts.append("Accessibility QC " + ", ".join(qc_bits) + ".")

    if passed:
        parts.append("Final compatibility QC accepted the multimodal package.")
    else:
        parts.append("Final compatibility QC did not fully accept the package.")
        if reasons:
            parts.append("Rejection / fail reasons: " + "; ".join(str(r) for r in reasons) + ".")
    if warnings:
        parts.append("Warnings: " + "; ".join(str(w) for w in warnings) + ".")
    if source in {"none", "", None} or assay in {"none", "", None}:
        parts.append(
            "No usable accessibility channel was retained; downstream scoring may operate as expression-only "
            "unless motif evidence was recovered independently."
        )
    return " ".join(parts)
