"""
Validate biological/technical compatibility between RNA + accessibility datasets.

Benchmark gold labels are not used for pass/fail (avoids using supervision to select data).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

_DEBUG_LOG_PATH = os.environ.get("GRN_AGENT_DEBUG_LOG_PATH", "").strip()
_DEBUG_SESSION = os.environ.get("GRN_AGENT_DEBUG_SESSION", "").strip()


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    if not _DEBUG_LOG_PATH:
        return
    payload = {
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    if _DEBUG_SESSION:
        payload["sessionId"] = _DEBUG_SESSION
    try:
        with Path(_DEBUG_LOG_PATH).open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except OSError:
        pass


def _norm(s: str | None) -> str:
    return str(s or "").strip().lower()


def _is_available_status(meta: dict[str, Any]) -> bool:
    source = _norm(meta.get("source"))
    status = _norm(meta.get("status"))
    if source in {"geo", "user_provided"}:
        return status not in {"", "not_found", "none", "incomplete", "no_usable_files", "unavailable"}
    return status == "released"


def canonical_species_label(s: str | None) -> str:
    """
    Map RNA CLI values (``mouse``) and ENCODE ``organism.scientific_name`` strings
    to a single slug so strict QC does not false-fail on vocabulary mismatch.
    """
    x = _norm(s)
    if not x or x in ("unknown", "na", "n/a"):
        return ""
    if "mus musculus" in x or x == "mouse" or x == "mus":
        return "mouse"
    if "homo sapiens" in x or x == "human":
        return "human"
    if "rattus norvegicus" in x or x == "rat":
        return "rat"
    if (
        "drosophila melanogaster" in x
        or x in {"drosophila", "fly", "dmel", "dm6"}
        or x.startswith("drosophila ")
    ):
        return "drosophila"
    return x


def _cell_type_match(rna_ct: str, acc_ct: str) -> bool:
    """
    Lightweight ontology-style matcher:
    - exact match
    - substring containment
    - token Jaccard overlap >= 0.6
    """
    a = _norm(rna_ct)
    b = _norm(acc_ct)
    if not a or not b:
        return False
    if a == b:
        return True
    if a in b or b in a:
        return True
    ta = {t for t in a.replace("-", " ").split() if t}
    tb = {t for t in b.replace("-", " ").split() if t}
    if not ta or not tb:
        return False
    j = len(ta & tb) / max(1, len(ta | tb))
    return j >= 0.6


def _semantic_text_match(query: str, text: str) -> bool:
    """Lightweight semantic-ish token overlap against free-text metadata."""
    q = _norm(query)
    t = _norm(text)
    if not q or not t:
        return False
    qt = {x for x in q.replace("-", " ").split() if x}
    tt = {x for x in t.replace("-", " ").split() if x}
    if not qt or not tt:
        return False
    overlap = qt & tt
    return len(overlap) >= 2 or (len(qt) <= 2 and len(overlap) == len(qt))


def _is_opaque_cell_type_label(value: str) -> bool:
    """
    Detect cell labels that are likely coded names (cell lines/IDs) rather than
    natural-language ontology phrases, e.g. ``ES-E14``.
    """
    v = _norm(value)
    if not v:
        return True
    toks = [t for t in v.replace("_", "-").split("-") if t]
    if not toks:
        return True
    # Opaque if every token is short or contains digits.
    return all((len(t) <= 4) or any(ch.isdigit() for ch in t) for t in toks)


def _norm_cell_line(value: str | None) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "", _norm(value))


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _text_token_overlap_score(a: str, b: str) -> float:
    aa = {t for t in _norm(a).replace("-", " ").split() if t}
    bb = {t for t in _norm(b).replace("-", " ").split() if t}
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / max(1, len(aa | bb))


def _match_score_exact_related(rna_value: str, acc_value: str) -> float:
    a = _norm(rna_value)
    b = _norm(acc_value)
    if not a or not b:
        return 0.5
    if a == b:
        return 1.0
    if a in b or b in a:
        return 0.85
    overlap = _text_token_overlap_score(a, b)
    if overlap >= 0.6:
        return 0.75
    if overlap >= 0.3:
        return 0.5
    return 0.0


def _protocol_match_score(assay: str) -> float:
    a = _norm(assay)
    if not a or a in {"unknown", "na", "n/a", "accessibility"}:
        return 0.5
    if "atac" in a or "dnase" in a:
        return 1.0
    return 0.5


def _promoter_score(promoter_cov: float | None) -> float:
    if promoter_cov is None:
        return 0.5
    p = _clip01(promoter_cov)
    if p < 0.25:
        return 0.0
    if p < 0.55:
        return (p - 0.25) / 0.30
    return 1.0


def _peak_distribution_score(promoter_peak_fraction: float | None) -> float:
    if promoter_peak_fraction is None:
        return 0.7
    r_prom = _clip01(promoter_peak_fraction)
    if r_prom < 0.05:
        return 0.3
    if 0.10 <= r_prom <= 0.60:
        return 1.0
    if r_prom > 0.70:
        return 0.5
    return 0.7


def _decision_from_score(score: float) -> str:
    if score >= 0.70:
        return "accept"
    if score >= 0.50:
        return "conditional_accept"
    return "reject"


def validate_dataset_compatibility(
    rna_meta: dict[str, Any],
    accessibility_meta: dict[str, Any],
    beeline_gold_genes: set[str],
    beeline_gold_tfs: set[str],
    strict: bool = True,
    min_promoter_coverage: float = 0.7,
) -> dict[str, Any]:
    """
    Enforce multimodal compatibility with a scored accessibility QC model.

    ``beeline_gold_genes`` and ``beeline_gold_tfs`` are kept for call-site compatibility
    but are not used to accept or reject a dataset.
    """
    reasons: list[str] = []
    warnings: list[str] = []
    run_id = str(accessibility_meta.get("accession") or rna_meta.get("accession") or "acq-run")
    _debug_log(
        run_id,
        "H1-H4",
        "compatibility.py:validate_dataset_compatibility:entry",
        "compatibility_input_snapshot",
        {
            "strict": strict,
            "rna_species": rna_meta.get("species"),
            "acc_species": accessibility_meta.get("species"),
            "rna_cell_type": rna_meta.get("cell_type"),
            "acc_cell_type": accessibility_meta.get("cell_type"),
            "rna_lineage": rna_meta.get("lineage"),
            "acc_lineage": accessibility_meta.get("lineage"),
            "rna_state": rna_meta.get("state"),
            "acc_state": accessibility_meta.get("state"),
            "perturbation": accessibility_meta.get("perturbation"),
            "rna_reps": rna_meta.get("n_replicates"),
            "acc_reps": accessibility_meta.get("n_replicates"),
            "promoter_cov": accessibility_meta.get("promoter_coverage_of_targets"),
            "min_promoter_coverage": min_promoter_coverage,
            "status_rna": rna_meta.get("status"),
            "status_acc": accessibility_meta.get("status"),
        },
    )
    
    weights = {
        "signal": 0.20,
        "tss": 0.20,
        "promoter_coverage": 0.15,
        "peak_distribution": 0.10,
        "replicate_concordance": 0.15,
        "biological_match": 0.15,
        "usability": 0.05,
    }

    # A. Biological match
    bio_pass = True
    rna_species = canonical_species_label(rna_meta.get("species", ""))
    acc_species = canonical_species_label(accessibility_meta.get("species", ""))
    if not acc_species:
        bio_pass = False
        reasons.append("Accessibility species missing/unknown (could not map organism metadata)")
    elif rna_species != acc_species:
        bio_pass = False
        reasons.append(f"Species mismatch: RNA={rna_species or _norm(rna_meta.get('species'))}, accessibility={acc_species}")
    
    rna_ct = _norm(rna_meta.get("cell_type", ""))
    acc_ct = _norm(accessibility_meta.get("cell_type", ""))
    desc = _norm(accessibility_meta.get("description", ""))
    cell_type_mismatch = False
    if rna_ct:
        if acc_ct and acc_ct not in ("unknown", "na", "n/a"):
            if not _cell_type_match(rna_ct, acc_ct):
                desc_match = bool(desc and _semantic_text_match(rna_ct, desc))
                opaque_acc = _is_opaque_cell_type_label(acc_ct)
                if (not desc_match) and (not opaque_acc):
                    cell_type_mismatch = True
                    bio_pass = False
                    reasons.append(f"Cell type mismatch: RNA={rna_ct}, accessibility={acc_ct}")
        elif desc:
            if not _semantic_text_match(rna_ct, desc):
                cell_type_mismatch = True
                bio_pass = False
                reasons.append("Cell type semantic mismatch from accessibility description")

    rna_line = _norm_cell_line(rna_meta.get("cell_line", ""))
    acc_line = _norm_cell_line(accessibility_meta.get("cell_line", ""))
    if rna_line and not acc_line:
        if strict:
            bio_pass = False
            reasons.append(
                f"Cell-line missing in accessibility metadata while RNA specifies line={_norm(rna_meta.get('cell_line'))}"
            )
        else:
            warnings.append(
                f"Cell-line missing in accessibility metadata while RNA specifies line={_norm(rna_meta.get('cell_line'))}"
            )
    elif rna_line and acc_line:
        if not (rna_line == acc_line or rna_line in acc_line or acc_line in rna_line):
            if strict:
                bio_pass = False
                reasons.append(
                    f"Cell-line mismatch: RNA={_norm(rna_meta.get('cell_line'))}, accessibility={_norm(accessibility_meta.get('cell_line'))}"
                )
            else:
                warnings.append(
                    f"Cell-line mismatch (non-strict): RNA={_norm(rna_meta.get('cell_line'))}, accessibility={_norm(accessibility_meta.get('cell_line'))}"
                )

    # Strict lineage/state compatibility (Acceptance A)
    rna_lineage = _norm(rna_meta.get("lineage", ""))
    acc_lineage = _norm(accessibility_meta.get("lineage", ""))
    if strict and rna_lineage and acc_lineage and rna_lineage != acc_lineage:
        bio_pass = False
        reasons.append(f"Cross-lineage mismatch: RNA={rna_lineage}, accessibility={acc_lineage}")
    elif (not strict) and rna_lineage and acc_lineage and rna_lineage != acc_lineage:
        warnings.append(f"Lineage mismatch (non-strict): RNA={rna_lineage}, accessibility={acc_lineage}")

    rna_state = _norm(rna_meta.get("state", ""))
    acc_state = _norm(accessibility_meta.get("state", ""))
    if strict and rna_state and acc_state and rna_state != acc_state:
        bio_pass = False
        reasons.append(f"State mismatch: RNA={rna_state}, accessibility={acc_state}")
    elif (not strict) and rna_state and acc_state and rna_state != acc_state:
        warnings.append(f"State mismatch (non-strict): RNA={rna_state}, accessibility={acc_state}")

    # No major perturbations unless explicitly modeled
    # Convention:
    #   rna_meta["allow_perturbation"] = True to permit.
    #   accessibility_meta["perturbation"] should describe treatment/engineering.
    allow_perturb = bool(rna_meta.get("allow_perturbation", False))
    perturb = _norm(accessibility_meta.get("perturbation", ""))
    if perturb in ("", "unknown", "na", "n/a"):
        pass
    elif perturb not in ("none", "control", "untreated", "naive") and not allow_perturb:
        bio_pass = False
        reasons.append(f"Strong treatment mismatch / perturbation present: {perturb}")
    _debug_log(
        run_id,
        "H1",
        "compatibility.py:validate_dataset_compatibility:bio",
        "post_biological_checks",
        {"bio_pass": bio_pass, "allow_perturbation": allow_perturb, "perturbation_norm": perturb, "reasons_so_far": list(reasons)},
    )
    
    m_species = 1.0 if rna_species and acc_species and rna_species == acc_species else 0.0
    desc_match_score = 1.0 if (desc and _semantic_text_match(rna_ct, desc)) else 0.0
    opaque_acc = _is_opaque_cell_type_label(acc_ct) if acc_ct else True
    if rna_ct and acc_ct and acc_ct not in ("unknown", "na", "n/a"):
        if _cell_type_match(rna_ct, acc_ct):
            m_celltype = 1.0
        elif desc_match_score > 0.0:
            m_celltype = 0.75
        elif opaque_acc:
            m_celltype = 0.5
        else:
            m_celltype = 0.0
    elif rna_ct and desc:
        m_celltype = 0.75 if _semantic_text_match(rna_ct, desc) else 0.0
    else:
        m_celltype = 0.5
    m_condition = 1.0 if perturb in ("", "unknown", "na", "n/a", "none", "control", "untreated", "naive") else 0.0
    m_development = _match_score_exact_related(rna_state, acc_state)
    if rna_lineage and acc_lineage:
        m_development = min(m_development, _match_score_exact_related(rna_lineage, acc_lineage))
    m_protocol = _protocol_match_score(accessibility_meta.get("assay", ""))
    q_match = (
        0.40 * m_species
        + 0.25 * m_celltype
        + 0.15 * m_condition
        + 0.10 * m_development
        + 0.10 * m_protocol
    )

    # B. Technical match
    tech_pass = True
    rna_genome = _norm(rna_meta.get("genome_build", ""))
    acc_genome = _norm(accessibility_meta.get("genome_build", ""))
    genome_known = bool(acc_genome and acc_genome not in {"unknown", "na", "n/a"})
    if rna_genome and acc_genome and rna_genome != acc_genome:
        tech_pass = False
        reasons.append(f"Genome mismatch (without liftover plan): RNA={rna_genome}, accessibility={acc_genome}")
    
    rna_reps = int(rna_meta.get("n_replicates", 0))
    acc_reps = int(accessibility_meta.get("n_replicates", 0))
    rna_source = _norm(rna_meta.get("source", ""))
    acc_source = _norm(accessibility_meta.get("source", ""))
    if rna_reps < 2:
        if rna_source != "user_provided":
            tech_pass = False
            reasons.append(f"RNA replicates < 2: {rna_reps}")
    if acc_reps < 2:
        if acc_source not in ("geo", "user_provided"):
            tech_pass = False
            reasons.append(f"Accessibility replicates < 2: {acc_reps}")
    
    # ENCODE has explicit released/in-progress state. GEO records in this
    # acquisition path only expose public availability, not ENCODE-style status.
    if not _is_available_status(rna_meta):
        tech_pass = False
        reasons.append(f"RNA data not available: {rna_meta.get('status')}")
    if not _is_available_status(accessibility_meta):
        tech_pass = False
        reasons.append(f"Accessibility data not available: {accessibility_meta.get('status')}")
    _debug_log(
        run_id,
        "H2",
        "compatibility.py:validate_dataset_compatibility:tech",
        "post_technical_checks",
        {
            "tech_pass": tech_pass,
            "rna_reps": rna_reps,
            "acc_reps": acc_reps,
            "rna_genome": rna_genome,
            "acc_genome": acc_genome,
            "reasons_so_far": list(reasons),
        },
    )
    
    # C. Feature compatibility — intentionally no gold-label gates (see module docstring).
    feat_pass = True

    # D. Accessibility QC scoring
    promoter_cov = _safe_float(accessibility_meta.get("promoter_coverage_of_targets", None), None)
    frip = _safe_float(
        accessibility_meta.get("frip", accessibility_meta.get("spot_score", accessibility_meta.get("signal_fraction", None))),
        None,
    )
    tss_enrichment = _safe_float(accessibility_meta.get("tss_enrichment", None), None)
    promoter_peak_fraction = _safe_float(accessibility_meta.get("promoter_peak_fraction", None), None)
    replicate_jaccard = _safe_float(accessibility_meta.get("replicate_jaccard", None), None)
    has_peak_file = bool(accessibility_meta.get("has_peak_file", bool(accessibility_meta.get("files"))))
    has_signal_file = bool(accessibility_meta.get("has_signal_file", False))
    metadata_terms = [
        accessibility_meta.get("cell_type"),
        accessibility_meta.get("description"),
        accessibility_meta.get("assay"),
        accessibility_meta.get("species"),
    ]
    metadata_present = sum(1 for x in metadata_terms if _norm(x))
    q_signal = 0.5 if frip is None else _clip01((frip - 0.05) / 0.20)
    q_tss = 0.5 if tss_enrichment is None else _clip01((tss_enrichment - 4.0) / 6.0)
    q_prom = _promoter_score(promoter_cov)
    q_peakdist = _peak_distribution_score(promoter_peak_fraction)
    if replicate_jaccard is not None:
        q_rep = _clip01((replicate_jaccard - 0.20) / 0.30)
    elif acc_reps < 2:
        q_rep = 0.5
    else:
        q_rep = 0.5
    f_peaks = 1.0 if has_peak_file else 0.0
    f_signal = 1.0 if has_signal_file else 0.0
    f_metadata = 1.0 if metadata_present >= 3 else (0.5 if metadata_present >= 1 else 0.0)
    f_genome = 1.0 if genome_known else 0.0
    q_usable = 0.40 * f_peaks + 0.30 * f_signal + 0.20 * f_metadata + 0.10 * f_genome
    qc_score = (
        weights["signal"] * q_signal
        + weights["tss"] * q_tss
        + weights["promoter_coverage"] * q_prom
        + weights["peak_distribution"] * q_peakdist
        + weights["replicate_concordance"] * q_rep
        + weights["biological_match"] * q_match
        + weights["usability"] * q_usable
    )
    qc_decision = _decision_from_score(qc_score)
    cov_pass = promoter_cov is None or promoter_cov >= 0.25
    _debug_log(
        run_id,
        "H3",
        "compatibility.py:validate_dataset_compatibility:coverage",
        "coverage_evaluation",
        {
            "promoter_cov": promoter_cov,
            "cov_pass": cov_pass,
            "frip": frip,
            "tss_enrichment": tss_enrichment,
            "replicate_jaccard": replicate_jaccard,
            "qc_score": qc_score,
            "qc_decision": qc_decision,
        },
    )

    # E. Hard reject conditions and audit flags
    hard_rejects: list[str] = []
    if not acc_species:
        hard_rejects.append("species_missing")
    elif rna_species and acc_species and rna_species != acc_species:
        hard_rejects.append("species_mismatch")
    if not genome_known:
        hard_rejects.append("genome_unresolved")
    if not has_peak_file and not has_signal_file:
        hard_rejects.append("missing_peak_and_signal_files")
    if not bio_pass:
        hard_rejects.append("severe_biological_mismatch")
    if cell_type_mismatch:
        hard_rejects.append("cell_type_mismatch")
    if q_signal < 0.2 and q_tss < 0.2:
        hard_rejects.append("critical_low_signal_and_tss")

    if accessibility_meta.get("qc_flags", {}).get("low_spot", False):
        warnings.append("DNase QC: low SPOT score")
    if accessibility_meta.get("qc_flags", {}).get("low_depth", False):
        warnings.append("DNase QC: low sequencing depth")

    if promoter_cov is not None and promoter_cov < 0.25:
        reasons.append(f"Promoter accessibility coverage < 25%: {promoter_cov:.2%}")
    if hard_rejects:
        overall = False
        for item in hard_rejects:
            reasons.append(f"Hard reject: {item}")
    else:
        overall = tech_pass and feat_pass and qc_decision in {"accept", "conditional_accept"}
    _debug_log(
        run_id,
        "H1-H4",
        "compatibility.py:validate_dataset_compatibility:exit",
        "compatibility_decision",
        {
            "overall_pass": overall,
            "bio_pass": bio_pass,
            "tech_pass": tech_pass,
            "feat_pass": feat_pass,
            "cov_pass": cov_pass,
            "qc_score": qc_score,
            "qc_decision": qc_decision,
            "hard_rejects": list(hard_rejects),
            "reasons": list(reasons),
            "warnings": list(warnings),
        },
    )
    return {
        "pass": overall,
        "biological_match": "pass" if bio_pass else "fail",
        "technical_match": "pass" if tech_pass else "fail",
        "feature_compatibility": "pass" if feat_pass else "fail",
        "accessibility_coverage": "pass" if cov_pass else "fail",
        "accessibility_qc": {
            "score": qc_score,
            "decision": qc_decision,
            "weights": weights,
            "components": {
                "Q_signal": q_signal,
                "Q_TSS": q_tss,
                "Q_prom": q_prom,
                "Q_peakdist": q_peakdist,
                "Q_rep": q_rep,
                "Q_match": q_match,
                "Q_usable": q_usable,
            },
            "subcomponents": {
                "M_species": m_species,
                "M_celltype": m_celltype,
                "M_condition": m_condition,
                "M_development": m_development,
                "M_protocol": m_protocol,
                "F_peaks": f_peaks,
                "F_signal": f_signal,
                "F_metadata": f_metadata,
                "F_genome": f_genome,
            },
            "metrics": {
                "frip": frip,
                "tss_enrichment": tss_enrichment,
                "promoter_coverage": promoter_cov,
                "promoter_peak_fraction": promoter_peak_fraction,
                "replicate_jaccard": replicate_jaccard,
            },
            "hard_rejects": hard_rejects,
        },
        "rejection_reasons": reasons,
        "warnings": warnings,
    }
