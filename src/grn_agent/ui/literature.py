"""Literature claim card helpers for the GUI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def pubmed_url(pmid: str) -> str:
    pmid = str(pmid).strip()
    return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"


def _split_pmids(raw: object) -> list[str]:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    parts = [p.strip() for p in text.replace(";", ",").split(",")]
    return [p for p in parts if p]


def load_literature_cards(
    *,
    source_tf: str,
    target_gene: str,
    literature_dir: str | Path | None = None,
    lit_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load per-abstract literature audit cards for one edge."""
    tf = str(source_tf).strip()
    gene = str(target_gene).strip()
    cards: list[dict[str, Any]] = []
    class_path: Path | None = None
    if literature_dir is not None:
        base = Path(literature_dir)
        candidates = [
            base / f"{tf}_{gene}_classifications.json",
            base / f"{tf.upper()}_{gene.upper()}_classifications.json",
        ]
        for path in candidates:
            if path.is_file():
                class_path = path
                break
        if class_path is None:
            matches = sorted(base.glob(f"{tf}_{gene}*_classifications.json"))
            class_path = matches[0] if matches else None

    if class_path is not None:
        raw = json.loads(class_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                pmid = str(item.get("pmid") or "").strip()
                cards.append(
                    {
                        "pmid": pmid,
                        "pubmed_url": pubmed_url(pmid) if pmid else "",
                        "supports": bool(
                            item.get("effective_support", item.get("supports_interaction", False))
                        ),
                        "evidence_type": item.get("evidence_type"),
                        "relationship": item.get("relationship"),
                        "confidence": item.get("confidence"),
                        "evidence_sentence": item.get("evidence_sentence") or "",
                        "cell_type_sentences": item.get("cell_type_sentences") or [],
                        "is_negated": bool(item.get("is_negated", False)),
                        "is_speculative": bool(item.get("is_speculative", False)),
                        "direction_correct": item.get("direction_correct", True),
                        "evidence_grounded": item.get("evidence_grounded"),
                    }
                )

    summary = {
        "source_tf": tf,
        "target_gene": gene,
        "lit_score": None,
        "n_papers": len(cards),
        "n_supporting": sum(1 for c in cards if c["supports"]),
        "conflict_detected": False,
        "pmids": [],
        "evidence_types": "",
        "relationships": "",
        "cards": cards,
        "classifications_path": str(class_path) if class_path else None,
        "note": "Post-hoc literature audit (does not change TF-EAGER model scores).",
    }
    if lit_row:
        summary["lit_score"] = lit_row.get("lit_score")
        summary["n_papers"] = lit_row.get("n_papers", summary["n_papers"])
        summary["n_supporting"] = lit_row.get("n_supporting", summary["n_supporting"])
        summary["conflict_detected"] = bool(lit_row.get("conflict_detected", False))
        summary["pmids"] = _split_pmids(lit_row.get("pmids"))
        summary["evidence_types"] = lit_row.get("evidence_types") or ""
        summary["relationships"] = lit_row.get("relationships") or ""
        if not cards and summary["pmids"]:
            for pmid in summary["pmids"]:
                cards.append(
                    {
                        "pmid": pmid,
                        "pubmed_url": pubmed_url(pmid),
                        "supports": True,
                        "evidence_type": summary["evidence_types"],
                        "relationship": summary["relationships"],
                        "confidence": lit_row.get("avg_conf"),
                        "evidence_sentence": "",
                        "cell_type_sentences": [],
                        "is_negated": False,
                        "is_speculative": False,
                        "direction_correct": True,
                        "evidence_grounded": None,
                    }
                )
            summary["cards"] = cards
    return summary
