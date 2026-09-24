from __future__ import annotations

from grn_agent.explain.reasoning import build_edge_explanation, format_acquisition_narrative


def test_build_edge_explanation_strong_support():
    row = {
        "source_tf": "SPI1",
        "target_gene": "CSF1R",
        "cell_type": "hematopoietic stem cell",
        "p_present": 0.91,
        "correlation": 0.62,
        "motif_present": True,
        "accessibility": 0.71,
        "ensemble_prior": 0.55,
        "window_vote_total": 5,
        "window_vote_fraction": 1.0,
        "p_present_std": 0.04,
    }
    explanation = build_edge_explanation(row)
    assert explanation["strength"] == "strong"
    assert "SPI1" in explanation["mechanism_reasoning"]
    assert "CSF1R" in explanation["mechanism_reasoning"]
    assert "coexpression" in explanation["mechanism_reasoning"].lower() or "coexpression" in str(explanation["factors"]).lower()
    assert explanation["missing_modalities"] == []
    assert "ensemble prior" not in explanation["mechanism_reasoning"].lower()
    assert all(factor.get("name") != "Ensemble prior" for factor in explanation["factors"])


def test_build_edge_explanation_marks_missing_modalities():
    explanation = build_edge_explanation(
        {"source_tf": "A", "target_gene": "B", "p_present": 0.2}
    )
    assert explanation["strength"] == "weak"
    assert "motif" in explanation["missing_modalities"]
    assert "accessibility" in explanation["missing_modalities"]
    assert "ensemble prior" not in explanation["missing_modalities"]


def test_acquisition_narrative_from_manifest():
    manifest = {
        "species": "mouse",
        "cell_type": "embryonic stem cell",
        "accessibility": {
            "source": "ENCODE",
            "assay": "ATAC-seq",
            "accession": "ENCSR000",
            "promoter_coverage_of_rnaseq_genes": 0.78,
        },
        "motifs": {"tf_motif_count": 100, "tf_overlap_with_rnaseq": 0.8, "status": "complete"},
        "qc": {"pass": True, "rejection_reasons": [], "warnings": ["low depth"]},
    }
    text = format_acquisition_narrative(manifest)
    assert "ENCODE" in text
    assert "accepted" in text.lower()
    assert "Warnings" in text
