"""
Build multimodal dataset manifest after validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_multimodal_manifest(
    dataset_id: str,
    species: str,
    cell_type: str,
    genome_build: str,
    rna_meta: dict[str, Any],
    accessibility_meta: dict[str, Any],
    motif_meta: dict[str, Any],
    qc_report: dict[str, Any],
    output_paths: dict[str, str],
    output_file: str | Path,
) -> None:
    """
    Write JSON manifest for a validated multimodal dataset.
    
    Args:
        dataset_id: unique identifier
        species: e.g., "mouse", "human"
        cell_type: e.g., "embryonic stem cell"
        genome_build: e.g., "mm10", "hg38"
        rna_meta: RNA dataset metadata
        accessibility_meta: DNase/ATAC metadata
        motif_meta: motif database info
        qc_report: output from validate_dataset_compatibility
        output_paths: dict of file paths (expression_matrix, promoter_accessibility, motif_hits, etc.)
        output_file: where to write manifest JSON
    """
    manifest = {
        "dataset_id": dataset_id,
        "species": species,
        "cell_type": cell_type,
        "genome_build": genome_build,
        "rna": {
            "source": rna_meta.get("source", "GEO"),
            "accession": rna_meta.get("accession", ""),
            "files": rna_meta.get("files", []),
            "n_replicates": rna_meta.get("n_replicates", 0),
            "n_genes": rna_meta.get("n_genes", 0),
        },
        "accessibility": {
            "source": accessibility_meta.get("source", "ENCODE"),
            "accession": accessibility_meta.get("accession", ""),
            "assay": accessibility_meta.get("assay", "DNase-seq"),
            "files": accessibility_meta.get("files", []),
            "n_replicates": accessibility_meta.get("n_replicates", 0),
            "promoter_coverage_of_rnaseq_genes": accessibility_meta.get("promoter_coverage_of_targets", 0.0),
            "promoter_peak_fraction": accessibility_meta.get("promoter_peak_fraction"),
            "frip": accessibility_meta.get("frip"),
            "tss_enrichment": accessibility_meta.get("tss_enrichment"),
            "has_peak_file": accessibility_meta.get("has_peak_file", False),
            "has_signal_file": accessibility_meta.get("has_signal_file", False),
        },
        "motifs": {
            "database": motif_meta.get("database", "JASPAR2026"),
            "tf_motif_count": motif_meta.get("tf_motif_count", 0),
            "tf_overlap_with_rnaseq": motif_meta.get("tf_overlap_with_rnaseq", 0.0),
            # Surface scan failures: a dropped motif channel silently degrades
            # downstream scoring to expression+accessibility only.
            "status": motif_meta.get("status", "unknown"),
            "error": motif_meta.get("error"),
        },
        "qc": qc_report,
        "qc_report": qc_report,
        "paths": output_paths,
    }
    out = Path(output_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
