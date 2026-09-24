#!/usr/bin/env python3
"""
Automated multimodal data acquisition for BEELINE benchmarks.

Fetches RNA + DNase/ATAC + motifs from ENCODE/GEO, validates compatibility, builds manifest.

Example:
    python scripts/acquire_multimodal_data.py \
      --beeline-dataset mESC \
      --beeline-gold data/beeline/mESC_refNetwork.csv \
      --beeline-expr data/beeline/mESC_ExpressionData.csv \
      --species mouse \
      --cell-type "embryonic stem cell" \
      --genome mm10 \
      --out-manifest data/mESC_multimodal_manifest.json
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import json
import logging
import os
from pathlib import Path
import re
import time

from grn_agent.acquisition import (
    ENCODEClient,
    GEOClient,
    build_multimodal_manifest,
    validate_dataset_compatibility,
    run_motif_integration,
)
from grn_agent.acquisition.gene_coords import load_gene_coords
from grn_agent.acquisition.compatibility import canonical_species_label
from grn_agent.acquisition.accessibility_processor import (
    compute_coverage_fraction,
    compute_promoter_peak_fraction,
    compute_promoter_accessibility,
    load_peaks_bed,
)
from grn_agent.acquisition.rna_processor import (
    identify_expressed_tfs,
    load_beeline_expression,
    load_beeline_gold_network,
    normalize_expression,
)
from grn_agent.acquisition.geo_client import is_likely_accessibility_sample

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
_DEBUG_LOG_PATH = os.environ.get("GRN_AGENT_DEBUG_LOG_PATH", "").strip()
_DEBUG_SESSION = os.environ.get("GRN_AGENT_DEBUG_SESSION", "").strip()
_USE_SEMANTIC_SLM = False
_SEMANTIC_SLM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_SEMANTIC_SLM_INSTANCE = None
_SEMANTIC_SLM_LOAD_FAILED = False
_GEO_ACCESSION_RE = re.compile(r"(?<![A-Z0-9])(?:GSE|GSM)\d+(?!\d)", re.IGNORECASE)


def _load_tf_symbols(path: str | Path) -> set[str]:
    """Load a one-column TF symbol file without relying on benchmark labels."""
    out: set[str] = set()
    with Path(path).open(encoding="utf-8") as fp:
        for i, line in enumerate(fp):
            raw = line.strip()
            if not raw:
                continue
            symbol = raw.split(",")[0].split("\t")[0].strip().strip('"').strip("'")
            if not symbol:
                continue
            lower = symbol.lower()
            if i == 0 and lower in {"tf", "tfs", "source", "source_tf", "regulator"}:
                continue
            out.add(symbol.upper())
    return out


def _select_expressed_tfs(
    expr_norm,
    gene_symbols: list[str],
    mean_expr,
    tf_universe: set[str],
    *,
    min_mean_expr: float = 0.1,
) -> tuple[set[str], str]:
    """
    Select expressed TFs without reading benchmark gold labels.

    If a TF universe is supplied, filter that list by expression. Otherwise use
    the existing expression-only fallback so acquisition remains label-free.
    """
    candidate_tfs = {str(tf).strip().upper() for tf in tf_universe if str(tf).strip()}
    if candidate_tfs:
        expressed = identify_expressed_tfs(expr_norm, gene_symbols, list(candidate_tfs), min_mean_expr=min_mean_expr)
        return {str(tf).strip().upper() for tf in expressed}, "tf_file"

    top_n = max(1, int(len(gene_symbols) * 0.1))
    top_idx = mean_expr.argsort()[-top_n:]
    return {str(gene_symbols[i]).strip().upper() for i in top_idx}, "expression_top_decile"


def _looks_like_peak_bed(path: Path) -> bool:
    """
    Lightweight sanity check for BED/narrowPeak-like text payloads.
    Accepts plain text or gzip-compressed text.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with path.open("rb") as fh:
            magic = fh.read(2)
        is_gz = magic == b"\x1f\x8b"
        opener = gzip.open if is_gz else open  # type: ignore[assignment]
        with opener(path, "rt", encoding="utf-8", errors="ignore") as fp:  # type: ignore[arg-type]
            for line in fp:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                parts = s.split("\t")
                if len(parts) < 3:
                    return False
                # chr/start/end
                _ = int(parts[1])
                _ = int(parts[2])
                return True
        return False
    except Exception:
        return False


def _debug_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # Optional debug telemetry; disabled by default unless env vars are set.
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


def _extract_geo_accessions_from_text(*texts: object) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for match in _GEO_ACCESSION_RE.finditer(str(text or "")):
            acc = match.group(0).upper()
            if acc not in seen:
                out.append(acc)
                seen.add(acc)
    return out


def _infer_request_geo_accessions(args: argparse.Namespace) -> list[str]:
    """
    Infer GEO provenance clues without requiring an explicit --rna-accession.

    Expression matrices are often downloaded from GEO and named with their
    source GSM/GSE, while richer protocol notes may live in --cell-context.
    """
    return _extract_geo_accessions_from_text(
        getattr(args, "rna_accession", ""),
        getattr(args, "expr", ""),
        getattr(args, "cell_context", ""),
        getattr(args, "dataset_id", ""),
    )


def _write_ranked_candidates(path: str | None, ranked: list[tuple]) -> None:
    """Write the top accessibility QC rankings without downloading files."""
    if not path:
        return
    rows = []
    for rank, item in enumerate(ranked, start=1):
        meta, score, detail = item
        meta = meta if isinstance(meta, dict) else {}
        detail = detail if isinstance(detail, dict) else {}
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            score_f = None
        rows.append(
            {
                "rank": rank,
                "accession": meta.get("accession"),
                "source": meta.get("source"),
                "assay": meta.get("assay"),
                "cell_type": meta.get("cell_type"),
                "genome_build": meta.get("genome_build"),
                "n_replicates": meta.get("n_replicates"),
                "score": score_f,
                "tier": detail.get("tier"),
                "s_bio": detail.get("s_bio"),
                "s_cond": detail.get("s_cond"),
                "s_quality": detail.get("s_quality"),
            }
        )
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acquire multimodal data with automatic ATAC/DNase search",
        epilog="""
Examples:
  # Minimal: expression + species (agent searches for ATAC)
  python acquire_multimodal_data.py \\
    --expr expression.csv \\
    --species mouse \\
    --out-manifest manifest.json
  
  # With cell type for better ATAC matching
  python acquire_multimodal_data.py \\
    --expr expression.csv \\
    --species mouse \\
    --cell-type "embryonic stem cell" \\
    --out-manifest manifest.json
  
  # User provides ATAC accession (skip search)
  python acquire_multimodal_data.py \\
    --expr expression.csv \\
    --species mouse \\
    --atac-accession ENCSR000CGE \\
    --out-manifest manifest.json
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # REQUIRED inputs
    parser.add_argument("--expr", required=True, help="Expression matrix (CSV: genes × cells or cells × genes)")
    parser.add_argument("--species", required=True, help="Species: mouse, human, etc.")
    parser.add_argument("--out-manifest", required=True, help="Output manifest JSON path")
    
    # OPTIONAL inputs (for validation and ATAC search)
    parser.add_argument("--gold-network", help="Optional gold network CSV; not needed for benchmark-blind acquisition")
    parser.add_argument("--tf-file", help="TF universe file used for motif scanning without reading gold labels")
    parser.add_argument("--cell-type", help="Cell type (improves ATAC search; e.g., 'embryonic stem cell')")
    parser.add_argument("--cell-line", help="Cell line identifier; used for strict matching/ranking")
    parser.add_argument("--lineage", help="Lineage/stage (e.g., 'hematopoietic', 'hepatocyte differentiation')")
    parser.add_argument("--state", help="Cell state (e.g., naive, primed, activated)")
    parser.add_argument(
        "--cell-context",
        help=(
            "Free-text RNA experiment context (protocol/timepoint/notes) used only "
            "for semantic candidate ranking, not strict metadata matching"
        ),
    )
    parser.add_argument(
        "--allow-perturbation",
        action="store_true",
        help="Allow treated/perturbed accessibility datasets (otherwise strict matching rejects them)",
    )
    parser.add_argument("--genome", help="Genome build (mm10, hg38); auto-inferred from species if not provided")
    parser.add_argument("--dataset-id", help="Dataset identifier (default: auto-generated)")
    parser.add_argument(
        "--rna-accession",
        help=(
            "RNA GEO accession used for paired GEO discovery. If a GSM is supplied, "
            "the agent searches sibling ATAC/DNase GSMs in the same GSE."
        ),
    )
    parser.add_argument(
        "--use-semantic-slm",
        action="store_true",
        help="Use small language model embeddings for cell-type semantic scoring",
    )
    parser.add_argument(
        "--semantic-slm-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name when --use-semantic-slm is enabled",
    )
    
    # OPTIONAL: user-provided ATAC (skips automatic search)
    parser.add_argument("--atac-accession", help="ENCODE/GEO accession for ATAC/DNase (if known)")
    parser.add_argument("--geo-accession", help="GEO series accession for accessibility data (e.g., GSE136447)")
    parser.add_argument("--atac-file", help="Local ATAC peaks BED file (if already downloaded)")
    
    # Validation and processing options
    parser.add_argument("--strict", action="store_true", help="Strict validation (reject on any QC failure)")
    parser.add_argument(
        "--min-promoter-coverage",
        type=float,
        default=0.6,
        help="Minimum promoter accessibility coverage required by QC (default: 0.7)",
    )
    parser.add_argument(
        "--coverage-denominator",
        choices=["rnaseq_expressed", "rnaseq_all"],
        default="rnaseq_expressed",
        help=(
            "Gene set used for promoter-coverage QC denominator: "
            "'rnaseq_expressed' (default) uses RNA-expressed genes; "
            "'rnaseq_all' uses all RNA genes."
        ),
    )
    parser.add_argument(
        "--coverage-min-mean-expr",
        type=float,
        default=0.1,
        help="Mean log1p-CPM threshold for rnaseq_expressed denominator (default: 0.1)",
    )
    parser.add_argument("--skip-atac-search", action="store_true", help="Skip automatic ATAC search (RNA-only mode)")
    parser.add_argument("--max-atac-candidates", type=int, default=5, help="Max ATAC candidates to evaluate (default: 5)")
    parser.add_argument(
        "--ranked-candidates-json",
        default="",
        help="Write the ranked accessibility candidates (top --max-atac-candidates) as JSON",
    )
    parser.add_argument(
        "--rank-only",
        action="store_true",
        help="Rank accessibility candidates and exit before download, motif scan, or manifest writing",
    )
    parser.add_argument(
        "--geo-search-workers",
        type=int,
        default=6,
        help="Parallel workers for GEO series metadata fetch/score during auto-search (default: 6)",
    )
    parser.add_argument("--cache-dir", default=".cache", help="Cache directory for API responses")

    # Motif integration options
    parser.add_argument(
        "--genome-fasta",
        default="",
        help=(
            "Genome FASTA (.fa, must be indexed with samtools faidx). "
            "If omitted, the genome is auto-downloaded from Ensembl and stored in GenomeDB "
            "(~/.cache/grn_agent/genomes/). Requires ~2-3 GB disk per species."
        ),
    )
    parser.add_argument(
        "--gtf",
        help="GTF file for gene TSS coordinates (optional; GenomeDB/BioMart queried if omitted)",
    )
    parser.add_argument(
        "--skip-motif", action="store_true",
        help="Skip motif integration entirely (use when FIMO/bedtools not installed)",
    )
    parser.add_argument(
        "--no-auto-genome", action="store_true",
        help="Disable automatic genome download (require --genome-fasta to be provided)",
    )
    parser.add_argument(
        "--motif-window", type=int, default=2000,
        help="Promoter window bp for peak-to-gene assignment (default: 2000)",
    )
    parser.add_argument(
        "--fimo-pvalue", type=float, default=1e-4,
        help="FIMO p-value threshold (default: 1e-4, standard field practice)",
    )
    parser.add_argument(
        "--fimo-workers", type=int, default=0,
        help="Parallel FIMO chunks (0 = CPU-bounded default). Full peak unions "
             "need >1 to finish within the timeout.",
    )
    parser.add_argument(
        "--fimo-timeout", type=int, default=3600,
        help="Per-chunk FIMO timeout in seconds (default: 3600)",
    )
    parser.add_argument(
        "--jaspar-release",
        type=int,
        default=2026,
        metavar="YEAR",
        help="JASPAR CORE MEME release year (default: 2026; use 2024 to pin older files)",
    )
    args = parser.parse_args()
    global _USE_SEMANTIC_SLM, _SEMANTIC_SLM_MODEL
    _USE_SEMANTIC_SLM = bool(args.use_semantic_slm)
    _SEMANTIC_SLM_MODEL = str(args.semantic_slm_model or "").strip() or _SEMANTIC_SLM_MODEL
    
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    
    # Auto-infer genome build if not provided
    genome = args.genome
    if not genome:
        genome_map = {
            "mouse": "mm10",
            "human": "hg38",
            "rat": "rn6",
            "drosophila": "dm6",
            "fly": "dm6",
        }
        genome = genome_map.get(args.species.lower(), "unknown")
        logger.info(f"Auto-inferred genome build: {genome}")
    
    # Load gold standard (optional diagnostics only; not used to choose TFs/data).
    gold_genes: set[str] = set()
    gold_tfs: set[str] = set()
    tf_universe: set[str] = set()
    if args.gold_network:
        logger.info("Loading gold standard network...")
        gold_genes, gold_tfs = load_beeline_gold_network(args.gold_network)
        logger.info(f"Gold standard: {len(gold_genes)} genes, {len(gold_tfs)} TFs")
    if args.tf_file:
        logger.info("Loading TF universe from TF file...")
        tf_universe = _load_tf_symbols(args.tf_file)
        logger.info(f"TF universe: {len(tf_universe)} TFs")
    
    # Load expression (REQUIRED)
    logger.info("Loading expression data...")
    expr, gene_symbols = load_beeline_expression(args.expr)
    # Canonical symbol convention for this script: uppercase everywhere.
    gene_symbols = [str(g).strip().upper() for g in gene_symbols]
    gold_genes = {str(g).strip().upper() for g in gold_genes}
    gold_tfs = {str(tf).strip().upper() for tf in gold_tfs}
    tf_universe = {str(tf).strip().upper() for tf in tf_universe}
    logger.info(f"Expression: {expr.shape[0]} cells, {expr.shape[1]} genes")
    expr_norm = normalize_expression(expr, method="log1p_cpm")
    mean_expr = expr_norm.mean(axis=0)
    expressed_genes = {
        gene_symbols[i]
        for i in range(len(gene_symbols))
        if float(mean_expr[i]) >= float(args.coverage_min_mean_expr)
    }
    
    # If no gold standard, use all genes as compatibility-call placeholders.
    # validate_dataset_compatibility keeps these args for call-site compatibility
    # but does not use benchmark labels to accept/reject datasets.
    if not gold_genes:
        gold_genes = set(gene_symbols)
        logger.info(f"No gold standard provided; using all {len(gold_genes)} genes as candidates")
    
    # Identify expressed TFs without using benchmark gold TFs.
    expressed_tfs, tf_source = _select_expressed_tfs(
        expr_norm,
        gene_symbols,
        mean_expr,
        tf_universe,
        min_mean_expr=0.1,
    )
    if tf_universe:
        logger.info(f"Expressed TFs: {len(expressed_tfs)}/{len(tf_universe)} source={tf_source}")
    else:
        logger.info(
            "No TF universe provided; using expression-only top 10%% fallback as TF candidates: %d",
            len(expressed_tfs),
        )
    
    inferred_req_perturbation = _infer_perturbation_label(
        args.state or "",
        args.cell_context or "",
        args.cell_type or "",
    )
    inferred_req_cell_line = (
        _extract_cell_line_from_label(args.cell_line or "")
        or _extract_cell_line_from_text(args.cell_line or "", args.cell_context or "", args.cell_type or "")
    )
    effective_req_cell_line = inferred_req_cell_line or (args.cell_line or "")
    effective_allow_perturbation = bool(args.allow_perturbation or inferred_req_perturbation)
    if inferred_req_perturbation and not args.allow_perturbation:
        logger.info(
            "Auto-enabling perturbation-aware matching from request context: detected=%s",
            inferred_req_perturbation,
        )
    request_geo_accessions = _infer_request_geo_accessions(args)
    if request_geo_accessions:
        logger.info("Inferred GEO accession clues from request context: %s", ", ".join(request_geo_accessions))

    # RNA metadata
    inferred_rna_accession = (args.rna_accession or (request_geo_accessions[0] if request_geo_accessions else "")).strip()
    rna_meta = {
        "source": "user_provided",
        "accession": inferred_rna_accession or "N/A",
        "rna_accession": inferred_rna_accession,
        "species": args.species,
        "cell_type": args.cell_type or "unknown",
        "cell_line": effective_req_cell_line or "",
        "lineage": args.lineage or "",
        "state": args.state or "",
        "cell_context": args.cell_context or "",
        "allow_perturbation": effective_allow_perturbation,
        "requested_perturbation": inferred_req_perturbation,
        "genome_build": genome,
        "n_replicates": 1,
        "n_genes": len(gene_symbols),
        "gene_symbols": gene_symbols,
        "expressed_tfs": list(expressed_tfs),
        "status": "released",
    }
    
    # Search for accessibility data (if not provided and not skipped)
    accessibility_meta: dict = {}
    
    if args.atac_file:
        logger.info(f"Using user-provided ATAC file: {args.atac_file}")
        accessibility_meta = {
            "source": "user_provided",
            "accession": "local_file",
            "assay": "ATAC-seq",
            "species": args.species,
            "cell_type": args.cell_type or "unknown",
            "cell_line": effective_req_cell_line or "",
            "lineage": args.lineage or "",
            "state": args.state or "",
            "perturbation": "none",
            "genome_build": genome,
            "n_replicates": 1,
            "files": [args.atac_file],
            "has_peak_file": True,
            "has_signal_file": False,
            "promoter_coverage_of_targets": 0.0,
            "frip": None,
            "tss_enrichment": None,
            "status": "user_provided",
            "qc_flags": {},
        }
    elif args.geo_accession:
        logger.info(f"Using user-provided GEO accessibility accession: {args.geo_accession}")
        geo = GEOClient(cache_dir=cache / "geo")
        try:
            geo_acc = str(args.geo_accession).upper().strip()
            if geo_acc.startswith("GSM"):
                sample = geo.get_sample_metadata(geo_acc)
                accessibility_meta = _extract_geo_sample_accessibility_meta(
                    sample,
                    genome=genome,
                    requested_species=args.species,
                )
                series = {
                    "title": sample.get("title", ""),
                    "organism": sample.get("organism", ""),
                    "samples": [geo_acc],
                    "supplementary_files": sample.get("supplementary_files", []),
                }
            else:
                series = geo.get_series_metadata(geo_acc)
                accessibility_meta = _extract_best_geo_accessibility_meta_from_series(
                    geo,
                    series,
                    genome=genome,
                    requested_species=args.species,
                    requested_cell_type=args.cell_type,
                    requested_cell_line=effective_req_cell_line,
                    requested_lineage=args.lineage,
                    requested_state=args.state,
                )
            _debug_log(
                str(geo_acc),
                "H4-H5",
                "acquire_multimodal_data.py:geo_series",
                "geo_series_metadata",
                {
                    "title": series.get("title", ""),
                    "organism": series.get("organism", ""),
                    "n_samples": len(series.get("samples", []) or []),
                    "n_supp_files": len(series.get("supplementary_files", []) or []),
                },
            )
        except Exception as e:
            logger.error(f"Failed to fetch GEO accession {args.geo_accession}: {e}")
    elif args.atac_accession:
        logger.info(f"Using user-provided ATAC accession: {args.atac_accession}")
        atac_acc = str(args.atac_accession).upper().strip()
        if atac_acc.startswith(("GSE", "GSM")):
            geo = GEOClient(cache_dir=cache / "geo")
            try:
                if atac_acc.startswith("GSM"):
                    sample = geo.get_sample_metadata(atac_acc)
                    accessibility_meta = _extract_geo_sample_accessibility_meta(
                        sample,
                        genome=genome,
                        requested_species=args.species,
                    )
                else:
                    series = geo.get_series_metadata(atac_acc)
                    accessibility_meta = _extract_best_geo_accessibility_meta_from_series(
                        geo,
                        series,
                        genome=genome,
                        requested_species=args.species,
                        requested_cell_type=args.cell_type,
                        requested_cell_line=effective_req_cell_line,
                        requested_lineage=args.lineage,
                        requested_state=args.state,
                    )
            except Exception as e:
                logger.error(f"Failed to fetch GEO accession {args.atac_accession}: {e}")
        else:
            encode = ENCODEClient(cache_dir=cache / "encode")
            try:
                exp = encode.get_experiment_metadata(args.atac_accession)
                accessibility_meta = _extract_accessibility_meta(exp, genome)
            except Exception as e:
                logger.error(f"Failed to fetch ATAC accession {args.atac_accession}: {e}")
    elif not args.skip_atac_search:
        logger.info("Searching for ATAC/DNase-seq data (ENCODE + GEO)...")
        encode = ENCODEClient(cache_dir=cache / "encode")
        geo = GEOClient(cache_dir=cache / "geo")
        organism_map = {
            "mouse": "Mus musculus",
            "human": "Homo sapiens",
            "rat": "Rattus norvegicus",
            "drosophila": "Drosophila melanogaster",
            "fly": "Drosophila melanogaster",
        }
        organism = organism_map.get(args.species.lower(), args.species)
        
        candidates: list[tuple[dict, float, dict]] = []
        seeded_candidate_keys: set[tuple[str, str, str]] = set()
        for rna_acc in request_geo_accessions:
            if rna_acc.startswith("GSM"):
                try:
                    seed_sample = geo.get_sample_metadata(rna_acc)
                    if is_likely_accessibility_sample(seed_sample):
                        meta = _extract_geo_sample_accessibility_meta(
                            seed_sample,
                            genome=genome,
                            requested_species=args.species,
                        )
                        meta["pairing_quality"] = meta.get("pairing_quality") or "request_geo_seed"
                        key = (
                            str(meta.get("source", "")),
                            str(meta.get("accession", "")),
                            str(meta.get("parent_series_accession", "")),
                        )
                        if key not in seeded_candidate_keys:
                            seeded_candidate_keys.add(key)
                            score, detail = _score_accessibility_candidate(
                                meta,
                                requested_species=args.species,
                                requested_cell_type=args.cell_type,
                                requested_cell_line=effective_req_cell_line,
                                requested_lineage=args.lineage,
                                requested_state=args.state,
                                requested_cell_context=args.cell_context,
                                requested_perturbation=inferred_req_perturbation,
                                allow_perturbation=effective_allow_perturbation,
                                filtered_hit=False,
                            )
                            meta["selection_score_breakdown"] = detail
                            candidates.append((meta, score, detail))
                            logger.info(
                                "Added request GEO seed as accessibility candidate: %s files=%d has_peak=%s",
                                meta.get("accession"),
                                len(meta.get("files") or []),
                                meta.get("has_peak_file"),
                            )
                    paired_samples = geo.find_accessibility_samples_for_rna(
                        rna_acc,
                        limit=max(5, int(args.max_atac_candidates)),
                    )
                    logger.info(
                        "Found %d same-GEO-series accessibility sample candidates for GEO seed %s",
                        len(paired_samples),
                        rna_acc,
                    )
                    for sample in paired_samples:
                        meta = _extract_geo_sample_accessibility_meta(
                            sample,
                            genome=genome,
                            requested_species=args.species,
                        )
                        key = (
                            str(meta.get("source", "")),
                            str(meta.get("accession", "")),
                            str(meta.get("parent_series_accession", "")),
                        )
                        if key in seeded_candidate_keys:
                            continue
                        seeded_candidate_keys.add(key)
                        score, detail = _score_accessibility_candidate(
                            meta,
                            requested_species=args.species,
                            requested_cell_type=args.cell_type,
                            requested_cell_line=effective_req_cell_line,
                            requested_lineage=args.lineage,
                            requested_state=args.state,
                            requested_cell_context=args.cell_context,
                            requested_perturbation=inferred_req_perturbation,
                            allow_perturbation=effective_allow_perturbation,
                            filtered_hit=False,
                        )
                        meta["selection_score_breakdown"] = detail
                        candidates.append((meta, score, detail))
                except Exception as e:
                    logger.warning("Paired GEO sample discovery failed for GEO seed %s: %s", rna_acc, e)
            elif rna_acc.startswith("GSE"):
                try:
                    series = geo.get_series_metadata(rna_acc)
                    meta = _extract_best_geo_accessibility_meta_from_series(
                        geo,
                        series,
                        genome=genome,
                        requested_species=args.species,
                        requested_cell_type=args.cell_type,
                        requested_cell_line=effective_req_cell_line,
                        requested_lineage=args.lineage,
                        requested_state=args.state,
                    )
                    meta["parent_series_accession"] = rna_acc
                    meta["pairing_quality"] = "same_series"
                    key = (
                        str(meta.get("source", "")),
                        str(meta.get("accession", "")),
                        str(meta.get("parent_series_accession", "")),
                    )
                    if key in seeded_candidate_keys:
                        continue
                    seeded_candidate_keys.add(key)
                    score, detail = _score_accessibility_candidate(
                        meta,
                        requested_species=args.species,
                        requested_cell_type=args.cell_type,
                        requested_cell_line=effective_req_cell_line,
                        requested_lineage=args.lineage,
                        requested_state=args.state,
                        requested_cell_context=args.cell_context,
                        requested_perturbation=inferred_req_perturbation,
                        allow_perturbation=effective_allow_perturbation,
                        filtered_hit=False,
                    )
                    meta["selection_score_breakdown"] = detail
                    candidates.append((meta, score, detail))
                except Exception as e:
                    logger.warning("Same-GEO-series accessibility discovery failed for %s: %s", rna_acc, e)
        # Reliability over speed: fetch a much deeper pool before ranking so valid
        # context-matched datasets are not missed due to portal ordering/pagination.
        fetch_limit = min(2000, max(400, int(args.max_atac_candidates) * 80))
        for assay in ["ATAC-seq", "DNase-seq"]:
            search_kwargs_base = {
                "assay_title": assay,
                "organism": organism,
                "status": "released",
                "limit": fetch_limit,
            }
            
            try:
                exps_filtered: list[dict] = []
                strict_filtered_ids: set[str] = set()
                if args.cell_type:
                    exps_filtered = encode.search_experiments(
                        **{**search_kwargs_base, "biosample_term_name": args.cell_type}
                    )
                    strict_filtered_ids = {
                        str(e.get("accession", "")).strip()
                        for e in exps_filtered
                        if str(e.get("accession", "")).strip()
                    }
                    logger.info(f"Found {len(exps_filtered)} {assay} experiments after cell-type filtering")
                    if not exps_filtered:
                        free_text_query = " ".join(
                            part
                            for part in (
                                args.cell_type,
                                effective_req_cell_line,
                                args.cell_context,
                            )
                            if str(part or "").strip()
                        ).strip()
                        if free_text_query:
                            exps_free = encode.search_experiments_free_text(
                                query_text=free_text_query,
                                assay_title=assay,
                                status="released",
                                limit=fetch_limit,
                            )
                            if exps_free:
                                exps_filtered = exps_free
                            logger.info(
                                "Found %d %s experiments via ENCODE free-text fallback query=%r",
                                len(exps_free),
                                assay,
                                free_text_query,
                            )
                exps_broad = encode.search_experiments(**search_kwargs_base)
                logger.info(f"Found {len(exps_broad)} {assay} experiments before filtering")
                merged: dict[str, dict] = {}
                # If user supplied a target cell type, keep retrieval context-constrained.
                # Do not flood ranking with unrelated tissues when filtered hits exist.
                pool = exps_filtered if (args.cell_type and exps_filtered) else exps_broad
                for exp in pool:
                    acc = str(exp.get("accession", "")).strip()
                    if acc and acc not in merged:
                        merged[acc] = exp
                for exp in list(merged.values())[:fetch_limit]:
                    meta = _extract_accessibility_meta(exp, genome)
                    score, detail = _score_accessibility_candidate(
                        meta,
                        requested_species=args.species,
                        requested_cell_type=args.cell_type,
                        requested_cell_line=effective_req_cell_line,
                        requested_lineage=args.lineage,
                        requested_state=args.state,
                        requested_cell_context=args.cell_context,
                        requested_perturbation=inferred_req_perturbation,
                        allow_perturbation=effective_allow_perturbation,
                        filtered_hit=bool(str(meta.get("accession", "")).strip() in strict_filtered_ids),
                    )
                    meta["selection_score_breakdown"] = detail
                    candidates.append((meta, score, detail))
            except Exception as e:
                logger.warning(f"ENCODE search failed for {assay}: {e}")

        # GEO auto-search in parallel path
        # Keep GEO fallback broad enough to recover useful accessibility series,
        # but cap fan-out so per-series metadata fetches do not dominate runtime.
        geo_limit = min(20, max(8, int(args.max_atac_candidates) * 4))
        try:
            geo_hits = geo.search_series(
                species=args.species,
                cell_type=args.cell_type,
                lineage=args.lineage,
                state=args.state,
                cell_context=args.cell_context,
                limit=geo_limit,
            )
            logger.info(f"Found {len(geo_hits)} GEO candidate series")
            geo_hits = [hit for hit in geo_hits if str(hit.get("accession", "")).strip()]
            geo_workers = max(1, min(int(args.geo_search_workers), len(geo_hits)))
            if geo_hits and geo_workers > 1:
                logger.info(
                    "Fetching/scoring GEO candidate series in parallel: series=%d workers=%d",
                    len(geo_hits),
                    geo_workers,
                )
            for result in _collect_parallel_geo_candidates(
                geo_hits,
                cache_dir=cache / "geo",
                genome=genome,
                requested_species=args.species,
                requested_cell_type=args.cell_type,
                requested_cell_line=effective_req_cell_line,
                requested_lineage=args.lineage,
                requested_state=args.state,
                requested_cell_context=args.cell_context,
                requested_perturbation=inferred_req_perturbation,
                allow_perturbation=effective_allow_perturbation,
                max_workers=geo_workers,
            ):
                candidates.append(result)
        except Exception as e:
            logger.warning(f"GEO search failed: {e}")
        
        if candidates:
            # ---- Step 6: filter to acceptable candidates --------------------
            acceptable = [
                c for c in candidates
                if _is_candidate_acceptable(
                    c[0], c[1], c[2],
                    requested_species=args.species,
                    requested_cell_type=args.cell_type,
                    requested_cell_line=effective_req_cell_line,
                    requested_cell_context=args.cell_context,
                )
            ]

            # ---- Step 7: tier-aware tie-breaking ----------------------------
            _tier_order = {"A_STRONG_MATCH": 0, "B_FALLBACK_MATCH": 1, "REJECT": 2}

            def _candidate_sort_key(c: tuple) -> tuple:
                m, s, d = c
                tier_rank  = _tier_order.get(str(d.get("tier", "REJECT")), 2)
                same_series_rank = 0 if _is_same_series_accessibility_candidate(m) else 1
                provenance = -float(d.get("s_study", 0.0))
                bio        = -float(d.get("s_bio", 0.0))
                assay_rank = 0 if "atac" in str(m.get("assay", "")).lower() else 1
                quality    = -float(d.get("s_quality", 0.0))
                n_reps     = -int(m.get("n_replicates", 0) or 0)
                build_rank = 0 if str(m.get("genome_build_match", "")) == "exact" else 1
                return (tier_rank, same_series_rank, provenance, bio, assay_rank, quality, n_reps, build_rank)

            if acceptable:
                acceptable.sort(key=_candidate_sort_key)
                best_meta, best_score, best_detail = acceptable[0]
                accessibility_meta = best_meta
            else:
                candidates_sorted = sorted(candidates, key=_candidate_sort_key)
                best_meta, best_score, best_detail = candidates_sorted[0]
                usable_rejected = [
                    c for c in candidates_sorted
                    if _is_rejected_but_usable_accessibility_candidate(c[0], c[2])
                ]
                if usable_rejected:
                    best_meta, best_score, best_detail = usable_rejected[0]
                    best_meta.setdefault("selection_warnings", [])
                    if isinstance(best_meta["selection_warnings"], list):
                        best_meta["selection_warnings"].append("selected_usable_accessibility_candidate_below_bio_threshold")
                    logger.warning(
                        "No candidate cleared biological/ranking thresholds; keeping usable accessibility candidate "
                        "for QC instead of RNA-only fallback (score=%.3f tier=%s accession=%s source=%s "
                        "has_peak=%s has_signal=%s s_bio=%.2f flags=%s).",
                        best_score,
                        best_detail.get("tier", "REJECT"),
                        best_meta.get("accession"),
                        best_meta.get("source"),
                        best_meta.get("has_peak_file"),
                        best_meta.get("has_signal_file"),
                        float(best_detail.get("s_bio", 0.0)),
                        {
                            k: v
                            for k, v in best_detail.items()
                            if k.endswith("mismatch") or k.endswith("penalty") or k.endswith("missing")
                        },
                    )
                    accessibility_meta = best_meta
                else:
                    logger.warning(
                        "All ranked accessibility candidates failed hard-rejection checks; "
                        "falling back to RNA-only manifest (top score=%.3f tier=%s accession=%s "
                        "source=%s has_peak=%s has_signal=%s s_bio=%.2f flags=%s).",
                        best_score,
                        best_detail.get("tier", "REJECT"),
                        best_meta.get("accession"),
                        best_meta.get("source"),
                        best_meta.get("has_peak_file"),
                        best_meta.get("has_signal_file"),
                        float(best_detail.get("s_bio", 0.0)),
                        {
                            k: v
                            for k, v in best_detail.items()
                            if k.endswith("mismatch") or k.endswith("penalty") or k.endswith("missing")
                        },
                    )
                    accessibility_meta = {}

            if accessibility_meta.get("source") == "ENCODE" and not accessibility_meta.get("files"):
                try:
                    full_exp = encode.get_experiment_metadata(str(accessibility_meta.get("accession", "")))
                    accessibility_meta = _extract_accessibility_meta(full_exp, genome)
                    accessibility_meta["selection_score_breakdown"] = best_detail
                except Exception as exc:
                    logger.warning(
                        "Failed to refresh ENCODE metadata for %s: %s",
                        accessibility_meta.get("accession"),
                        exc,
                    )
            if accessibility_meta:
                _augment_encode_qc_metrics(encode, accessibility_meta)
                logger.info(
                    "Selected best accessibility candidate: %s assay=%s score=%.3f tier=%s details=%s",
                    accessibility_meta.get("accession"),
                    accessibility_meta.get("assay"),
                    best_score,
                    best_detail.get("tier", "?"),
                    best_detail,
                )
            # Log top-5 from full candidate list for diagnostics
            candidates_sorted_all = sorted(candidates, key=_candidate_sort_key)
            ranked_limit = max(1, int(args.max_atac_candidates))
            for i, (m, s, d) in enumerate(candidates_sorted_all[:ranked_limit], start=1):
                logger.info(
                    "Ranked candidate #%d: %s assay=%s score=%.3f tier=%s "
                    "cell_type=%s s_bio=%.2f s_cond=%.2f s_quality=%.2f",
                    i,
                    m.get("accession"),
                    m.get("assay"),
                    s,
                    d.get("tier", "?"),
                    m.get("cell_type"),
                    d.get("s_bio", 0.0),
                    d.get("s_cond", 0.0),
                    d.get("s_quality", 0.0),
                )
            _write_ranked_candidates(
                str(args.ranked_candidates_json or ""),
                candidates_sorted_all[:ranked_limit],
            )
            if args.rank_only:
                logger.info("rank-only: wrote accessibility candidates and stopped before download")
                return
        else:
            logger.warning("No ATAC/DNase data found in ENCODE")

    if args.rank_only:
        ranked_path = str(args.ranked_candidates_json or "")
        if ranked_path and not Path(ranked_path).is_file():
            _write_ranked_candidates(ranked_path, [])
        logger.info("rank-only: stopped before download and harmonization")
        return
    
    if not accessibility_meta:
        logger.warning("No accessibility data available; creating RNA-only manifest")
        accessibility_meta = {
            "source": "none",
            "accession": "N/A",
            "assay": "none",
            "species": args.species,
            "cell_type": args.cell_type or "unknown",
            "lineage": args.lineage or "",
            "state": args.state or "",
            "perturbation": "",
            "genome_build": genome,
            "n_replicates": 0,
            "has_peak_file": False,
            "has_signal_file": False,
            "promoter_coverage_of_targets": 0.0,
            "frip": None,
            "tss_enrichment": None,
            "status": "not_found",
            "qc_flags": {},
        }

    # If ENCODE candidate provides file accessions, download one peak BED locally
    # so motif/ATAC integration can consume actual file content.
    if (
        accessibility_meta.get("source") == "ENCODE"
        and not args.atac_file
        and isinstance(accessibility_meta.get("files"), list)
        and accessibility_meta.get("files")
    ):
        encode_files = [str(x or "").strip() for x in accessibility_meta.get("files", []) if str(x or "").strip()]
        if encode_files and encode_files[0].startswith("ENCFF"):
            dataset_id_dl = args.dataset_id or f"{args.species}_{args.cell_type or 'unknown'}".replace(" ", "_")
            out_dir_dl = Path(args.out_manifest).parent / dataset_id_dl
            out_dir_dl.mkdir(parents=True, exist_ok=True)
            enc = ENCODEClient(cache_dir=cache / "encode")
            downloaded = False
            for file_acc in encode_files:
                if not file_acc.startswith("ENCFF"):
                    continue
                bed_gz = out_dir_dl / f"{file_acc}.bed.gz"
                bed_plain = out_dir_dl / f"{file_acc}.bed"
                try:
                    logger.info("Downloading ENCODE peak file for multimodal integration: %s", file_acc)
                    enc.download_file(file_acc, bed_gz)
                    if not _looks_like_peak_bed(bed_gz):
                        raise RuntimeError(f"Downloaded payload for {file_acc} is not BED/narrowPeak text")
                    # Normalize to plain .bed when gzipped, else keep as text file path.
                    try:
                        with bed_gz.open("rb") as fh:
                            is_gz = fh.read(2) == b"\x1f\x8b"
                        if is_gz:
                            with gzip.open(bed_gz, "rb") as src, bed_plain.open("wb") as dst:
                                dst.write(src.read())
                            if not _looks_like_peak_bed(bed_plain):
                                raise RuntimeError(f"Decompressed file for {file_acc} is not BED text")
                            accessibility_meta["files"] = [str(bed_plain)]
                        else:
                            # Rename non-gz text to .bed so downstream readers don't assume gzip.
                            bed_gz.rename(bed_plain)
                            accessibility_meta["files"] = [str(bed_plain)]
                    except Exception:
                        # Keep gz path only if it passes sanity.
                        if not _looks_like_peak_bed(bed_gz):
                            raise
                        accessibility_meta["files"] = [str(bed_gz)]
                    downloaded = True
                    break
                except Exception as exc:
                    logger.warning("Failed to download ENCODE peak file %s: %s", file_acc, exc)
            if not downloaded:
                logger.warning(
                    "No ENCODE peak file could be downloaded from %d candidate files for %s",
                    len(encode_files),
                    accessibility_meta.get("accession", ""),
                )

    # GEO: ``files`` are supplementary URLs from FTP listing — download a peak BED locally
    # so promoter accessibility + motif steps can read a real path (same as ENCODE above).
    if (
        accessibility_meta.get("source") == "GEO"
        and not args.atac_file
        and isinstance(accessibility_meta.get("files"), list)
        and accessibility_meta.get("files")
    ):
        raw_urls = [str(u) for u in accessibility_meta["files"]]
        first_ref = raw_urls[0] if raw_urls else ""
        need_download = first_ref.startswith(("http://", "https://", "ftp://")) or (
            first_ref and not Path(first_ref).is_file()
        )
        if need_download:
            peak_url = _pick_geo_peak_file_url(raw_urls)
            if peak_url:
                dataset_id_geo = args.dataset_id or f"{args.species}_{args.cell_type or 'unknown'}".replace(" ", "_")
                out_dir_geo = Path(args.out_manifest).parent / dataset_id_geo
                out_dir_geo.mkdir(parents=True, exist_ok=True)
                safe_name = peak_url.rstrip("/").split("/")[-1].split("?")[0] or "geo_peaks.bed.gz"
                local_peaks = out_dir_geo / safe_name
                try:
                    logger.info(
                        "Downloading GEO supplementary peak file for integration: %s",
                        peak_url[:160] + ("…" if len(peak_url) > 160 else ""),
                    )
                    _download_geo_supplementary(
                        GEOClient(cache_dir=cache / "geo"), peak_url, local_peaks
                    )
                    resolved = local_peaks.resolve()
                    if _geo_file_is_peak_count_matrix(str(resolved)):
                        bed_out = out_dir_geo / "peaks_from_fragment_counts.bed"
                        n_peaks = _convert_peak_count_matrix_to_bed(resolved, bed_out)
                        if n_peaks < 100:
                            raise RuntimeError(
                                f"Peak-count matrix converted to only {n_peaks} intervals"
                            )
                        logger.info(
                            "Converted GEO peak×cell matrix → BED: %d peaks → %s",
                            n_peaks,
                            bed_out,
                        )
                        accessibility_meta["files"] = [str(bed_out.resolve())]
                        accessibility_meta["peak_source_matrix"] = str(resolved)
                    else:
                        accessibility_meta["files"] = [str(resolved)]
                    accessibility_meta["has_peak_file"] = True
                except Exception as exc:
                    logger.warning(
                        "Failed to download GEO peak file from %s: %s",
                        peak_url[:120],
                        exc,
                    )
            else:
                logger.warning(
                    "GEO series %s has no peak-like supplementary file (bed/narrowPeak) in listing; "
                    "provide --atac-file or a different GEO accession.",
                    accessibility_meta.get("accession"),
                )
        # #region agent log
        _debug_log(
            str(accessibility_meta.get("accession") or "acq-run"),
            "H1",
            "acquire_multimodal_data.py:geo_peak_download",
            "geo_files_after_download_attempt",
            {
                "accession": accessibility_meta.get("accession"),
                "n_urls": len(raw_urls),
                "first_ref": first_ref[:200],
                "need_download": need_download,
                "resolved_files0": (accessibility_meta.get("files") or [None])[0],
                "resolved_is_file": bool(
                    Path(str((accessibility_meta.get("files") or [""])[0])).is_file()
                )
                if accessibility_meta.get("files")
                else False,
            },
        )
        # #endregion
    
    # Build promoter accessibility file if local peaks are available.
    promoter_accessibility_path: str | None = None
    peak_file_for_processing: str | None = None
    if args.atac_file and Path(args.atac_file).is_file():
        peak_file_for_processing = args.atac_file
    elif accessibility_meta.get("files"):
        cand0 = str(accessibility_meta["files"][0])
        if Path(cand0).is_file():
            peak_file_for_processing = cand0

    if peak_file_for_processing:
        try:
            peaks_df = load_peaks_bed(peak_file_for_processing)
            coord_gtf_path: str | None = (args.gtf or None)
            if coord_gtf_path is None:
                # For explicit build-style genome keys (mm10/hg38/rn6...), prefer
                # a build-matched local GTF from GenomeDB over BioMart.
                gk = str(genome or "").strip().lower()
                if re.match(r"^(mm|hg|rn|ce|dm|danrer|tair)\d+", gk):
                    try:
                        from grn_agent.acquisition.genome_db import ensure_genome

                        _, _gtf = ensure_genome(genome)
                        coord_gtf_path = str(_gtf)
                    except Exception as exc:
                        logger.warning("Could not resolve GenomeDB GTF for %s: %s", genome, exc)
            coord_df = load_gene_coords(
                genome,
                gtf_path=coord_gtf_path,
                cache_dir=cache / "gene_coords",
                gene_symbols=list(set(gene_symbols)),
            )
            if coord_df.empty and not args.gtf:
                # If BioMart is unavailable, hydrate GenomeDB and retry using its local GTF.
                try:
                    from grn_agent.acquisition.genome_db import get_genome_db

                    db = get_genome_db()
                    if db.get(genome) is None:
                        db.ensure_genome(genome)
                    entry = db.get(genome)
                    if entry and Path(entry.gtf_path).is_file():
                        coord_df = load_gene_coords(
                            genome,
                            gtf_path=entry.gtf_path,
                            cache_dir=cache / "gene_coords",
                            gene_symbols=list(set(gene_symbols)),
                        )
                except Exception as exc:
                    logger.warning("GenomeDB-assisted gene coordinate load failed: %s", exc)
            if not coord_df.empty:
                coord_df = coord_df.copy()
                coord_df["gene_symbol"] = coord_df["gene_symbol"].astype(str).str.upper()
                coord_df = coord_df.drop_duplicates(subset=["gene_symbol"], keep="first")
                acc = compute_promoter_accessibility(peaks_df, coord_df, window=args.motif_window)
                dataset_id_pa = args.dataset_id or f"{args.species}_{args.cell_type or 'unknown'}".replace(" ", "_")
                out_dir_pa = Path(args.out_manifest).parent / dataset_id_pa
                out_dir_pa.mkdir(parents=True, exist_ok=True)
                pa_path = out_dir_pa / "promoter_accessibility.bed"
                with pa_path.open("w", encoding="utf-8") as out:
                    for g in gene_symbols:
                        out.write(f"{g}\t{float(acc.get(str(g).strip().upper(), 0.0))}\n")
                promoter_accessibility_path = str(pa_path)
                # Leakage-safe denominator options derived only from RNA.
                cov_targets = (
                    set(expressed_genes)
                    if args.coverage_denominator == "rnaseq_expressed"
                    else set(gene_symbols)
                )
                accessibility_meta["promoter_coverage_of_targets"] = compute_coverage_fraction(acc, cov_targets)
                accessibility_meta["promoter_peak_fraction"] = compute_promoter_peak_fraction(
                    peaks_df,
                    coord_df,
                    window=args.motif_window,
                )
                accessibility_meta["total_peak_count"] = int(len(peaks_df))
                accessibility_meta["coverage_denominator"] = args.coverage_denominator
                accessibility_meta["coverage_gene_count"] = len(cov_targets)
                logger.info(
                    "Promoter accessibility built: %s (coverage=%.2f%%; promoter-peak-frac=%.2f; denominator=%s n=%d)",
                    pa_path,
                    100.0 * float(accessibility_meta.get("promoter_coverage_of_targets", 0.0)),
                    float(accessibility_meta.get("promoter_peak_fraction", 0.0)),
                    args.coverage_denominator,
                    len(cov_targets),
                )
            else:
                logger.warning("Gene coordinate table is empty; promoter accessibility file not generated.")
        except Exception as exc:
            logger.warning("Failed to compute promoter accessibility from peaks: %s", exc)

    # ── Motif integration (JASPAR → bedtools getfasta → FIMO) ─────────────
    motif_hits_path: str | None = None
    motif_meta: dict = {
        "database": f"JASPAR{args.jaspar_release}",
        "scanner": "FIMO",
        "tf_motif_count": 0,
        "tf_overlap_with_rnaseq": 0.0,
        "status": "skipped",
    }

    has_peaks = accessibility_meta.get("status") not in ("not_found", "none")

    auto_genome = not getattr(args, "no_auto_genome", False)

    if args.skip_motif:
        logger.info("Motif integration skipped (--skip-motif)")
    elif not has_peaks:
        logger.info("Motif integration skipped (no accessibility data)")
    else:
        # Resolve genome FASTA: user-provided or GenomeDB auto-download
        resolved_genome_fasta: str | None = args.genome_fasta.strip() if args.genome_fasta else None
        resolved_gtf: str | None = args.gtf

        if not resolved_genome_fasta:
            if auto_genome:
                logger.info(
                    "No --genome-fasta provided — will auto-download '%s' via GenomeDB.", genome
                )
            else:
                logger.info(
                    "Motif integration skipped — no --genome-fasta and --no-auto-genome set."
                )

        # Resolve peak file: prefer user-provided file, else first ENCODE file
        peak_file: str | None = None
        if args.atac_file:
            peak_file = args.atac_file
        elif accessibility_meta.get("files"):
            candidate = str(accessibility_meta["files"][0])
            if Path(candidate).is_file():
                peak_file = candidate

        if not peak_file:
            logger.warning(
                "Motif integration skipped — no local peak BED file available.\n"
                "  Use --atac-file peaks.bed to provide peaks directly."
            )
        else:
            dataset_id_local = args.dataset_id or (
                f"{args.species}_{args.cell_type or 'unknown'}".replace(" ", "_")
            )
            out_dir_local = Path(args.out_manifest).parent / dataset_id_local
            motif_tsv = out_dir_local / "motif_hits.tsv"
            out_dir_local.mkdir(parents=True, exist_ok=True)

            logger.info(
                "Running motif integration:\n"
                "  Protocol : bedtools getfasta → FIMO (p ≤ %.0e) → aggregate\n"
                "  Peaks    : %s\n"
                "  Genome   : %s\n"
                "  TFs      : %d expressed",
                args.fimo_pvalue,
                peak_file,
                resolved_genome_fasta or f"auto-download({genome})",
                len(expressed_tfs),
            )
            try:
                motif_df = run_motif_integration(
                    peaks_bed=peak_file,
                    tf_names=sorted(expressed_tfs),
                    species_or_genome=genome,
                    genome_fasta=resolved_genome_fasta or None,
                    output_tsv=motif_tsv,
                    gtf_path=resolved_gtf,
                    promoter_window=args.motif_window,
                    fimo_pvalue=args.fimo_pvalue,
                    jaspar_cache_dir=cache / "jaspar",
                    jaspar_release=args.jaspar_release,
                    gene_cache_dir=cache / "gene_coords",
                    pair_filter=None,
                    auto_download_genome=auto_genome,
                    fimo_workers=args.fimo_workers,
                    fimo_timeout=args.fimo_timeout,
                )
                if not motif_df.empty:
                    motif_df["source_tf"] = motif_df["source_tf"].astype(str).str.strip().str.upper()
                    motif_df["target_gene"] = motif_df["target_gene"].astype(str).str.strip().str.upper()
                    motif_df.to_csv(motif_tsv, sep="\t", index=False)
                n_pairs = len(motif_df)
                n_present = int(motif_df["motif_present"].sum()) if not motif_df.empty else 0
                tf_with_hits = motif_df["source_tf"].nunique() if not motif_df.empty else 0
                logger.info(
                    "Motif integration complete: %d TF-gene pairs, %d with motif present",
                    n_pairs, n_present,
                )
                motif_hits_path = str(motif_tsv)
                motif_meta = {
                    "database": f"JASPAR{args.jaspar_release}",
                    "scanner": "FIMO",
                    "fimo_pvalue_threshold": args.fimo_pvalue,
                    "tf_motif_count": tf_with_hits,
                    "tf_overlap_with_rnaseq": tf_with_hits / max(1, len(expressed_tfs)),
                    "n_tf_gene_pairs": n_pairs,
                    "n_pairs_with_motif": n_present,
                    "status": "complete",
                }
            except RuntimeError as exc:
                logger.warning("Motif integration failed: %s", exc)
                motif_meta["status"] = "failed"
                motif_meta["error"] = str(exc)
    
    # Validate compatibility
    logger.info("Validating dataset compatibility...")
    _debug_log(
        str(accessibility_meta.get("accession") or "acq-run"),
        "H1-H5",
        "acquire_multimodal_data.py:before_qc",
        "pre_qc_meta",
        {
            "rna_meta": {
                "species": rna_meta.get("species"),
                "cell_type": rna_meta.get("cell_type"),
                "cell_line": rna_meta.get("cell_line"),
                "lineage": rna_meta.get("lineage"),
                "state": rna_meta.get("state"),
                "n_replicates": rna_meta.get("n_replicates"),
                "allow_perturbation": rna_meta.get("allow_perturbation"),
            },
            "accessibility_meta": {
                "source": accessibility_meta.get("source"),
                "accession": accessibility_meta.get("accession"),
                "species": accessibility_meta.get("species"),
                "cell_type": accessibility_meta.get("cell_type"),
                "cell_line": accessibility_meta.get("cell_line"),
                "lineage": accessibility_meta.get("lineage"),
                "state": accessibility_meta.get("state"),
                "assay": accessibility_meta.get("assay"),
                "perturbation": accessibility_meta.get("perturbation"),
                "n_replicates": accessibility_meta.get("n_replicates"),
                "promoter_cov": accessibility_meta.get("promoter_coverage_of_targets"),
                "status": accessibility_meta.get("status"),
            },
            "strict": bool(args.strict),
        },
    )
    qc = validate_dataset_compatibility(
        rna_meta,
        accessibility_meta,
        gold_genes,
        gold_tfs,
        strict=args.strict,
        min_promoter_coverage=float(args.min_promoter_coverage),
    )
    logger.info(f"QC result: {qc['pass']}")
    if qc["rejection_reasons"]:
        logger.warning(f"Rejection reasons: {qc['rejection_reasons']}")
    if qc["warnings"]:
        logger.info(f"Warnings: {qc['warnings']}")
    
    # ── Output paths ───────────────────────────────────────────────────────
    dataset_id = args.dataset_id or f"{args.species}_{args.cell_type or 'unknown'}".replace(" ", "_")
    out_dir = Path(args.out_manifest).parent / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save gene symbols
    genes_file = out_dir / "genes.txt"
    genes_file.write_text("\n".join(gene_symbols), encoding="utf-8")

    output_paths: dict[str, str] = {
        "expression_matrix": str(Path(args.expr).resolve()),
        "gene_symbols": str(genes_file),
    }

    if promoter_accessibility_path and Path(promoter_accessibility_path).is_file():
        output_paths["promoter_accessibility"] = promoter_accessibility_path
    if motif_hits_path:
        output_paths["motif_hits"] = motif_hits_path
    
    # Build manifest
    logger.info(f"Writing manifest to {args.out_manifest}")
    build_multimodal_manifest(
        dataset_id=dataset_id,
        species=args.species,
        cell_type=args.cell_type or "unknown",
        genome_build=genome,
        rna_meta=rna_meta,
        accessibility_meta=accessibility_meta,
        motif_meta=motif_meta,
        qc_report=qc,
        output_paths=output_paths,
        output_file=args.out_manifest,
    )
    logger.info("Done.")


def _tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", str(s or "").lower()) if t}


def _load_geo_series_candidate(
    hit: dict,
    *,
    cache_dir: Path,
    genome: str,
    requested_species: str,
    requested_cell_type: str | None,
    requested_cell_line: str | None,
    requested_lineage: str | None,
    requested_state: str | None,
    requested_cell_context: str | None,
    requested_perturbation: str,
    allow_perturbation: bool,
) -> tuple[dict, float, dict] | None:
    acc = str(hit.get("accession", "")).strip()
    if not acc:
        return None

    geo = GEOClient(cache_dir=cache_dir)
    series = geo.get_series_metadata(acc, geo_id=str(hit.get("geo_id", "")).strip() or None)
    meta = _extract_best_geo_accessibility_meta_from_series(
        geo,
        series,
        genome=genome,
        requested_species=requested_species,
        requested_cell_type=requested_cell_type,
        requested_cell_line=requested_cell_line,
        requested_lineage=requested_lineage,
        requested_state=requested_state,
    )
    score, detail = _score_accessibility_candidate(
        meta,
        requested_species=requested_species,
        requested_cell_type=requested_cell_type,
        requested_cell_line=requested_cell_line,
        requested_lineage=requested_lineage,
        requested_state=requested_state,
        requested_cell_context=requested_cell_context,
        requested_perturbation=requested_perturbation,
        allow_perturbation=allow_perturbation,
        filtered_hit=False,
    )
    meta["selection_score_breakdown"] = detail
    return meta, score, detail


def _collect_parallel_geo_candidates(
    geo_hits: list[dict],
    *,
    cache_dir: Path,
    genome: str,
    requested_species: str,
    requested_cell_type: str | None,
    requested_cell_line: str | None,
    requested_lineage: str | None,
    requested_state: str | None,
    requested_cell_context: str | None,
    requested_perturbation: str,
    allow_perturbation: bool,
    max_workers: int,
) -> list[tuple[dict, float, dict]]:
    if not geo_hits:
        return []

    if max_workers <= 1:
        out: list[tuple[dict, float, dict]] = []
        for hit in geo_hits:
            acc = str(hit.get("accession", "")).strip()
            try:
                result = _load_geo_series_candidate(
                    hit,
                    cache_dir=cache_dir,
                    genome=genome,
                    requested_species=requested_species,
                    requested_cell_type=requested_cell_type,
                    requested_cell_line=requested_cell_line,
                    requested_lineage=requested_lineage,
                    requested_state=requested_state,
                    requested_cell_context=requested_cell_context,
                    requested_perturbation=requested_perturbation,
                    allow_perturbation=allow_perturbation,
                )
                if result is not None:
                    out.append(result)
            except Exception as exc:
                logger.warning(f"GEO series parse failed for {acc}: {exc}")
        return out

    out: list[tuple[dict, float, dict]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _load_geo_series_candidate,
                hit,
                cache_dir=cache_dir,
                genome=genome,
                requested_species=requested_species,
                requested_cell_type=requested_cell_type,
                requested_cell_line=requested_cell_line,
                requested_lineage=requested_lineage,
                requested_state=requested_state,
                requested_cell_context=requested_cell_context,
                requested_perturbation=requested_perturbation,
                allow_perturbation=allow_perturbation,
            ): str(hit.get("accession", "")).strip()
            for hit in geo_hits
        }
        for future in as_completed(futures):
            acc = futures[future]
            try:
                result = future.result()
                if result is not None:
                    out.append(result)
            except Exception as exc:
                logger.warning(f"GEO series parse failed for {acc}: {exc}")
    return out


def _infer_perturbation_label(*texts: str) -> str:
    """Infer a compact perturbation label from free-text metadata."""
    blob = " ".join(str(t or "").lower() for t in texts if str(t or "").strip())
    if not blob:
        return ""
    tok = _tokens(blob)
    if ("lipopolysaccharide" in blob) or ("lps" in tok):
        return "lps"
    if ("interferon" in blob) or ("ifn" in tok):
        return "interferon"
    if ("tumor necrosis factor" in blob) or ("tnf" in tok):
        return "tnf"
    return ""


def _text_overlap_score(query: str, text: str) -> float:
    q = _tokens(query)
    t = _tokens(text)
    if not q or not t:
        return 0.0
    return float(len(q & t) / max(1, len(q | t)))


def _get_semantic_slm():
    global _SEMANTIC_SLM_INSTANCE, _SEMANTIC_SLM_LOAD_FAILED
    if _SEMANTIC_SLM_INSTANCE is not None:
        return _SEMANTIC_SLM_INSTANCE
    if _SEMANTIC_SLM_LOAD_FAILED:
        return None
    try:
        from sentence_transformers import SentenceTransformer

        _SEMANTIC_SLM_INSTANCE = SentenceTransformer(_SEMANTIC_SLM_MODEL)
        logger.info("Semantic SLM enabled for scoring: %s", _SEMANTIC_SLM_MODEL)
        return _SEMANTIC_SLM_INSTANCE
    except Exception as exc:
        _SEMANTIC_SLM_LOAD_FAILED = True
        logger.warning(
            "Semantic SLM unavailable (%s). Falling back to token-overlap semantic score.",
            exc,
        )
        return None


def _semantic_similarity(query: str, text: str) -> float:
    """
    Hybrid semantic similarity in [0, 1] with a TRUE zero baseline.

    Problem with the naive 0.5*(cos+1) mapping:
      all-MiniLM-L6-v2 produces cosine ~0.2-0.4 for *unrelated* biomedical text
      pairs, which maps to ~0.60-0.70 after the affine shift — the same range
      used by the rubric to assign s_bio=1.0 (exact match).  This caused
      Ammon's horn tissue to score identically to hESC endoderm.

    Fix: use raw cosine clamped to [0, 1].  Unrelated biomedical pairs now
    produce ~0.2-0.4, genuinely matched pairs produce 0.6-0.9, exact
    repetitions produce >0.95.  The rubric thresholds (0.55/0.40/0.25/0.15)
    are preserved and now correctly discriminate.

    Token-overlap is kept as a fallback and as a floor when SLM is off.
    """
    base = _text_overlap_score(query, text)
    if not _USE_SEMANTIC_SLM:
        return base
    model = _get_semantic_slm()
    if model is None:
        return base
    try:
        emb = model.encode([str(query or ""), str(text or "")], normalize_embeddings=True)
        qv = emb[0]
        tv = emb[1]
        # Raw cosine of L2-normalised vectors; clamp negatives to 0.
        # This preserves a true zero for orthogonal/unrelated pairs.
        cos = float(sum(float(a) * float(b) for a, b in zip(qv, tv)))
        cos_clamped = max(0.0, min(1.0, cos))
        return max(base, cos_clamped)
    except Exception:
        return base


def _norm_cell_line(s: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s or "").lower())


def _extract_cell_line_from_label(label: str | None) -> str:
    text = str(label or "").strip()
    if not text:
        return ""
    for cp in (r"\b[a-z]{1,3}-[a-z]?\d{1,3}[a-z]?\b", r"\b[a-z]{2,6}\d{1,4}[a-z]?\b"):
        m = re.search(cp, text, flags=re.IGNORECASE)
        if m:
            return str(m.group(0)).strip()
    return ""


def _extract_cell_line_from_text(*texts: str) -> str:
    blob = " ".join(str(t or "") for t in texts)
    # Only trust matches when near explicit cell-line cues.
    cue_windows = (
        r"(cell\s*line|line|clone|esc|mesc)[^\\n\\r;,:()]{0,60}",
    )
    candidate_pats = (
        r"\b[a-z]{1,4}[-_ ]?\d{1,4}[a-z]{0,2}\b",  # e.g. es-e14, h9, j1
        r"\b[a-z]{2,8}\d{1,4}[a-z]{0,2}\b",        # e.g. e14tg2a, hek293t
    )
    for win_pat in cue_windows:
        for win in re.finditer(win_pat, blob, flags=re.IGNORECASE):
            seg = win.group(0)
            for cp in candidate_pats:
                m = re.search(cp, seg, flags=re.IGNORECASE)
                if m:
                    return str(m.group(0)).strip()
    # Multiome / barnyard GEO titles often encode the line as an underscore
    # token (e.g. scATAC_mESC_K562) without a "cell line" cue nearby.
    for m in re.finditer(
        r"(?:^|[_\-/])([A-Za-z]{1,4}\d{2,4}[A-Za-z]?)(?=$|[_\-/\s])",
        blob,
    ):
        tok = str(m.group(1)).strip()
        if len(tok) >= 3:
            return tok
    # No unconstrained fallback: standalone alnum tokens in summaries can be genes
    # (e.g., SOX17) and should not be interpreted as cell-line identifiers.
    return ""

def _assign_tier(score_01: float) -> str:
    """Map a normalised [0,1] composite score to a selection tier."""
    if score_01 >= 0.75:
        return "A_STRONG_MATCH"
    if score_01 >= 0.60:
        return "B_FALLBACK_MATCH"
    return "REJECT"


# ---------------------------------------------------------------------------
# Hard-rejection filter  (Step 3 of the protocol)
# ---------------------------------------------------------------------------

_STRONG_BIO_MISMATCH_PAIRS: list[tuple[set[str], set[str]]] = [
    # embryonic / stem-cell context vs clearly differentiated somatic cell/tissue types.
    # NOTE: "tissue" is intentionally absent here — it is too generic and would
    # block valid stem-cell studies whose metadata includes the word "tissue".
    # Named anatomical regions (hippocampus, ammon, horn) are included because
    # those are never a valid substitute for ESC/pluripotent context.
    ({"embryonic", "stem", "esc", "hesc", "mesc", "ipsc", "pluripotent"},
     {"fibroblast", "neuron", "cortex", "liver", "hepatocyte", "heart",
      "muscle", "lung", "kidney", "adipocyte", "osteoblast",
      "hippocampus", "ammon", "horn"}),
    # liver vs neural
    ({"hepatocyte", "liver", "hepatic"},
     {"neuron", "neuronal", "cortex", "hippocampus", "cerebellar", "glial"}),
    # blood/immune vs epithelial solid-tissue types
    ({"pbmc", "blood", "lymphocyte", "tcell", "bcell", "monocyte"},
     {"epithelial", "keratinocyte", "hepatocyte", "cardiomyocyte"}),
]


def _has_strong_bio_mismatch(req_blob: str, meta_blob: str) -> bool:
    """Return True if req and meta tokens match opposite poles of a known
    biologically-incompatible pair."""
    req_tok = _tokens(req_blob)
    meta_tok = _tokens(meta_blob)
    for req_set, meta_set in _STRONG_BIO_MISMATCH_PAIRS:
        if (req_tok & req_set) and (meta_tok & meta_set):
            return True
        if (meta_tok & req_set) and (req_tok & meta_set):
            return True
    return False


def _is_candidate_acceptable(
    meta: dict,
    score_01: float,
    detail: dict,
    *,
    requested_species: str | None = None,
    requested_cell_type: str | None = None,
    requested_cell_line: str | None = None,
    requested_cell_context: str | None = None,
) -> bool:
    """
    Hard-rejection gate (Step 3).  Every rule here is a hard veto:
    a candidate that trips any rule is NEVER selected, regardless of
    its numeric score.
    """
    # 3.1  Species mismatch
    if detail.get("species_mismatch"):
        return False

    assay_lc = str(meta.get("assay", "") or "").lower()
    if "rna" in assay_lc and not any(tok in assay_lc for tok in ("atac", "dnase", "accessibility")):
        return False

    # 3.2  Strong biological mismatch: e.g. embryonic stem cell vs tissue
    req_blob = " ".join(filter(None, [requested_cell_type, requested_cell_context]))
    meta_blob = " ".join(
        str(meta.get(k) or "")
        for k in ("cell_type", "lineage", "state", "description",
                  "aliases", "synonyms", "biosample_summary")
    )
    if req_blob and meta_blob and _has_strong_bio_mismatch(req_blob, meta_blob):
        return False

    # 3.3  Strong condition mismatch (perturbation when not allowed)
    if detail.get("condition_mismatch"):
        return False

    # 3.4  Missing required cell-line evidence when one was explicitly requested
    if detail.get("cell_line_missing"):
        return False

    # 3.5  Cell-type anchor tokens not found anywhere in candidate metadata.
    # This is a ranking/QC signal, not a file-usability veto. A peak-bearing
    # ATAC/DNase candidate should remain selectable so downstream QC reports
    # the true biology/coverage issue instead of an RNA-only missing-file error.

    # 3.6  Technically unusable: no peak files and no signal.
    # GEO RNA samples may have matrix/count supplementary files; those are not
    # usable accessibility evidence.
    has_peak = bool(meta.get("has_peak_file"))
    has_signal = bool(meta.get("signal_files") or meta.get("signal") or meta.get("has_signal_file"))
    has_files = bool(meta.get("files"))
    if not has_peak and not has_signal:
        status = str(meta.get("status", "")).lower()
        source = str(meta.get("source", "")).upper()
        # ENCODE search rows can omit files until the selected experiment is
        # refreshed. GEO candidates without any file/signal URL are unusable.
        if not (source == "ENCODE" and status == "released"):
            return False

    # 3.x  The numeric tier is used for ranking, not for deciding whether real
    # accessibility data exists. Hard vetoes above still reject unsafe matches.

    return True


def _is_same_series_accessibility_candidate(meta: dict) -> bool:
    return bool(
        meta.get("matched_rna_accession")
        or str(meta.get("pairing_quality") or "") in {"same_series_sample", "same_series", "request_geo_seed"}
    )


def _is_rejected_but_usable_accessibility_candidate(meta: dict, detail: dict) -> bool:
    """Keep a real file-bearing accessibility candidate for downstream QC/audit.

    This prevents a misleading RNA-only fallback when GEO found a peak-bearing
    ATAC/DNase sample that failed only the biological similarity threshold.
    Species and condition mismatches remain vetoes.
    """
    if str(meta.get("source") or "").upper() != "GEO":
        return False
    assay = str(meta.get("assay") or "").lower()
    if "rna" in assay and not any(tok in assay for tok in ("atac", "dnase", "accessibility")):
        return False
    if detail.get("species_mismatch") or detail.get("condition_mismatch") or detail.get("cell_line_missing"):
        return False
    return bool(meta.get("has_peak_file") or meta.get("has_signal_file"))


# ---------------------------------------------------------------------------
# Component scoring  (Steps 4-5 of the protocol)
# ---------------------------------------------------------------------------

def _score_accessibility_candidate(
    meta: dict,
    *,
    requested_species: str,
    requested_cell_type: str | None,
    requested_cell_line: str | None,
    requested_lineage: str | None,
    requested_state: str | None,
    requested_cell_context: str | None,
    requested_perturbation: str | None,
    allow_perturbation: bool,
    filtered_hit: bool = False,
) -> tuple[float, dict]:
    """
    Compute a normalised [0, 1] composite score following the protocol:

        S = 0.35*s_bio + 0.20*s_cond + 0.15*s_stage
          + 0.10*s_study + 0.05*s_assay + 0.10*s_quality + 0.05*s_build

    Returns (composite_score_01, detail_dict).
    detail_dict carries component scores AND hard-rejection flags so
    _is_candidate_acceptable() can test them without re-running logic.
    """
    why: dict[str, float | bool] = {}

    # ---- species (prerequisite; treated as hard flag, not a component) ------
    req_species = canonical_species_label(requested_species) or str(requested_species or "").lower()
    meta_species = canonical_species_label(meta.get("species")) or str(meta.get("species") or "").lower()
    species_ok = bool(req_species and meta_species and req_species == meta_species)
    species_unknown = not req_species or not meta_species
    if not species_ok and not species_unknown:
        why["species_mismatch"] = True
    why["species_ok"] = float(species_ok)

    # ---- build context blobs ------------------------------------------------
    context_blob = " ".join(
        str(meta.get(k) or "")
        for k in ("cell_type", "lineage", "state", "description",
                  "aliases", "synonyms", "biosample_summary")
    )
    req_tok = _tokens(requested_cell_type or "")
    ctx_tok = _tokens(context_blob)
    meta_ct = str(meta.get("cell_type") or "").strip().lower()

    # =========================================================
    # 5.1  Biological identity score  s_bio  (weight 0.35)
    # =========================================================
    if not requested_cell_type:
        s_bio = 0.50  # no information, neutral
        why["s_bio"] = s_bio
    else:
        sem = _semantic_similarity(requested_cell_type, context_blob)
        why["cell_type_semantic_raw"] = sem

        anchors = {t for t in req_tok if len(t) >= 5}
        anchors_found = bool(anchors & ctx_tok) or filtered_hit

        if not anchors_found:
            # Hard flag for _is_candidate_acceptable
            why["cell_type_anchor_penalty"] = True
            s_bio = 0.00
        elif filtered_hit:
            # Exact ENCODE biosample-term match
            s_bio = max(0.90, min(1.00, sem + 0.50))
        else:
            # Map sem [0,1] -> s_bio rubric
            if sem >= 0.55:
                s_bio = 1.00   # exact / very close
            elif sem >= 0.40:
                s_bio = 0.90   # exact cell type, line unknown
            elif sem >= 0.25:
                s_bio = 0.75   # ontology-equivalent / close synonym
            elif sem >= 0.15:
                s_bio = 0.60   # same lineage / tissue compartment
            elif sem >= 0.08:
                s_bio = 0.30   # same broad tissue only
            else:
                s_bio = 0.00   # mismatch

        # Downgrade when cell_type field is opaque
        if meta_ct in ("", "unknown", "na", "n/a") and s_bio > 0:
            # allow rescue only if semantic evidence is strong
            if sem >= 0.25:
                s_bio = min(s_bio, 0.75)
                why["cell_type_summary_rescue"] = True
            else:
                s_bio = min(s_bio, 0.30)

        why["s_bio"] = s_bio

    # =========================================================
    # 5.2  Condition score  s_cond  (weight 0.20)
    # =========================================================
    perturb = str(meta.get("perturbation", "")).strip().lower()
    perturb_is_baseline = (
        not perturb
        or perturb in ("none", "control", "untreated", "naive", "unknown", "")
    )
    req_pert = str(requested_perturbation or "").strip().lower()
    if req_pert:
        req_tok = _tokens(req_pert)
        perturb_tok = _tokens(perturb)
        perturb_unknown = perturb in ("unknown", "na", "n/a", "")
        if perturb_unknown:
            # Unknown condition should not outrank explicit condition matches.
            s_cond = 0.30
            why["condition_unknown"] = True
        elif req_tok and (req_tok & perturb_tok):
            s_cond = 1.00
            why["condition_match"] = True
        elif perturb_is_baseline:
            s_cond = 0.10
            if not allow_perturbation:
                why["condition_mismatch"] = True
        else:
            s_cond = 0.20
            if not allow_perturbation:
                why["condition_mismatch"] = True
    else:
        if not allow_perturbation and not perturb_is_baseline:
            s_cond = 0.00
            why["condition_mismatch"] = True
        elif perturb_is_baseline:
            s_cond = 1.00
        else:
            # perturbation present and allow_perturbation=True — still slightly penalise
            # unknown compatibility
            s_cond = 0.50
    why["s_cond"] = s_cond

    # =========================================================
    # 5.3  Stage / time score  s_stage  (weight 0.15)
    # =========================================================
    if requested_state:
        stage_sim = _semantic_similarity(requested_state, context_blob)
        if stage_sim >= 0.40:
            s_stage = 1.00
        elif stage_sim >= 0.20:
            s_stage = 0.80
        else:
            s_stage = 0.50   # unknown, not contradictory
        why["state_semantic_raw"] = stage_sim
    else:
        s_stage = 0.50   # no stage requested
    why["s_stage"] = s_stage

    # =========================================================
    # 5.4  Provenance / study score  s_study  (weight 0.10)
    # =========================================================
    # Same-series RNA->ATAC GEO pairing or ENCODE biosample-term match are
    # high-provenance signals.
    if meta.get("matched_rna_accession") or meta.get("pairing_quality") in {"same_series_sample", "same_series"}:
        s_study = 1.00
    elif filtered_hit:
        s_study = 1.00
    else:
        s_study = 0.30   # unrelated study (default; no cross-reference available)
    why["s_study"] = s_study

    # =========================================================
    # 5.5  Assay score  s_assay  (weight 0.05)
    # =========================================================
    assay = str(meta.get("assay", "")).lower()
    if "atac" in assay:
        s_assay = 1.00
    elif "dnase" in assay:
        s_assay = 0.85
    else:
        s_assay = 0.20
    why["s_assay"] = s_assay

    # =========================================================
    # 5.6  Quality score  s_quality  (weight 0.10)
    # =========================================================
    n_reps = int(meta.get("n_replicates", 0) or 0)
    has_peak = bool(meta.get("has_peak_file"))
    has_signal = bool(meta.get("signal_files") or meta.get("signal") or meta.get("has_signal_file"))
    status_available = str(meta.get("status", "")).lower() in {"released", "public", "available", "user_provided"}

    q = 0.0
    if n_reps >= 2:
        q += 0.40
    elif n_reps == 1:
        q += 0.20
    if has_peak or has_signal:
        q += 0.30
    if status_available:
        q += 0.30
    s_quality = min(q, 1.00)
    why["s_quality"] = s_quality

    # =========================================================
    # 5.7  Genome build score  s_build  (weight 0.05)
    # =========================================================
    # genome build compatibility is stored in meta if inferred upstream
    build_match = str(meta.get("genome_build_match", "unknown")).lower()
    if build_match == "exact":
        s_build = 1.00
    elif build_match == "liftover":
        s_build = 0.50
    elif build_match == "incompatible":
        s_build = 0.00
    else:
        s_build = 0.20   # unclear
    why["s_build"] = s_build

    # =========================================================
    # Cell-line check  (hard flag, not a scoring component)
    # =========================================================
    if requested_cell_line:
        req_line = _norm_cell_line(requested_cell_line)
        meta_line = _norm_cell_line(str(meta.get("cell_line") or ""))
        # Rescue line tokens buried in multiome titles/summaries
        # (e.g. scATAC_mESC_K562) when structured cell_line is empty.
        if req_line and not meta_line and req_line in _norm_cell_line(context_blob):
            meta_line = req_line
            why["cell_line_from_context"] = True
        if req_line and not meta_line:
            why["cell_line_missing"] = True
        elif req_line and meta_line:
            if req_line == meta_line or req_line in meta_line or meta_line in req_line:
                why["cell_line_match"] = True
                # Confirmed cell-line match: boost s_bio and clear anchor penalty.
                # The anchor check was run on cell-type tokens only; a direct
                # cell-line match is stronger evidence and overrides that signal.
                s_bio = 1.00
                why["s_bio"] = s_bio
                why.pop("cell_type_anchor_penalty", None)
            else:
                why["cell_line_mismatch"] = True
                s_bio = 0.00
                why["s_bio"] = s_bio

    # Prefer real ATAC / multiome over ADT/HTO library titles when the
    # request context asks for paired scATAC / multiome accessibility.
    req_ctx_lc = str(requested_cell_context or "").lower()
    title_lc = " ".join(
        str(meta.get(k) or "") for k in ("description", "biosample_summary", "cell_type")
    ).lower()
    wants_atac = any(tok in req_ctx_lc for tok in ("scatac", "atac", "multiome", "dnase"))
    if wants_atac and any(tok in title_lc for tok in ("scadt", "sc_hto", "schto", "adt_", "hto_")):
        s_assay = min(float(s_assay), 0.10)
        why["s_assay"] = s_assay
        why["adt_hto_assay_penalty"] = True
    # =========================================================
    # Lineage hint  (soft booster for s_bio when lineage given)
    # =========================================================
    if requested_lineage:
        lin_sim = _text_overlap_score(requested_lineage, context_blob)
        s_bio = min(1.00, s_bio + 0.10 * lin_sim)
        why["s_bio"] = s_bio
        why["lineage_overlap"] = lin_sim

    # =========================================================
    # Cell-context semantic (extra semantic signal for s_bio)
    # =========================================================
    if requested_cell_context and not why.get("cell_type_anchor_penalty"):
        ctx_sim = _semantic_similarity(requested_cell_context, context_blob)
        s_bio = min(1.00, s_bio + 0.15 * ctx_sim)
        why["s_bio"] = s_bio
        why["cell_context_semantic_raw"] = ctx_sim

    # =========================================================
    # Composite score (normalised [0, 1])
    # =========================================================
    composite = (
        0.35 * s_bio
        + 0.20 * s_cond
        + 0.15 * s_stage
        + 0.10 * s_study
        + 0.05 * s_assay
        + 0.10 * s_quality
        + 0.05 * s_build
    )
    why["composite"] = composite
    why["tier"] = _assign_tier(composite)

    # Hard-override: species mismatch collapses score to 0
    if why.get("species_mismatch"):
        composite = 0.0
        why["composite"] = 0.0
        why["tier"] = "REJECT"

    return composite, why


def _organism_scientific_name_from_experiment(exp: dict) -> str:
    """
    ENCODE ``/search/`` experiment objects usually omit ``biosample_ontology.organism``;
    organism lives on ``replicates[].library.biosample.organism.scientific_name``.
    """
    bo = exp.get("biosample_ontology")
    if isinstance(bo, dict):
        org = bo.get("organism")
        if isinstance(org, dict):
            sn = str(org.get("scientific_name", "")).strip()
            if sn:
                return sn
    for rep in exp.get("replicates") or []:
        if not isinstance(rep, dict):
            continue
        lib = rep.get("library")
        if not isinstance(lib, dict):
            continue
        bs = lib.get("biosample")
        if not isinstance(bs, dict):
            continue
        org = bs.get("organism")
        if isinstance(org, dict):
            sn = str(org.get("scientific_name", "")).strip()
            if sn:
                return sn
    top = exp.get("organism")
    if isinstance(top, dict):
        sn = str(top.get("scientific_name", "")).strip()
        if sn:
            return sn
    return ""


def _iter_nested_objects(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_nested_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_nested_objects(child)


def _extract_numeric_metric(payload: dict, aliases: tuple[str, ...]) -> float | None:
    alias_set = {a.strip().lower() for a in aliases}
    for obj in _iter_nested_objects(payload):
        if not isinstance(obj, dict):
            continue
        for key, raw in obj.items():
            key_norm = str(key).strip().lower().replace(" ", "_").replace("-", "_")
            if key_norm in alias_set:
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    continue
    return None


def _extract_encode_accessibility_qc_metrics(metrics_payloads: list[dict]) -> dict[str, float]:
    frip_aliases = (
        "frip",
        "frip_score",
        "fraction_of_reads_in_peaks",
        "fraction_reads_in_peaks",
        "spot_score",
        "signal_fraction",
    )
    tss_aliases = (
        "tss_enrichment",
        "tss_enrichment_score",
        "tss_score",
        "tss",
    )
    out: dict[str, float] = {}
    for payload in metrics_payloads:
        if "frip" not in out:
            value = _extract_numeric_metric(payload, frip_aliases)
            if value is not None:
                if value > 1.0 and "spot" not in json.dumps(payload).lower():
                    value = value / 100.0
                out["frip"] = value
        if "tss_enrichment" not in out:
            value = _extract_numeric_metric(payload, tss_aliases)
            if value is not None:
                out["tss_enrichment"] = value
        if "frip" in out and "tss_enrichment" in out:
            break
    return out


def _augment_encode_qc_metrics(encode: ENCODEClient, accessibility_meta: dict) -> None:
    if str(accessibility_meta.get("source") or "").upper() != "ENCODE":
        return
    accessions: list[str] = []
    for key in ("peak_file_accessions", "signal_file_accessions"):
        values = accessibility_meta.get(key, []) or []
        for value in values:
            acc = str(value or "").strip()
            if acc and acc not in accessions:
                accessions.append(acc)
    if not accessions:
        return
    payloads: list[dict] = []
    for acc in accessions:
        try:
            payload = encode.get_qc_metrics(acc)
        except Exception as exc:
            logger.warning("Failed to fetch ENCODE QC metrics for %s: %s", acc, exc)
            continue
        if payload:
            payloads.append(payload)
    metrics = _extract_encode_accessibility_qc_metrics(payloads)
    if "frip" in metrics and accessibility_meta.get("frip") is None:
        accessibility_meta["frip"] = float(metrics["frip"])
    if "tss_enrichment" in metrics and accessibility_meta.get("tss_enrichment") is None:
        accessibility_meta["tss_enrichment"] = float(metrics["tss_enrichment"])


def _extract_accessibility_meta(exp: dict, genome: str) -> dict:
    """Extract relevant metadata from ENCODE experiment."""
    asm_map = {
        "hg38": {"hg38", "grch38"},
        "grch38": {"hg38", "grch38"},
        "hg19": {"hg19", "grch37"},
        "grch37": {"hg19", "grch37"},
        "mm10": {"mm10", "grcm38"},
        "grcm38": {"mm10", "grcm38"},
    }
    genome_key = str(genome or "").strip().lower()
    acceptable_assemblies = asm_map.get(genome_key, {genome_key})

    files = exp.get("files", [])

    def _is_peak_like_encode_file(f: dict) -> bool:
        out_type = str(f.get("output_type") or "").strip().lower()
        file_fmt = str(f.get("file_format") or "").strip().lower()
        file_fmt_type = str(f.get("file_format_type") or "").strip().lower()
        is_peak = "peak" in out_type
        is_bed_like = (
            file_fmt in {"bed", "narrowpeak", "broadpeak"}
            or file_fmt_type in {"narrowpeak", "broadpeak"}
            or ("bed" in file_fmt)
        )
        return is_peak and is_bed_like and str(f.get("status") or "").strip().lower() == "released"

    def _encode_peak_rank_key(f: dict) -> tuple:
        out_type = str(f.get("output_type") or "").strip().lower()
        fmt_type = str(f.get("file_format_type") or "").strip().lower()
        fmt = str(f.get("file_format") or "").strip().lower()
        assembly = str(f.get("assembly") or "").strip().lower()
        is_exact_assembly = 1 if assembly in acceptable_assemblies else 0
        is_idr_optimal = 1 if (("idr" in out_type) and ("optimal" in out_type)) else 0
        if "narrowpeak" in fmt_type or "narrowpeak" in fmt:
            peak_type_rank = 3
        elif "broadpeak" in fmt_type or "broadpeak" in fmt:
            peak_type_rank = 2
        elif "bed" in fmt:
            peak_type_rank = 1
        else:
            peak_type_rank = 0
        return (is_exact_assembly, is_idr_optimal, peak_type_rank)

    peak_files_exact = [
        f for f in files
        if _is_peak_like_encode_file(f)
        and str(f.get("assembly") or "").strip().lower() in acceptable_assemblies
    ]
    peak_files_any = [f for f in files if _is_peak_like_encode_file(f)]
    peak_files = sorted((peak_files_exact if peak_files_exact else peak_files_any), key=_encode_peak_rank_key, reverse=True)
    signal_files = [
        f for f in files
        if (
            str(f.get("file_format") or "").strip().lower() in {"bigwig", "bw"}
            or "signal" in str(f.get("output_type") or "").strip().lower()
        )
        and str(f.get("status") or "").strip().lower() == "released"
    ]
    has_signal_file = bool(signal_files)
    
    biosample = exp.get("biosample_ontology", {})
    replicates = exp.get("replicates", [])
    scientific_name = _organism_scientific_name_from_experiment(exp)
    species_slug = canonical_species_label(scientific_name) or ""
    species_short = species_slug if species_slug else "unknown"
    lineage = (
        str(biosample.get("classification", "")).strip()
        or str(biosample.get("biosample_type", "")).strip()
        or ""
    )
    term_name = str(biosample.get("term_name", "")).strip()
    aliases = biosample.get("aliases", []) if isinstance(biosample, dict) else []
    synonyms = biosample.get("synonyms", []) if isinstance(biosample, dict) else []
    term_lower = term_name.lower()
    state = ""
    for s in ("naive", "primed", "activated", "quiescent", "differentiated"):
        if s in term_lower:
            state = s
            break

    # Collect treatment / perturbation hints from replicate biosamples if present
    perturb_tags: list[str] = []
    for rep in replicates:
        try:
            libs = rep.get("library", {}) if isinstance(rep, dict) else {}
            bs = libs.get("biosample", {}) if isinstance(libs, dict) else {}
            treatments = bs.get("treatments", []) if isinstance(bs, dict) else []
            for t in treatments:
                if isinstance(t, dict):
                    name = str(t.get("treatment_term_name", "")).strip()
                    if name:
                        perturb_tags.append(name.lower())
        except Exception:
            continue
    perturbation = ",".join(sorted(set(perturb_tags))) if perturb_tags else "none"
    
    # Determine genome build compatibility from available peak files
    peak_assemblies = {
        str(f.get("assembly") or "").strip().lower()
        for f in peak_files_any
        if str(f.get("assembly") or "").strip()
    }
    if peak_assemblies & acceptable_assemblies:
        genome_build_match = "exact"
    elif peak_assemblies:
        # Has files but no assembly in acceptable set — might need liftover
        genome_build_match = "liftover"
    else:
        genome_build_match = "unknown"

    return {
        "source": "ENCODE",
        "accession": exp.get("accession", ""),
        "assay": exp.get("assay_title", "DNase-seq"),
        "species": species_short,
        "cell_type": term_name,
        "cell_line": _extract_cell_line_from_label(term_name) or _extract_cell_line_from_text(
            term_name, str(exp.get("biosample_summary", "") or "")
        ),
        "lineage": lineage,
        "state": state,
        "aliases": ", ".join(str(x) for x in aliases if x),
        "synonyms": ", ".join(str(x) for x in synonyms if x),
        "biosample_summary": str(exp.get("biosample_summary", "") or "").strip(),
        "perturbation": perturbation,
        "description": str(exp.get("biosample_summary", "") or "").strip(),
        "genome_build": genome,
        "genome_build_match": genome_build_match,
        "n_replicates": len(replicates),
        "files": [str(f.get("accession") or "").strip() for f in peak_files if str(f.get("accession") or "").strip()],
        "peak_file_accessions": [str(f.get("accession") or "").strip() for f in peak_files if str(f.get("accession") or "").strip()],
        "signal_file_accessions": [str(f.get("accession") or "").strip() for f in signal_files if str(f.get("accession") or "").strip()],
        "has_peak_file": bool(peak_files),
        "has_signal_file": has_signal_file,
        # Unknown until local peak BED is parsed downstream.
        "promoter_coverage_of_targets": None,
        "frip": None,
        "tss_enrichment": None,
        "status": exp.get("status", ""),
        "qc_flags": {},
    }


def _pick_geo_peak_file_url(urls: list[str]) -> str | None:
    """Pick a supplementary file URL usable as peaks (BED/narrowPeak or peak×cell counts).

    Prefer classical peak calls. Fall back to scATAC peak×cell matrices
    (``*fragment_counts*.tsv.gz``, ``*peak*counts*``) which can be converted to BED.
    Skip BigWig/BAM and RNA count matrices.
    """
    scored: list[tuple[int, str]] = []
    for raw in urls:
        u = str(raw or "").strip()
        if not u:
            continue
        low = u.lower()
        name = low.rsplit("/", 1)[-1]
        if any(bad in low for bad in ("bigwig", ".bw", ".bam", ".bai", "fpkm", "tpm")):
            continue
        # Raw 10x fragments are not a peak universe by themselves.
        if "fragments.tsv" in name and "fragment_counts" not in name and "peak" not in name:
            continue
        score = 0
        if "narrowpeak" in low:
            score += 25
        elif "broadpeak" in low:
            score += 20
        elif ".bed" in low:
            score += 18
        elif "fragment_counts" in name or "peak_counts" in name or "peakcount" in name:
            # Peak×cell count matrix — convertible to BED intervals.
            score += 14
        elif ("peaks" in low or "_peak" in low) and any(
            ext in low for ext in (".tsv", ".txt", ".csv", ".bed", ".gz")
        ):
            score += 12
        else:
            continue
        if low.endswith(".gz") or low.endswith(".bgz"):
            score += 3
        scored.append((score, u))
    if not scored:
        return None
    scored.sort(key=lambda x: (-x[0], -len(x[1])))
    return scored[0][1]


def _geo_download_cgi_url(url: str) -> str:
    """Map a GEO sample supplementary FTP path to the ``geo/download`` CGI form."""
    from urllib.parse import quote, urlparse

    parsed = urlparse(str(url or ""))
    if not parsed.netloc.lower().endswith("ncbi.nlm.nih.gov"):
        return ""
    m = re.search(r"/geo/samples/[^/]+/(GSM\d+)/suppl/(.+)$", parsed.path, flags=re.IGNORECASE)
    if not m:
        return ""
    return (
        "https://www.ncbi.nlm.nih.gov/geo/download/"
        f"?acc={m.group(1).upper()}&format=file&file={quote(m.group(2), safe='')}"
    )


def _download_geo_supplementary(client: GEOClient, url: str, dest: Path) -> None:
    """Download a GEO supplementary file, falling back to the download CGI.

    The FTP mirror intermittently answers 403 for sample supplementary files
    that the CGI endpoint still serves.
    """
    try:
        client.download_supplementary_file(url, dest)
        return
    except Exception as exc:
        cgi_url = _geo_download_cgi_url(url)
        if not cgi_url:
            raise
        logger.warning("GEO FTP download failed (%s); retrying via download endpoint", exc)
    client.download_supplementary_file(cgi_url, dest)


def _geo_file_is_peak_count_matrix(path_or_url: str) -> bool:
    name = str(path_or_url).lower().rsplit("/", 1)[-1]
    return any(
        tok in name
        for tok in ("fragment_counts", "peak_counts", "peakcount", "peak_matrix", "peaks_matrix")
    )


def _convert_peak_count_matrix_to_bed(matrix_path: Path, bed_out: Path) -> int:
    """Write unique chrom/start/end intervals from a peak×cell count matrix index."""
    import gzip
    import re

    opener = gzip.open if str(matrix_path).endswith(".gz") else open
    seen: set[tuple[str, int, int]] = set()
    rows: list[tuple[str, int, int]] = []
    with opener(matrix_path, "rt", encoding="utf-8", errors="replace") as fh:
        header = fh.readline()  # noqa: F841 — skip barcode header
        for line in fh:
            peak_id = line.split("\t", 1)[0].strip().strip('"')
            if not peak_id:
                continue
            chrom = start = end = None
            if ":" in peak_id and "-" in peak_id:
                chrom, rest = peak_id.split(":", 1)
                a, b = rest.split("-", 1)
                start, end = int(a), int(b)
            elif peak_id.count("-") >= 2:
                chrom, a, b = peak_id.split("-", 2)
                if re.fullmatch(r"\d+", a) and re.fullmatch(r"\d+", b):
                    start, end = int(a), int(b)
            if chrom is None or start is None or end is None:
                continue
            if end < start:
                start, end = end, start
            key = (chrom, start, end)
            if key in seen:
                continue
            seen.add(key)
            rows.append(key)
    rows.sort(key=lambda x: (x[0], x[1], x[2]))
    bed_out.parent.mkdir(parents=True, exist_ok=True)
    bed_out.write_text("".join(f"{c}\t{s}\t{e}\n" for c, s, e in rows), encoding="utf-8")
    return len(rows)


def _geo_sample_text(sample_meta: dict) -> str:
    parts = [
        sample_meta.get("title", ""),
        sample_meta.get("source_name", ""),
        sample_meta.get("organism", ""),
        sample_meta.get("library_strategy", ""),
        sample_meta.get("library_source", ""),
        sample_meta.get("library_selection", ""),
        sample_meta.get("extract_protocol", ""),
        sample_meta.get("data_processing", ""),
        sample_meta.get("characteristics_text", ""),
        sample_meta.get("parent_series_title", ""),
        " ".join(str(u) for u in sample_meta.get("supplementary_files", []) or []),
    ]
    return " ".join(str(p or "") for p in parts)


def _geo_sample_characteristic(sample_meta: dict, *names: str) -> str:
    chars = sample_meta.get("characteristics", {})
    if not isinstance(chars, dict):
        return ""
    for name in names:
        vals = chars.get(name)
        if isinstance(vals, list) and vals:
            return str(vals[0]).strip()
        if isinstance(vals, str) and vals.strip():
            return vals.strip()
    return ""


def _extract_geo_sample_accessibility_meta(
    sample_meta: dict,
    *,
    genome: str,
    requested_species: str,
) -> dict:
    """Build acquisition metadata from a GEO sample (GSM...) record."""
    sample_text = _geo_sample_text(sample_meta)
    text_blob = sample_text.lower()
    library_strategy = str(sample_meta.get("library_strategy", "") or "").strip().lower()
    assay = "ATAC-seq" if "atac" in text_blob else ("DNase-seq" if "dnase" in text_blob else "accessibility")
    if library_strategy in {"rna-seq", "rna seq", "scrna-seq", "scrna seq", "single-cell rna-seq"}:
        assay = "RNA-seq"
    organism = sample_meta.get("organism") or requested_species
    species_norm = canonical_species_label(organism) or canonical_species_label(requested_species) or "unknown"
    cell_type_meta = (
        _geo_sample_characteristic(sample_meta, "cell_type", "cell line", "cell_line")
        or _geo_sample_characteristic(sample_meta, "tissue", "organ", "source_name")
        or str(sample_meta.get("source_name", "")).strip()
        or "unknown"
    )
    inferred_line = _extract_cell_line_from_label(cell_type_meta) or _extract_cell_line_from_text(cell_type_meta, sample_text)
    supp_files = list(sample_meta.get("supplementary_files", []) or [])
    has_peak_file = bool(_pick_geo_peak_file_url(supp_files))
    has_signal_file = any(
        any(token in str(url).lower() for token in ("bigwig", ".bw", "signal", "coverage"))
        for url in supp_files
    )
    parent_series = (
        str(sample_meta.get("parent_series_accession", "")).strip()
        or next((str(x).strip() for x in sample_meta.get("series_accessions", []) or [] if str(x).strip()), "")
    )

    return {
        "source": "GEO",
        "accession": str(sample_meta.get("accession", "")).strip(),
        "sample_accession": str(sample_meta.get("accession", "")).strip(),
        "parent_series_accession": parent_series,
        "matched_rna_accession": str(sample_meta.get("matched_rna_accession", "")).strip(),
        "pairing_quality": str(sample_meta.get("pairing_quality", "")).strip(),
        "assay": assay,
        "species": species_norm,
        "cell_type": cell_type_meta,
        "cell_line": inferred_line,
        "lineage": _geo_sample_characteristic(sample_meta, "tissue", "organ", "developmental_stage"),
        "state": _geo_sample_characteristic(sample_meta, "state", "condition", "developmental_stage"),
        "aliases": "",
        "synonyms": "",
        "biosample_summary": sample_text.strip(),
        "perturbation": _infer_perturbation_label(sample_text) or "unknown",
        "description": sample_text.strip(),
        "genome_build": genome,
        # GEO sample records generally do not expose a reliable assembly field.
        "genome_build_match": "unknown",
        "n_replicates": 1,
        "files": supp_files,
        "has_peak_file": has_peak_file,
        "has_signal_file": has_signal_file,
        "promoter_coverage_of_targets": None,
        "frip": None,
        "tss_enrichment": None,
        "status": "public" if assay != "RNA-seq" and (has_peak_file or has_signal_file) else "no_usable_files",
        "qc_flags": {},
    }


def _geo_sample_sort_key(sample_meta: dict) -> tuple:
    files = list(sample_meta.get("supplementary_files", []) or [])
    text = _geo_sample_text(sample_meta).lower()
    species_rank = 0 if sample_meta.get("_requested_species_match") is True else 1
    has_peak = 1 if _pick_geo_peak_file_url(files) else 0
    is_atac = 1 if "atac" in text else 0
    is_dnase = 1 if "dnase" in text else 0
    pairing_score = float(sample_meta.get("pairing_score", 0.0) or 0.0)
    return (species_rank, -has_peak, -is_atac, -is_dnase, -pairing_score, str(sample_meta.get("accession", "")))


def _extract_best_geo_accessibility_meta_from_series(
    geo: GEOClient,
    series_meta: dict,
    *,
    genome: str,
    requested_species: str,
    requested_cell_type: str | None,
    requested_cell_line: str | None,
    requested_lineage: str | None,
    requested_state: str | None,
) -> dict:
    sample_candidates: list[dict] = []
    requested_species_norm = canonical_species_label(requested_species)
    for raw_sample in series_meta.get("sample_metadata", []) or []:
        if not isinstance(raw_sample, dict):
            continue
        sample = dict(raw_sample)
        acc = str(sample.get("accession", "")).strip()
        if acc and not sample.get("supplementary_files"):
            try:
                full_sample = geo.get_sample_metadata(acc)
                merged = dict(sample)
                merged.update({k: v for k, v in full_sample.items() if v not in (None, "", [])})
                sample = merged
            except Exception:
                pass
        sample.setdefault("parent_series_accession", series_meta.get("accession", ""))
        sample.setdefault("parent_series_title", series_meta.get("title", ""))
        if is_likely_accessibility_sample(sample):
            sample_species_norm = canonical_species_label(sample.get("organism"))
            if requested_species_norm and sample_species_norm and sample_species_norm != requested_species_norm:
                continue
            sample["_requested_species_match"] = bool(
                requested_species_norm and sample_species_norm and sample_species_norm == requested_species_norm
            )
            sample_candidates.append(sample)

    if sample_candidates:
        sample_candidates.sort(key=_geo_sample_sort_key)
        return _extract_geo_sample_accessibility_meta(
            sample_candidates[0],
            genome=genome,
            requested_species=requested_species,
        )

    return _extract_geo_accessibility_meta(
        series_meta,
        genome=genome,
        requested_species=requested_species,
        requested_cell_type=requested_cell_type,
        requested_cell_line=requested_cell_line,
        requested_lineage=requested_lineage,
        requested_state=requested_state,
    )


def _extract_geo_accessibility_meta(
    series_meta: dict,
    *,
    genome: str,
    requested_species: str,
    requested_cell_type: str | None,
    requested_cell_line: str | None,
    requested_lineage: str | None,
    requested_state: str | None,
) -> dict:
    """Build acquisition metadata record from a GEO series (GSE...) response."""
    title = str(series_meta.get("title", "")).strip()
    summary = str(series_meta.get("summary", "")).strip()
    experiment_type = str(series_meta.get("experiment_type", "")).strip()
    text_blob = f"{title} {summary} {experiment_type}".lower()
    assay = "ATAC-seq" if "atac" in text_blob else ("DNase-seq" if "dnase" in text_blob else "accessibility")
    species_norm = canonical_species_label(series_meta.get("organism")) or canonical_species_label(requested_species) or "unknown"
    cell_type_meta = str(series_meta.get("cell_type", "")).strip() or "unknown"
    inferred_line = _extract_cell_line_from_label(cell_type_meta) or _extract_cell_line_from_text(cell_type_meta, title, summary)
    # IMPORTANT: do NOT backfill missing GEO cell-line from the user request.
    # That would make ranking and QC circular (a missing line would always "match").
    cell_line_meta = inferred_line
    supp_files = list(series_meta.get("supplementary_files", []) or [])
    has_peak_file = bool(_pick_geo_peak_file_url(supp_files))
    has_signal_file = any(
        any(token in str(url).lower() for token in ("bigwig", ".bw", "signal", "coverage"))
        for url in supp_files
    )
    # GEO does not expose an ENCODE-style released/in-progress status in the
    # metadata used here. Public retrievable records are marked by availability.
    geo_status = "public" if has_peak_file or has_signal_file else "no_usable_files"

    return {
        "source": "GEO",
        "accession": str(series_meta.get("accession", "")).strip(),
        "assay": assay,
        "species": species_norm,
        "cell_type": cell_type_meta,
        "cell_line": cell_line_meta,
        # GEO series metadata often lacks lineage/state; keep empty so scoring/QC uses
        # real evidence instead of copying the query into candidate metadata.
        "lineage": "",
        "state": "",
        # Populate fields used by _score_accessibility_candidate context_blob so
        # semantic scoring is not silently blinded for GEO entries.
        "aliases": "",
        "synonyms": "",
        "biosample_summary": f"{title} {summary}".strip(),
        "perturbation": _infer_perturbation_label(title, summary) or "unknown",
        "description": f"{title}\n{summary}".strip(),
        "genome_build": genome,
        # GEO entries have no reliable assembly metadata; mark as unknown.
        "genome_build_match": "unknown",
        "n_replicates": max(1, int(len(series_meta.get("samples", []) or []))),
        "files": supp_files,
        "has_peak_file": has_peak_file,
        "has_signal_file": has_signal_file,
        # Unknown until local peak BED is parsed downstream.
        "promoter_coverage_of_targets": None,
        "frip": None,
        "tss_enrichment": None,
        "status": geo_status,
        "qc_flags": {},
    }


if __name__ == "__main__":
    main()
