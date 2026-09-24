"""
Standard-field motif integration pipeline.

Protocol (same as BEELINE / ENCODE / GTRD community practice)
-------------------------------------------------------------
1.  Download JASPAR CORE MEME file  (once, cached; default release 2026, fallback 2024)
2.  Filter MEME file to expressed TFs    (faster FIMO)
3.  Extract promoter sequences           (bedtools getfasta)
4.  Scan with FIMO                       (MEME suite, threshold p ≤ 1e-4)
5.  Parse FIMO TSV output
6.  Map peaks → genes via TSS ±2 kb     (gene_coords module)
7.  Aggregate per (TF, gene) pair:
        motif_present   bool   ≥1 hit in any promoter peak
        max_score_pct   float  best FIMO score / PWM max score  (0–1)
        peak_count      int    # promoter peaks with a hit

Output TSV columns:
    source_tf, target_gene, motif_id,
    motif_present, max_score_pct, peak_count

Dependencies (must be on PATH)
-------------------------------
    bedtools   ≥2.30   https://bedtools.readthedocs.io
    fimo       ≥5.5    https://meme-suite.org/meme/tools/fimo
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import pandas as pd

from grn_agent.acquisition.gene_coords import build_peak_to_gene_map, load_gene_coords
from grn_agent.acquisition.jaspar_client import (
    download_jaspar_meme_file,
    filter_meme_for_tfs,
    parse_meme_tf_map,
)

logger = logging.getLogger(__name__)

# Promoter scans over a full peak union (~30k peaks) exceed a 1h single-process
# budget, so parallel chunking is the default rather than an opt-in.
DEFAULT_FIMO_WORKERS = 8
DEFAULT_FIMO_TIMEOUT = 3600


def _normalize_species_for_jaspar(species_or_genome: str) -> str:
    """Map genome-build aliases to species labels accepted by JASPAR client."""
    s = str(species_or_genome or "").strip().lower()
    if s in {"mouse", "mm10", "mm39"}:
        return "mouse"
    if s in {"human", "hg38", "hg19"}:
        return "human"
    if s in {"rat", "rn6", "rn7"}:
        return "rat"
    if s in {"zebrafish", "danrer10", "danrer11"}:
        return "zebrafish"
    if s in {"drosophila", "dm6"}:
        return "drosophila"
    if s in {"arabidopsis", "tair10"}:
        return "arabidopsis"
    return s


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_motif_integration(
    peaks_bed: str | Path,
    tf_names: list[str],
    species_or_genome: str,
    genome_fasta: str | Path | None = None,
    output_tsv: str | Path = "motif_hits.tsv",
    *,
    gtf_path: str | Path | None = None,
    promoter_window: int = 2000,
    fimo_pvalue: float = 1e-4,
    jaspar_cache_dir: str | Path | None = None,
    jaspar_release: int | None = None,
    gene_cache_dir: str | Path | None = None,
    pair_filter: set[tuple[str, str]] | None = None,
    keep_tmp: bool = False,
    auto_download_genome: bool = True,
    fimo_workers: int | None = None,
    fimo_timeout: int = DEFAULT_FIMO_TIMEOUT,
) -> pd.DataFrame:
    """
    Standard motif integration pipeline (bedtools + FIMO + JASPAR).

    Args:
        peaks_bed:         BED file of DNase/ATAC peaks (≥3 columns: chr, start, end).
        tf_names:          Expressed TF gene symbols to scan.
        species_or_genome: "mouse" | "human" | "mm10" | "hg38" | etc.
        genome_fasta:      Genome FASTA (.fa, must be indexed with .fai).
                           If None and auto_download_genome=True, the genome is
                           fetched automatically from Ensembl via GenomeDB.
        output_tsv:        Path to write the feature table TSV.
        gtf_path:          GTF for gene TSS coordinates; GenomeDB/BioMart used if omitted.
        promoter_window:   bp around TSS for peak→gene assignment (default 2000).
        fimo_pvalue:       FIMO p-value threshold (default 1e-4).
        jaspar_cache_dir:  Cache for JASPAR MEME file and filtered copies.
        jaspar_release:    JASPAR year (e.g. 2026). None = try 2026 then 2024.
        gene_cache_dir:    Cache for gene TSS table (BioMart / GTF results).
        pair_filter:       Optional {(tf, gene)} whitelist — only these pairs
                           are written to the output (reduces table size).
        keep_tmp:          Keep temporary files (useful for debugging).
        auto_download_genome: Download genome from Ensembl if fasta not provided (default True).
        fimo_workers:      Parallel FIMO chunks; None picks a CPU-bounded default.
        fimo_timeout:      Per-chunk FIMO timeout in seconds.

    Returns:
        DataFrame with motif feature table.

    Raises:
        RuntimeError: if bedtools or fimo are not found on PATH.
    """
    peaks_bed = Path(peaks_bed)

    # ── Auto-resolve genome FASTA + GTF via GenomeDB ─────────────────────────
    if genome_fasta is None:
        if auto_download_genome:
            try:
                from grn_agent.acquisition.genome_db import ensure_genome
                logger.info(
                    "No genome FASTA provided — fetching '%s' from GenomeDB (Ensembl FTP)…",
                    species_or_genome,
                )
                _fasta, _gtf = ensure_genome(species_or_genome)
                genome_fasta = _fasta
                if gtf_path is None:
                    gtf_path = _gtf
                    logger.info("Using GenomeDB GTF: %s", gtf_path)
            except (ValueError, RuntimeError) as exc:
                raise RuntimeError(
                    f"Genome FASTA not provided and auto-download failed for '{species_or_genome}': {exc}\n"
                    "Either provide --genome-fasta <path> or ensure the species is in GenomeDB."
                ) from exc
        else:
            raise RuntimeError(
                "genome_fasta is required when auto_download_genome=False."
            )

    genome_fasta = Path(genome_fasta)
    output_tsv = Path(output_tsv)
    output_tsv.parent.mkdir(parents=True, exist_ok=True)

    _require_tools()

    jcache = Path(jaspar_cache_dir) if jaspar_cache_dir else Path(".cache") / "jaspar"
    jcache.mkdir(parents=True, exist_ok=True)

    if keep_tmp:
        _tmp = tempfile.mkdtemp(prefix="motif_scan_")
        tmp_ctx = nullcontext(_tmp)
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="motif_scan_")

    with tmp_ctx as _tmp:
        tmp = Path(_tmp)

        # ── Step 1: JASPAR MEME file ─────────────────────────────────────
        logger.info("Step 1 — Fetching JASPAR MEME file …")
        jaspar_species = _normalize_species_for_jaspar(species_or_genome)
        meme_full = download_jaspar_meme_file(
            jaspar_species, cache_dir=jcache, jaspar_release=jaspar_release
        )

        # ── Step 2: Filter to expressed TFs ──────────────────────────────
        logger.info("Step 2 — Filtering MEME to %d expressed TFs …", len(tf_names))
        meme_filtered = jcache / f"filtered_{'_'.join(sorted(tf_names)[:5])}_etc.meme"
        meme_filtered = filter_meme_for_tfs(meme_full, tf_names, meme_filtered)

        # Build motif_id → tf_name lookup
        motif_to_tf = parse_meme_tf_map(meme_filtered)
        if not motif_to_tf:
            logger.warning("No matching motifs found in JASPAR for %d TFs — returning empty table", len(tf_names))
            return _empty_df()

        # ── Step 3: Gene TSS coordinates ─────────────────────────────────
        logger.info("Step 3 — Loading gene TSS coordinates …")
        gene_coords = load_gene_coords(
            species_or_genome,
            gtf_path=gtf_path,
            cache_dir=gene_cache_dir,
        )
        if gene_coords.empty:
            logger.warning("No gene TSS coordinates available — cannot map peaks to genes")
            return _empty_df()

        # ── Step 4: Build promoter BED ────────────────────────────────────
        logger.info("Step 4 — Building promoter peak BED …")
        peaks = _load_peaks(peaks_bed)
        peak_to_genes = build_peak_to_gene_map(peaks, gene_coords, window=promoter_window)
        promoter_peaks = _peaks_with_genes(peaks, peak_to_genes)

        if promoter_peaks.empty:
            logger.warning("No peaks overlap any gene promoter — returning empty table")
            return _empty_df()

        logger.info("  %d / %d peaks overlapping promoters", len(promoter_peaks), len(peaks))

        promoter_peaks = _normalize_peak_chromosomes_to_fasta(promoter_peaks, genome_fasta)
        promoter_bed = tmp / "promoter_peaks.bed"
        _write_named_bed(promoter_peaks, promoter_bed)

        # ── Step 5: bedtools getfasta ─────────────────────────────────────
        logger.info("Step 5 — Extracting sequences with bedtools getfasta …")
        peaks_fasta = tmp / "promoter_peaks.fa"
        _bedtools_getfasta(promoter_bed, genome_fasta, peaks_fasta)

        # ── Step 6: FIMO ──────────────────────────────────────────────────
        n_workers = resolve_fimo_workers(fimo_workers)
        logger.info(
            "Step 6 — Running FIMO (p ≤ %.0e, workers=%d, timeout=%ds) …",
            fimo_pvalue, n_workers, fimo_timeout,
        )
        fimo_out = tmp / "fimo_out"
        _run_fimo(
            meme_filtered,
            peaks_fasta,
            fimo_out,
            fimo_pvalue,
            workers=n_workers,
            timeout=fimo_timeout,
        )

        # ── Step 7: Parse & aggregate ─────────────────────────────────────
        logger.info("Step 7 — Parsing FIMO output and aggregating …")
        fimo_df = _parse_fimo(fimo_out / "fimo.tsv")
        if fimo_df.empty:
            logger.info("FIMO found no significant hits at p ≤ %.0e", fimo_pvalue)
            return _empty_df()

        feature_df = _aggregate(fimo_df, motif_to_tf, peak_to_genes, pair_filter)

    feature_df.to_csv(output_tsv, sep="\t", index=False)
    n_present = int(feature_df["motif_present"].sum())
    logger.info(
        "Motif hits written: %d TF-gene pairs (%d with motif present) → %s",
        len(feature_df), n_present, output_tsv,
    )
    return feature_df


# ─────────────────────────────────────────────────────────────────────────────
# External tool wrappers
# ─────────────────────────────────────────────────────────────────────────────

def _require_tools() -> None:
    """Raise a clear error if bedtools or fimo are not on PATH."""
    missing: list[str] = []
    for tool in ("bedtools", "fimo"):
        try:
            subprocess.run([tool, "--version"], capture_output=True, check=True, timeout=10)
        except FileNotFoundError:
            missing.append(tool)
        except subprocess.CalledProcessError:
            pass  # tool exists but returned non-zero (e.g. fimo --version returns 1 on some builds)
    if missing:
        raise RuntimeError(
            f"Required tools not found on PATH: {missing}.\n"
            "Install with:\n"
            "  bedtools: conda install -c bioconda bedtools\n"
            "  fimo:     conda install -c bioconda meme"
        )


def _bedtools_getfasta(
    bed: Path,
    fasta: Path,
    out_fasta: Path,
) -> None:
    """
    Run ``bedtools getfasta -nameOnly`` to extract sequences using only the
    peak name column as the FASTA header — so FIMO ``sequence_name`` can be
    mapped back to peak IDs without coordinate re-parsing.

    Requires the genome FASTA to be indexed (samtools faidx genome.fa).
    """
    cmd = [
        "bedtools", "getfasta",
        "-fi", str(fasta),
        "-bed", str(bed),
        "-fo", str(out_fasta),
        "-nameOnly",      # use BED name only (no "::chr:start-end" suffix)
        "-s",             # force strandedness (required for TSS orientation)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"bedtools getfasta timed out after {exc.timeout} seconds") from exc
    if result.returncode != 0:
        raise RuntimeError(f"bedtools getfasta failed:\n{result.stderr}")
    if not out_fasta.is_file() or out_fasta.stat().st_size == 0:
        raise RuntimeError("bedtools getfasta produced no output")
    logger.debug("bedtools getfasta → %d bytes", out_fasta.stat().st_size)


def resolve_fimo_workers(requested: int | None) -> int:
    """Clamp the requested FIMO worker count to at least 1 CPU-bounded process."""
    cpus = os.cpu_count() or 1
    if requested is None or int(requested) <= 0:
        return max(1, min(DEFAULT_FIMO_WORKERS, cpus))
    return max(1, min(int(requested), cpus))


def _split_fasta(fasta: Path, n_parts: int, dest_dir: Path) -> list[Path]:
    """Round-robin FASTA records into ``n_parts`` files, returning non-empty ones.

    Round-robin (rather than contiguous blocks) keeps per-chunk runtime even
    when promoter sequences vary a lot in length.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    parts = [dest_dir / f"part_{i:03d}.fa" for i in range(n_parts)]
    handles = [p.open("w", encoding="utf-8") for p in parts]
    counts = [0] * n_parts
    try:
        idx = -1
        current = None
        with fasta.open(encoding="utf-8") as fh:
            for line in fh:
                if line.startswith(">"):
                    idx += 1
                    current = idx % n_parts
                    counts[current] += 1
                if current is None:
                    continue
                handles[current].write(line)
    finally:
        for h in handles:
            h.close()
    return [p for p, c in zip(parts, counts, strict=True) if c > 0]


def _merge_fimo_tsvs(part_tsvs: list[Path], merged: Path) -> None:
    """Concatenate FIMO TSVs, keeping a single header and dropping comments."""
    merged.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = False
    with merged.open("w", encoding="utf-8") as out:
        for tsv in part_tsvs:
            if not tsv.is_file():
                continue
            with tsv.open(encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if not line.strip() or line.startswith("#"):
                        continue
                    if i == 0 or line.startswith("motif_id"):
                        if not wrote_header:
                            out.write(line)
                            wrote_header = True
                        continue
                    out.write(line)


def _fimo_cmd(meme_file: Path, fasta: Path, out_dir: Path, pvalue: float) -> list[str]:
    return [
        "fimo",
        "--oc", str(out_dir),
        "--thresh", str(pvalue),
        "--no-qvalue",
        str(meme_file),
        str(fasta),
    ]


def _run_fimo(
    meme_file: Path,
    fasta: Path,
    out_dir: Path,
    pvalue: float,
    *,
    workers: int = 1,
    timeout: int = DEFAULT_FIMO_TIMEOUT,
) -> None:
    """
    Run FIMO with standard parameters used in the field.

    Key flags:
        --thresh      p-value threshold (1e-4 is standard for promoter scans)
        --no-qvalue   skip q-value computation (not needed for single-gene analysis)

    NOTE:
        We intentionally do NOT use ``--parse-genomic-coord`` here because it
        rewrites FASTA headers into (chrom, start, stop) fields and changes
        ``sequence_name`` to bare chromosome labels (e.g. ``chr1``). Downstream
        aggregation maps FIMO hits by peak IDs from BED names, so preserving
        the original header is required.

    With ``workers > 1`` the FASTA is split by record and scanned in parallel,
    then the per-chunk TSVs are concatenated. This is exact rather than an
    approximation: ``--no-qvalue`` means no statistic is pooled across
    sequences, so per-sequence hits do not depend on how records are batched.
    ``timeout`` applies per chunk.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if workers <= 1:
        try:
            result = subprocess.run(
                _fimo_cmd(meme_file, fasta, out_dir, pvalue),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"FIMO timed out after {exc.timeout} seconds "
                f"(raise fimo_timeout or fimo_workers)"
            ) from exc
        if result.returncode != 0:
            raise RuntimeError(f"FIMO failed (exit {result.returncode}):\n{result.stderr[:500]}")
        logger.debug("FIMO stdout: %s", result.stdout[:200])
        return

    chunk_dir = out_dir / "chunks"
    chunks = _split_fasta(fasta, workers, chunk_dir)
    if not chunks:
        raise RuntimeError(f"No FASTA records to scan in {fasta}")
    logger.info("  FIMO parallel scan: %d chunks x %d workers", len(chunks), workers)

    def _scan(chunk: Path) -> Path:
        chunk_out = chunk_dir / f"{chunk.stem}_out"
        try:
            res = subprocess.run(
                _fimo_cmd(meme_file, chunk, chunk_out, pvalue),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"FIMO chunk {chunk.name} timed out after {exc.timeout} seconds "
                f"(raise fimo_timeout or fimo_workers)"
            ) from exc
        if res.returncode != 0:
            raise RuntimeError(
                f"FIMO chunk {chunk.name} failed (exit {res.returncode}):\n{res.stderr[:500]}"
            )
        return chunk_out / "fimo.tsv"

    with ThreadPoolExecutor(max_workers=workers) as pool:
        part_tsvs = list(pool.map(_scan, chunks))

    _merge_fimo_tsvs(part_tsvs, out_dir / "fimo.tsv")


# ─────────────────────────────────────────────────────────────────────────────
# Parsing & aggregation
# ─────────────────────────────────────────────────────────────────────────────

def _parse_fimo(fimo_tsv: Path) -> pd.DataFrame:
    """
    Parse FIMO output TSV.

    FIMO columns (≥5.5):
        motif_id  motif_alt_id  sequence_name  start  stop
        strand    score         p-value        q-value  matched_sequence

    Older versions may omit q-value or matched_sequence.

    Important: do **not** pass ``comment='#'`` to ``read_csv``. Peak IDs may
    contain ``#`` (e.g. ``peak#12`` after duplicate-name uniquification); pandas
    would then truncate ``sequence_name`` at ``#`` and null out remaining fields,
    yielding an empty table even when FIMO found hits.
    """
    if not fimo_tsv.is_file():
        return pd.DataFrame()

    # Skip full-line comments only (lines beginning with '#').
    rows: list[str] = []
    with fimo_tsv.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            rows.append(line)
    if not rows:
        return pd.DataFrame()
    from io import StringIO

    df = pd.read_csv(StringIO("".join(rows)), sep="\t")
    if df.empty:
        return df

    # Normalise column names (versions differ)
    df.columns = [c.strip().replace("-", "_").lower() for c in df.columns]
    required = {"motif_id", "sequence_name", "score", "p_value"}
    if not required.issubset(set(df.columns)):
        logger.warning("Unexpected FIMO columns: %s", list(df.columns))
        return pd.DataFrame()

    df = df.dropna(subset=["motif_id", "sequence_name", "score"])
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)
    df["p_value"] = pd.to_numeric(df["p_value"], errors="coerce")
    return df


def _aggregate(
    fimo_df: pd.DataFrame,
    motif_to_tf: dict[str, str],
    peak_to_genes: dict[str, list[str]],
    pair_filter: set[tuple[str, str]] | None,
) -> pd.DataFrame:
    """
    Aggregate FIMO hits into a per-(TF, gene) feature table.

    Score normalisation: FIMO scores are in bits.  We normalise by the
    maximum observed score for that motif across all hits, giving a
    0–1 ``max_score_pct`` that is comparable across motifs.
    """
    # Per-motif max score for normalisation
    motif_max: dict[str, float] = (
        fimo_df.groupby("motif_id")["score"].max().to_dict()
    )

    # (tf, gene) → {max_score_pct, peak_set}
    pair_best: dict[tuple[str, str], float] = {}
    pair_peaks: dict[tuple[str, str], set[str]] = {}
    pair_motif: dict[tuple[str, str], str] = {}
    pair_filter_norm: set[tuple[str, str]] | None = None
    if pair_filter:
        pair_filter_norm = {(str(tf).upper(), str(g).upper()) for tf, g in pair_filter}

    for _, row in fimo_df.iterrows():
        motif_id = str(row["motif_id"])
        tf = motif_to_tf.get(motif_id, "")
        if not tf:
            continue

        # sequence_name from bedtools -name is the BED name column
        # Strip strand suffix added by bedtools: "peakX(+)" → "peakX"
        raw_seq_name = str(row["sequence_name"])
        peak_id = raw_seq_name.split("(")[0]

        score = float(row["score"])
        max_s = motif_max.get(motif_id, 1.0) or 1.0
        score_pct = score / max_s

        genes = peak_to_genes.get(peak_id, [])
        # Composite JASPAR names (e.g. "POU5F1::SOX2") should be allowed to
        # match per-TF pair filters.
        tf_tokens = [tf]
        if "::" in tf or "+" in tf or "-" in tf:
            tokenized = tf.replace("::", " ").replace("+", " ").replace("-", " ").split()
            tf_tokens.extend(tokenized)
        tf_tokens = [t for t in dict.fromkeys(tf_tokens) if t]

        for gene in genes:
            if pair_filter_norm is None:
                candidate_tfs = [tf]
            else:
                candidate_tfs = [
                    t for t in tf_tokens if (t.upper(), str(gene).upper()) in pair_filter_norm
                ]
            for tf_out in candidate_tfs:
                k = (tf_out, gene)
                if k not in pair_best or score_pct > pair_best[k]:
                    pair_best[k] = score_pct
                    pair_motif[k] = motif_id
                pair_peaks.setdefault(k, set()).add(peak_id)

    rows = [
        {
            "source_tf": tf,
            "target_gene": gene,
            "motif_id": pair_motif[(tf, gene)],
            "motif_present": True,
            "max_score_pct": round(pair_best[(tf, gene)], 4),
            "peak_count": len(pair_peaks[(tf, gene)]),
        }
        for (tf, gene) in pair_best
    ]

    if not rows:
        return _empty_df()

    df = pd.DataFrame(rows)
    df["motif_present"] = df["motif_present"].astype(bool)
    return df.sort_values(["source_tf", "target_gene"]).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# BED helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_peaks(bed_path: Path) -> pd.DataFrame:
    df = pd.read_csv(bed_path, sep="\t", header=None, comment="#")
    cols = ["chr", "start", "end"]
    if df.shape[1] >= 4:
        cols.append("name")
    df = df.iloc[:, : len(cols)].copy()
    df.columns = cols
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    # Use a non-genomic token to prevent FIMO from interpreting IDs as
    # genomic coordinates (which would collapse sequence_name to chromosome).
    coord_id = "peak_" + df.index.astype(str)
    if "name" not in df.columns:
        df["name"] = coord_id
    else:
        # ENCODE narrowPeak BEDs often use "." for all names; those collide and
        # break peak→gene maps. Replace placeholders with coordinate IDs.
        # Also treat generic "peak" (common when writing scored BED4/5) as a
        # placeholder so we never uniquify to "peak#N" — the "#" breaks FIMO
        # parsers that treat "#" as a comment character.
        name_str = df["name"].astype(str).str.strip()
        bad_name = name_str.isin({"", ".", "nan", "NA", "N/A", "null", "None", "peak"})
        df.loc[bad_name, "name"] = coord_id[bad_name]

    # Ensure name uniqueness so each peak can be mapped independently.
    # Use "_" (not "#") so IDs remain safe for comment-aware TSV parsers.
    dup_mask = df["name"].duplicated(keep=False)
    if dup_mask.any():
        dup_idx = df.groupby("name").cumcount().astype(str)
        df.loc[dup_mask, "name"] = df.loc[dup_mask, "name"] + "_" + dup_idx[dup_mask]
    return df


def _read_fasta_contigs(fasta_path: Path) -> set[str]:
    contigs: set[str] = set()
    with fasta_path.open(encoding="utf-8", errors="ignore") as fp:
        for line in fp:
            if line.startswith(">"):
                token = line[1:].strip().split(None, 1)[0]
                if token:
                    contigs.add(token)
    return contigs


def _chrom_aliases(chrom: str) -> list[str]:
    c = str(chrom).strip()
    if not c:
        return []
    aliases = [c]
    if c.startswith("chr"):
        bare = c[3:]
        aliases.append(bare)
        if bare == "M":
            aliases.append("MT")
    else:
        aliases.append(f"chr{c}")
        if c == "MT":
            aliases.append("chrM")
    out: list[str] = []
    seen: set[str] = set()
    for item in aliases:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _normalize_peak_chromosomes_to_fasta(peaks: pd.DataFrame, fasta_path: Path) -> pd.DataFrame:
    contigs = _read_fasta_contigs(fasta_path)
    if not contigs or peaks.empty:
        return peaks
    normalized = peaks.copy()
    remapped = 0
    rewritten: list[str] = []
    for idx, chrom in normalized["chr"].astype(str).items():
        target = None
        for alias in _chrom_aliases(chrom):
            if alias in contigs:
                target = alias
                break
        if target is not None and target != chrom:
            normalized.at[idx, "chr"] = target
            remapped += 1
            if len(rewritten) < 5:
                rewritten.append(f"{chrom}->{target}")
    if remapped:
        logger.info(
            "Normalized %d peak chromosomes to match FASTA contigs (%s)",
            remapped,
            ", ".join(rewritten),
        )
    return normalized


def _peaks_with_genes(
    peaks: pd.DataFrame,
    peak_to_genes: dict[str, list[str]],
) -> pd.DataFrame:
    """Return only the peaks that overlap at least one gene promoter."""
    mask = peaks["name"].map(lambda n: bool(peak_to_genes.get(n)))
    return peaks[mask].reset_index(drop=True)


def _write_named_bed(peaks: pd.DataFrame, out_path: Path) -> None:
    """Write a 4-column BED (chr, start, end, name) for bedtools."""
    peaks[["chr", "start", "end", "name"]].to_csv(
        out_path, sep="\t", header=False, index=False
    )


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "source_tf", "target_gene", "motif_id",
            "motif_present", "max_score_pct", "peak_count",
        ]
    )
