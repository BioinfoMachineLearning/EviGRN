"""
Gene TSS coordinate lookup for peak-to-gene mapping.

Priority of data sources (first available wins):
  1. User-supplied GTF file (parsed locally)
  2. Ensembl BioMart REST API
  3. Fallback: empty table (caller must warn)
"""

from __future__ import annotations

import gzip
import io
import logging
import re
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# Ensembl BioMart base URL
_BIOMART_URL = "https://www.ensembl.org/biomart/martservice"

# Map species/genome to BioMart dataset name
_BIOMART_DATASETS: dict[str, str] = {
    "human": "hsapiens_gene_ensembl",
    "hg38": "hsapiens_gene_ensembl",
    "hg19": "hsapiens_gene_ensembl",
    "mouse": "mmusculus_gene_ensembl",
    "mm10": "mmusculus_gene_ensembl",
    "mm39": "mmusculus_gene_ensembl",
    "rat": "rnorvegicus_gene_ensembl",
    "rn6": "rnorvegicus_gene_ensembl",
    "zebrafish": "drerio_gene_ensembl",
}

# Columns we need: chr, start, end, strand, gene_symbol
_GENE_COORD_COLS = ["gene_symbol", "chr", "tss", "strand"]


def load_gene_coords(
    species_or_genome: str,
    gtf_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    gene_symbols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load TSS coordinates for each gene symbol.

    Args:
        species_or_genome: "mouse", "human", "mm10", "hg38", etc.
        gtf_path:          Path to local GTF/GTF.gz (preferred if provided).
        cache_dir:         Directory for caching BioMart results.
        gene_symbols:      Restrict to these gene symbols (speeds up BioMart query).

    Returns:
        DataFrame with columns: gene_symbol, chr, tss (int), strand (+/-)
        One row per gene (TSS position from smallest start on + strand or
        largest end on – strand).
    """
    key = species_or_genome.lower()

    # --- Source 1: local GTF ---
    if gtf_path is not None:
        logger.info("Parsing GTF: %s", gtf_path)
        df = _parse_gtf(Path(gtf_path), gene_symbols)
        if not df.empty:
            return df
        logger.warning("GTF parsing returned empty table; falling back to BioMart")

    # --- Source 2: BioMart ---
    dataset = _BIOMART_DATASETS.get(key)
    if dataset:
        return _fetch_biomart(dataset, gene_symbols, cache_dir)

    # --- Source 3: GenomeDB GTF (auto-download if needed) ---
    try:
        from grn_agent.acquisition.genome_db import get_genome_db
        db = get_genome_db()
        entry = db.get(key)
        if entry and Path(entry.gtf_path).is_file():
            logger.info("Using GenomeDB GTF for '%s': %s", key, entry.gtf_path)
            df = _parse_gtf(Path(entry.gtf_path), gene_symbols)
            if not df.empty:
                return df
    except Exception as _e:
        logger.debug("GenomeDB GTF lookup failed: %s", _e)

    # --- Fallback ---
    logger.warning(
        "No gene coordinate source found for '%s'. "
        "Provide --gtf, ensure species is in %s, "
        "or run ensure_genome('%s') to auto-download.",
        species_or_genome,
        list(_BIOMART_DATASETS.keys()),
        species_or_genome,
    )
    return pd.DataFrame(columns=_GENE_COORD_COLS)


# ---------------------------------------------------------------------------
# GTF parser
# ---------------------------------------------------------------------------

def _normalize_gtf_chrom(chrom_raw: str) -> str:
    """Normalize GTF contig names to match peak BED / GenomeDB FASTA.

    Mammalian Ensembl GTFs use bare ``1``/``X``; our peak beds are UCSC
    ``chr1``/``chrX``, so we add the ``chr`` prefix there.

    Drosophila (dm6) Ensembl already uses ``2L``/``2R``/… matching the FASTA
    and scATAC peak beds — do **not** rewrite those to ``chr2L``.
    """
    c = str(chrom_raw).strip()
    if not c:
        return c
    if c.lower().startswith("chr"):
        return c
    if re.match(r"^(\d+|X|Y|M|MT)$", c, flags=re.IGNORECASE):
        return "chr" + ("M" if c.upper() == "MT" else c)
    return c


def _parse_gtf(gtf_path: Path, gene_symbols: set[str] | list[str] | None) -> pd.DataFrame:
    """
    Parse a GTF/GTF.gz file and extract one TSS per gene symbol.
    Only 'gene' or 'transcript' feature rows are used.
    """
    sym_set = {str(s).upper() for s in gene_symbols} if gene_symbols else None
    records: dict[str, dict] = {}  # symbol → {chr, tss, strand}

    opener = gzip.open if gtf_path.suffix == ".gz" else open
    with opener(gtf_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            feature = parts[2]
            if feature not in ("gene", "transcript"):
                continue

            chrom = _normalize_gtf_chrom(parts[0])
            start = int(parts[3])
            end = int(parts[4])
            strand = parts[6]
            attrs = parts[8]

            symbol = _gtf_attr(attrs, "gene_name") or _gtf_attr(attrs, "gene_id", "")
            if not symbol:
                continue
            if sym_set and symbol.upper() not in sym_set:
                continue

            tss = start if strand == "+" else end

            # Keep only the most canonical TSS (smallest start for +, largest end for -)
            if symbol not in records:
                records[symbol] = {"chr": chrom, "tss": tss, "strand": strand}
            else:
                prev_tss = records[symbol]["tss"]
                if strand == "+" and tss < prev_tss:
                    records[symbol]["tss"] = tss
                elif strand == "-" and tss > prev_tss:
                    records[symbol]["tss"] = tss

    rows = [{"gene_symbol": sym, **vals} for sym, vals in records.items()]
    return pd.DataFrame(rows, columns=_GENE_COORD_COLS) if rows else pd.DataFrame(columns=_GENE_COORD_COLS)


_ATTR_RE = re.compile(r'(\w+)\s+"([^"]+)"')


def _gtf_attr(attrs: str, key: str, default: str | None = None) -> str | None:
    for m in _ATTR_RE.finditer(attrs):
        if m.group(1) == key:
            return m.group(2)
    return default


# ---------------------------------------------------------------------------
# BioMart fetcher
# ---------------------------------------------------------------------------

def _fetch_biomart(
    dataset: str,
    gene_symbols: list[str] | None,
    cache_dir: str | Path | None,
) -> pd.DataFrame:
    """
    Query Ensembl BioMart for TSS coordinates.
    """
    cache_path: Path | None = None
    if cache_dir:
        cache_path = Path(cache_dir) / f"biomart_{dataset}.tsv"
        if cache_path.is_file():
            logger.info("Using cached BioMart TSS table: %s", cache_path)
            return _post_process_biomart(pd.read_csv(cache_path, sep="\t"))

    logger.info("Querying BioMart dataset: %s", dataset)
    xml_query = _build_biomart_xml(dataset, gene_symbols)
    try:
        resp = requests.post(_BIOMART_URL, data={"query": xml_query}, timeout=120)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), sep="\t", header=None)
        df.columns = ["gene_symbol", "chr", "start", "end", "strand_enc"]
        df = _post_process_biomart(df)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(cache_path, sep="\t", index=False)
        return df
    except Exception as exc:
        logger.warning("BioMart query failed: %s", exc)
        return pd.DataFrame(columns=_GENE_COORD_COLS)


def _build_biomart_xml(dataset: str, gene_symbols: list[str] | None) -> str:
    """Build BioMart XML query string."""
    # Filters block (only if specific gene symbols are requested)
    filter_block = ""
    if gene_symbols:
        sym_list = ",".join(gene_symbols[:2000])  # BioMart has a limit
        filter_block = f'<Filter name="external_gene_name" value="{sym_list}"/>'

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="0"
       uniqueRows="1" count="" datasetConfigVersion="0.6">
  <Dataset name="{dataset}" interface="default">
    {filter_block}
    <Attribute name="external_gene_name"/>
    <Attribute name="chromosome_name"/>
    <Attribute name="start_position"/>
    <Attribute name="end_position"/>
    <Attribute name="strand"/>
  </Dataset>
</Query>"""


def _post_process_biomart(df: pd.DataFrame) -> pd.DataFrame:
    """Convert BioMart output to standard format."""
    if df.empty:
        return pd.DataFrame(columns=_GENE_COORD_COLS)

    # Cached tables already contain processed columns: gene_symbol, chr, tss, strand.
    if {"gene_symbol", "chr", "tss"}.issubset(df.columns):
        out_df = df.copy()
        out_df["gene_symbol"] = out_df["gene_symbol"].astype(str)
        out_df["chr"] = out_df["chr"].astype(str).str.strip()
        out_df["chr"] = out_df["chr"].apply(
            lambda c: c if c.lower().startswith("chr") else f"chr{c}"
        )
        out_df["tss"] = pd.to_numeric(out_df["tss"], errors="coerce")
        out_df = out_df.dropna(subset=["gene_symbol", "chr", "tss"])
        out_df["tss"] = out_df["tss"].astype(int)
        if "strand" not in out_df.columns:
            out_df["strand"] = "+"
        out_df["strand"] = out_df["strand"].astype(str).replace({"1": "+", "-1": "-"})
        out_df = out_df.sort_values("tss").drop_duplicates("gene_symbol", keep="first")
        return out_df.loc[:, _GENE_COORD_COLS].reset_index(drop=True)

    # Rename if coming from cache with headers
    if "gene_symbol" not in df.columns:
        df.columns = ["gene_symbol", "chr", "start", "end", "strand_enc"]

    out: list[dict] = []
    for _, row in df.iterrows():
        strand = "+" if str(row.get("strand_enc", "1")) in ("1", "+") else "-"
        try:
            start = int(row.get("start", 0))
            end = int(row.get("end", 0))
        except (ValueError, TypeError):
            continue
        raw_chr = str(row.get("chr", "")).strip()
        if not raw_chr:
            continue
        chrom = raw_chr if raw_chr.lower().startswith("chr") else f"chr{raw_chr}"
        tss = start if strand == "+" else end
        out.append(
            {
                "gene_symbol": str(row.get("gene_symbol", "")),
                "chr": chrom,
                "tss": tss,
                "strand": strand,
            }
        )
    result = pd.DataFrame(out, columns=_GENE_COORD_COLS)
    # Deduplicate: keep one TSS per gene
    result = result.sort_values("tss").drop_duplicates("gene_symbol", keep="first")
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Utility: build peak → gene map from TSS table
# ---------------------------------------------------------------------------

def build_peak_to_gene_map(
    peaks: pd.DataFrame,
    gene_coords: pd.DataFrame,
    window: int = 2000,
) -> dict[str, list[str]]:
    """
    For each peak (chr, start, end), find genes whose TSS lies within ±window bp.

    Args:
        peaks:       DataFrame with columns: chr, start, end, (optional) name
        gene_coords: DataFrame with columns: gene_symbol, chr, tss
        window:      bp around TSS

    Returns:
        {peak_id: [gene_symbol, ...]}
        peak_id = name column if present, else "chr:start-end"
    """
    result: dict[str, list[str]] = {}

    # Build chromosome-grouped gene index for fast lookup
    chr_genes: dict[str, pd.DataFrame] = {
        chrom: grp for chrom, grp in gene_coords.groupby("chr")
    }

    for _, peak in peaks.iterrows():
        chrom = str(peak["chr"])
        start = int(peak["start"])
        end = int(peak["end"])

        if "name" in peak and pd.notna(peak.get("name")):
            peak_id = str(peak["name"])
        else:
            peak_id = f"{chrom}:{start}-{end}"

        genes_on_chr = chr_genes.get(chrom)
        if genes_on_chr is None:
            result[peak_id] = []
            continue

        # Genes whose TSS overlaps with the peak region ± window
        tss = genes_on_chr["tss"]
        mask = (tss >= start - window) & (tss <= end + window)
        result[peak_id] = genes_on_chr.loc[mask, "gene_symbol"].tolist()

    return result
