"""
Genome Database — auto-download, index, and serve reference genomes per species.

When a new species/build is first needed, the database:
  1. Resolves the canonical Ensembl build (FASTA + GTF)
  2. Downloads to <project_root>/GenomeDatabase/<build>/
  3. Decompresses (gunzip)
  4. Indexes FASTA with samtools faidx
  5. Records entry in GenomeDatabase/genome_db.json for future use

Subsequent requests are instant (path returned from registry).

Supported species (auto-extended — just add to GENOME_REGISTRY):
  mouse  / mm10  / mm39
  human  / hg38  / hg19
  rat    / rn7   / rn6
  zebrafish / danRer11 / danRer10
  drosophila / dm6
  worm   / ce11

Usage:
    from grn_agent.acquisition.genome_db import GenomeDB
    db = GenomeDB()
    fasta, gtf = db.ensure("mouse")   # downloads if not cached
    fasta, gtf = db.ensure("mm10")    # same result, UCSC alias accepted
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

# ── Global genome registry ────────────────────────────────────────────────────
# Each entry: (ensembl_species_dir, assembly_name, ensembl_release, gtf_species_prefix)
# UCSC alias → canonical key for lookup.

@dataclass
class _GenomeBuild:
    """Static metadata for one reference build."""
    key: str                 # canonical key, e.g. "mm10"
    species_common: str      # "mouse"
    species_latin: str       # "mus_musculus"
    assembly: str            # "GRCm38"   (Ensembl assembly name)
    ensembl_release: int     # 102
    ucsc_name: str           # "mm10"     (UCSC build alias)
    taxid: int               # 10090


# Canonical builds — add more here for new species
_REGISTRY: list[_GenomeBuild] = [
    # Mouse
    _GenomeBuild("mm10",  "mouse",      "mus_musculus",           "GRCm38",    102, "mm10",      10090),
    _GenomeBuild("mm39",  "mouse",      "mus_musculus",           "GRCm39",    110, "mm39",      10090),
    # Human
    _GenomeBuild("hg38",  "human",      "homo_sapiens",           "GRCh38",    110, "hg38",       9606),
    _GenomeBuild("hg19",  "human",      "homo_sapiens",           "GRCh37",     75, "hg19",       9606),
    # Rat
    _GenomeBuild("rn7",   "rat",        "rattus_norvegicus",      "mRatBN7.2", 109, "rn7",       10116),
    _GenomeBuild("rn6",   "rat",        "rattus_norvegicus",      "Rnor_6.0",   96, "rn6",       10116),
    # Zebrafish
    _GenomeBuild("danRer11", "zebrafish", "danio_rerio",          "GRCz11",    109, "danRer11",   7955),
    _GenomeBuild("danRer10", "zebrafish", "danio_rerio",          "GRCz10",     91, "danRer10",   7955),
    # Drosophila (BDGP6.46 toplevel first appears at Ensembl 110; 109 still has BDGP6.32)
    _GenomeBuild("dm6",   "drosophila", "drosophila_melanogaster","BDGP6.46",  110, "dm6",        7227),
    # C. elegans
    _GenomeBuild("ce11",  "worm",       "caenorhabditis_elegans", "WBcel235",  109, "ce11",       6239),
    # Arabidopsis
    _GenomeBuild("tair10","arabidopsis","arabidopsis_thaliana",   "TAIR10",     51, "tair10",     3702),
]

# Build lookup index: canonical key, UCSC name, common name → _GenomeBuild
# For species_common (e.g. "mouse"), FIRST entry in registry wins as the default build.
_BUILD_INDEX: dict[str, _GenomeBuild] = {}
for _b in _REGISTRY:
    _BUILD_INDEX[_b.key] = _b
    _BUILD_INDEX[_b.ucsc_name] = _b
    _BUILD_INDEX.setdefault(_b.species_common, _b)  # first listed build is default for common name


# ── Persistent registry entry ─────────────────────────────────────────────────

@dataclass
class GenomeEntry:
    key: str
    fasta_path: str
    fai_path: str
    gtf_path: str
    fasta_indexed: bool = False
    downloaded_at: str = ""
    source: str = "ensembl"
    extra: dict[str, Any] = field(default_factory=dict)


# ── Main class ────────────────────────────────────────────────────────────────

class GenomeDB:
    """
    Persistent genome database.  Auto-downloads + indexes reference genomes.

    Parameters
    ----------
    db_root : Path
        Root storage directory.  Defaults to ~/.cache/grn_agent/genomes/
    """

    _DB_FILE = "genome_db.json"
    _ENSEMBL_FTP = "https://ftp.ensembl.org/pub"
    _CHUNK = 1 << 20  # 1 MB download chunks

    # Default root: <project_root>/GenomeDatabase/
    # src/grn_agent/acquisition/genome_db.py → parents[3] = project root
    _DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "GenomeDatabase"

    def __init__(self, db_root: str | Path | None = None) -> None:
        self._root = Path(db_root) if db_root else self._DEFAULT_ROOT
        self._root.mkdir(parents=True, exist_ok=True)
        self._db_file = self._root / self._DB_FILE
        self._registry: dict[str, GenomeEntry] = {}
        self._load_db()

    # ── Public API ────────────────────────────────────────────────────────────

    def ensure(self, species_or_build: str) -> tuple[Path, Path]:
        """
        Return (fasta_path, gtf_path) for the requested species/build.
        Downloads + indexes if not already cached.

        Parameters
        ----------
        species_or_build : str
            Common name ("mouse"), UCSC build ("mm10"), or canonical key ("GRCm38").

        Returns
        -------
        (fasta_path, gtf_path)  — both are ready-to-use local files.

        Raises
        ------
        ValueError  if the species is unknown and cannot be auto-resolved.
        RuntimeError if download or indexing fails.
        """
        build = self._resolve_build(species_or_build)
        key = build.key

        if key in self._registry:
            entry = self._registry[key]
            if Path(entry.fasta_path).is_file() and Path(entry.gtf_path).is_file():
                log.info("Genome '%s' already in DB: %s", key, entry.fasta_path)
                return Path(entry.fasta_path), Path(entry.gtf_path)
            log.warning("Genome '%s' in DB but files missing — re-downloading.", key)

        # Recover from already-downloaded files on disk even if DB metadata is missing/stale.
        recovered = self._discover_existing_entry(build)
        if recovered is not None:
            self._registry[key] = recovered
            self._save_db()
            log.info("Recovered genome '%s' from local files: %s", key, recovered.fasta_path)
            return Path(recovered.fasta_path), Path(recovered.gtf_path)

        log.info("Genome '%s' not in local DB — downloading now…", key)
        entry = self._download_build(build)
        self._registry[key] = entry
        self._save_db()
        return Path(entry.fasta_path), Path(entry.gtf_path)

    def get(self, species_or_build: str) -> GenomeEntry | None:
        """Return registry entry without downloading (returns None if not present)."""
        try:
            build = self._resolve_build(species_or_build)
            return self._registry.get(build.key)
        except ValueError:
            return None

    def list_cached(self) -> list[dict]:
        """Return summary of all cached genomes."""
        return [
            {
                "key": k,
                "fasta": v.fasta_path,
                "gtf": v.gtf_path,
                "indexed": v.fasta_indexed,
                "downloaded_at": v.downloaded_at,
            }
            for k, v in self._registry.items()
        ]

    def register_existing(
        self,
        species_or_build: str,
        fasta_path: str | Path,
        gtf_path: str | Path,
    ) -> GenomeEntry:
        """
        Register user-provided FASTA + GTF files without downloading.
        Indexes FASTA if not already indexed (.fai missing).
        """
        build = self._resolve_build(species_or_build)
        fasta = Path(fasta_path).resolve()
        gtf = Path(gtf_path).resolve()
        if not fasta.is_file():
            raise FileNotFoundError(f"FASTA not found: {fasta}")
        if not gtf.is_file():
            raise FileNotFoundError(f"GTF not found: {gtf}")

        fai = Path(str(fasta) + ".fai")
        indexed = fai.is_file()
        if not indexed:
            log.info("Indexing user-provided FASTA with samtools faidx…")
            indexed = self._faidx(fasta)

        entry = GenomeEntry(
            key=build.key,
            fasta_path=str(fasta),
            fai_path=str(fai),
            gtf_path=str(gtf),
            fasta_indexed=indexed,
            downloaded_at=datetime.utcnow().isoformat(),
            source="user_provided",
        )
        self._registry[build.key] = entry
        self._save_db()
        log.info("Registered '%s' in genome DB.", build.key)
        return entry

    # ── Internal: build resolution ─────────────────────────────────────────────

    def _resolve_build(self, query: str) -> _GenomeBuild:
        key = query.strip().lower()
        build = _BUILD_INDEX.get(key) or _BUILD_INDEX.get(query.strip())
        if build:
            return build
        raise ValueError(
            f"Unknown species/build: '{query}'.\n"
            f"Known keys: {sorted(_BUILD_INDEX.keys())}\n"
            "To add a new species, use genome_db.register_existing() or extend GENOME_REGISTRY."
        )

    # ── Internal: download ─────────────────────────────────────────────────────

    def _download_build(self, build: _GenomeBuild) -> GenomeEntry:
        dest = self._root / build.key
        dest.mkdir(parents=True, exist_ok=True)

        fasta_path = self._download_fasta(build, dest)
        gtf_path = self._download_gtf(build, dest)
        indexed = self._faidx(fasta_path)

        return GenomeEntry(
            key=build.key,
            fasta_path=str(fasta_path),
            fai_path=str(fasta_path) + ".fai",
            gtf_path=str(gtf_path),
            fasta_indexed=indexed,
            downloaded_at=datetime.utcnow().isoformat(),
            source=f"ensembl_release_{build.ensembl_release}",
        )

    def _discover_existing_entry(self, build: _GenomeBuild) -> GenomeEntry | None:
        """
        Discover an already-downloaded genome under GenomeDatabase/<build.key>/.
        This avoids unnecessary downloads when files exist but registry JSON is absent/stale.
        """
        dest = self._root / build.key
        fasta = dest / "genome.fa"
        gtf = dest / "annotation.gtf"
        if not (fasta.is_file() and gtf.is_file()):
            return None

        fai = Path(str(fasta) + ".fai")
        indexed = fai.is_file()
        if not indexed:
            indexed = self._faidx(fasta)

        return GenomeEntry(
            key=build.key,
            fasta_path=str(fasta),
            fai_path=str(fai),
            gtf_path=str(gtf),
            fasta_indexed=indexed,
            downloaded_at=datetime.utcnow().isoformat(),
            source="discovered_local",
        )

    def _ensembl_ftp_fasta_url(self, build: _GenomeBuild) -> str:
        """
        Resolve Ensembl FTP URL for the toplevel / primary_assembly FASTA.
        Human uses primary_assembly (no patches); others use toplevel.
        """
        rel = build.ensembl_release
        sp = build.species_latin          # e.g. "mus_musculus"
        # Ensembl filename convention: Genus capitalized, species lower-case
        # (e.g. Mus_musculus, Homo_sapiens).
        sp_ensembl = self._ensembl_species_prefix(sp)
        suffix = "dna.primary_assembly" if sp == "homo_sapiens" else "dna.toplevel"
        fname = f"{sp_ensembl}.{build.assembly}.{suffix}.fa.gz"
        return f"{self._ENSEMBL_FTP}/release-{rel}/fasta/{sp}/dna/{fname}"

    def _ensembl_ftp_fasta_urls(self, build: _GenomeBuild) -> list[str]:
        """
        Build a prioritized list of candidate FASTA URLs for one build.
        This is intentionally generic (no build-specific hardcoding).
        """
        rel = build.ensembl_release
        sp = build.species_latin
        sp_ensembl = self._ensembl_species_prefix(sp)
        suffixes = ["dna.toplevel", "dna.primary_assembly"]
        urls: list[str] = []
        for suffix in suffixes:
            fname = f"{sp_ensembl}.{build.assembly}.{suffix}.fa.gz"
            urls.append(f"{self._ENSEMBL_FTP}/release-{rel}/fasta/{sp}/dna/{fname}")
        return urls

    def _ensembl_ftp_gtf_url(self, build: _GenomeBuild) -> str:
        rel = build.ensembl_release
        sp = build.species_latin
        sp_ensembl = self._ensembl_species_prefix(sp)
        fname = f"{sp_ensembl}.{build.assembly}.{rel}.gtf.gz"
        return f"{self._ENSEMBL_FTP}/release-{rel}/gtf/{sp}/{fname}"

    def _ensembl_ftp_gtf_urls(self, build: _GenomeBuild) -> list[str]:
        """
        Build a prioritized list of candidate GTF URLs for one build.
        This is intentionally generic (no build-specific hardcoding).
        """
        rel = build.ensembl_release
        sp = build.species_latin
        sp_ensembl = self._ensembl_species_prefix(sp)
        fname_with_rel = f"{sp_ensembl}.{build.assembly}.{rel}.gtf.gz"
        fname_no_rel = f"{sp_ensembl}.{build.assembly}.gtf.gz"
        urls = [
            f"{self._ENSEMBL_FTP}/release-{rel}/gtf/{sp}/{fname_with_rel}",
            f"{self._ENSEMBL_FTP}/release-{rel}/gtf/{sp}/{fname_no_rel}",
        ]
        # Also try whichever files currently exist on Ensembl mirrors.
        release_dir = f"{self._ENSEMBL_FTP}/release-{rel}/gtf/{sp}/"
        current_dir = f"{self._ENSEMBL_FTP}/current_gtf/{sp}/"
        for base in (release_dir, current_dir):
            for discovered in self._discover_gtf_urls_from_listing(base, build.assembly):
                if discovered not in urls:
                    urls.append(discovered)
        return urls

    @staticmethod
    def _ensembl_species_prefix(species_latin: str) -> str:
        """
        Convert 'mus_musculus' -> 'Mus_musculus' for Ensembl file names.
        """
        parts = [p for p in species_latin.split("_") if p]
        if not parts:
            return species_latin
        return "_".join([parts[0].capitalize()] + [p.lower() for p in parts[1:]])

    def _discover_gtf_urls_from_listing(self, base_url: str, assembly: str) -> list[str]:
        """
        Parse an Ensembl FTP directory listing and return matching GTF URLs.
        """
        try:
            resp = requests.get(base_url, timeout=30)
            resp.raise_for_status()
            names = set(re.findall(r'href="([^"]+\.gtf\.gz)"', resp.text))
            # Prefer files containing the requested assembly; keep others as fallback.
            ordered = sorted(
                names,
                key=lambda n: (
                    0 if assembly in n else 1,
                    1 if "abinitio" in n.lower() else 0,
                    1 if "chr_patch" in n.lower() else 0,
                    1 if ".chr." in n.lower() else 0,
                    n,
                ),
            )
            return [base_url + n for n in ordered]
        except Exception:
            return []

    def _download_fasta(self, build: _GenomeBuild, dest: Path) -> Path:
        gz_path = dest / "genome.fa.gz"
        fa_path = dest / "genome.fa"
        if fa_path.is_file():
            log.info("FASTA already decompressed: %s", fa_path)
            return fa_path

        # Build-agnostic URL strategy:
        # 1) Ensembl release URLs (multiple FASTA flavors)
        # 2) UCSC goldenPath fallback using the build's UCSC alias
        fasta_urls = self._ensembl_ftp_fasta_urls(build)
        fasta_urls.append(
            f"https://hgdownload.soe.ucsc.edu/goldenPath/{build.ucsc_name}/bigZips/{build.ucsc_name}.fa.gz"
        )
        last_exc: Exception | None = None
        for url in fasta_urls:
            try:
                log.info("Downloading FASTA from %s", url)
                self._stream_download(url, gz_path, label=f"{build.key} FASTA")
                log.info("Decompressing FASTA…")
                self._gunzip(gz_path, fa_path)
                gz_path.unlink(missing_ok=True)
                return fa_path
            except Exception as exc:
                last_exc = exc
                gz_path.unlink(missing_ok=True)
                fa_path.unlink(missing_ok=True)
                log.warning("FASTA download failed via %s: %s", url, exc)
        raise RuntimeError(f"Unable to download FASTA for {build.key}") from last_exc

    def _download_gtf(self, build: _GenomeBuild, dest: Path) -> Path:
        gz_path = dest / "annotation.gtf.gz"
        gtf_path = dest / "annotation.gtf"
        if gtf_path.is_file():
            log.info("GTF already decompressed: %s", gtf_path)
            return gtf_path

        # Build-agnostic GTF strategy (Ensembl release variants).
        gtf_urls = self._ensembl_ftp_gtf_urls(build)
        last_exc: Exception | None = None
        for url in gtf_urls:
            try:
                log.info("Downloading GTF from %s", url)
                self._stream_download(url, gz_path, label=f"{build.key} GTF")
                log.info("Decompressing GTF…")
                self._gunzip(gz_path, gtf_path)
                gz_path.unlink(missing_ok=True)
                return gtf_path
            except Exception as exc:
                last_exc = exc
                gz_path.unlink(missing_ok=True)
                gtf_path.unlink(missing_ok=True)
                log.warning("GTF download failed via %s: %s", url, exc)
        raise RuntimeError(f"Unable to download GTF for {build.key}") from last_exc

    def _stream_download(self, url: str, dest: Path, label: str = "") -> None:
        """Stream a file from URL to dest with progress reporting."""
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(url, stream=True, timeout=60)
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                downloaded = 0
                last_pct = -1
                with dest.open("wb") as fp:
                    for chunk in resp.iter_content(chunk_size=self._CHUNK):
                        fp.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            pct = int(downloaded * 100 / total)
                            if pct != last_pct and pct % 10 == 0:
                                mb = downloaded / 1_048_576
                                total_mb = total / 1_048_576
                                print(
                                    f"\r[genome_db] {label}: {pct}% ({mb:.0f}/{total_mb:.0f} MB)",
                                    end="", flush=True,
                                )
                                last_pct = pct
                print(f"\r[genome_db] {label}: done ({downloaded/1_048_576:.0f} MB)    ", flush=True)
                return
            except Exception as exc:
                log.warning("Download attempt %d/%d failed for %s: %s", attempt, max_retries, url, exc)
                if attempt < max_retries:
                    time.sleep(5 * attempt)
                    dest.unlink(missing_ok=True)
                else:
                    raise RuntimeError(f"Failed to download {url} after {max_retries} attempts: {exc}") from exc

    @staticmethod
    def _gunzip(src: Path, dst: Path) -> None:
        """Decompress gzip file."""
        with gzip.open(src, "rb") as f_in, dst.open("wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    @staticmethod
    def _faidx(fasta_path: Path) -> bool:
        """Index FASTA with samtools faidx. Returns True on success."""
        try:
            result = subprocess.run(
                ["samtools", "faidx", str(fasta_path)],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode == 0:
                log.info("samtools faidx: indexed %s", fasta_path.name)
                return True
            log.warning("samtools faidx failed: %s", result.stderr[:300])
        except FileNotFoundError:
            log.warning(
                "samtools not found — FASTA not indexed. "
                "Install: conda install -c bioconda samtools"
            )
        except Exception as exc:
            log.warning("samtools faidx error: %s", exc)
        return False

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load_db(self) -> None:
        if not self._db_file.is_file():
            return
        try:
            raw = json.loads(self._db_file.read_text(encoding="utf-8"))
            self._registry = {k: GenomeEntry(**v) for k, v in raw.items()}
            log.debug("Loaded %d genomes from DB.", len(self._registry))
        except Exception as exc:
            log.warning("Failed to load genome_db.json: %s", exc)

    def _save_db(self) -> None:
        data = {k: asdict(v) for k, v in self._registry.items()}
        self._db_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log.debug("Saved %d genomes to %s", len(self._registry), self._db_file)


# ── Module-level singleton ────────────────────────────────────────────────────

_DEFAULT_DB: GenomeDB | None = None


def get_genome_db(db_root: str | Path | None = None) -> GenomeDB:
    """Return (and lazily create) the default GenomeDB singleton."""
    global _DEFAULT_DB
    if _DEFAULT_DB is None or db_root is not None:
        _DEFAULT_DB = GenomeDB(db_root)
    return _DEFAULT_DB


def ensure_genome(species_or_build: str, db_root: str | Path | None = None) -> tuple[Path, Path]:
    """
    Convenience function — get (fasta, gtf) paths, downloading if needed.

    Genomes are stored in <project_root>/GenomeDatabase/<build>/ by default.

    >>> fasta, gtf = ensure_genome("mouse")   # → GenomeDatabase/mm10/genome.fa
    >>> fasta, gtf = ensure_genome("mm10")
    >>> fasta, gtf = ensure_genome("hg38")    # → GenomeDatabase/hg38/genome.fa
    """
    return get_genome_db(db_root).ensure(species_or_build)
