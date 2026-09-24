# config.py
# Central configuration for the literature mining agent.
# Edit these values before running the pipeline.

from dotenv import load_dotenv as _load_dotenv
_load_dotenv()

# --- NCBI / PubMed ---
# Email alone does NOT raise the rate limit — only an API key does.
# Without key: 3 req/sec. With key: 10 req/sec.
# Get a free key at https://www.ncbi.nlm.nih.gov/account/settings/  (My NCBI → API Key Management)
# Set via env var:  export NCBI_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
import os as _os
#
NCBI_EMAIL = _os.environ.get("NCBI_EMAIL", "user@example.com")  # NCBI requires an email for contact info, but it does NOT affect rate limits.
NCBI_API_KEY = _os.environ.get("NCBI_API_KEY", "")
NCBI_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
MAX_ABSTRACTS_PER_PAIR = 40          # cap how many abstracts to fetch per TF-Target pair
PUBMED_DELAY_SEC = 0.11 if NCBI_API_KEY else 0.34   # 10 req/s w/ key, 3 req/s w/o
PUBMED_MAX_RETRIES = 5               # for 429 / 5xx; uses exponential backoff
PUBMED_BACKOFF_BASE_SEC = 1.0        # first retry waits 1s, then 2s, 4s, 8s, ...

# --- Composite ranking (replaces PubMed relevance for top-N selection) ---
# When True: ESearch widens to POOL_SIZE candidates, ranks by weighted composite
# of citation impact + relevance/recency + full-text availability + title keywords,
# takes top MAX_ABSTRACTS_PER_PAIR. Surfaces foundational papers PubMed buries.
USE_COMPOSITE_RANKING = True
POOL_SIZE = 1000                     # candidate pool BEFORE ranking
                                     # cost: ~20s extra EFetch per pair, no LLM impact
RANK_WEIGHTS = {
    "citations": 0.3,                # field-normalized RCR percentile (or raw count fallback)
    "relevance_recency": 0.2,        # 0.5 * pubmed_rank + 0.5 * year_recency
    "full_text": 0.2,                # PMC full text available
    "title_keywords": 0.3,           # categorical fraction (TF, target, cell-type, assay) + pair bonus
}
REVIEW_PENALTY = 0.7                 # multiplier for non-research-article (reviews etc.)
TITLE_PAIR_BONUS = 0.3               # added to title score if BOTH TF and target in title

# Title-keyword assay/regulation terms — used to detect mechanism language in titles.
# Substring matching, so "regulates" matches "upregulates" / "downregulates".
TITLE_ASSAY_TERMS = [
    "ChIP", "ChIP-seq", "luciferase", "EMSA", "knockdown", "siRNA",
    "regulates", "binds", "promoter", "enhancer", "transcriptional",
    "direct target", "binding site", "regulation", "represses", "activates",
]

# --- Local LLM (Ollama) ---
OLLAMA_MODEL = "gemma:7b"
OLLAMA_BASE_URL = "http://localhost:11434"

# --- Score Aggregation ---
# lit_score = avg_confidence + beta_dir (0.10) + beta_rep (0.10), capped at 1.0
# beta_dir: bonus for direct physical evidence (ChIP-seq, EMSA, etc.)
# beta_rep: bonus for reproducibility (K > 1 supporting papers)
BETA_DIRECT = 0.10
BETA_REPRODUCIBILITY = 0.10

# --- Paths ---
from pathlib import Path as _Path
_DATA_DIR = _Path(__file__).resolve().parents[3] / "data"
INPUT_CSV = ""
ABSTRACTS_CACHE_DIR = str(_DATA_DIR / ".cache")
RESULTS_DIR = str(_DATA_DIR / "results")
REGULATORY_VERBS_FILE = str(_DATA_DIR / "dictionaries" / "regulatory_verbs.txt")
EXPERIMENTAL_METHODS_FILE = str(_DATA_DIR / "dictionaries" / "experimental_methods.txt")


# --- Filtering Thresholds ---
MIN_COOCCURRENCE_SCORE = 0.1        # abstracts below this are dropped in Stage 2
MIN_REGULATORY_SCORE = 0.05         # abstracts below this are dropped in Stage 2

# --- Cell Type Aliases ---
# Keys match the prefix of BEELINE network filenames (e.g. "hESC" from hESC-ChIP-seq-network.csv).
# Values cover how each cell type is referred to in literature — used for both
# PubMed query enrichment and the hard cell-type filter in Stage 2.
CELL_TYPE_ALIASES: dict[str, list[str]] = {
    # hESC: human pluripotent / ESC literature uses many synonyms.
    # Include generic "embryonic stem cell" — most hESC papers omit "human"
    # in the abstract once the species is established. Also iPSC because
    # human iPSC studies share the same regulatory biology.
    "hESC": [
        "hESC", "hESCs",
        "human embryonic stem cell", "human embryonic stem cells",
        "embryonic stem cell", "embryonic stem cells",
        "ES cells", "ESC", "ESCs",
        "pluripotent stem cell", "pluripotent stem cells",
        "human pluripotent",
        "iPSC", "iPSCs", "induced pluripotent",
        "H1", "H9", "WA01", "WA09", "HUES", "BG01", "BG02",
    ],
    # HepG2: hepatocellular carcinoma cell line and hepatocyte synonyms
    "HepG2": [
        "HepG2", "Hep G2", "hepatocellular carcinoma",
        "hepatocyte", "hepatocytes", "hepatic", "liver",
        "Huh7", "Huh-7",
    ],
    # mESC: mouse pluripotent
    "mESC": [
        "mESC", "mESCs",
        "mouse embryonic stem cell", "mouse embryonic stem cells",
        "murine embryonic stem cell", "murine embryonic stem cells",
        "embryonic stem cell", "embryonic stem cells",
        "ES cells", "ESC", "ESCs",
        "E14", "J1", "R1", "CCE", "AB2.2",
        "mouse pluripotent", "murine pluripotent",
        "miPSC", "miPSCs",
    ],
    # mDC: mouse dendritic cell — many literature aliases
    "mDC": [
        "mDC", "mDCs",
        "dendritic cell", "dendritic cells",
        "BMDC", "BMDCs", "bone marrow-derived dendritic",
        "cDC1", "cDC2", "pDC", "pDCs",
        "splenic DC", "Flt3L",
        "CD11c+",
    ],
    # mHSC: mouse hematopoietic stem cell (granulocyte-macrophage lineage)
    "mHSC": [
        "mHSC", "mHSCs",
        "hematopoietic stem cell", "hematopoietic stem cells",
        "haematopoietic stem cell", "haematopoietic stem cells",
        "HSC", "HSCs",
        "LSK", "Lin-Sca-1+c-Kit+",
        "bone marrow",
        "myeloid progenitor", "MPP", "CMP", "GMP",
        "long-term HSC", "LT-HSC", "ST-HSC",
        "myeloid", "myeloid cells", "myeloid lineage",
        "granulocyte", "granulocytes", "macrophage", "macrophages",
        "monocyte", "monocytes",
        "leukemia", "AML", "acute myeloid leukemia",
        "GM progenitor", "GM-CSF",
        "CD34+", "CD34", "progenitor cell", "progenitor cells",
    ],
}
