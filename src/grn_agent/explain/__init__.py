"""Deterministic scientific explanation helpers for GRNAgent."""

from .attention import (
    forward_with_attention,
    occlusion_deltas,
    summarize_attention_for_gene,
)
from .reasoning import (
    build_edge_explanation,
    enrich_scored_row,
    explanation_to_mechanism_text,
    format_acquisition_narrative,
    strength_label,
)

__all__ = [
    "build_edge_explanation",
    "enrich_scored_row",
    "explanation_to_mechanism_text",
    "format_acquisition_narrative",
    "forward_with_attention",
    "occlusion_deltas",
    "strength_label",
    "summarize_attention_for_gene",
]
