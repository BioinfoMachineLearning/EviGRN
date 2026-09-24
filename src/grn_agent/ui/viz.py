"""Plotly visualization helpers for the GRNAgent GUI."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover
    px = None  # type: ignore[assignment]
    go = None  # type: ignore[assignment]


SCI_COLORS = {
    "ink": "#0F2744",
    "teal": "#1F6F7A",
    "slate": "#4A6075",
    "sand": "#E8EEF4",
    "accent": "#C45C26",
    "good": "#2F6B4F",
    "warn": "#A67C00",
    "bad": "#8B2E2E",
}


def _require_plotly() -> None:
    if px is None or go is None:
        raise ImportError("plotly is required for GUI visualizations. Install with: pip install -e '.[gui]'")


def coverage_figure(manifest: dict[str, Any]):
    _require_plotly()
    qc = manifest.get("qc") or manifest.get("qc_report") or {}
    acc_qc = qc.get("accessibility_qc") or {}
    components = acc_qc.get("components") or {}
    metrics = acc_qc.get("metrics") or {}
    rows: list[dict[str, Any]] = []
    for name, value in components.items():
        try:
            rows.append({"metric": str(name), "value": float(value), "group": "QC component"})
        except (TypeError, ValueError):
            continue
    coverage = metrics.get("promoter_coverage")
    if coverage is None:
        coverage = (manifest.get("accessibility") or {}).get("promoter_coverage_of_rnaseq_genes")
    if coverage is not None:
        try:
            rows.append({"metric": "promoter_coverage", "value": float(coverage), "group": "Coverage"})
        except (TypeError, ValueError):
            pass
    motif_overlap = (manifest.get("motifs") or {}).get("tf_overlap_with_rnaseq")
    if motif_overlap is not None:
        try:
            rows.append({"metric": "motif_tf_overlap", "value": float(motif_overlap), "group": "Motif"})
        except (TypeError, ValueError):
            pass
    if not rows:
        fig = go.Figure()
        fig.update_layout(title="No coverage / QC numeric fields available")
        return fig
    df = pd.DataFrame(rows)
    fig = px.bar(
        df,
        x="metric",
        y="value",
        color="group",
        color_discrete_sequence=[SCI_COLORS["teal"], SCI_COLORS["accent"], SCI_COLORS["good"]],
        title="Multimodal coverage and QC components",
    )
    fig.update_layout(
        font=dict(size=15, color=SCI_COLORS["ink"]),
        paper_bgcolor="white",
        plot_bgcolor=SCI_COLORS["sand"],
        yaxis_title="Score / fraction",
        xaxis_title="",
        legend_title="",
        legend=dict(orientation="h", yanchor="top", y=-0.22),
        margin=dict(l=48, r=24, t=56, b=110),
        height=420,
    )
    fig.update_xaxes(tickangle=-30, automargin=True)
    fig.update_yaxes(range=[0, 1.05])
    return fig


def evidence_factor_figure(explanation: dict[str, Any]):
    _require_plotly()
    rows = []
    for factor in explanation.get("factors") or []:
        name = str(factor.get("name"))
        value = factor.get("value")
        if isinstance(value, bool):
            rows.append({"factor": name, "value": 1.0 if value else 0.0})
        elif isinstance(value, (int, float)):
            rows.append({"factor": name, "value": float(value)})
    if not rows:
        fig = go.Figure()
        fig.update_layout(title="No quantitative factors available")
        return fig
    df = pd.DataFrame(rows)
    fig = px.bar(
        df,
        x="factor",
        y="value",
        title="Evidence factors for selected edge",
        color_discrete_sequence=[SCI_COLORS["teal"]],
    )
    fig.update_layout(
        font=dict(size=15, color=SCI_COLORS["ink"]),
        paper_bgcolor="white",
        plot_bgcolor=SCI_COLORS["sand"],
        showlegend=False,
        margin=dict(l=48, r=24, t=56, b=100),
        height=420,
    )
    fig.update_xaxes(tickangle=-30, automargin=True)
    return fig


def load_tf_window(windows_path: str | Path, source_tf: str, *, max_scan: int = 5000) -> dict[str, Any] | None:
    tf = str(source_tf).strip().upper()
    path = Path(windows_path)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if i >= max_scan:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if str(rec.get("source_tf", "")).strip().upper() == tf:
                return rec
    return None


def tf_evidence_graph_figure(window: dict[str, Any], *, top_n: int = 40):
    _require_plotly()
    tf = str(window.get("source_tf", "TF")).upper()
    genes = list(window.get("genes") or [])[: max(1, int(top_n))]
    nodes = [{"id": "TF", "label": tf, "kind": "tf", "x": 0.0, "y": 0.0}]
    edges = []
    n = max(len(genes), 1)
    for i, gene in enumerate(genes):
        target = str(gene.get("target_gene", f"G{i}")).upper()
        evidence = gene.get("evidence") or {}
        motif = bool(evidence.get("motif_present", False))
        acc = float(evidence.get("accessibility") or 0.0)
        corr = float(evidence.get("correlation") or 0.0)
        angle = (i / n) * (2.0 * math.pi)
        nodes.append(
            {
                "id": target,
                "label": target,
                "kind": "gene",
                "x": 1.4 * math.cos(angle),
                "y": 1.4 * math.sin(angle),
                "motif": motif,
                "accessibility": acc,
                "correlation": corr,
            }
        )
        edges.append({"source": "TF", "target": target, "corr": corr, "motif": motif, "acc": acc})

    node_x = [n["x"] for n in nodes]
    node_y = [n["y"] for n in nodes]
    node_text = [
        (
            f"{n['label']}<br>motif={n.get('motif')}<br>acc={n.get('accessibility', 0):.2f}<br>"
            f"corr={n.get('correlation', 0):.2f}"
            if n["kind"] == "gene"
            else n["label"]
        )
        for n in nodes
    ]
    node_color = [SCI_COLORS["accent"] if n["kind"] == "tf" else SCI_COLORS["teal"] for n in nodes]
    node_size = [28 if n["kind"] == "tf" else 12 for n in nodes]
    show_labels = len(nodes) <= 18

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    lookup = {n["id"]: n for n in nodes}
    for edge in edges:
        s = lookup[edge["source"]]
        t = lookup[edge["target"]]
        edge_x.extend([s["x"], t["x"], None])
        edge_y.extend([s["y"], t["y"], None])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1, color=SCI_COLORS["slate"]),
            hoverinfo="none",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text" if show_labels else "markers",
            text=[n["label"] for n in nodes] if show_labels else None,
            textposition="top center",
            textfont=dict(size=11),
            marker=dict(size=node_size, color=node_color, line=dict(width=1, color="white")),
            hovertext=node_text,
            hoverinfo="text",
            showlegend=False,
        )
    )
    fig.update_layout(
        title=f"TF-centered evidence neighborhood: {tf}",
        font=dict(size=15, color=SCI_COLORS["ink"]),
        paper_bgcolor="white",
        plot_bgcolor=SCI_COLORS["sand"],
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
        margin=dict(l=24, r=24, t=56, b=24),
        height=620,
    )
    return fig


def network_figure(df: pd.DataFrame, *, threshold: float = 0.5, max_edges: int = 200):
    _require_plotly()
    work = df.copy()
    if "p_present" not in work.columns:
        raise ValueError("network dataframe requires p_present")
    work = work[work["p_present"] >= float(threshold)].copy()
    work = work.sort_values("p_present", ascending=False).head(int(max_edges))
    if work.empty:
        fig = go.Figure()
        fig.update_layout(title="No edges above threshold")
        return fig

    tfs = sorted(work["source_tf"].astype(str).str.upper().unique().tolist())
    genes = sorted(work["target_gene"].astype(str).str.upper().unique().tolist())
    tf_x = {tf: 0.0 for tf in tfs}
    tf_y = {tf: i for i, tf in enumerate(tfs)}
    gene_x = {g: 1.0 for g in genes}
    gene_y = {g: i * (len(tfs) / max(len(genes), 1)) for i, g in enumerate(genes)}

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for _, row in work.iterrows():
        tf = str(row["source_tf"]).upper()
        gene = str(row["target_gene"]).upper()
        edge_x.extend([tf_x[tf], gene_x[gene], None])
        edge_y.extend([tf_y[tf], gene_y[gene], None])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1, color="#9AA8B5"),
            hoverinfo="none",
            showlegend=False,
        )
    )
    label_tfs = len(tfs) <= 24
    label_genes = len(genes) <= 24
    fig.add_trace(
        go.Scatter(
            x=list(tf_x.values()),
            y=list(tf_y.values()),
            mode="markers+text" if label_tfs else "markers",
            text=list(tf_x.keys()) if label_tfs else None,
            textposition="middle left",
            marker=dict(size=14, color=SCI_COLORS["accent"]),
            name="TF",
            hovertext=list(tf_x.keys()),
            hoverinfo="text",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(gene_x.values()),
            y=list(gene_y.values()),
            mode="markers+text" if label_genes else "markers",
            text=list(gene_x.keys()) if label_genes else None,
            textposition="middle right",
            marker=dict(size=10, color=SCI_COLORS["teal"]),
            name="Target",
            hovertext=list(gene_x.keys()),
            hoverinfo="text",
        )
    )
    span = max(len(tfs), len(genes), 1)
    fig.update_layout(
        title=f"Final network (p_present >= {threshold:.2f}, top {len(work)} edges)",
        font=dict(size=15, color=SCI_COLORS["ink"]),
        paper_bgcolor="white",
        plot_bgcolor=SCI_COLORS["sand"],
        xaxis=dict(visible=False, range=[-0.35, 1.35]),
        yaxis=dict(visible=False),
        margin=dict(l=90, r=90, t=72, b=36),
        height=min(920, max(480, 28 * span + 120)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


_CURVE_COLORS = ["#1F6F7A", "#C45C26", "#2F6B4F", "#0F2744", "#A67C00", "#4A6075", "#8B2E2E"]


def _layout(fig, *, title: str, height: int = 460) -> None:
    fig.update_layout(
        title=title,
        font=dict(size=15, color=SCI_COLORS["ink"]),
        paper_bgcolor="white",
        plot_bgcolor=SCI_COLORS["sand"],
        margin=dict(l=56, r=24, t=64, b=72),
        height=height,
        legend=dict(orientation="h", yanchor="top", y=-0.18),
    )


def metrics_bar_figure(frame: pd.DataFrame):
    """Grouped bars for AUROC, AUPRC, and EP@100 across negative ratios."""
    _require_plotly()
    if frame.empty:
        fig = go.Figure()
        _layout(fig, title="No ranking metrics in this report")
        return fig
    value_cols = [col for col in ("AUROC", "AUPRC", "EP@100") if col in frame.columns]
    long = frame.melt(id_vars=["ratio_label"], value_vars=value_cols, var_name="metric", value_name="value")
    long = long.dropna(subset=["value"])
    if long.empty:
        fig = go.Figure()
        _layout(fig, title="AUROC, AUPRC, and EP@100 are not numeric in this report")
        return fig
    fig = px.bar(
        long,
        x="ratio_label",
        y="value",
        color="metric",
        barmode="group",
        color_discrete_sequence=[SCI_COLORS["teal"], SCI_COLORS["accent"], SCI_COLORS["good"]],
        title="AUROC, AUPRC, and EP@100 by negative ratio",
    )
    _layout(fig, title="AUROC, AUPRC, and EP@100 by negative ratio", height=440)
    fig.update_yaxes(title="Score", range=[0, 1.05], automargin=True)
    fig.update_xaxes(title="Negative ratio", automargin=True)
    return fig


def pr_curves_figure(series: list[dict[str, Any]]):
    _require_plotly()
    fig = go.Figure()
    if not series:
        _layout(fig, title="No precision-recall curve is stored in this report")
        return fig
    for index, item in enumerate(series):
        color = _CURVE_COLORS[index % len(_CURVE_COLORS)]
        recall = item["recall"]
        precision = item["precision"]
        fig.add_trace(
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=item["label"],
                line=dict(color=color, width=2.4),
            )
        )
        std = item.get("precision_std")
        if std is not None and len(std) == len(precision):
            upper = [min(1.0, p + s) for p, s in zip(precision, std)]
            lower = [max(0.0, p - s) for p, s in zip(precision, std)]
            fig.add_trace(
                go.Scatter(
                    x=list(recall) + list(reversed(recall)),
                    y=upper + list(reversed(lower)),
                    fill="toself",
                    fillcolor=color,
                    opacity=0.15,
                    line=dict(width=0),
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"{item['label']} std",
                )
            )
        prevalence = item.get("prevalence")
        if prevalence is not None:
            fig.add_hline(y=float(prevalence), line=dict(color=color, width=1, dash="dot"))
    _layout(fig, title="Precision-recall curves")
    fig.update_xaxes(title="Recall", range=[0, 1], automargin=True)
    fig.update_yaxes(title="Precision", range=[0, 1.02], automargin=True)
    return fig


def roc_curves_figure(series: list[dict[str, Any]]):
    _require_plotly()
    fig = go.Figure()
    if not series:
        _layout(fig, title="No ROC curve coordinates are stored in this report")
        return fig
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="chance",
            line=dict(color="#9AA8B5", width=1, dash="dash"),
        )
    )
    for index, item in enumerate(series):
        color = _CURVE_COLORS[index % len(_CURVE_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=item["fpr"],
                y=item["tpr"],
                mode="lines",
                name=item["label"],
                line=dict(color=color, width=2.4),
            )
        )
    _layout(fig, title="ROC curves")
    fig.update_xaxes(title="False positive rate", range=[0, 1], automargin=True)
    fig.update_yaxes(title="True positive rate", range=[0, 1.02], automargin=True)
    return fig


def attention_figure(channels: pd.DataFrame, local: pd.DataFrame | None = None):
    """One edge: channel bars. Several edges: heatmap of local-target attention."""
    _require_plotly()
    if channels.empty and (local is None or local.empty):
        fig = go.Figure()
        _layout(fig, title="No attention summary available")
        return fig
    edges = []
    if not channels.empty and "edge" in channels.columns:
        edges = channels["edge"].dropna().astype(str).unique().tolist()
    elif local is not None and not local.empty:
        edges = local["edge"].dropna().astype(str).unique().tolist()
    if len(edges) <= 1 and not channels.empty:
        title = f"Cross-attention: {edges[0]}" if edges else "Cross-attention"
        fig = px.bar(
            channels,
            x="channel",
            y="attention_mass",
            color="stage",
            barmode="group",
            title=title,
            color_discrete_sequence=[SCI_COLORS["teal"], SCI_COLORS["accent"], SCI_COLORS["ink"]],
        )
        _layout(fig, title=title, height=460)
        fig.update_yaxes(title="Attention mass", range=[0, 1.05], automargin=True)
        fig.update_xaxes(title="", tickangle=-20, automargin=True)
        return fig
    source = local if local is not None and not local.empty else channels
    if "region" in source.columns:
        focus = source[source["region"].astype(str) == "local_target_tokens"]
        if focus.empty:
            focus = source
        pivot = focus.pivot(index="edge", columns="stage", values="attention_mass").fillna(0.0)
        heat_title = "Local-target attention by selected edge"
    else:
        pivot = (
            source.groupby(["edge", "stage"], as_index=False)["attention_mass"]
            .mean()
            .pivot(index="edge", columns="stage", values="attention_mass")
            .fillna(0.0)
        )
        heat_title = "Mean channel attention by selected edge"
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.to_numpy(),
            x=[str(col) for col in pivot.columns],
            y=[str(idx) for idx in pivot.index],
            colorscale=[[0, "#F3F6F9"], [1, SCI_COLORS["teal"]]],
            zmin=0,
            zmax=1,
            colorbar=dict(title="mass"),
            hovertemplate="edge=%{y}<br>stage=%{x}<br>mass=%{z:.3f}<extra></extra>",
        )
    )
    _layout(fig, title=heat_title, height=max(420, 36 * len(pivot) + 160))
    fig.update_xaxes(title="Decoder stage", automargin=True)
    fig.update_yaxes(title="", automargin=True)
    return fig


def pca_figure(frame: pd.DataFrame, variance: tuple[float, float], *, title: str):
    _require_plotly()
    if frame.empty:
        fig = go.Figure()
        _layout(fig, title=title)
        return fig
    work = frame.copy()
    work["role"] = work["selected"].map(lambda flag: "selected" if bool(flag) else "other")
    color_col = "p_present" if work["p_present"].notna().any() else None
    fig = px.scatter(
        work,
        x="pc1",
        y="pc2",
        color=color_col,
        symbol="role",
        hover_name="edge",
        color_continuous_scale=["#E8EEF4", "#1F6F7A", "#0F2744"],
        title=title,
    )
    pc1 = 100.0 * float(variance[0])
    pc2 = 100.0 * float(variance[1])
    _layout(fig, title=title, height=520)
    fig.update_xaxes(title=f"PC1 ({pc1:.1f}% variance)", automargin=True)
    fig.update_yaxes(title=f"PC2 ({pc2:.1f}% variance)", automargin=True)
    if color_col is not None:
        fig.update_layout(coloraxis_colorbar=dict(title="p_present"))
    return fig
