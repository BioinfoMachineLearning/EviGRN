#!/usr/bin/env python3
"""EviGRN scientific explorer GUI (Streamlit).

Additive interface over existing artifacts and scripts.
Does not modify model architecture or acquisition/training cores.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from grn_agent.explain.reasoning import (  # noqa: E402
    build_edge_explanation,
    format_acquisition_narrative,
)
from grn_agent.ui.artifacts import discover_workflow_dirs, load_json, resolve_workflow_artifacts  # noqa: E402
from grn_agent.ui.eval_view import metric_warnings, metrics_frame, pr_series, roc_series  # noqa: E402
from grn_agent.ui.interpret import attention_for_edges, evidence_pca_frame, latent_pca_for_edges  # noqa: E402
from grn_agent.ui.literature import load_literature_cards  # noqa: E402
from grn_agent.ui.readme_view import how_evigrn_works, overview_figure_path, project_github_url  # noqa: E402
from grn_agent.ui.run_page import render_run_page  # noqa: E402
from grn_agent.ui.runner import launch_integrated_workflow  # noqa: E402
from grn_agent.ui.theme import inject_theme  # noqa: E402
from grn_agent.ui.viz import (  # noqa: E402
    attention_figure,
    coverage_figure,
    evidence_factor_figure,
    load_tf_window,
    metrics_bar_figure,
    network_figure,
    pca_figure,
    pr_curves_figure,
    roc_curves_figure,
    tf_evidence_graph_figure,
)
from grn_agent.ui.workflow_view import derive_handoffs, load_workflow_runtime, summarize_stages  # noqa: E402


st.set_page_config(
    page_title="EviGRN",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


def _repo_root() -> Path:
    return ROOT


def _safe_read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def _chips(parts: list[str]) -> None:
    body = "".join(f'<span class="metric-chip">{part}</span>' for part in parts)
    st.markdown(f'<div class="chip-row">{body}</div>', unsafe_allow_html=True)


def _hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero-banner"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def page_overview(artifact_root: Path) -> None:
    root = _repo_root()
    _hero(
        "EviGRN",
        "Evidence-aware gene regulatory inference. This page shows the method overview, "
        "how EviGRN works, and the GitHub repository.",
    )
    github = project_github_url(root)
    if github:
        st.markdown(
            f'<a class="github-link" href="{github}" target="_blank" rel="noopener noreferrer">GitHub repository</a>',
            unsafe_allow_html=True,
        )
    else:
        st.info("No GitHub remote was found for this checkout.")

    st.markdown("### EviGRN overview")
    figure = overview_figure_path(root)
    if figure is None:
        st.info("No EviGRN overview figure was found under docs/.")
    else:
        st.image(str(figure), width="stretch")

    readme_path = root / "README.md"
    st.markdown("### How EviGRN works")
    if readme_path.is_file():
        readme_text = readme_path.read_text(encoding="utf-8")
        explanation = how_evigrn_works(readme_text)
        if explanation:
            st.markdown(explanation)
        else:
            st.info("README.md does not contain a method explanation before the install section.")
        with st.expander("Full README"):
            st.markdown(readme_text)
    else:
        st.info("README.md was not found in the repository root.")

    conf_dir = _repo_root() / "conf"
    templates = sorted(conf_dir.rglob("*.yml")) if conf_dir.is_dir() else []
    st.markdown("### Workspace")
    st.write(f"Repository: `{_repo_root()}`")
    st.write(f"Artifact root: `{artifact_root}`")
    st.markdown("### Launch existing integrated workflow")
    st.caption("This calls the existing script unchanged. Use for new runs; explore finished artifacts below.")
    options = [str(p.relative_to(_repo_root())) for p in templates]
    default_idx = 0
    for i, opt in enumerate(options):
        if "tf_eager_integrated_standard.yml" in opt:
            default_idx = i
            break
    chosen = st.selectbox("YAML config", options, index=default_idx if options else 0)
    force = st.checkbox("Force recompute", value=False)
    if st.button("Start integrated workflow", type="primary"):
        try:
            info = launch_integrated_workflow(_repo_root() / chosen, force_recompute=force)
            st.success(f"Started PID {info['pid']}: {' '.join(info['cmd'])}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not start workflow: {exc}")
    st.markdown("### Install GUI extras")
    st.code('pip install -e ".[gui]"\nstreamlit run app.py', language="bash")


def page_acquisition(arts: dict) -> None:
    _hero("Acquisition Trace", "Coverage, QC decisions, and plain-language multimodal package narrative.")
    manifest_path = arts.get("multimodal_manifest")
    if manifest_path is None:
        st.warning("No multimodal_manifest.json found in this workflow. Run acquisition or choose another workspace.")
        return
    manifest = load_json(manifest_path)
    narrative = format_acquisition_narrative(manifest)
    st.markdown(f'<div class="rationale-box">{narrative}</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    acc = manifest.get("accessibility") or {}
    motifs = manifest.get("motifs") or {}
    qc = manifest.get("qc") or {}
    c1.metric("Species", str(manifest.get("species", "—")))
    c2.metric("Cell type", str(manifest.get("cell_type", "—")))
    c3.metric("QC pass", "yes" if qc.get("pass") else "no")

    st.markdown("### Selected accessibility package")
    _chips(
        [
            f"source: {acc.get('source')}",
            f"assay: {acc.get('assay')}",
            f"accession: {acc.get('accession')}",
            f"coverage: {acc.get('promoter_coverage_of_rnaseq_genes')}",
            f"motifs: {motifs.get('tf_motif_count')}",
        ]
    )
    try:
        st.plotly_chart(coverage_figure(manifest), use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        st.info(f"Coverage chart unavailable: {exc}")

    reasons = qc.get("rejection_reasons") or []
    warnings = qc.get("warnings") or []
    if reasons:
        st.markdown("### Rejection / fail reasons")
        for reason in reasons:
            st.markdown(f"- {reason}")
    if warnings:
        st.markdown("### Warnings")
        for warning in warnings:
            st.markdown(f"- {warning}")
    with st.expander("Raw multimodal manifest"):
        st.json(manifest)


def page_workflow(arts: dict) -> None:
    _hero("Workflow Trace", "Stage status and artifact handoffs from existing workflow_runtime.json.")
    runtime_path = arts.get("runtime")
    if runtime_path is None:
        st.warning("No workflow_runtime.json found for this workspace.")
        return
    runtime = load_workflow_runtime(runtime_path)
    _chips(
        [
            f"status: {runtime.get('status')}",
            f"workflow: {runtime.get('workflow_id')}",
            f"elapsed: {runtime.get('total_elapsed_seconds')} s",
        ]
    )
    stages = summarize_stages(runtime)
    if stages:
        st.dataframe(pd.DataFrame(stages).drop(columns=["outputs"], errors="ignore"), use_container_width=True)
    handoffs = derive_handoffs(runtime)
    st.markdown("### Handoffs")
    if not handoffs:
        st.info("No handoff records found.")
    else:
        for item in handoffs:
            st.markdown(f"- **{item['stage']}** · `{item['event']}` — {item['detail']}")


def _edge_signature(edges: list[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(tf).strip().upper(), str(gene).strip().upper()) for tf, gene in edges))


def _render_edge_actions(arts: dict, edges: list[tuple[str, str]], key_prefix: str) -> None:
    """Buttons that draw attention or PCA for the edges currently selected."""
    st.markdown("### Attention and PCA")
    st.caption(
        "Attention reads the workflow checkpoint and the TF window for each selected edge. "
        "PCA uses those final decoder states when a checkpoint and windows file are available; "
        "otherwise it uses numeric evidence columns on the scored edges."
    )
    signature = _edge_signature(edges)
    state_key = f"{key_prefix}_viz"
    left, right = st.columns(2)
    run_attention = left.button(
        "Attention for selected edges",
        key=f"{key_prefix}_attn",
        disabled=not edges,
        use_container_width=True,
    )
    run_pca = right.button(
        "PCA visualization",
        key=f"{key_prefix}_pca",
        disabled=not edges,
        use_container_width=True,
    )
    checkpoint = arts.get("checkpoint")
    windows = arts.get("windows")
    scored = _safe_read_csv(arts.get("scored_edges"))

    if run_attention:
        if checkpoint is None or windows is None:
            st.session_state[state_key] = {
                "mode": "error",
                "sig": signature,
                "message": "Attention needs a TF-EAGER checkpoint and test_windows.jsonl in this workflow.",
            }
        else:
            try:
                payload = attention_for_edges(
                    checkpoint=checkpoint,
                    windows_path=windows,
                    edges=list(signature),
                )
                st.session_state[state_key] = {"mode": "attention", "sig": signature, "payload": payload}
            except Exception as exc:  # noqa: BLE001
                st.session_state[state_key] = {"mode": "error", "sig": signature, "message": str(exc)}

    if run_pca:
        payload = None
        latent_error = None
        if checkpoint is not None and windows is not None:
            try:
                payload = latent_pca_for_edges(
                    checkpoint=checkpoint,
                    windows_path=windows,
                    edges=list(signature),
                    scored=scored,
                )
            except Exception as exc:  # noqa: BLE001
                latent_error = str(exc)
        if payload is None:
            try:
                payload = evidence_pca_frame(scored, list(signature))
                if latent_error:
                    payload["note"] = (
                        "PCA of scored-edge evidence features. "
                        f"Model latent PCA was not used: {latent_error}"
                    )
                elif checkpoint is None or windows is None:
                    payload["note"] = (
                        "PCA of scored-edge evidence features. "
                        "No checkpoint and window file were both available for model latents."
                    )
                st.session_state[state_key] = {"mode": "pca", "sig": signature, "payload": payload}
            except Exception as exc:  # noqa: BLE001
                message = str(exc)
                if latent_error:
                    message = f"{message} Model latent PCA: {latent_error}"
                st.session_state[state_key] = {"mode": "error", "sig": signature, "message": message}
        else:
            st.session_state[state_key] = {"mode": "pca", "sig": signature, "payload": payload}

    stored = st.session_state.get(state_key)
    if not stored or stored.get("sig") != signature:
        return
    if stored.get("mode") == "error":
        st.warning(str(stored.get("message")))
        return
    payload = stored.get("payload") or {}
    st.caption(str(payload.get("note") or ""))
    if stored.get("mode") == "attention":
        channels = payload.get("channels")
        local = payload.get("local")
        if channels is None or channels.empty:
            st.info("Attention ran, but no channel masses were returned.")
            return
        try:
            st.plotly_chart(attention_figure(channels, local), use_container_width=True)
        except Exception as exc:  # noqa: BLE001
            st.warning(str(exc))
        st.dataframe(channels, use_container_width=True, hide_index=True)
        return
    frame = payload.get("frame")
    variance = payload.get("variance") or (0.0, 0.0)
    if frame is None or frame.empty:
        st.info("PCA did not return coordinates.")
        return
    kind = "Model latent PCA" if payload.get("kind") == "model_latent" else "Evidence-feature PCA"
    try:
        st.plotly_chart(
            pca_figure(frame, (float(variance[0]), float(variance[1])), title=kind),
            use_container_width=True,
        )
    except Exception as exc:  # noqa: BLE001
        st.warning(str(exc))
    preview = frame.sort_values("selected", ascending=False).head(40)
    st.dataframe(preview, use_container_width=True, hide_index=True)


def page_explainer(arts: dict) -> None:
    _hero(
        "Prediction Explainer",
        "Quantitative evidence factors and deterministic scientific rationale for EviGRN predictions. "
        "Literature cards are post-hoc and do not change model scores.",
    )
    scored_path = arts.get("scored_edges")
    if scored_path is None:
        st.warning("No scored edges CSV found. Run inference or select another workflow.")
        return
    df = _safe_read_csv(scored_path)
    if df.empty:
        st.warning("Scored edges file is empty.")
        return

    tfs = sorted(df["source_tf"].astype(str).unique().tolist())
    selected_tf = st.selectbox("Transcription factor", tfs)
    sub = df[df["source_tf"].astype(str) == selected_tf].sort_values("p_present", ascending=False)
    genes = sub["target_gene"].astype(str).tolist()
    selected_gene = st.selectbox("Target gene", genes)
    row = sub[sub["target_gene"].astype(str) == selected_gene].iloc[0].to_dict()
    explanation = build_edge_explanation(row)

    m1, m2, m3 = st.columns(3)
    m1.metric("p_present", f'{explanation["p_present"]:.3f}')
    m2.metric("Strength", explanation["strength"])
    m3.metric("Cell type", str(explanation.get("cell_type") or "—"))
    missing = ", ".join(explanation["missing_modalities"]) or "none"
    _chips([f"Missing modalities: {missing}"])

    st.markdown("### Scientific rationale")
    st.markdown(f'<div class="rationale-box">{explanation["mechanism_reasoning"]}</div>', unsafe_allow_html=True)
    try:
        st.plotly_chart(evidence_factor_figure(explanation), use_container_width=True)
    except Exception as exc:  # noqa: BLE001
        st.info(f"Factor chart unavailable: {exc}")

    st.markdown("### Evidence table")
    st.dataframe(pd.DataFrame(explanation["factors"]), use_container_width=True, hide_index=True)
    _render_edge_actions(arts, [(selected_tf, selected_gene)], "explainer")

    st.markdown("### Literature audit")
    lit_df = _safe_read_csv(arts.get("literature_validated"))
    lit_row = None
    if not lit_df.empty and {"source_tf", "target_gene"}.issubset(lit_df.columns):
        hits = lit_df[
            (lit_df["source_tf"].astype(str) == selected_tf)
            & (lit_df["target_gene"].astype(str) == selected_gene)
        ]
        if not hits.empty:
            lit_row = hits.iloc[0].to_dict()
    cards = load_literature_cards(
        source_tf=selected_tf,
        target_gene=selected_gene,
        literature_dir=arts.get("literature_dir"),
        lit_row=lit_row,
    )
    st.caption(cards["note"])
    if cards.get("lit_score") is not None:
        _chips(
            [
                f'lit_score: {cards["lit_score"]}',
                f'supporting: {cards["n_supporting"]} / {cards["n_papers"]}',
                f'conflict: {cards["conflict_detected"]}',
            ]
        )
    if not cards["cards"]:
        st.info("No literature classifications found for this edge. Run literature validation to populate claim cards.")
    for card in cards["cards"]:
        support = "supporting" if card["supports"] else "not supporting"
        link = card.get("pubmed_url") or ""
        pmid = card.get("pmid") or "unknown"
        quote = card.get("evidence_sentence") or ""
        st.markdown(
            f'<div class="lit-card"><div><strong>{support}</strong> · {card.get("evidence_type")} · '
            f'{card.get("relationship")} · conf={card.get("confidence")}</div>'
            f'<div><a href="{link}" target="_blank" rel="noopener noreferrer">PubMed {pmid}</a></div>'
            f'<div class="quote">{quote}</div></div>',
            unsafe_allow_html=True,
        )


def page_visuals(arts: dict) -> None:
    _hero("Visual Explorer", "Coverage, TF-evidence neighborhoods, and thresholded final networks.")
    tab1, tab2, tab3 = st.tabs(["Coverage", "TF evidence graph", "Final network"])

    with tab1:
        manifest_path = arts.get("multimodal_manifest")
        if manifest_path is None:
            st.info("No multimodal manifest available.")
        else:
            try:
                st.plotly_chart(coverage_figure(load_json(manifest_path)), use_container_width=True)
            except Exception as exc:  # noqa: BLE001
                st.warning(str(exc))

    with tab2:
        windows_path = arts.get("windows")
        scored = _safe_read_csv(arts.get("scored_edges"))
        if windows_path is None:
            st.info("No test_windows.jsonl found.")
        else:
            default_tf = None
            if not scored.empty:
                default_tf = str(scored.sort_values("p_present", ascending=False).iloc[0]["source_tf"])
            tf = st.text_input("TF symbol for neighborhood graph", value=default_tf or "SPI1")
            top_n = st.slider("Max genes in neighborhood", min_value=10, max_value=100, value=40, step=5)
            window = load_tf_window(windows_path, tf)
            if window is None:
                st.warning(f"No window found for TF={tf} in the first scanned records.")
            else:
                try:
                    st.plotly_chart(tf_evidence_graph_figure(window, top_n=top_n), use_container_width=True)
                except Exception as exc:  # noqa: BLE001
                    st.warning(str(exc))

    with tab3:
        net = _safe_read_csv(arts.get("network"))
        if net.empty:
            net = _safe_read_csv(arts.get("scored_edges"))
        if net.empty:
            st.info("No network / scored edges available.")
        else:
            threshold = st.slider("Probability threshold", 0.0, 1.0, 0.5, 0.05)
            max_edges = st.slider("Max edges drawn", 20, 500, 200, 20)
            try:
                st.plotly_chart(network_figure(net, threshold=threshold, max_edges=max_edges), use_container_width=True)
            except Exception as exc:  # noqa: BLE001
                st.warning(str(exc))
            preview = net.sort_values("p_present", ascending=False).head(100)
            st.dataframe(preview, use_container_width=True, hide_index=True)
            labels = [
                f"{row.source_tf} → {row.target_gene}"
                for row in preview.itertuples(index=False)
            ]
            chosen_labels = st.multiselect(
                "Edges for attention or PCA",
                labels,
                default=labels[:1] if labels else [],
            )
            if len(chosen_labels) > 12:
                st.caption("Only the first 12 selected edges are sent to attention or PCA.")
                chosen_labels = chosen_labels[:12]
            chosen_edges = []
            for label in chosen_labels:
                tf, gene = label.split(" → ", 1)
                chosen_edges.append((tf, gene))
            _render_edge_actions(arts, chosen_edges, "network")


def page_results(arts: dict) -> None:
    _hero("Results", "AUROC, AUPRC, early precision, and ROC or PR curves when the report stores them.")
    report_path = arts.get("eval_report")
    if report_path is None:
        st.info("No evaluation report found for this workflow.")
    else:
        payload = load_json(report_path)
        frame = metrics_frame(payload)
        st.markdown("### Ranking metrics")
        if frame.empty:
            st.warning("This report does not contain AUROC, AUPRC, or early-precision fields.")
        else:
            st.caption("EP@k is precision among the top k scored edges.")
            display = frame.drop(columns=["negative_ratio"], errors="ignore")
            st.dataframe(display, use_container_width=True, hide_index=True)
            try:
                st.plotly_chart(metrics_bar_figure(frame), use_container_width=True)
            except Exception as exc:  # noqa: BLE001
                st.info(f"Metric chart unavailable: {exc}")
        for note in metric_warnings(payload):
            st.warning(note)

        st.markdown("### Precision-recall")
        curves = pr_series(payload)
        if curves:
            try:
                st.plotly_chart(pr_curves_figure(curves), use_container_width=True)
            except Exception as exc:  # noqa: BLE001
                st.info(f"PR chart unavailable: {exc}")
        else:
            pr_image = arts.get("pr_curve")
            if pr_image is not None and Path(pr_image).is_file():
                st.image(str(pr_image), use_container_width=True)
            else:
                st.info("No precision-recall curve is stored for this report.")

        st.markdown("### ROC")
        roc = roc_series(payload)
        if roc:
            try:
                st.plotly_chart(roc_curves_figure(roc), use_container_width=True)
            except Exception as exc:  # noqa: BLE001
                st.info(f"ROC chart unavailable: {exc}")
        else:
            roc_image = arts.get("roc_curve")
            if roc_image is not None and Path(roc_image).is_file():
                st.image(str(roc_image), use_container_width=True)
            else:
                st.info("This report does not include ROC curve coordinates. AUROC is still listed when present.")

    scored = _safe_read_csv(arts.get("scored_edges"))
    if not scored.empty:
        st.markdown("### Top scored edges")
        st.dataframe(
            scored.sort_values("p_present", ascending=False).head(50),
            use_container_width=True,
            hide_index=True,
        )
    optional = {"literature_dir", "checkpoint", "roc_curve", "workflow_dir"}
    missing = [k for k, v in arts.items() if v is None and k not in optional]
    if missing:
        st.caption("Missing artifacts: " + ", ".join(missing))


def main() -> None:
    inject_theme()
    artifact_root = _repo_root() / "artifacts"
    with st.sidebar:
        st.markdown("## EviGRN")
        st.caption("Scientific explorer")
        page = st.radio(
            "Navigate",
            [
                "Overview",
                "Run inference",
                "Acquisition Trace",
                "Workflow Trace",
                "Prediction Explainer",
                "Visual Explorer",
                "Results",
            ],
        )
        active_run = st.session_state.get("gui_run_dir")
        workflow_dir = Path(active_run) if active_run else artifact_root
        if active_run:
            st.caption(f"Active run: {active_run}")
        with st.expander("Open a finished run"):
            workflows = discover_workflow_dirs(artifact_root)
            labels = [str(p.relative_to(artifact_root)) for p in workflows] if workflows else []
            if labels:
                chosen_label = st.selectbox("Finished run", labels, index=0)
                if st.button("Use this run"):
                    st.session_state["gui_run_dir"] = str(artifact_root / chosen_label)
                    workflow_dir = artifact_root / chosen_label
            else:
                st.caption("No finished runs under artifacts/ yet.")
        st.markdown('<p class="statusbar">Mechanistic text is deterministic from evidence fields. Literature is post-hoc.</p>', unsafe_allow_html=True)

    arts = resolve_workflow_artifacts(workflow_dir)
    if page == "Overview":
        page_overview(artifact_root)
    elif page == "Run inference":
        render_run_page(_repo_root())
    elif page == "Acquisition Trace":
        page_acquisition(arts)
    elif page == "Workflow Trace":
        page_workflow(arts)
    elif page == "Prediction Explainer":
        page_explainer(arts)
    elif page == "Visual Explorer":
        page_visuals(arts)
    else:
        page_results(arts)


if __name__ == "__main__":
    main()
