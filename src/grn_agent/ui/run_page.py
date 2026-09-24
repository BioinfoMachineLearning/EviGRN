"""Streamlit page that runs EviGRN inference from user files."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from grn_agent.explain.reasoning import format_acquisition_narrative
from grn_agent.ui.artifacts import load_json
from grn_agent.ui.eval_view import metrics_frame, pr_series
from grn_agent.ui.inference_job import (
    acquire_harmonize_command,
    acquire_rank_command,
    discover_example_inputs,
    literature_command,
    load_ranked_candidates,
    safe_dataset_id,
    write_inference_config,
)
from grn_agent.ui.runner import launch_logged
from grn_agent.ui.viz import metrics_bar_figure, pr_curves_figure


def _store_upload(uploaded: Any, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(uploaded.getvalue())
    return dest


def _resolve_input(path_text: str, uploaded: Any, dest: Path, repo_root: Path) -> Path | None:
    text = str(path_text or "").strip()
    if text:
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = repo_root / path
        if path.is_file():
            return path
    if uploaded is not None:
        return _store_upload(uploaded, dest)
    return None


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def _tail(path: str | Path | None, n: int = 40) -> str:
    if not path:
        return ""
    file_path = Path(path)
    if not file_path.is_file():
        return ""
    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def _job_status(st: Any, key: str) -> None:
    job = st.session_state.get(key) or {}
    if not job:
        return
    alive = _pid_alive(job.get("pid"))
    label = "running" if alive else "finished"
    st.caption(f"{key.replace('_', ' ')}: {label} (pid {job.get('pid')})")
    log = _tail(job.get("log"))
    if log:
        st.code(log, language="text")


def _list_checkpoints(repo_root: Path) -> list[Path]:
    """Training checkpoints. Multicontext artifact paths are not listed."""
    trainings = repo_root / "trainings"
    if not trainings.is_dir():
        return []
    matches = [path for path in trainings.rglob("*.pt") if path.is_file() and "multicontext" not in path.as_posix()]
    unique = {path.resolve(): path for path in matches}

    def _order(path: Path) -> tuple[int, str]:
        if path.name == "tf_eager_bootstrap_v2.pt":
            return (0, path.as_posix())
        if "no_tf_identity" in path.name:
            return (1, path.as_posix())
        return (2, path.name)

    return sorted(unique.values(), key=_order)


def render_run_page(repo_root: Path) -> None:
    import streamlit as st

    st.markdown(
        '<div class="hero-banner"><h1>Run EviGRN</h1>'
        "<p>Provide expression, a TF list, and optional gold edges. "
        "Rank accessibility candidates, choose one or enter an accession, "
        "then run QC, inference, gold-standard evaluation, and literature verification.</p></div>",
        unsafe_allow_html=True,
    )
    runs_root = repo_root / "artifacts" / "gui_runs"
    template = repo_root / "conf" / "tf_eager_integrated_standard.yml"
    acquire_script = repo_root / "scripts" / "acquire_multimodal_data.py"
    literature_script = repo_root / "scripts" / "run_literature_validation.py"

    example = discover_example_inputs(repo_root)
    st.markdown("### 1. Expression, TFs, and context")
    st.caption(
        "Expression is a genes-by-cells table. The TF file is a symbol list. "
        "A gold network is optional and is used only for evaluation."
    )
    if example:
        st.caption(f"Sample files from `{example['config']}`.")
        if st.button("Load example dataset"):
            st.session_state["expr_path"] = example["expression"]
            st.session_state["tf_path"] = example["tf_file"]
            st.session_state["gold_path"] = example["gold"]
            st.session_state["species"] = example["species"]
            st.session_state["cell_type"] = example["cell_type"]
            st.session_state["cell_line"] = example["cell_line"]
            st.session_state["cell_context"] = example["cell_context"]
            st.session_state["dataset_name"] = example["dataset_id"]
            if example["checkpoint"]:
                st.session_state["checkpoint_choice"] = "Enter a path"
                st.session_state["checkpoint_path"] = example["checkpoint"]
            st.rerun()
    else:
        st.caption("No local example expression and TF list were found under this checkout.")
    c1, c2 = st.columns(2)
    expr_path = c1.text_input(
        "Expression matrix path",
        key="expr_path",
        placeholder=(example or {}).get("expression") or "ExpressionData.csv",
    )
    expr_file = c2.file_uploader("Or upload expression CSV", type=["csv", "tsv", "txt"], key="expr_file")
    tf_path = c1.text_input(
        "TF list path",
        key="tf_path",
        placeholder=(example or {}).get("tf_file") or "TFs.csv",
    )
    tf_file = c2.file_uploader("Or upload TF list", type=["csv", "tsv", "txt"], key="tf_file")
    gold_path = c1.text_input(
        "Gold network path (optional)",
        key="gold_path",
        placeholder=(example or {}).get("gold") or "refNetwork.csv",
    )
    gold_file = c2.file_uploader("Or upload gold network", type=["csv", "tsv", "txt"], key="gold_file")
    species = st.text_input(
        "Species",
        key="species",
        placeholder=(example or {}).get("species") or "mouse",
        help="Examples: human, mouse, drosophila",
    )
    d1, d2, d3 = st.columns(3)
    cell_type = d1.text_input(
        "Cell type",
        key="cell_type",
        placeholder=example["cell_type"] if example else "embryonic stem cell",
    )
    cell_line = d2.text_input(
        "Cell line",
        key="cell_line",
        placeholder=example["cell_line"] if example else "",
    )
    cell_context = d3.text_input(
        "Cell context",
        key="cell_context",
        placeholder=example["cell_context"] if example else "culture condition or time point",
    )
    dataset_name = st.text_input(
        "Run name",
        key="dataset_name",
        placeholder=(example or {}).get("dataset_id") or "example_dataset",
        help="Used as the output folder name.",
    )

    def _collect() -> dict[str, Any] | None:
        name = safe_dataset_id(dataset_name or Path(expr_path or "expression").stem)
        run_dir = runs_root / name
        inputs = run_dir / "inputs"
        expression = _resolve_input(expr_path, expr_file, inputs / "ExpressionData.csv", repo_root)
        tfs = _resolve_input(tf_path, tf_file, inputs / "TFs.csv", repo_root)
        gold = _resolve_input(gold_path, gold_file, inputs / "gold_network.csv", repo_root)
        if expression is None or tfs is None:
            st.error("Expression matrix and TF list are required.")
            return None
        if not species.strip():
            st.error("Species is required.")
            return None
        run_dir.mkdir(parents=True, exist_ok=True)
        st.session_state["gui_run_dir"] = str(run_dir)
        return {
            "run_dir": run_dir,
            "dataset_id": name,
            "expression": expression,
            "tf_file": tfs,
            "gold": gold,
            "species": species.strip(),
            "cell_type": cell_type.strip(),
            "cell_line": cell_line.strip(),
            "cell_context": cell_context.strip(),
        }

    st.markdown("### 2. Ranked accessibility QC")
    st.caption("Ranking queries ENCODE and GEO, writes the top candidates, and does not download peak files.")
    if st.button("Rank accessibility candidates", type="primary"):
        collected = _collect()
        if collected is not None:
            run_dir = collected["run_dir"]
            ranked_json = run_dir / "acquisition" / "ranked_accessibility.json"
            manifest = run_dir / "acquisition" / "multimodal_manifest.json"
            cmd = acquire_rank_command(
                python=sys.executable,
                script=acquire_script,
                expression=collected["expression"],
                species=collected["species"],
                tf_file=collected["tf_file"],
                out_manifest=manifest,
                ranked_json=ranked_json,
                dataset_id=collected["dataset_id"],
                cell_type=collected["cell_type"],
                cell_line=collected["cell_line"],
                cell_context=collected["cell_context"],
                gold=collected["gold"],
                cache_dir=run_dir / ".cache",
            )
            st.session_state["rank_job"] = launch_logged(cmd, run_dir / "rank.log")
    _job_status(st, "rank_job")

    run_dir_text = st.session_state.get("gui_run_dir")
    ranked_rows: list[dict[str, Any]] = []
    if run_dir_text:
        ranked_rows = load_ranked_candidates(Path(run_dir_text) / "acquisition" / "ranked_accessibility.json")
    if ranked_rows:
        st.dataframe(pd.DataFrame(ranked_rows), use_container_width=True, hide_index=True)
        labels = [f"{row.get('rank')}. {row.get('accession')} ({row.get('assay')}, {row.get('tier')})" for row in ranked_rows]
        chosen_label = st.radio("Top ranked candidates", labels, index=0)
        chosen_accession = str(ranked_rows[labels.index(chosen_label)].get("accession") or "")
    else:
        chosen_accession = ""
        if run_dir_text and not _pid_alive((st.session_state.get("rank_job") or {}).get("pid")):
            st.info("No ranked accessibility candidates yet. Rank candidates, or enter an accession below.")
    manual_accession = st.text_input(
        "ATAC / DNase accession",
        key="manual_accession",
        placeholder="ENCSR... or GSE.../GSM...",
        help="ENCODE or GEO accession. If filled, this overrides the selected ranked candidate.",
    )
    skip_motif = st.checkbox("Skip motif scan during harmonization", value=False)

    st.markdown("### 3. QC and harmonization")
    if st.button("Run QC and harmonization"):
        collected = _collect()
        if collected is not None:
            run_dir = collected["run_dir"]
            accession = manual_accession.strip() or chosen_accession
            cmd = acquire_harmonize_command(
                python=sys.executable,
                script=acquire_script,
                expression=collected["expression"],
                species=collected["species"],
                tf_file=collected["tf_file"],
                out_manifest=run_dir / "acquisition" / "multimodal_manifest.json",
                dataset_id=collected["dataset_id"],
                cell_type=collected["cell_type"],
                cell_line=collected["cell_line"],
                cell_context=collected["cell_context"],
                gold=collected["gold"],
                cache_dir=run_dir / ".cache",
                atac_accession=accession,
                skip_motif=skip_motif,
            )
            st.session_state["qc_job"] = launch_logged(cmd, run_dir / "qc.log")
    _job_status(st, "qc_job")
    if run_dir_text:
        manifest_path = Path(run_dir_text) / "acquisition" / "multimodal_manifest.json"
        if manifest_path.is_file():
            manifest = load_json(manifest_path)
            st.markdown(f'<div class="rationale-box">{format_acquisition_narrative(manifest)}</div>', unsafe_allow_html=True)
            qc = manifest.get("qc") or {}
            for reason in qc.get("rejection_reasons") or []:
                st.warning(str(reason))
            for warning in qc.get("warnings") or []:
                st.info(str(warning))

    st.markdown("### 4. Inference and gold-standard evaluation")
    st.caption("Inference uses a trained checkpoint. Gold-standard AUROC, AUPRC, and EP are computed only when a gold network was provided.")
    checkpoints = _list_checkpoints(repo_root)
    checkpoint_labels = ["Enter a path"] + [
        path.relative_to(repo_root).as_posix() if path.is_relative_to(repo_root) else path.as_posix()
        for path in checkpoints
    ]
    checkpoint_choice = st.selectbox("Checkpoint", checkpoint_labels, key="checkpoint_choice")
    if checkpoint_choice == "Enter a path":
        checkpoint_text = st.text_input(
            "Checkpoint path",
            key="checkpoint_path",
            placeholder=(example or {}).get("checkpoint") or "tf_eager.pt",
        )
    else:
        checkpoint_text = checkpoint_choice
        st.caption(checkpoint_choice)
    device = st.selectbox("Inference device", ["cpu", "cuda"], index=0)
    if st.button("Run inference", type="primary"):
        collected = _collect()
        checkpoint = Path(checkpoint_text).expanduser() if checkpoint_text.strip() else None
        if checkpoint is not None and not checkpoint.is_file():
            rooted = repo_root / checkpoint
            checkpoint = rooted if rooted.is_file() else checkpoint
        if collected is None:
            pass
        elif checkpoint is None or not checkpoint.is_file():
            st.error("A trained checkpoint file is required for inference.")
        elif not template.is_file():
            st.error(f"Workflow template not found: {template}")
        else:
            run_dir = collected["run_dir"]
            manifest = run_dir / "acquisition" / "multimodal_manifest.json"
            if not manifest.is_file():
                st.error("Run QC and harmonization first so the multimodal manifest exists.")
            else:
                config_path = write_inference_config(
                    template,
                    run_dir / "gui_inference.yml",
                    run_dir=run_dir,
                    dataset_id=collected["dataset_id"],
                    expression=collected["expression"],
                    tf_file=collected["tf_file"],
                    species=collected["species"],
                    checkpoint=checkpoint,
                    cell_type=collected["cell_type"],
                    cell_line=collected["cell_line"],
                    cell_context=collected["cell_context"],
                    gold=collected["gold"],
                    manifest=manifest,
                    device=device,
                )
                cmd = [sys.executable, str(repo_root / "scripts" / "run_integrated_tf_eager_workflow.py"), "--config", str(config_path)]
                st.session_state["infer_job"] = launch_logged(cmd, run_dir / "infer.log")
    _job_status(st, "infer_job")

    if run_dir_text:
        report = Path(run_dir_text) / "evaluation" / "eval_test_by_ratio.json"
        scored = Path(run_dir_text) / "tf_eager" / "test_scored_edges.csv"
        st.markdown("### Results")
        if report.is_file():
            frame = metrics_frame(load_json(report))
            if frame.empty:
                st.info("The evaluation report has no AUROC, AUPRC, or EP fields.")
            else:
                st.caption("Gold-standard evaluation. EP@k is precision among the top k scored edges.")
                st.dataframe(frame.drop(columns=["negative_ratio"], errors="ignore"), use_container_width=True, hide_index=True)
                try:
                    st.plotly_chart(metrics_bar_figure(frame), use_container_width=True)
                    curves = pr_series(load_json(report))
                    if curves:
                        st.plotly_chart(pr_curves_figure(curves), use_container_width=True)
                except Exception as exc:  # noqa: BLE001
                    st.info(f"Result chart unavailable: {exc}")
        elif scored.is_file():
            st.info("Scored edges are ready. Gold-standard evaluation was skipped because no gold network was provided.")
        if scored.is_file():
            edges = pd.read_csv(scored).sort_values("p_present", ascending=False).head(50)
            st.dataframe(edges, use_container_width=True, hide_index=True)

    st.markdown("### 5. Literature verification")
    st.caption("Post hoc PubMed check. The API key is passed only to this run and is not written into the repository.")
    email = st.text_input("NCBI email", key="ncbi_email", placeholder="name@institution.edu")
    api_key = st.text_input("NCBI API key", type="password", key="ncbi_key", placeholder="NCBI API key")
    lit_limit = st.number_input("Maximum edges to verify", min_value=1, max_value=500, value=20, step=1)
    if st.button("Run literature verification"):
        if not run_dir_text:
            st.error("Start a run before literature verification.")
        else:
            scored = Path(run_dir_text) / "tf_eager" / "test_scored_edges.csv"
            if not scored.is_file():
                st.error("Inference has not written scored edges yet.")
            elif not email.strip():
                st.error("NCBI requires a contact email.")
            else:
                output = Path(run_dir_text) / "literature_validated.csv"
                cmd = literature_command(
                    python=sys.executable,
                    script=literature_script,
                    scored_csv=scored,
                    output=output,
                    cell_type=cell_type.strip(),
                    limit=int(lit_limit),
                )
                st.session_state["lit_job"] = launch_logged(
                    cmd,
                    Path(run_dir_text) / "literature.log",
                    env={"NCBI_EMAIL": email.strip(), "NCBI_API_KEY": api_key.strip()},
                )
    _job_status(st, "lit_job")
    if run_dir_text:
        lit_csv = Path(run_dir_text) / "literature_validated.csv"
        if lit_csv.is_file():
            st.dataframe(pd.read_csv(lit_csv).head(50), use_container_width=True, hide_index=True)
