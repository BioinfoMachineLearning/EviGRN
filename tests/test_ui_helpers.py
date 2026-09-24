from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from grn_agent.ui.artifacts import resolve_workflow_artifacts
from grn_agent.ui.eval_view import metrics_frame, pr_series, roc_series
from grn_agent.ui.inference_job import (
    acquire_rank_command,
    discover_example_inputs,
    load_ranked_candidates,
    prepare_inference_config,
    safe_dataset_id,
)
from grn_agent.ui.interpret import attention_frame, discover_checkpoint, evidence_pca_frame, pca_scores
from grn_agent.ui.literature import load_literature_cards, pubmed_url
from grn_agent.ui.readme_view import how_evigrn_works, normalize_github_remote, overview_figure_path, project_github_url
from grn_agent.ui.viz import attention_figure, metrics_bar_figure, pr_curves_figure, roc_curves_figure
from grn_agent.ui.workflow_view import derive_handoffs, summarize_stages


def test_pubmed_url():
    assert pubmed_url("12345") == "https://pubmed.ncbi.nlm.nih.gov/12345/"


def test_resolve_workflow_artifacts(tmp_path: Path):
    wf = tmp_path / "demo"
    (wf / "tf_eager").mkdir(parents=True)
    (wf / "acquisition").mkdir(parents=True)
    (wf / "workflow_runtime.json").write_text("{}", encoding="utf-8")
    (wf / "tf_eager" / "test_scored_edges.csv").write_text("source_tf,target_gene,p_present\nA,B,0.9\n", encoding="utf-8")
    (wf / "acquisition" / "multimodal_manifest.json").write_text("{}", encoding="utf-8")
    arts = resolve_workflow_artifacts(wf)
    assert arts["runtime"] is not None
    assert arts["scored_edges"] is not None
    assert arts["multimodal_manifest"] is not None


def test_workflow_handoffs_from_runtime():
    runtime = {
        "stages": {
            "acquisition": {
                "status": "ran",
                "started_at": "2026-01-01T00:00:00+00:00",
                "outputs": {"out_manifest": "artifacts/x/multimodal_manifest.json"},
            },
            "infer_tf_eager": {
                "status": "skipped_reuse",
                "started_at": "2026-01-01T00:01:00+00:00",
                "reuse_path": "artifacts/x/tf_eager.pt",
                "outputs": {"scored_csv": "artifacts/x/test_scored_edges.csv"},
            },
        }
    }
    stages = summarize_stages(runtime)
    assert [s["stage"] for s in stages] == ["acquisition", "infer_tf_eager"]
    handoffs = derive_handoffs(runtime)
    assert any(h["event"] == "reuse" for h in handoffs)
    assert any("scored_csv" in h["detail"] for h in handoffs)


def test_literature_cards_from_json(tmp_path: Path):
    lit_dir = tmp_path / "literature_classifications"
    lit_dir.mkdir()
    payload = [
        {
            "pmid": "999",
            "effective_support": True,
            "evidence_type": "ChIP-seq",
            "relationship": "activates",
            "confidence": 0.9,
            "evidence_sentence": "SPI1 binds CSF1R.",
        }
    ]
    (lit_dir / "SPI1_CSF1R_classifications.json").write_text(json.dumps(payload), encoding="utf-8")
    cards = load_literature_cards(source_tf="SPI1", target_gene="CSF1R", literature_dir=lit_dir)
    assert cards["n_supporting"] == 1
    assert cards["cards"][0]["pubmed_url"].endswith("/999/")


def test_metrics_frame_uses_auroc_auprc_and_early_precision():
    payload = {
        "results_by_ratio": {
            "1.0": {
                "auroc": 0.81,
                "auprc_macro": 0.42,
                "precision_at_k": {"present@10": 0.7, "present@50": 0.4, "present@100": 0.25},
                "n_positive": 10,
                "n_negative": 10,
                "n_matched": 20,
                "pr_curve": {
                    "precision": [1.0, 0.5],
                    "recall": [0.0, 1.0],
                    "positive_prevalence": 0.5,
                },
            },
            "5": {
                "auroc_macro": 0.66,
                "aucpr_macro": 0.2,
                "ep_at_100": 0.11,
                "roc_curve": {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]},
            },
        }
    }
    frame = metrics_frame(payload)
    assert list(frame["ratio_label"]) == ["1:1", "1:5"]
    assert frame.loc[0, "AUROC"] == 0.81
    assert frame.loc[0, "AUPRC"] == 0.42
    assert frame.loc[0, "EP@10"] == 0.7
    assert frame.loc[0, "EP@100"] == 0.25
    assert frame.loc[1, "EP@100"] == 0.11
    assert len(pr_series(payload)) == 1
    assert len(roc_series(payload)) == 1
    assert roc_series(payload)[0]["label"] == "1:5"
    assert metrics_bar_figure(frame).data
    assert pr_curves_figure(pr_series(payload)).data
    assert roc_curves_figure(roc_series(payload)).data


def test_attention_and_evidence_pca_frames():
    channels = attention_frame(
        [
            {
                "edge": "SPI1 → GENE1",
                "source_tf": "SPI1",
                "target_gene": "GENE1",
                "by_stage_channel_group": {"stage3": {"mechanistic": 0.2, "functional": 0.8}},
            }
        ]
    )
    assert set(channels["channel"]) == {"mechanistic", "functional"}
    assert attention_figure(channels).data

    scored = pd.DataFrame(
        {
            "source_tf": ["A", "A", "B", "B"],
            "target_gene": ["G1", "G2", "G3", "G4"],
            "correlation": [0.1, 0.4, -0.2, 0.8],
            "accessibility": [0.2, 0.9, 0.3, 0.1],
            "motif_present": [True, False, True, False],
            "p_present": [0.9, 0.2, 0.4, 0.7],
        }
    )
    result = evidence_pca_frame(scored, [("A", "G1")])
    assert result["kind"] == "evidence_features"
    assert bool(result["frame"].loc[result["frame"]["target_gene"] == "G1", "selected"].iloc[0])
    scores, variance = pca_scores(np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]))
    assert scores.shape == (4, 2)
    assert variance.shape == (2,)


def test_readme_uses_evigrn_module_framing():
    text = Path("README.md").read_text(encoding="utf-8")
    lowered = text.lower()
    assert "multi-agent" not in lowered
    assert "split agent" not in lowered
    assert "graph reasoning agent" not in lowered
    excerpt = how_evigrn_works(text)
    assert "evidence-adaptive graph reasoning" in excerpt.lower()
    assert "4.9×" in excerpt
    assert "pip install" not in excerpt.lower()


def test_readme_overview_and_github_link(tmp_path: Path):
    assert normalize_github_remote("https://github.com/aghktb/GRN_Agent.git") == "https://github.com/aghktb/GRN_Agent"
    assert normalize_github_remote("git@github.com:aghktb/GRN_Agent.git") == "https://github.com/aghktb/GRN_Agent"
    assert normalize_github_remote("https://example.com/not-github") is None
    docs = tmp_path / "docs" / "figures_and_tables" / "figures"
    docs.mkdir(parents=True)
    figure = docs / "fig00_overview_evigrn.png"
    figure.write_bytes(b"png")
    assert overview_figure_path(tmp_path) == figure
    readme = "# EviGRN\n\n![skip](fig.png)\n\n**EviGRN** scores TF to gene edges.\n\n## Install\n\npip\n"
    assert "scores TF" in how_evigrn_works(readme)
    assert "pip" not in how_evigrn_works(readme)
    (tmp_path / "README.md").write_text("See https://github.com/aghktb/GRN_Agent for source.\n", encoding="utf-8")
    assert project_github_url(tmp_path) == "https://github.com/aghktb/GRN_Agent"


def test_discover_example_inputs_prefers_gold_network(tmp_path: Path):
    data = tmp_path / "Data" / "demo"
    data.mkdir(parents=True)
    (data / "ExpressionData.csv").write_text("gene,c1\nA,1\n", encoding="utf-8")
    (data / "TFs.csv").write_text("TF\nA\n", encoding="utf-8")
    (data / "gold.csv").write_text("Gene1,Gene2\nA,B\n", encoding="utf-8")
    conf = tmp_path / "conf"
    conf.mkdir()
    (conf / "plain.yml").write_text(
        "dataset:\n  expression_path: Data/demo/ExpressionData.csv\n  tf_file: Data/demo/TFs.csv\n  species: mouse\n",
        encoding="utf-8",
    )
    (conf / "with_gold.yml").write_text(
        "\n".join(
            [
                "dataset:",
                "  dataset_id: demo",
                "  species: mouse",
                "  expression_path: Data/demo/ExpressionData.csv",
                "  tf_file: Data/demo/TFs.csv",
                "acquisition:",
                "  cell_type: embryonic stem cell",
                "  cell_line: ES-E14",
                "  gold_network: Data/demo/gold.csv",
            ]
        ),
        encoding="utf-8",
    )
    example = discover_example_inputs(tmp_path)
    assert example is not None
    assert example["expression"] == "Data/demo/ExpressionData.csv"
    assert example["tf_file"] == "Data/demo/TFs.csv"
    assert example["gold"] == "Data/demo/gold.csv"
    assert example["species"] == "mouse"
    assert example["cell_type"] == "embryonic stem cell"
    assert example["config"] == "conf/with_gold.yml"


def test_discover_example_inputs_uses_mhsc_l_not_ecoli(tmp_path: Path):
    mhsc = tmp_path / "Data" / "sc-RNA-seq" / "mHSC-L"
    mhsc.mkdir(parents=True)
    (mhsc / "ExpressionData.csv").write_text("gene,c1\nA,1\n", encoding="utf-8")
    (mhsc / "TFs.csv").write_text("TF\nA\n", encoding="utf-8")
    (mhsc / "mHSC-ChIP-seq-network.csv").write_text("Gene1,Gene2\nA,B\n", encoding="utf-8")
    ecoli = tmp_path / "Data" / "ecoli_grnformer"
    ecoli.mkdir()
    (ecoli / "ExpressionData.csv").write_text("gene,c1\nA,1\n", encoding="utf-8")
    (ecoli / "TFs.csv").write_text("TF\nA\n", encoding="utf-8")
    (ecoli / "refNetwork.csv").write_text("Gene1,Gene2\nA,B\n", encoding="utf-8")
    conf = tmp_path / "conf"
    conf.mkdir()
    (conf / "ecoli.yml").write_text(
        "dataset:\n  dataset_id: ecoli\n  species: Escherichia coli\n"
        "  expression_path: Data/ecoli_grnformer/ExpressionData.csv\n"
        "  tf_file: Data/ecoli_grnformer/TFs.csv\n",
        encoding="utf-8",
    )
    trainings = tmp_path / "trainings" / "tf_eager copy"
    trainings.mkdir(parents=True)
    (trainings / "tf_eager_bootstrap_v2.pt").write_bytes(b"pt")
    example = discover_example_inputs(tmp_path)
    assert example is not None
    assert example["dataset_id"] == "mHSC-L"
    assert "ecoli" not in example["expression"]
    assert example["gold"].endswith("mHSC-ChIP-seq-network.csv")
    assert example["checkpoint"].endswith("tf_eager_bootstrap_v2.pt")
    assert "multicontext" not in example["checkpoint"]


def test_inference_config_uses_gold_and_skips_training(tmp_path: Path):
    template = {
        "workflow": {},
        "acquisition": {},
        "dataset": {},
        "cell_context": {},
        "split": {},
        "tf_eager": {},
        "train_tf_eager": {},
        "build_train_windows": {},
        "build_test_windows": {},
        "infer_tf_eager": {},
        "evaluation": {},
        "literature_validation": {},
    }
    cfg = prepare_inference_config(
        template,
        run_dir=tmp_path / "run_a",
        dataset_id="demo set",
        expression=tmp_path / "expr.csv",
        tf_file=tmp_path / "tfs.csv",
        species="human",
        checkpoint=tmp_path / "model.pt",
        gold=tmp_path / "gold.csv",
        cell_type="k562",
    )
    assert safe_dataset_id("demo set") == "demo_set"
    assert cfg["workflow"]["id"] == "demo_set"
    assert cfg["train_tf_eager"]["enabled"] is False
    assert cfg["disable_priors"] is True
    assert cfg["evaluation"]["enabled"] is True
    assert cfg["evaluation"]["gold_edges"].endswith("gold.csv")
    assert cfg["literature_validation"]["enabled"] is False
    no_gold = prepare_inference_config(
        template,
        run_dir=tmp_path / "run_b",
        dataset_id="b",
        expression=tmp_path / "expr.csv",
        tf_file=tmp_path / "tfs.csv",
        species="mouse",
        checkpoint=tmp_path / "model.pt",
    )
    assert no_gold["evaluation"]["enabled"] is False
    cmd = acquire_rank_command(
        python="python",
        script="scripts/acquire_multimodal_data.py",
        expression="expr.csv",
        species="human",
        tf_file="tfs.csv",
        out_manifest="manifest.json",
        ranked_json="ranked.json",
        dataset_id="demo",
        cache_dir=".cache",
    )
    assert "--rank-only" in cmd
    assert "--max-atac-candidates" in cmd
    ranked = tmp_path / "ranked.json"
    ranked.write_text(json.dumps([{"rank": 1, "accession": "ENCSR1", "tier": "A_STRONG_MATCH"}]), encoding="utf-8")
    rows = load_ranked_candidates(ranked)
    assert rows[0]["accession"] == "ENCSR1"


def test_discover_checkpoint_prefers_named_file(tmp_path: Path):
    tf_dir = tmp_path / "tf_eager"
    tf_dir.mkdir()
    (tf_dir / "other.pt").write_bytes(b"nope")
    named = tf_dir / "tf_eager_run.pt"
    named.write_bytes(b"ckpt")
    found = discover_checkpoint(tmp_path)
    assert found == named
