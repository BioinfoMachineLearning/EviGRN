# EviGRN: Evidence-adaptive graph reasoning for Gene Regulatory Network inference

[Source code](https://github.com/aghktb/GRN_Agent)

![EviGRN overview](docs/figures_and_tables/figures/fig00_overview_evigrn.png)

Inferring gene regulatory networks (GRNs) from multimodal biological data requires evidence beyond transcript abundance. RNA-seq, chromatin accessibility, and transcription factor (TF) motif priors provide complementary signals, but finding, harmonizing, and assessing these sources requires substantial manual effort, and modalities are often missing across biological contexts. Existing pipelines commonly require evidence preparation as a separate preprocessing step, while incomplete or unavailable modalities are often not explicitly represented during inference.

**EviGRN** (Evidence-adaptive graph reasoning for Gene Regulatory Network inference) is a multi-omics evidence-aware reasoning framework with three connected modules:

1. **Automated evidence acquisition, quality control, and harmonization.** Quality-control decisions determine which modalities enter TF-centered candidate neighborhoods. Rejected or unavailable modalities stay explicit missing evidence rather than being silently dropped.
2. **TF-centered candidate subnetwork construction and evidence-adaptive graph reasoning.** The reasoning model encodes expression, motif, and accessibility evidence with confidence, context, and availability, then applies staged cross-attention over motif/accessibility, expression, and joint evidence to score candidate TF–target interactions.
3. **Grounded post hoc literature verification.** Literature support is attached after scoring and does not change model scores.

Across 24 full-matrix settings from two held-out cell types and 10 baselines, EviGRN achieves approximately 4.9× higher mean AUPRC and 3.1× higher EP@100 than the strongest respective baselines, with the largest gains on cell-type-specific ChIP-seq references. Without target-context labels or retraining, EviGRN achieves AUPRC 0.705 and EP@10 0.812 on unseen cell types under sampled GRN evaluation. The mammalian-trained model transfers without retraining to *E. coli* and *Drosophila*, recovering tissue-relevant Notch and Hedgehog regulatory programs in the *Drosophila* eye-antennal disc. Grounded post hoc literature verification supports 17% of the top 100 benchmark-absent predictions in a hematopoietic context.

For labeling, negatives, split policy, and model knobs, see [`docs/TF-EAGER.md`](docs/TF-EAGER.md).

---

## Three modules

Code lives under `src/grn_agent/`. The integrated run is `scripts/run_integrated_tf_eager_workflow.py`, which executes the enabled YAML stages in order. Checkpoint and config keys still use the `tf_eager` stage id.

| Module | Role | Typical artifact |
|--------|------|------------------|
| **(i) Acquisition, QC, harmonization** | Resolves RNA and optional ATAC/motif inputs, records QC decisions, and aligns genes to the expression context. | `multimodal_manifest.json` |
| **Holdout** | Builds a TF-aware split so train/val/test follow the configured strategy (for example leave-one-TF-out). | `split_manifest.csv` |
| **(ii) TF-centered subnetworks** | Samples a TF-centered candidate neighborhood and writes one evidence graph per window. | `train_windows.jsonl`, `test_windows.jsonl` |
| **(ii) Evidence-adaptive graph reasoning** | Trains or runs staged cross-attention over motif/accessibility, expression, and joint evidence. | `tf_eager_bootstrap_v2.pt` |
| **Scores** | Writes per-edge scores, an optional thresholded network, and flattened evidence. | `test_scored_edges.csv`, `test_network.csv`, `test_flat_evidence.jsonl` |
| **Evaluation** | Held-out metrics with split constraints and optional negative-ratio sweeps (`scripts/eval_grn_agent.py`). | `evaluation/eval_test_by_ratio.json` |
| **(iii) Literature verification** | Post hoc PubMed support for top predictions. Does not modify `p_present`. | `literature_validated.csv`, `literature_classifications/` |

---

## Install

From the repository root:

```bash
pip install -e .
pip install -e ".[torch]"          # training and GPU inference
pip install -e ".[acquisition]"   # optional: multimodal acquisition (ATAC/motif, etc.)
pip install -e ".[dev]"           # optional: tests
```

---

## How to run (by goal)

| Goal | Command | Config entry points |
|------|---------|---------------------|
| Full single-dataset pipeline (acquire → split → windows → train → infer → eval) | `python scripts/run_integrated_tf_eager_workflow.py --config <YAML>` | [Integrated single-context](#config-file-templates) |
| One shared model across many contexts | `python scripts/run_multicontext_tf_eager_workflow.py --config <YAML>` | [Multicontext](#multicontext) |
| Blind inference / blind eval over many DataContext folders | `python scripts/run_blind_tf_eager_datacontext_eval.py --config <YAML>` | [Blind datacontext](#blind-datacontext-inference-and-evaluation) |
| Manual build → train → infer (debugging or custom wiring) | `build_tf_eager_windows.py` → `train_tf_eager.py` → `infer_tf_eager.py` | [Manual pipeline](#manual-pipeline-build-train-infer) |

Artifacts live under `artifacts/` (exact layout follows `workflow.id` and `artifact_root` in your YAML).

---

## Config file templates

Copy a template and point it at your expression matrix, `TFs.csv`, gold edges, and genome settings as needed.

### Generic integrated template

| File | Purpose |
|------|---------|
| [`conf/tf_eager_integrated_standard.yml`](conf/tf_eager_integrated_standard.yml) | Annotated master YAML: `workflow`, `acquisition`, `dataset`, `split`, window build, `train_tf_eager`, `infer_tf_eager`, `evaluation`. |

### Integrated single-context

| File | Purpose |
|------|---------|
| [`conf/mESC_tf500_tf_eager_integrated.yml`](conf/mESC_tf500_tf_eager_integrated.yml) | mESC tf500 integrated workflow. |
| [`conf/mESC/mESC_tf_eager_integrated.yml`](conf/mESC/mESC_tf_eager_integrated.yml) | mESC integrated variant. |
| [`conf/mHSC-GM_celltype_specific_chipseq_Tf1000/tf_eager_integrated_mHSC-GM.yml`](conf/mHSC-GM_celltype_specific_chipseq_Tf1000/tf_eager_integrated_mHSC-GM.yml) | mHSC–GM example. |
| [`conf/ecoli/tf_eager_integrated_ecoli.yml`](conf/ecoli/tf_eager_integrated_ecoli.yml) | *E. coli* example. |

### Train / infer YAML (manual or scripted steps)

| File | Purpose |
|------|---------|
| [`conf/mESC_tf500_tf_eager_train.yml`](conf/mESC_tf500_tf_eager_train.yml) | Window build + train + infer settings for `build_tf_eager_windows.py`, `train_tf_eager.py`, `infer_tf_eager.py`. |

### Multicontext

| File | Purpose |
|------|---------|
| [`conf/multicontext_tf_eager/all_datacontext_contexts_neg2.yml`](conf/multicontext_tf_eager/all_datacontext_contexts_neg2.yml) | Neg2 variant. |
| [`conf/multicontext_tf_eager/all_datacontext_contexts_neg2_single_stage.yml`](conf/multicontext_tf_eager/all_datacontext_contexts_neg2_single_stage.yml) | Neg2 single-stage. |
| [`conf/multicontext_tf_eager/all_datacontext_contexts_neg2_functional_only.yml`](conf/multicontext_tf_eager/all_datacontext_contexts_neg2_functional_only.yml) | Neg2 functional-only. |

**Resume or partial recompute:**

```bash
python scripts/run_multicontext_tf_eager_workflow.py \
  --config conf/multicontext_tf_eager/all_datacontext_contexts.yml \
  --force-recompute \
  --start-from <context_id>
```

### Blind datacontext

| File | Purpose |
|------|---------|
| [`conf/blind_tf_eager_datacontext_eval_neg2.yml`](conf/blind_tf_eager_datacontext_eval_neg2.yml) | Neg2 blind. |
| [`conf/blind_tf_eager_datacontext_eval_neg2_exhaustive.yml`](conf/blind_tf_eager_datacontext_eval_neg2_exhaustive.yml) | Exhaustive neg2 blind sweep. |

---

## Integrated workflow (single YAML)

Runs acquisition, TF-centered windows, evidence-adaptive graph reasoning, inference, and evaluation when those stages are enabled in the config.

```bash
python scripts/run_integrated_tf_eager_workflow.py --config conf/mESC_tf500_tf_eager_integrated.yml
```

**Stages:** optional acquisition → split manifest → train windows → train the evidence-adaptive graph reasoning model → test windows → infer (score + export) → evaluation.

**Default inference outputs** (paths can be overridden in YAML):

- `test_scored_edges.csv`
- `test_network.csv`
- `test_flat_evidence.jsonl`

**Evaluation:** e.g. `evaluation/eval_test_by_ratio.json` under the workflow artifact directory when ratio sweeps are configured.

---

## Literature verification

Module (iii) checks top predictions against PubMed after inference. Support counts and quotes are stored beside the original scores. They do not change `p_present`.

Add an NCBI email in the `NCBI_EMAIL` section of `src/grn_agent/agents/lit_config.py`. For faster queries, create an API key at https://www.ncbi.nlm.nih.gov/account/settings/ and put `NCBI_API_KEY` in a `.env` file.

## Literature verification artifacts

If you run the literature validation stage (e.g., via `scripts/run_literature_validation.py`), the following files are produced in your artifact directory:

- **`literature_validated.csv`**: The primary results table. It contains the original model scores plus literature-derived metrics: `lit_score`, `n_supporting` (papers), `pmids`, and `evidence_types` (e.g., ChIP-seq, knockdown).
- **`literature_classifications/`**: A directory containing detailed audit logs for every interaction.
  - Files are named `{TF}_{Target}_classifications.json`.
  - **Key fields**:
    - `evidence_sentence`: The **exact quote** from the paper that supports the interaction.
    - `cell_type_sentences`: Quotes from the paper that grounded the study in your specific cell type.
    - `effective_support`: A boolean flag indicating if the paper passed all quality gates (grounded, correct direction, not negated).
    - `confidence`: The LLM's confidence score for that specific abstract.

---

## Multicontext

Trains **one** evidence-adaptive graph reasoning model on combined training windows from multiple contexts, then runs test inference and evaluation **per context**.

```bash
python scripts/run_multicontext_tf_eager_workflow.py \
  --config conf/multicontext_tf_eager/all_datacontext_contexts.yml
```

---

## Blind datacontext (inference and evaluation)

Batch blind runs across prepared DataContext directories (see [`scripts/run_blind_tf_eager_datacontext_eval.py`](scripts/run_blind_tf_eager_datacontext_eval.py)):

```bash
python scripts/run_blind_tf_eager_datacontext_eval.py --config conf/blind_tf_eager_datacontext_eval.yml
```

Use the `neg2` YAML variants to compare negative-construction settings.

The integrated workflow can also build **blind-style** test windows when no split manifest is configured; see [`scripts/run_integrated_tf_eager_workflow.py`](scripts/run_integrated_tf_eager_workflow.py) and your YAML.

---

## Manual pipeline (build → train → infer)

```bash
python scripts/build_tf_eager_windows.py --config conf/mESC_tf500_tf_eager_train.yml
python scripts/train_tf_eager.py --config conf/mESC_tf500_tf_eager_train.yml
python scripts/infer_tf_eager.py --config conf/mESC_tf500_tf_eager_train.yml
```

---

## Train, infer, and evaluate (scripts)

| Step | Script | Notes |
|------|--------|--------|
| Build windows | `scripts/build_tf_eager_windows.py` | YAML with `tf_eager` and paths for split subset and output JSONL |
| Train graph reasoning | `scripts/train_tf_eager.py` | Requires training `windows_jsonl` and checkpoint `out` in config |
| Infer + export | `scripts/infer_tf_eager.py` | Checkpoint + test `windows_jsonl`; writes CSV / optional JSONL |
| Evaluate | `scripts/eval_grn_agent.py` | Invoked from integrated configs for held-out, ratio-based eval; needs scored outputs, gold edges, and split manifest when doing split-aware eval |

---

## Interactive GUI (scientific explorer)

Additive Streamlit interface over existing artifacts. It does **not** change model architecture or acquisition/training cores. Mechanistic reasoning text is generated deterministically from evidence fields (no LLM). Literature cards remain post-hoc.

```bash
pip install -e ".[gui]"
streamlit run app.py
```

Views:

| Page | What you see |
|------|----------------|
| **Overview** | EviGRN overview, how the method works, and the GitHub link |
| **Run inference** | Expression, TF list, optional gold network, ranked accessibility QC, harmonization, inference, gold-standard metrics, and literature verification |
| **Acquisition Trace** | Multimodal package narrative, QC reasons, coverage chart |
| **Workflow Trace** | Stage status and artifact handoffs from `workflow_runtime.json` |
| **Prediction Explainer** | Quantitative evidence factors, scientific rationale, PubMed-linked literature claims |
| **Visual Explorer** | Coverage, TF-centered evidence neighborhood, thresholded final network |
| **Results** | Eval report / PR curve / top scored edges |

Select a finished workflow under `artifacts/` in the sidebar. Large readable fonts and a clean scientific theme are applied by default.

### Mechanistic weak-coexpression gold case study

Same-graph case study (Full multimodal scored edges only; does not compare rebuilt no-motif candidate graphs):

```bash
python scripts/build_mech_weakcorr_case_study.py \
  --corr-threshold 0.1 \
  --min-p 0.5 \
  --top-n 12
```

Outputs under `artifacts/InferenceAblation/.../case_study_mech_weakcorr/` (`RESULTS.md`, CSV, PNG/PDF).

---

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -q
```

Relevant tests include `tests/test_tf_eager_*.py` and `tests/test_integrated_tf_eager_workflow.py`.

---

## Further reading

| Doc | Content |
|-----|---------|
| [`docs/TF-EAGER.md`](docs/TF-EAGER.md) | Evidence-graph model behavior, knobs, minimal commands |
| [`docs/TF_EAGER_CONFIG_TUTORIAL.md`](docs/TF_EAGER_CONFIG_TUTORIAL.md) | Config walkthrough |
| [`src/grn_agent/acquisition/README.md`](src/grn_agent/acquisition/README.md) | Multimodal acquisition |
