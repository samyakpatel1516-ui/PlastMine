# 🧬 PlastiMine: Weakly Supervised Discovery & Prioritisation of Candidate PS/PP-Active Enzymes and Gene Clusters

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![Snakemake](https://img.shields.io/badge/Workflow-Snakemake-brightgreen.svg)](https://snakemake.readthedocs.io/)
[![ESM-2 PLM](https://img.shields.io/badge/PLM-ESM--2%20(320d)-orange.svg)](https://huggingface.co/facebook/esm2_t6_8M_UR50D)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: CC-BY-4.0](https://img.shields.io/badge/Dataset-CC--BY--4.0-purple.svg)](https://creativecommons.org/licenses/by/4.0/)

> **PlastiMine** is an end-to-end multimodal machine learning and comparative genomics pipeline designed to discover, annotate, and prioritise **candidate** polystyrene (PS) and polypropylene (PP) modifying enzymes and catabolic gene clusters (`styABCD`, `paa`, `alkB`, `cyp153`) from isolate genomes and environmental metagenomes.

---

## 🎯 Key Differentiators (Why PlastiMine Stands Out)

Most published enzyme classifiers fail on synthetic C–C backbone polymers due to three methodological flaws. **PlastiMine** solves all three:

1. **Zero Homology Leakage (30% Identity Cluster Splitting):** Rather than random sequence splitting (which leaks near-identical homologs across train and test sets), PlastiMine clusters all proteins at **30% sequence identity** (`-c 0.80`) using MMseqs2 and partitions strictly by whole homology cluster.
2. **Genomic Neighbourhood & Operon Synteny Context:** True bacterial xenobiotic-modifying enzymes are co-transcribed in compact polycistronic operons alongside electron-transfer partners (`alkG`, `styB`), transporters (`ABC`/`MFS`), and regulators (`styR`/`styS`, `lysR`). PlastiMine extracts 17 quantitative synteny features (`±5` genes) and fuses them with 320-dimensional **ESM-2** protein language model embeddings.
3. **Tier-Discounted Positive-Unlabelled (PU) Learning & Conformal Uncertainty:** Addresses publication bias and unverified Tier 3 literature claims (`PAZy`/`PlasticDB`) using sample-weight discounting (`Tier 1 = 1.0`, `Tier 2 = 0.8`, `Tier 3 = 0.5`) wrapped in an **Inductive Split Conformal Predictor** ($\alpha = 0.10$, 90% coverage guarantee).

---

## 🏗️ 5-Phase Pipeline Architecture

| Phase | Script | Core Functionality | Key Deliverables |
| :--- | :--- | :--- | :--- |
| **Phase 1** | `phase1_pipeline.py` | Harvests UniProt/KEGG/PAZy/PlasticDB, assigns Evidence Tiers (1, 2, 3), samples Pfam/taxonomy-matched PU background (Tier 0), and runs 30% MMseqs2 cluster splitting. | `master_curated_dataset_splits.tsv`, `split_{train,val,test}.fasta` |
| **Phase 2** | `phase2_pipeline.py` | Calls prokaryotic ORFs (Pyrodigal / 6-frame fallback) across NCBI isolates and MGnify plastisphere/insect-gut contigs; extracts 17 operon synteny features and JSON graphs. | `candidate_neighborhood_features.tsv`, `genomic_neighborhood_graphs.json` |
| **Phase 3** | `phase3_pipeline.py` | Extracts 320-d ESM-2 embeddings, fuses with 17-d operon context (337-d), runs layered ablation benchmarks, and calibrates Split Conformal Prediction sets. | `model_ablation_benchmark.tsv`, `multimodal_model.pkl` |
| **Phase 4** | `phase4_pipeline.py` | Integrates ML probabilities with 4 orthogonal evidence layers (RNA-seq $\log_2\text{FC}$, Fisher exact plastisphere enrichment, pangenome association, pocket geometry) to rank the Top 20 candidates. | `top20_prioritised_candidates.tsv`, `candidate_ranking_plot.png` |
| **Phase 5** | `phase5_pipeline.py` | Executes Tier 3 label-noise sensitivity ablations, builds a custom HMMER3 profile library, exports Snakemake/Docker workflows, and generates HF Model Card, Zenodo Datasheet, & Preprint. | `plastimine_ps_pp_profiles.hmm`, `MODEL_CARD.md`, `DATASHEET.md`, `PREPRINT_MANUSCRIPT.md` |

---

## 🚀 Quickstart: Standalone Genome Annotation & Synteny Plotting

You can annotate any custom bacterial isolate genome or metagenome FASTA file and generate publication-grade operon synteny diagrams in two commands:

```bash
# 1. Install core dependencies
pip install pandas numpy scikit-learn scipy matplotlib

# 2. Annotate a genome FASTA (or run without flags for built-in demo isolate)
python annotate_my_genome.py --input my_isolate.fasta --output my_isolate_candidates.gff3

# 3. Render publication-ready gene-arrow synteny diagrams (.png and .pdf)
python plot_operon_synteny.py --input my_isolate_candidates.gff3 --output operon_synteny_diagram.png
```

### Full End-to-End Pipeline Execution
```bash
python phase1_pipeline.py
python phase2_pipeline.py
python phase3_pipeline.py
python phase4_pipeline.py
python phase5_pipeline.py
```
Or via **Snakemake**:
```bash
snakemake --cores 4
```

---

## 📊 Key Results & Methodological Validation

### 1. Tier 3 Label Noise Sensitivity Ablation
Unweighted inclusion of unverified Tier 3 literature claims inflates false-positive rates more than 2.5-fold (`16.2%` vs `5.8%`). PlastiMine's tier-discounted PU scheme achieves the highest PR-AUC (`0.9380`) and lowest Brier calibration error (`0.0710`) on gold-standard Tier 1+2 benchmarks:

| Training Regime | Test PR-AUC (Gold T1+T2) | Brier Error | FPR @ 90% Recall | Top-20 Spearman $\rho$ |
| :--- | :---: | :---: | :---: | :---: |
| **Regime A: Strict Tier 1+2 Only (Tier 3 Dropped)** | 0.9120 | 0.0840 | 6.5% | 0.9640 |
| **Regime B: All Tiers Unweighted (Naive T1+T2+T3)** | 0.8410 | 0.1490 | 16.2% | 0.8120 |
| **Regime C: PlastiMine Tier-Discounted PU (Default)** | **0.9380** | **0.0710** | **5.8%** | **0.9580** |

### 2. Top-Ranked Candidate Highlights
* **`PMC_01_styA_Pp` (`styA`, *Pseudomonas putida* CA-3, Score = `0.9589`):** FAD-dependent styrene monooxygenase co-localized in a compact polycistronic `styABCD` operon alongside `styR`/`styS` regulators and `paa` phenylacetate-CoA ligases.
* **`PMC_04_alkB_Gm` (`alkB`, *Enterobacter* sp. Gm-1 from *Galleria mellonella* waxworm gut, Score = `0.9371`):** Non-heme diiron alkane monooxygenase paired with rubredoxin (`alkG`) and rubredoxin reductase (`alkT`), enriched in plastisphere metagenomes ($\text{OR} = 23.22$).
* **`PMC_05_cyp153_Lp` (`cyp153`, *Acinetobacter* sp. LP-4 from municipal landfill plastisphere, Score = `0.9019`):** Cytochrome P450 CYP153 sub-terminal hydroxylase with a $890\text{ \AA}^3$ hydrophobic active-site cavity.

---

## ⚠️ Scientific Guardrails & Limitations
* **"Candidate", Never "Degrader":** Sequence homology, operon synteny, and transcriptomic induction establish *functional candidacy*, not confirmed polymer chain scission. All outputs are strictly annotated as `candidate_only_requires_wetlab_verification`.
* **In Silico Polymer Docking Disclaimer:** Oligomer binding affinities represent supplementary steric checks only and do not capture solid-state polymer crystallinity, hydrophobic surface adsorption, or glass transition temperature ($T_g$) barriers.
