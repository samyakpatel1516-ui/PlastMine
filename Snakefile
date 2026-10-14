# ==============================================================================
# PlastiMine: Reproducible Snakemake Pipeline (Phases 1 to 5)
# Weakly supervised discovery and prioritisation of candidate PS/PP-active
# enzymes and gene clusters from genomes and metagenomes.
# ==============================================================================

configfile: "config.yaml"

rule all:
    input:
        "plasti_mine_data/processed/master_curated_dataset_splits.tsv",
        "plasti_mine_data/phase2_output/candidate_neighborhood_features.tsv",
        "plasti_mine_data/phase3_output/model_ablation_benchmark.tsv",
        "plasti_mine_data/phase4_output/top20_prioritised_candidates.tsv",
        "plasti_mine_data/phase5_output/genome_plastic_candidates.gff3",
        "plasti_mine_data/phase5_output/plastimine_ps_pp_profiles.hmm",
        "plasti_mine_data/phase5_output/tier3_label_noise_ablation.tsv",
        "plasti_mine_data/phase5_output/MODEL_CARD.md",
        "plasti_mine_data/phase5_output/DATASHEET.md",
        "plasti_mine_data/phase5_output/PREPRINT_MANUSCRIPT.md"

rule phase1_curate_and_split:
    output:
        splits="plasti_mine_data/processed/master_curated_dataset_splits.tsv",
        positives="plasti_mine_data/processed/positives_tiered.tsv",
        background="plasti_mine_data/processed/unlabelled_background.tsv"
    container:
        "docker://plastimine/plastimine-env:1.0"
    shell:
        "python3 phase1_pipeline.py"

rule phase2_genomic_context:
    input:
        "plasti_mine_data/processed/master_curated_dataset_splits.tsv"
    output:
        features="plasti_mine_data/phase2_output/candidate_neighborhood_features.tsv",
        graphs="plasti_mine_data/phase2_output/genomic_neighborhood_graphs.json",
        gff="plasti_mine_data/phase2_output/annotated_contigs.gff3"
    container:
        "docker://plastimine/plastimine-env:1.0"
    shell:
        "python3 phase2_pipeline.py"

rule phase3_multimodal_modelling:
    input:
        splits="plasti_mine_data/processed/master_curated_dataset_splits.tsv",
        context="plasti_mine_data/phase2_output/candidate_neighborhood_features.tsv"
    output:
        ablation="plasti_mine_data/phase3_output/model_ablation_benchmark.tsv",
        preds="plasti_mine_data/phase3_output/test_predictions_conformal.tsv",
        model="plasti_mine_data/phase3_output/multimodal_model.pkl"
    container:
        "docker://plastimine/plastimine-env:1.0"
    shell:
        "python3 phase3_pipeline.py"

rule phase4_prioritise_candidates:
    input:
        model="plasti_mine_data/phase3_output/multimodal_model.pkl",
        context="plasti_mine_data/phase2_output/candidate_neighborhood_features.tsv"
    output:
        top20_tsv="plasti_mine_data/phase4_output/top20_prioritised_candidates.tsv",
        top20_json="plasti_mine_data/phase4_output/top20_prioritised_candidates.json",
        plot="plasti_mine_data/phase4_output/candidate_ranking_plot.png"
    container:
        "docker://plastimine/plastimine-env:1.0"
    shell:
        "python3 phase4_pipeline.py"

rule phase5_deliverables_and_gff3:
    input:
        top20="plasti_mine_data/phase4_output/top20_prioritised_candidates.tsv"
    output:
        gff3="plasti_mine_data/phase5_output/genome_plastic_candidates.gff3",
        hmm="plasti_mine_data/phase5_output/plastimine_ps_pp_profiles.hmm",
        tier3="plasti_mine_data/phase5_output/tier3_label_noise_ablation.tsv",
        model_card="plasti_mine_data/phase5_output/MODEL_CARD.md",
        datasheet="plasti_mine_data/phase5_output/DATASHEET.md",
        preprint="plasti_mine_data/phase5_output/PREPRINT_MANUSCRIPT.md"
    container:
        "docker://plastimine/plastimine-env:1.0"
    shell:
        "python3 phase5_pipeline.py"
