# ==============================================================================
# PlastiMine Reproducible Container Image (Docker / Apptainer / Singularity)
# ==============================================================================
FROM condaforge/mambaforge:24.3.0-0

LABEL maintainer="PlastiMine Research Group"
LABEL description="Weakly supervised discovery and prioritisation of candidate PS/PP-active enzymes"
LABEL version="1.0.0"

WORKDIR /workspace

# Install bioinformatics and machine learning dependencies via bioconda & conda-forge
RUN mamba install -y -c conda-forge -c bioconda     python=3.11     pandas=2.2.*     numpy=1.26.*     scikit-learn=1.5.*     scipy=1.13.*     matplotlib=3.9.*     pyrodigal=3.4.*     mmseqs2=15.6f452     hmmer=3.4     snakemake-minimal=8.11.* &&     mamba clean -afy

COPY phase1_pipeline.py phase2_pipeline.py phase3_pipeline.py phase4_pipeline.py phase5_pipeline.py Snakefile /workspace/

ENTRYPOINT ["snakemake", "--cores", "4"]
