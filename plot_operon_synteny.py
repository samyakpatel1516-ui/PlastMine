#!/usr/bin/env python3
"""
plot_operon_synteny.py
PlastiMine Publication-Grade Gene Cluster & Operon Synteny Plotter
-------------------------------------------------------------------------
Author: PlastiMine Team
Description:
  Reads a PlastiMine-annotated GFF3 file (from annotate_my_genome.py or
  phase5_pipeline.py) and renders a publication-ready gene-arrow synteny
  diagram (.png and .pdf) showing:
    - Directional CDS gene arrows (+ strand right, - strand left)
    - Functional pathway color-coding (sty cluster, paa cluster, alkB/cyp153
      alkane machinery, hydrolytic lipA, transporters, regulators, redox defense)
    - Intergenic spacing callouts (highlighting compact <=60 bp operon junctions
      vs. divergent promoter regions)
    - PlastiMine multimodal confidence scores and Operon Context Scores
    - Conformal prediction status badges (High-Confidence vs. Ambiguous)

  * Zero-Click VS Code Support: If run without arguments, it automatically
    locates 'custom_genome_plastic_candidates.gff3' or
    'plasti_mine_data/phase5_output/genome_plastic_candidates.gff3' (or invokes
    a built-in demo GFF3) and saves 'operon_synteny_diagram.png'.

Usage:
  python plot_operon_synteny.py --input custom_genome_plastic_candidates.gff3 --output operon_synteny_diagram.png
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("PlastiMine-SyntenyPlotter")

# ==============================================================================
# 1. FUNCTIONAL CATEGORY COLOR PALETTE & DEFAULT DEMO GFF3
# ==============================================================================

CATEGORY_COLORS = {
    "sty_cluster":      {"color": "#1f77b4", "label": "Styrene Pathway Candidate (styA/B/C/D)"},
    "paa_cluster":      {"color": "#17becf", "label": "Phenylacetate Pathway Candidate (paaA/K/Z)"},
    "alkane_machinery": {"color": "#2ca02c", "label": "Polyolefin/Alkane Monooxygenase (alkB/cyp153/alkG)"},
    "hydrolytic":       {"color": "#ff7f0e", "label": "Secreted Hydrolase/Esterase Candidate (lipA)"},
    "transporter":      {"color": "#9467bd", "label": "Hydrophobic Uptake/Efflux Transporter (abc/mfs)"},
    "regulator":        {"color": "#d62728", "label": "Transcriptional Regulator (styR/S, lysR, tetR)"},
    "redox_defense":    {"color": "#e377c2", "label": "Oxidative Stress & Redox Defense (katE, sod)"},
    "other":            {"color": "#7f7f7f", "label": "Unassigned / Hypothetical CDS"}
}

GENE_TO_CATEGORY = {
    "stya": "sty_cluster", "styb": "sty_cluster", "styc": "sty_cluster", "styd": "sty_cluster",
    "styr": "regulator", "stys": "regulator",
    "paaa": "paa_cluster", "paak": "paa_cluster", "paaz": "paa_cluster",
    "alkb": "alkane_machinery", "alkb_cand": "alkane_machinery",
    "alkg": "alkane_machinery", "alkt": "alkane_machinery",
    "cyp153": "alkane_machinery", "cyp153_mono": "alkane_machinery",
    "lipa": "hydrolytic", "lipa_cand": "hydrolytic",
    "abc_perm": "transporter", "mfs_pump": "transporter",
    "kate": "redox_defense", "cat_scav": "redox_defense"
}

FALLBACK_DEMO_GFF3 = """##gff-version 3
#!processor PlastiMine-Standalone-Annotator v1.0
#!caution All annotations denote CANDIDATE plastic-associated enzymes or pathway components, NEVER confirmed degraders.
##sequence-region ISOLATE_CONTIG_01_STY_PAA_CLUSTER 1 2838
ISOLATE_CONTIG_01_STY_PAA_CLUSTER\tPlastiMine_v1\tCDS\t145	441	0.4930	-	0	ID=ISOLATE_CONTIG_01_STY_PAA_CLUSTER_CDS_0001;Name=styR;role=Accessory_Operon_Partner;hmm_profile=PM_CTX_0101:Accessory_StyR_Regulator;target_substrate=Regulatory_component;plastimine_confidence=0.4930;ml_probability=0.4121;operon_context_score=0.643;conformal_set=Candidate_(Uncertain_/_Ambiguous);ko=K10597;ec=-;validation_status=candidate_only_requires_wetlab_verification
ISOLATE_CONTIG_01_STY_PAA_CLUSTER\tPlastiMine_v1\tCDS\t682	1113	0.9403	+	0	ID=ISOLATE_CONTIG_01_STY_PAA_CLUSTER_CDS_0002;Name=styA;role=Primary_Candidate_Enzyme;hmm_profile=PM_HMM_0001:PlastiMine_StyA_SMO;target_substrate=Polystyrene_(PS)_/_Styrene_oligomers;plastimine_confidence=0.9403;ml_probability=0.9800;operon_context_score=0.867;conformal_set=High-Confidence_Candidate;ko=K00499;ec=1.14.14.11;validation_status=candidate_only_requires_wetlab_verification
ISOLATE_CONTIG_01_STY_PAA_CLUSTER\tPlastiMine_v1\tCDS\t1138	1563	0.9403	+	0	ID=ISOLATE_CONTIG_01_STY_PAA_CLUSTER_CDS_0003;Name=styB;role=Primary_Candidate_Enzyme;hmm_profile=PM_HMM_0002:PlastiMine_StyB_FADH2;target_substrate=Polystyrene_(PS)_/_Styrene_oligomers;plastimine_confidence=0.9403;ml_probability=0.9800;operon_context_score=0.867;conformal_set=High-Confidence_Candidate;ko=K10599;ec=1.5.1.36;validation_status=candidate_only_requires_wetlab_verification
ISOLATE_CONTIG_01_STY_PAA_CLUSTER\tPlastiMine_v1\tCDS\t1588	2007	0.9403	+	0	ID=ISOLATE_CONTIG_01_STY_PAA_CLUSTER_CDS_0004;Name=styC;role=Primary_Candidate_Enzyme;hmm_profile=PM_HMM_0003:PlastiMine_StyC_SOI;target_substrate=Polystyrene_(PS)_/_Epoxystyrene;plastimine_confidence=0.9403;ml_probability=0.9800;operon_context_score=0.867;conformal_set=High-Confidence_Candidate;ko=K02610;ec=5.3.99.7;validation_status=candidate_only_requires_wetlab_verification
ISOLATE_CONTIG_01_STY_PAA_CLUSTER\tPlastiMine_v1\tCDS\t2044	2361	0.5518	+	0	ID=ISOLATE_CONTIG_01_STY_PAA_CLUSTER_CDS_0005;Name=paaA;role=Primary_Candidate_Enzyme;hmm_profile=PM_HMM_0006:PlastiMine_PaaA_RingOx;target_substrate=Polystyrene_(PS)_/_Phenylacetic_acid_catabolism;plastimine_confidence=0.5518;ml_probability=0.4630;operon_context_score=0.717;conformal_set=Candidate_(Uncertain_/_Ambiguous);ko=K02609;ec=1.14.13.149;validation_status=candidate_only_requires_wetlab_verification
ISOLATE_CONTIG_01_STY_PAA_CLUSTER\tPlastiMine_v1\tCDS\t2410	2694	0.6463	+	0	ID=ISOLATE_CONTIG_01_STY_PAA_CLUSTER_CDS_0006;Name=abc_perm;role=Accessory_Operon_Partner;hmm_profile=PM_CTX_0103:Accessory_ABC_Transporter;target_substrate=Hydrophobic_uptake_transporter;plastimine_confidence=0.6463;ml_probability=0.5761;operon_context_score=0.777;conformal_set=Candidate_(Uncertain_/_Ambiguous);ko=K02003;ec=-;validation_status=candidate_only_requires_wetlab_verification
##sequence-region ISOLATE_CONTIG_02_ALKB_CYP153_CLUSTER 1 2280
ISOLATE_CONTIG_02_ALKB_CYP153_CLUSTER\tPlastiMine_v1\tCDS\t145	567	0.8610	+	0	ID=ISOLATE_CONTIG_02_ALKB_CYP153_CLUSTER_CDS_0001;Name=alkB;role=Primary_Candidate_Enzyme;hmm_profile=PM_HMM_0004:PlastiMine_AlkB_Diiron;target_substrate=Polypropylene_(PP)_/_Polyethylene_(PE);plastimine_confidence=0.8610;ml_probability=0.9800;operon_context_score=0.640;conformal_set=High-Confidence_Candidate;ko=K00496;ec=1.14.15.3;validation_status=candidate_only_requires_wetlab_verification
ISOLATE_CONTIG_02_ALKB_CYP153_CLUSTER\tPlastiMine_v1\tCDS\t592	867	0.7341	+	0	ID=ISOLATE_CONTIG_02_ALKB_CYP153_CLUSTER_CDS_0002;Name=alkG;role=Accessory_Operon_Partner;hmm_profile=PM_CTX_0102:Accessory_AlkG_Rubredoxin;target_substrate=Electron_transfer_partner;plastimine_confidence=0.7341;ml_probability=0.7848;operon_context_score=0.640;conformal_set=Candidate_(Uncertain_/_Ambiguous);ko=K00539;ec=-;validation_status=candidate_only_requires_wetlab_verification
ISOLATE_CONTIG_02_ALKB_CYP153_CLUSTER\tPlastiMine_v1\tCDS\t904	1323	0.8610	+	0	ID=ISOLATE_CONTIG_02_ALKB_CYP153_CLUSTER_CDS_0003;Name=cyp153;role=Primary_Candidate_Enzyme;hmm_profile=PM_HMM_0005:PlastiMine_CYP153_P450;target_substrate=Polypropylene_(PP)_/_Branched_polyolefins;plastimine_confidence=0.8610;ml_probability=0.9800;operon_context_score=0.640;conformal_set=High-Confidence_Candidate;ko=K00483;ec=1.14.15.1;validation_status=candidate_only_requires_wetlab_verification
ISOLATE_CONTIG_02_ALKB_CYP153_CLUSTER\tPlastiMine_v1\tCDS\t1348	1794	0.8588	+	0	ID=ISOLATE_CONTIG_02_ALKB_CYP153_CLUSTER_CDS_0004;Name=lipA;role=Primary_Candidate_Enzyme;hmm_profile=PM_HMM_0007:PlastiMine_LipA_Hydrolase;target_substrate=Polystyrene_(PS)_/_Ester-linked_oligomers;plastimine_confidence=0.8588;ml_probability=0.9766;operon_context_score=0.640;conformal_set=High-Confidence_Candidate;ko=K01046;ec=3.1.1.3;validation_status=candidate_only_requires_wetlab_verification
ISOLATE_CONTIG_02_ALKB_CYP153_CLUSTER\tPlastiMine_v1\tCDS\t1843	2136	0.7361	+	0	ID=ISOLATE_CONTIG_02_ALKB_CYP153_CLUSTER_CDS_0005;Name=katE;role=Accessory_Operon_Partner;hmm_profile=PM_CTX_0104:Accessory_Catalase_Redox;target_substrate=Oxidative_stress_defense;plastimine_confidence=0.7361;ml_probability=0.8202;operon_context_score=0.580;conformal_set=Candidate_(Uncertain_/_Ambiguous);ko=K03781;ec=1.11.1.6;validation_status=candidate_only_requires_wetlab_verification
"""

# ==============================================================================
# 2. GFF3 PARSER
# ==============================================================================

def parse_plastimine_gff3(gff_path: Path) -> tuple:
    """Parses a PlastiMine GFF3 file into contig lengths and gene feature records."""
    if not gff_path.exists():
        alt_paths = [
            Path("custom_genome_plastic_candidates.gff3"),
            Path("plasti_mine_data/phase5_output/genome_plastic_candidates.gff3"),
            Path("plasti_mine_data/phase2_output/annotated_contigs.gff3")
        ]
        found = None
        for p in alt_paths:
            if p.exists() and p.stat().st_size > 0:
                found = p
                break
        if found is not None:
            logger.info("Using discovered GFF3 file: %s", found)
            gff_path = found
        else:
            logger.info("No GFF3 file found on disk. Writing built-in demo GFF3 -> %s", gff_path)
            with open(gff_path, "w", encoding="utf-8") as f:
                f.write(FALLBACK_DEMO_GFF3)

    contig_lengths = {}
    contigs_genes = defaultdict(list)

    with open(gff_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("##sequence-region"):
                parts = line.split()
                if len(parts) >= 4:
                    contig_lengths[parts[1]] = int(parts[3])
                continue
            if line.startswith("#"):
                continue

            cols = line.split("\t")
            if len(cols) < 9:
                continue
            contig_id, source, ftype, start_s, end_s, score_s, strand, phase, attr_s = cols[:9]
            if ftype != "CDS":
                continue

            start = int(start_s)
            end = int(end_s)
            conf = float(score_s) if score_s != "." else 0.75

            attrs = {}
            for item in attr_s.split(";"):
                if "=" in item:
                    k, v = item.split("=", 1)
                    attrs[k.strip()] = v.strip()

            name = attrs.get("Name", attrs.get("ID", "CDS"))
            role = attrs.get("role", "Primary_Candidate_Enzyme")
            ctx_score = float(attrs.get("operon_context_score", 0.65))
            conformal = attrs.get("conformal_set", "Candidate")
            hmm_prof = attrs.get("hmm_profile", "-")

            cat = GENE_TO_CATEGORY.get(name.lower(), "other")
            if cat == "other":
                if "sty" in hmm_prof.lower(): cat = "sty_cluster"
                elif "paa" in hmm_prof.lower(): cat = "paa_cluster"
                elif "alk" in hmm_prof.lower() or "cyp" in hmm_prof.lower(): cat = "alkane_machinery"
                elif "lip" in hmm_prof.lower(): cat = "hydrolytic"

            contigs_genes[contig_id].append({
                "gene_id": attrs.get("ID", name),
                "name": name,
                "start": start,
                "end": end,
                "strand": strand,
                "confidence": conf,
                "context_score": ctx_score,
                "conformal": conformal,
                "role": role,
                "category": cat
            })

    # Sort genes per contig by start coordinate
    for cid in contigs_genes:
        contigs_genes[cid].sort(key=lambda g: g["start"])
        if cid not in contig_lengths:
            max_end = max(g["end"] for g in contigs_genes[cid])
            contig_lengths[cid] = int(max_end * 1.08)

    return contig_lengths, contigs_genes


# ==============================================================================
# 3. GENE-ARROW POLYGON & SYNTENY RENDERER
# ==============================================================================

def draw_gene_arrow(ax, start: float, end: float, strand: str, y_center: float,
                    height: float, facecolor: str, edgecolor: str = "#1a1a1a",
                    is_high_conf: bool = True):
    """Draws a directional gene arrow polygon on a Matplotlib axis."""
    width = max(10.0, end - start)
    head_len = min(width * 0.32, 95.0)
    half_h = height / 2.0
    shaft_h = half_h * 0.68

    if strand == "+":
        x0 = start
        x_neck = end - head_len
        x_tip = end
        coords = [
            (x0, y_center - shaft_h),
            (x_neck, y_center - shaft_h),
            (x_neck, y_center - half_h),
            (x_tip, y_center),
            (x_neck, y_center + half_h),
            (x_neck, y_center + shaft_h),
            (x0, y_center + shaft_h)
        ]
    else:
        x_tip = start
        x_neck = start + head_len
        x1 = end
        coords = [
            (x1, y_center - shaft_h),
            (x_neck, y_center - shaft_h),
            (x_neck, y_center - half_h),
            (x_tip, y_center),
            (x_neck, y_center + half_h),
            (x_neck, y_center + shaft_h),
            (x1, y_center + shaft_h)
        ]

    lw = 1.8 if is_high_conf else 1.2
    ls = "-" if is_high_conf else "--"
    poly = Polygon(coords, closed=True, facecolor=facecolor, edgecolor=edgecolor,
                   linewidth=lw, linestyle=ls, alpha=0.92, zorder=3)
    ax.add_patch(poly)


def render_synteny_figure(contig_lengths: dict, contigs_genes: dict, out_png: Path, out_pdf: Path):
    """Renders the multi-contig operon synteny diagram."""
    contig_ids = list(contigs_genes.keys())
    n_contigs = len(contig_ids)
    if n_contigs == 0:
        raise ValueError("No CDS features found in GFF3 file to plot.")

    fig_height = max(5.8, 3.2 * n_contigs + 1.4)
    fig, axes = plt.subplots(n_contigs, 1, figsize=(13.5, fig_height), squeeze=False)

    used_categories = set()

    for idx, cid in enumerate(contig_ids):
        ax = axes[idx, 0]
        genes = contigs_genes[cid]
        c_len = contig_lengths.get(cid, max(g["end"] for g in genes) + 200)

        # Draw backbone contig track line
        ax.plot([0, c_len], [0, 0], color="#444444", linewidth=2.5, zorder=1)
        ax.plot([0, 0], [-0.08, 0.08], color="#444444", linewidth=2.0, zorder=1)
        ax.plot([c_len, c_len], [-0.08, 0.08], color="#444444", linewidth=2.0, zorder=1)

        # Draw each gene arrow and annotation callout
        for g_idx, g in enumerate(genes):
            cat = g["category"]
            used_categories.add(cat)
            color = CATEGORY_COLORS.get(cat, CATEGORY_COLORS["other"])["color"]
            is_hc = "High-Confidence" in g["conformal"]

            draw_gene_arrow(
                ax, g["start"], g["end"], g["strand"],
                y_center=0.0, height=0.52,
                facecolor=color,
                edgecolor="#111111" if is_hc else "#555555",
                is_high_conf=is_hc
            )

            mid_x = 0.5 * (g["start"] + g["end"])

            # Gene symbol & strand label above arrow
            badge = "★" if is_hc else "◇"
            gene_label = f"{badge} {g['name']} ({g['strand']})"
            y_label = 0.36 if (g_idx % 2 == 0) else 0.48
            ax.text(
                mid_x, y_label, gene_label,
                ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", style="italic", color="#111111",
                bbox=dict(boxstyle="round,pad=0.18", facecolor="#ffffff", edgecolor=color, alpha=0.90, lw=1.1),
                zorder=5
            )

            # Confidence & Context score below arrow
            score_text = f"Conf: {g['confidence']:.2f} | Ctx: {g['context_score']:.2f}"
            y_score = -0.35 if (g_idx % 2 == 0) else -0.47
            ax.text(
                mid_x, y_score, score_text,
                ha="center", va="top",
                fontsize=7.8, color="#222222",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.85, lw=0.8),
                zorder=4
            )

            # Intergenic distance callout between adjacent genes
            if g_idx < len(genes) - 1:
                next_g = genes[g_idx + 1]
                gap_bp = next_g["start"] - g["end"] - 1
                gap_mid_x = 0.5 * (g["end"] + next_g["start"])
                same_strand = (g["strand"] == next_g["strand"])
                is_operon_junction = (-30 <= gap_bp <= 60 and same_strand)

                gap_color = "#1b7837" if is_operon_junction else "#b2182b"
                gap_tag = f"{gap_bp} bp" if gap_bp <= 80 else f"← {gap_bp} bp →"
                ax.annotate(
                    gap_tag,
                    xy=(gap_mid_x, 0.0),
                    xytext=(gap_mid_x, 0.22 if (g_idx % 2 == 1) else -0.22),
                    ha="center", va="center",
                    fontsize=7.5, fontweight="bold", color=gap_color, bbox=dict(boxstyle="round,pad=0.1", facecolor="#ffffff", edgecolor="none", alpha=0.85),
                    zorder=6
                )

        # Contig header title
        mean_ctx = np.mean([g["context_score"] for g in genes])
        ax.set_title(
            f"Contig: {cid}  |  Length: {c_len:,} bp  |  Annotated CDS: {len(genes)}  |  Mean Operon Context Score: {mean_ctx:.3f}",
            loc="left", fontsize=10.5, fontweight="bold", pad=8, color="#1a2530"
        )

        ax.set_xlim(-int(c_len * 0.03), int(c_len * 1.03))
        ax.set_ylim(-0.78, 0.78)
        ax.set_yticks([])
        ax.set_xlabel("Genomic Coordinate (bp)", fontsize=9.5, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", linestyle=":", alpha=0.45)

    # Overall Figure Title & Guardrail Subtitle
    fig.suptitle(
        "PlastiMine: Genomic Neighborhood & Operon Synteny of Candidate PS/PP-Associated Gene Clusters\n"
        "[★ = High-Confidence Candidate (Solid Border)  |  ◇ = Ambiguous/Accessory Partner (Dashed Border)  |  Green bp = Compact Operon Junction]",
        fontsize=10.5, fontweight="bold", y=0.97
    )

    # Legend at the bottom
    legend_handles = [
        Patch(facecolor=CATEGORY_COLORS[cat]["color"], edgecolor="#111111", label=CATEGORY_COLORS[cat]["label"])
        for cat in CATEGORY_COLORS if cat in used_categories
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(3, len(legend_handles)),
        frameon=True,
        fontsize=8.8,
        bbox_to_anchor=(0.5, 0.01)
    )

    plt.tight_layout(rect=[0.0, 0.09, 1.0, 0.92])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    logger.info("Saved high-resolution synteny diagram -> %s (and %s)", out_png, out_pdf)


# ==============================================================================
# 4. MAIN CLI ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PlastiMine Publication-Grade Operon Synteny Diagram Generator"
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=Path("custom_genome_plastic_candidates.gff3"),
        help="Input PlastiMine GFF3 file (default: custom_genome_plastic_candidates.gff3)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("operon_synteny_diagram.png"),
        help="Output PNG image path (default: operon_synteny_diagram.png)"
    )
    args = parser.parse_args()

    out_pdf = args.output.with_suffix(".pdf")

    logger.info("==========================================================")
    logger.info("PlastiMine Operon Synteny Visualization Engine")
    logger.info("==========================================================")

    contig_lengths, contigs_genes = parse_plastimine_gff3(args.input)
    render_synteny_figure(contig_lengths, contigs_genes, args.output, out_pdf)

    logger.info("==========================================================")
    logger.info("Synteny Plotting Complete!")
    logger.info("  -> PNG Image: %s", args.output)
    logger.info("  -> PDF Vector:%s", out_pdf)
    logger.info("==========================================================")

if __name__ == "__main__":
    main()
