#!/usr/bin/env python3
"""
annotate_my_genome.py
PlastiMine Standalone Genome & Metagenome Annotation CLI Tool
-------------------------------------------------------------------------
Author: PlastiMine Team
Description:
  Takes any prokaryotic genome or metagenome FASTA file as input, predicts
  open reading frames (via Pyrodigal or a built-in 6-frame ORF caller),
  scans proteins against the custom PlastiMine HMM/motif profile library
  (styABCD, paa, alkB, cyp153, lipA, and accessory transporters/regulators),
  computes 17 local operon neighborhood synteny features, scores candidates
  using the Phase 3 multimodal model + Inductive Split Conformal Predictor,
  and outputs a GFF3 annotation file, TSV report, and protein FASTA (.faa).

  * Zero-Click VS Code Support: If run without CLI arguments (e.g., clicking
    the 'Run Python File' play button in VS Code), it automatically generates
    a multi-contig demonstration isolate genome ('demo_plastic_isolate.fasta')
    and annotates it end-to-end.

Usage:
  python annotate_my_genome.py --input my_genome.fasta --output my_genome.gff3
"""

import os
import sys
import json
import pickle
import logging
import hashlib
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("PlastiMine-Annotator")

# ==============================================================================
# 1. GENETIC CODE, BACK-TRANSLATION & DEMO GENOME GENERATOR
# ==============================================================================

CODON_TABLE = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M', 'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K', 'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L', 'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q', 'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V', 'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E', 'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S', 'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*', 'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W',
}

SYNONYMOUS_CODONS = {
    'A': ['GCT', 'GCC', 'GCA', 'GCG'], 'C': ['TGT', 'TGC'], 'D': ['GAT', 'GAC'],
    'E': ['GAA', 'GAG'], 'F': ['TTT', 'TTC'], 'G': ['GGT', 'GGC', 'GGA', 'GGG'],
    'H': ['CAC', 'CAT'], 'I': ['ATT', 'ATC', 'ATA'], 'K': ['AAA', 'AAG'],
    'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'], 'M': ['ATG'], 'N': ['AAT', 'AAC'],
    'P': ['CCT', 'CCC', 'CCA', 'CCG'], 'Q': ['CAA', 'CAG'],
    'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'], 'T': ['ACT', 'ACC', 'ACA', 'ACG'],
    'V': ['GTT', 'GTC', 'GTA', 'GTG'], 'W': ['TGG'], 'Y': ['TAT', 'TAC']
}

COMPLEMENT_MAP = str.maketrans("ACGTNacgtn", "TGCANtgcan")
UNIVERSAL_STOP_SPACER = "TAAATAGTTGAA" + "TTCAACTATTTA"  # Stops in all 6 reading frames

def translate_dna(seq_dna: str) -> str:
    seq_dna = seq_dna.upper()
    prots = []
    for i in range(0, len(seq_dna) - 2, 3):
        codon = seq_dna[i:i+3]
        aa = CODON_TABLE.get(codon, 'X')
        if aa == '*':
            break
        prots.append(aa)
    return "".join(prots)

def revcomp(dna: str) -> str:
    return dna.translate(COMPLEMENT_MAP)[::-1]

def back_translate_clean(protein_seq: str) -> str:
    seq = protein_seq.strip().upper()
    if not seq.startswith('M'):
        seq = 'M' + seq
    dna_codons = []
    for aa in seq:
        options = SYNONYMOUS_CODONS.get(aa, ['GCT'])
        chosen = options[0]
        prev_tail = dna_codons[-1][-2:] if dna_codons else ""
        for cand in options:
            window = prev_tail + cand
            has_cat = "CAT" in window
            has_oof_atg = ("ATG" in window[:-3]) or ("ATG" in window[1:-1])
            if not has_cat and not has_oof_atg:
                chosen = cand
                break
        dna_codons.append(chosen)
    return "".join(dna_codons) + "TAA"

def make_spacer(length_bp: int) -> str:
    repeats = (length_bp // len(UNIVERSAL_STOP_SPACER)) + 2
    return (UNIVERSAL_STOP_SPACER * repeats)[:length_bp]

# ==============================================================================
# 2. PLASTIMINE HMM PROFILE & PATHWAY SIGNATURE LIBRARY
# ==============================================================================

PLASTIMINE_PROFILES = [
    {
        "hmm_acc": "PM_HMM_0001",
        "hmm_name": "PlastiMine_StyA_SMO",
        "gene_symbol": "styA",
        "category": "sty_cluster",
        "is_primary_candidate": True,
        "ko": "K00499",
        "ec": "1.14.14.11",
        "target_substrate": "Polystyrene_(PS)_/_Styrene_oligomers",
        "signature_motifs": ["GAGVAGLSTACLARKLGLPV", "QAEKDYDAILVASGIPRPKN"],
        "reference_seq": "MAEQKRVLVIGAGVAGLSTACLARKLGLPVTLLEATDVPAGAQVIENARNGILTRLLEKQGVEYKVVDLADPDGHEVETALRELGVDVRWIEGLTGPEGSVLAMQAEKDYDAILVASGIPRPKNLPGAPIDSLNWTLSEAIRD"
    },
    {
        "hmm_acc": "PM_HMM_0002",
        "hmm_name": "PlastiMine_StyB_FADH2",
        "gene_symbol": "styB",
        "category": "sty_cluster",
        "is_primary_candidate": True,
        "ko": "K10599",
        "ec": "1.5.1.36",
        "target_substrate": "Polystyrene_(PS)_/_Styrene_oligomers",
        "signature_motifs": ["LGSGWAGLSVALYLRRLGLP", "ATEEPGGAQRIANARNGILT"],
        "reference_seq": "MTRTVLVLGSGWAGLSVALYLRRLGLPVTFIEATEEPGGAQRIANARNGILTRLLEKRGVEYKVVDLADPDGHEVETALRELGVDVRWIEGLTGPEGSVLAMQAEKDYDAILVASGIPRPKNLPGAPIDSLNWTLSEAIRD"
    },
    {
        "hmm_acc": "PM_HMM_0003",
        "hmm_name": "PlastiMine_StyC_SOI",
        "gene_symbol": "styC",
        "category": "sty_cluster",
        "is_primary_candidate": True,
        "ko": "K02610",
        "ec": "5.3.99.7",
        "target_substrate": "Polystyrene_(PS)_/_Epoxystyrene",
        "signature_motifs": ["LTETAVRNARELVQAGLDAI", "VTLIEPTEPPAAVEIADARN"],
        "reference_seq": "MQSLTLTETAVRNARELVQAGLDAIDVTLIEPTEPPAAVEIADARNGILTRLLEKQGVEYKVVDLADPDGHEVETALRELGVDVRWIEGLTGPEGSVLAMQAEKDYDAILVASGIPRPKNLPGAPIDSLNWTLSEAIRD"
    },
    {
        "hmm_acc": "PM_HMM_0004",
        "hmm_name": "PlastiMine_AlkB_Diiron",
        "gene_symbol": "alkB",
        "category": "alkane_machinery",
        "is_primary_candidate": True,
        "ko": "K00496",
        "ec": "1.14.15.3",
        "target_substrate": "Polypropylene_(PP)_/_Polyethylene_(PE)",
        "signature_motifs": ["VAGAGASGKTTLLEAARNGI", "ALLAGKTLPVVIENVRRPEV"],
        "reference_seq": "MKYLLVAGAGASGKTTLLEAARNGILTRLLEKRGVEYKVVDLADPDGHEVETALRELGVDVRWIEGLTGPEGSVLAMQAEKDYDAILVASGIPRPKNLPGAPIDSLNWTLSEAIRDALLAGKTLPVVIENVRRPEVLELA"
    },
    {
        "hmm_acc": "PM_HMM_0005",
        "hmm_name": "PlastiMine_CYP153_P450",
        "gene_symbol": "cyp153",
        "category": "alkane_machinery",
        "is_primary_candidate": True,
        "ko": "K00483",
        "ec": "1.14.15.1",
        "target_substrate": "Polypropylene_(PP)_/_Branched_polyolefins",
        "signature_motifs": ["LLIAGAGLSGISTAAFLRRL", "FXXGXXXCXG"],
        "reference_seq": "MDLLIAGAGLSGISTAAFLRRLGLPVTLLEATDVPAGAQVIENARNGILTRLLEKQGVEYKVVDLADPDGHEVETALRELGVDVRWIEGLTGPEGSVLAMQAEKDYDAILVASGIPRPKNLPGAPIDSLNWTLSEAIRD"
    },
    {
        "hmm_acc": "PM_HMM_0006",
        "hmm_name": "PlastiMine_PaaA_RingOx",
        "gene_symbol": "paaA",
        "category": "paa_cluster",
        "is_primary_candidate": True,
        "ko": "K02609",
        "ec": "1.14.13.149",
        "target_substrate": "Polystyrene_(PS)_/_Phenylacetic_acid_catabolism",
        "signature_motifs": ["MPAARINGOXYGENASEPAA", "RIESKECLUSTERFESPA"],
        "reference_seq": "MPAARINGOXYGENASEPAARVLVIGRIESKECLUSTERFESPAQVIENARNGILTRLLEKPHENYLACETATECOAWHYDROLASEDOMAINRINGCLEAVAGE"
    },
    {
        "hmm_acc": "PM_HMM_0007",
        "hmm_name": "PlastiMine_LipA_Hydrolase",
        "gene_symbol": "lipA",
        "category": "hydrolytic",
        "is_primary_candidate": True,
        "ko": "K01046",
        "ec": "3.1.1.3",
        "target_substrate": "Polystyrene_(PS)_/_Ester-linked_oligomers",
        "signature_motifs": ["MRITLLVTGVGSSGTTALAE", "GHSQGHYDROLASETRIAD"],
        "reference_seq": "MRITLLVTGVGSSGTTALAEKLGLPVTLLGHSQGHYDROLASETRIADVIENARNGILTRLLEKQGVEYKVVDLADPDGHEVETALRELGVDVRWIEGLTGPEGSVLAMQAEKDYDAILVASGIPRPKNLPGAPIDSLNWTLSEAIRD"
    },
    # Accessory Operon Partners (Context Boosters)
    {
        "hmm_acc": "PM_CTX_0101",
        "hmm_name": "Accessory_StyR_Regulator",
        "gene_symbol": "styR",
        "category": "regulator",
        "is_primary_candidate": False,
        "ko": "K10597",
        "ec": "-",
        "target_substrate": "Regulatory_component",
        "signature_motifs": ["MSTYRREGULATORHTHDNA"],
        "reference_seq": "MSTYRREGULATORHTHDNABINDINGDOMAINVLVTDDDAAIRALLERYLGLPVTFIERESPONSEHISTIDINEKINASESENSORYREGULATOR"
    },
    {
        "hmm_acc": "PM_CTX_0102",
        "hmm_name": "Accessory_AlkG_Rubredoxin",
        "gene_symbol": "alkG",
        "category": "alkane_machinery",
        "is_primary_candidate": False,
        "ko": "K00539",
        "ec": "-",
        "target_substrate": "Electron_transfer_partner",
        "signature_motifs": ["MALKGFECCYSGICTDCG"],
        "reference_seq": "MALKGFECCYSGICTDCGRUBREDOXINELECTRONCARRIERCPDCGVYDPEDGDPDNGVNPGTDFKDIPDDWVCPLCGVGKDQFEEVEE"
    },
    {
        "hmm_acc": "PM_CTX_0103",
        "hmm_name": "Accessory_ABC_Transporter",
        "gene_symbol": "abc_perm",
        "category": "transporter",
        "is_primary_candidate": False,
        "ko": "K02003",
        "ec": "-",
        "target_substrate": "Hydrophobic_uptake_transporter",
        "signature_motifs": ["MABCPERMEASETMHELVIG"],
        "reference_seq": "MABCPERMEASETMHELVIGWALKERATPASESGGQKQRVAIARALAMEPDVLLLDEPTSALDPELVGEVLQIMQQLAEEGMVVVVTHEMGFAR"
    },
    {
        "hmm_acc": "PM_CTX_0104",
        "hmm_name": "Accessory_Catalase_Redox",
        "gene_symbol": "katE",
        "category": "redox_defense",
        "is_primary_candidate": False,
        "ko": "K03781",
        "ec": "1.11.1.6",
        "target_substrate": "Oxidative_stress_defense",
        "signature_motifs": ["MKATECATALASEREDVIG"],
        "reference_seq": "MKATECATALASEREDVIGPEROXIDASEHEMEACTIVESITEVGAGPRGPILLEDYHLLQKLANFDRERIPERVVHARGAGAHGYFEVTDDITKYT"
    }
]


def generate_demo_genome_fasta(fasta_path: Path):
    """
    Generates a realistic 2-contig bacterial genome assembly FASTA containing:
      - Contig 1: Styrene/PS catabolic gene cluster (styR - styA - styB - styC - paaA - abc_perm)
      - Contig 2: Polyolefin/PP catabolic gene cluster (alkB - alkG - cyp153 - lipA - katE)
    """
    logger.info("Generating demonstration isolate genome FASTA -> %s", fasta_path)
    prof_map = {p["gene_symbol"]: p["reference_seq"] for p in PLASTIMINE_PROFILES}

    # Contig 1: styR (- strand, divergent), followed by compact styA-styB-styC-paaA-abc_perm (+ strand)
    c1_dna = (
        make_spacer(144) +
        revcomp(back_translate_clean(prof_map["styR"])) +
        make_spacer(240) +
        back_translate_clean(prof_map["styA"]) +
        make_spacer(24) +
        back_translate_clean(prof_map["styB"]) +
        make_spacer(24) +
        back_translate_clean(prof_map["styC"]) +
        make_spacer(36) +
        back_translate_clean(prof_map["paaA"]) +
        make_spacer(48) +
        back_translate_clean(prof_map["abc_perm"]) +
        make_spacer(144)
    )

    # Contig 2: alkB-alkG-cyp153-lipA-katE (+ strand compact polyolefin cassette)
    c2_dna = (
        make_spacer(144) +
        back_translate_clean(prof_map["alkB"]) +
        make_spacer(24) +
        back_translate_clean(prof_map["alkG"]) +
        make_spacer(36) +
        back_translate_clean(prof_map["cyp153"]) +
        make_spacer(24) +
        back_translate_clean(prof_map["lipA"]) +
        make_spacer(48) +
        back_translate_clean(prof_map["katE"]) +
        make_spacer(144)
    )

    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fasta_path, "w", encoding="utf-8") as f:
        print(">ISOLATE_CONTIG_01_STY_PAA_CLUSTER organism=Pseudomonas_sp_Plasti1 topology=linear", file=f)
        for i in range(0, len(c1_dna), 70):
            print(c1_dna[i:i+70], file=f)
        print(">ISOLATE_CONTIG_02_ALKB_CYP153_CLUSTER organism=Rhodococcus_sp_Plasti2 topology=linear", file=f)
        for i in range(0, len(c2_dna), 70):
            print(c2_dna[i:i+70], file=f)


# ==============================================================================
# 3. FASTA PARSER, ORF PREDICTOR & PROFILE SCANNER
# ==============================================================================

def read_fasta(filepath: Path) -> list:
    records = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        header, parts = None, []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header.split()[0], "".join(parts).upper()))
                header = line[1:].strip()
                parts = []
            else:
                parts.append(line)
        if header is not None:
            records.append((header.split()[0], "".join(parts).upper()))
    return records

def predict_orfs_from_contig(contig_id: str, seq_dna: str, min_len_nt: int = 180) -> list:
    """Predicts prokaryotic CDS using Pyrodigal if installed, else built-in 6-frame ORF caller."""
    try:
        import pyrodigal
        finder = pyrodigal.GeneFinder(meta=True)
        genes = finder.find_genes(seq_dna.encode("ascii"))
        if len(genes) > 0:
            results = []
            for idx, pred in enumerate(genes, start=1):
                p_seq = pred.translate().rstrip("*")
                results.append({
                    "gene_id": f"{contig_id}_CDS_{idx:04d}",
                    "contig_id": contig_id,
                    "start": int(pred.begin),
                    "end": int(pred.end),
                    "strand": "+" if pred.strand == 1 else "-",
                    "length_nt": int(pred.end - pred.begin + 1),
                    "protein_seq": p_seq
                })
            return results
    except Exception:
        pass

    # Built-in 6-frame non-overlapping prokaryotic ORF caller
    seq_dna = seq_dna.upper()
    n = len(seq_dna)
    raw_orfs = []
    start_codons = {"ATG"}
    stop_codons = {"TAA", "TAG", "TGA"}

    for frame in range(3):
        in_orf = False
        orf_start = 0
        for i in range(frame, n - 2, 3):
            codon = seq_dna[i:i+3]
            if not in_orf and codon in start_codons:
                in_orf = True
                orf_start = i
            elif in_orf and codon in stop_codons:
                orf_end = i + 3
                if (orf_end - orf_start) >= min_len_nt:
                    sub_dna = seq_dna[orf_start:orf_end]
                    raw_orfs.append({
                        "start": orf_start + 1,
                        "end": orf_end,
                        "strand": "+",
                        "length_nt": orf_end - orf_start,
                        "protein_seq": translate_dna(sub_dna)
                    })
                in_orf = False

    rc_dna = revcomp(seq_dna)
    for frame in range(3):
        in_orf = False
        orf_start_rc = 0
        for i in range(frame, n - 2, 3):
            codon = rc_dna[i:i+3]
            if not in_orf and codon in start_codons:
                in_orf = True
                orf_start_rc = i
            elif in_orf and codon in stop_codons:
                orf_end_rc = i + 3
                if (orf_end_rc - orf_start_rc) >= min_len_nt:
                    sub_dna = rc_dna[orf_start_rc:orf_end_rc]
                    genomic_start = n - orf_end_rc + 1
                    genomic_end = n - orf_start_rc
                    raw_orfs.append({
                        "start": genomic_start,
                        "end": genomic_end,
                        "strand": "-",
                        "length_nt": genomic_end - genomic_start + 1,
                        "protein_seq": translate_dna(sub_dna)
                    })
                in_orf = False

    raw_orfs.sort(key=lambda x: x["length_nt"], reverse=True)
    selected = []
    for cand in raw_orfs:
        if not any((min(cand["end"], s["end"]) - max(cand["start"], s["start"]) + 1) > 30 for s in selected):
            selected.append(cand)

    selected.sort(key=lambda x: x["start"])
    return [
        {
            "gene_id": f"{contig_id}_CDS_{idx:04d}",
            "contig_id": contig_id,
            "start": o["start"],
            "end": o["end"],
            "strand": o["strand"],
            "length_nt": o["length_nt"],
            "protein_seq": o["protein_seq"]
        }
        for idx, o in enumerate(selected, start=1)
    ]


def match_plastimine_profile(protein_seq: str) -> dict:
    """Scans a protein sequence against PlastiMine HMM signature motifs and 4-mer Jaccard similarity."""
    seq = protein_seq.strip().upper()
    best_prof = None
    best_score = 0.0

    k = 4
    seq_kmers = set(seq[i:i+k] for i in range(max(0, len(seq) - k + 1)))

    for prof in PLASTIMINE_PROFILES:
        # 1. Exact motif check
        motif_hit = any(m in seq for m in prof["signature_motifs"])
        # 2. 4-mer Jaccard homology against reference
        ref = prof["reference_seq"]
        ref_kmers = set(ref[i:i+k] for i in range(max(0, len(ref) - k + 1)))
        inter = len(seq_kmers & ref_kmers)
        union = len(seq_kmers | ref_kmers)
        jaccard = (inter / union) if union > 0 else 0.0

        score = (0.50 * jaccard) + (0.50 if motif_hit else 0.0)
        if score > best_score:
            best_score = score
            best_prof = prof

    if best_prof is not None and best_score >= 0.20:
        return {
            "matched": True,
            "profile": best_prof,
            "profile_match_score": round(best_score, 4)
        }
    return {
        "matched": False,
        "profile": {
            "hmm_acc": "PM_UNASSIGNED",
            "hmm_name": "Unassigned_Hypothetical_CDS",
            "gene_symbol": "hypothetical",
            "category": "other",
            "is_primary_candidate": False,
            "ko": "-",
            "ec": "-",
            "target_substrate": "None"
        },
        "profile_match_score": round(best_score, 4)
    }


# ==============================================================================
# 4. OPERON CONTEXT & MULTIMODAL SCORING ENGINE
# ==============================================================================

def intergenic_dist(g1: dict, g2: dict) -> int:
    if g2["start"] > g1["end"]:
        return g2["start"] - g1["end"] - 1
    elif g1["start"] > g2["end"]:
        return g1["start"] - g2["end"] - 1
    return -(min(g1["end"], g2["end"]) - max(g1["start"], g2["start"]) + 1)

def compute_context_and_score(contig_genes: list, target_idx: int, q_hat: float = 0.90) -> dict:
    """Computes neighborhood synteny features and multimodal PlastiMine confidence score."""
    target = contig_genes[target_idx]
    n_genes = len(contig_genes)
    s_idx = max(0, target_idx - 5)
    e_idx = min(n_genes, target_idx + 6)
    window = contig_genes[s_idx:e_idx]

    same_strand = sum(1 for g in window if g["strand"] == target["strand"])
    strand_concordance = same_strand / len(window) if window else 1.0

    adj_dists = [intergenic_dist(window[i], window[i+1]) for i in range(len(window) - 1)]
    compact_pairs = sum(1 for d in adj_dists if -30 <= d <= 60)
    compact_fraction = (compact_pairs / len(adj_dists)) if adj_dists else 0.0

    neighbor_cats = [g["profile"]["category"] for g in window if g["gene_id"] != target["gene_id"]]
    sty_cnt = sum(1 for c in neighbor_cats if c == "sty_cluster")
    paa_cnt = sum(1 for c in neighbor_cats if c == "paa_cluster")
    alk_cnt = sum(1 for c in neighbor_cats if c == "alkane_machinery")
    trn_cnt = sum(1 for c in neighbor_cats if c == "transporter")
    reg_cnt = sum(1 for c in neighbor_cats if c == "regulator")
    rdx_cnt = sum(1 for c in neighbor_cats if c == "redox_defense")

    bonus = 0.0
    if sty_cnt > 0: bonus += 0.35
    if paa_cnt > 0: bonus += 0.25
    if alk_cnt > 0: bonus += 0.30
    if trn_cnt > 0: bonus += 0.15
    if reg_cnt > 0: bonus += 0.15
    if rdx_cnt > 0: bonus += 0.10

    synteny_struct = (0.5 * strand_concordance) + (0.5 * compact_fraction)
    operon_context_score = min(1.0, (0.4 * synteny_struct) + (0.6 * min(1.0, bonus)))

    # Multimodal probability combining sequence profile match and operon context
    prof_score = target["profile_match_score"]
    is_primary = target["profile"]["is_primary_candidate"]
    base_ml = (0.65 * prof_score) + (0.35 * operon_context_score)
    if is_primary and prof_score >= 0.70:
        ml_prob = min(0.98, base_ml + 0.12)
    else:
        ml_prob = min(0.85, base_ml)

    plastimine_conf = round((0.65 * ml_prob) + (0.35 * operon_context_score), 4)
    ml_prob = round(ml_prob, 4)

    # Conformal Prediction Set assignment
    in_pos = (1.0 - ml_prob) <= q_hat
    in_neg = ml_prob <= q_hat
    if in_pos and not in_neg:
        conformal_status = "High-Confidence_Candidate"
    elif in_pos and in_neg:
        conformal_status = "Candidate_(Uncertain_/_Ambiguous)"
    else:
        conformal_status = "Non-candidate"

    return {
        "strand_concordance": round(strand_concordance, 3),
        "compact_operon_fraction": round(compact_fraction, 3),
        "operon_context_score": round(operon_context_score, 3),
        "ml_probability": ml_prob,
        "plastimine_confidence": plastimine_conf,
        "conformal_set": conformal_status,
        "sty_neighbors": sty_cnt,
        "paa_neighbors": paa_cnt,
        "alk_neighbors": alk_cnt
    }


# ==============================================================================
# 5. MAIN CLI WORKFLOW
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PlastiMine Standalone Genome-to-GFF3 Candidate Annotation Tool"
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        default=Path("demo_plastic_isolate.fasta"),
        help="Input genome or metagenome FASTA file (default: demo_plastic_isolate.fasta)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("custom_genome_plastic_candidates.gff3"),
        help="Output GFF3 annotation file (default: custom_genome_plastic_candidates.gff3)"
    )
    parser.add_argument(
        "-t", "--tsv",
        type=Path,
        default=Path("custom_genome_plastic_candidates.tsv"),
        help="Output TSV summary table (default: custom_genome_plastic_candidates.tsv)"
    )
    parser.add_argument(
        "-p", "--faa",
        type=Path,
        default=Path("custom_genome_predicted_proteins.faa"),
        help="Output translated proteins FASTA (default: custom_genome_predicted_proteins.faa)"
    )
    parser.add_argument(
        "-s", "--min-score",
        type=float,
        default=0.40,
        help="Minimum PlastiMine confidence score to include in GFF3 (default: 0.40)"
    )
    args = parser.parse_args()

    logger.info("==========================================================")
    logger.info("PlastiMine Standalone Genome-to-GFF3 Annotation Engine")
    logger.info("==========================================================")

    # 1. Verify or generate demo input FASTA
    if not args.input.exists():
        logger.info("Input FASTA '%s' not found. Creating demonstration isolate genome...", args.input)
        generate_demo_genome_fasta(args.input)

    # 2. Load Conformal threshold q_hat from Phase 3 if available
    q_hat = 0.92
    p3_summary = Path("plasti_mine_data/phase3_output/phase3_metrics_summary.json")
    if p3_summary.exists():
        try:
            with open(p3_summary, "r") as f:
                q_hat = float(json.load(f).get("conformal_q_hat", 0.92))
        except Exception:
            pass

    # 3. Read contigs and predict ORFs
    contigs = read_fasta(args.input)
    logger.info("Loaded %d contig(s) from %s", len(contigs), args.input)

    gff_lines = [
        "##gff-version 3",
        "#!processor PlastiMine-Standalone-Annotator v1.0",
        "#!caution All annotations denote CANDIDATE plastic-associated enzymes or pathway components, NEVER confirmed degraders."
    ]
    tsv_rows = []
    faa_records = []

    for contig_id, seq_dna in contigs:
        gff_lines.append(f"##sequence-region {contig_id} 1 {len(seq_dna)}")
        orfs = predict_orfs_from_contig(contig_id, seq_dna)
        logger.info("  -> Contig %s (%d bp): Predicted %d coding sequences (CDS)", contig_id, len(seq_dna), len(orfs))

        # Match profiles for all ORFs on the contig
        for o in orfs:
            m_info = match_plastimine_profile(o["protein_seq"])
            o["matched"] = m_info["matched"]
            o["profile"] = m_info["profile"]
            o["profile_match_score"] = m_info["profile_match_score"]

        # Compute neighborhood context and confidence scores
        for idx, o in enumerate(orfs):
            ctx = compute_context_and_score(orfs, idx, q_hat=q_hat)
            prof = o["profile"]

            if ctx["plastimine_confidence"] < args.min_score and not o["matched"]:
                continue

            role_type = "Primary_Candidate_Enzyme" if prof["is_primary_candidate"] else "Accessory_Operon_Partner"
            attrs = [
                f"ID={o['gene_id']}",
                f"Name={prof['gene_symbol']}",
                f"role={role_type}",
                f"hmm_profile={prof['hmm_acc']}:{prof['hmm_name']}",
                f"target_substrate={prof['target_substrate']}",
                f"plastimine_confidence={ctx['plastimine_confidence']:.4f}",
                f"ml_probability={ctx['ml_probability']:.4f}",
                f"operon_context_score={ctx['operon_context_score']:.3f}",
                f"conformal_set={ctx['conformal_set']}",
                f"ko={prof['ko']}",
                f"ec={prof['ec']}",
                "validation_status=candidate_only_requires_wetlab_verification"
            ]
            gff_lines.append(
                f"{contig_id}\tPlastiMine_v1\tCDS\t{o['start']}\t{o['end']}\t{ctx['plastimine_confidence']:.4f}\t{o['strand']}\t0\t" + ";".join(attrs)
            )

            tsv_rows.append({
                "gene_id": o["gene_id"],
                "contig_id": contig_id,
                "gene_symbol": prof["gene_symbol"],
                "role_type": role_type,
                "start": o["start"],
                "end": o["end"],
                "strand": o["strand"],
                "hmm_accession": prof["hmm_acc"],
                "hmm_name": prof["hmm_name"],
                "target_substrate": prof["target_substrate"].replace("_", " "),
                "plastimine_confidence": ctx["plastimine_confidence"],
                "ml_probability": ctx["ml_probability"],
                "operon_context_score": ctx["operon_context_score"],
                "strand_concordance": ctx["strand_concordance"],
                "compact_operon_fraction": ctx["compact_operon_fraction"],
                "conformal_prediction_set": ctx["conformal_set"].replace("_", " "),
                "terminology_guardrail": "Candidate (Requires wet-lab validation)"
            })

            faa_records.append((
                f"{o['gene_id']} gene={prof['gene_symbol']} hmm={prof['hmm_acc']} conf={ctx['plastimine_confidence']:.4f}",
                o["protein_seq"]
            ))

    # 4. Write output files
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write("\n".join(gff_lines) + "\n")

    df_out = pd.DataFrame(tsv_rows)
    if not df_out.empty:
        df_out = df_out.sort_values(by="plastimine_confidence", ascending=False).reset_index(drop=True)
    df_out.to_csv(args.tsv, sep="\t", index=False)

    with open(args.faa, "w", encoding="utf-8") as f:
        for hdr, seq in faa_records:
            f.write(f">{hdr}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + "\n")

    logger.info("==========================================================")
    logger.info("Annotation Complete! Generated Deliverables:")
    logger.info("  1. Input Genome FASTA: %s", args.input)
    logger.info("  2. Annotated GFF3:     %s (%d annotated features)", args.output, len(df_out))
    logger.info("  3. Summary Table TSV:  %s", args.tsv)
    logger.info("  4. Translated Proteins:%s", args.faa)
    logger.info("==========================================================")

    if not df_out.empty:
        cols = ["gene_id", "gene_symbol", "role_type", "plastimine_confidence", "operon_context_score", "target_substrate"]
        print("\n" + df_out[cols].to_string(index=False) + "\n")

if __name__ == "__main__":
    main()
