# Template Collapse and Information-Theoretic Limits in Camera rPPG Pulse Morphology Restoration

**arXiv:2606.03802**
Achraf Ben Ahmed, PlesmoSense SARL

---

## Overview

This repository contains the full source code for our arXiv paper on rPPG morphological restoration. We investigate whether a deep generative prior trained on clinical-grade contact PPG can decode subject-specific pulse morphology from consumer camera rPPG signals.

**Key finding:** Template collapse is an information-theoretic limit of the camera input signal — not an architectural failure. All 16 tested architectures collapse to the population-average waveform. The SupCon contrastive family confirms this: 6 independent variants all converge to log(N) = 4.844, the theoretical null for a batch of N = 127 samples.

---

## Architecture naming

| Paper name | Internal ID | Description |
|---|---|---|
| VAE-Base | V5-B | Baseline VAE + CameraEncoder (CHROM input) |
| VAE-Orth | V6 | Orthogonal macro/micro disentanglement (cascaded decoder) |
| VAE-Large | A1 | Double latent dimension (z = 64) |
| VAE-Flow | A2 | Conditional normalising flow decoder |
| VQ-VAE | A3 | Vector-quantised discrete codebook |
| Trans-Multi | A4 | Multi-cycle Transformer encoder |
| Trans-rPPG | A4-B | Trans-Multi restricted to rPPG-only input |
| Two-Stage+Div | A5-v4 | Frozen VAE-Base + trainable RefineNet with subject-mean diversity penalty |
| RGB-Window | A6-D | Raw RGB window encoder (camera-only, bypasses rPPG preprocessing) |
| RGB-Physics | A7 | Physics-informed RGB encoder |
| RGB-FPS | A8-v2 | FPS-agnostic camera-only encoder |
| Diffusion-z | A9 | DDPM in latent z-space with classifier-free guidance |
| DPS-rPPG | A10 | Diffusion posterior sampling + rPPG likelihood |
| VMD-6ch | A11 | VMD 6-channel feature encoder |
| VMD-Peak | A12 | VMD peak-aligned cycle encoder |
| SupCon | A13 | Supervised contrastive (information-theoretic null, 6 sub-variants) |

---

## Datasets

| Dataset | Subjects | PPG Hz | Availability |
|---|---|---|---|
| UBFC-rPPG | 42 | 64 (CMS50E) | Public — see below |
| UBFC-PHYS | 56 | 64 (CMS50E) | Public — see below |
| DS1 (In-House) | 9 | 1000 (Polymate) | Not publicly available |
| DS2 (In-House) | 46 | 500 (Polymate) | Not publicly available |

**Download public datasets:**
- UBFC-rPPG: https://sites.google.com/view/ybenezeth/ubfcrppg
- UBFC-PHYS: https://sites.google.com/view/ybenezeth/ubfc-phys

The in-house Polymate datasets (DS1/DS2) are used to train the Stage 1 VAE prior and are not publicly available. Reproducing the full paper results requires these datasets. Results on UBFC data alone can be obtained by training the VAE on UBFC-PHYS (64 Hz), which will yield a less morphologically rich prior — see the paper discussion.

---

## Setup

### 1. Install dependencies

```bash
conda create -n rppg_morph python=3.10
conda activate rppg_morph
pip install -r requirements.txt
```

### 2. Configure paths

Edit `morph_research_pipeline/config/paths.py` — set the two blocks marked **EDIT THIS**:

```python
# Results/checkpoints root
ROOT_E = Path(r'C:\your\results\rPPG_Morphology_Restore')

# Raw dataset roots
UBFC_RPPG_RAW_ROOT = r'C:\your\raw\UBFC_2'
UBFC_PHYS_RAW_ROOT = r'C:\your\raw\UBFC-Phys'
```

---

## Reproduction pipeline

### Step 1 — Parse raw video to RGB patch CSVs

```bash
python -m morph_research_pipeline.extraction.morph_parse_ubfc
```

### Step 2 — Extract rPPG signals

```bash
python -m morph_research_pipeline.extraction.extract_rppg
```

### Step 3 — Extract cardiac cycles

```bash
python -m morph_research_pipeline.extraction.extract_cycles
python -m morph_research_pipeline.extraction.patch_chrom_cycles
```

### Step 4 — Train Stage 1 VAE (morphological prior)

```bash
python -m morph_research_pipeline.training.train_vae
```

### Step 5 — Train Stage 2 encoder for each architecture

```bash
# Example: VAE-Base (V5-B)
python -m morph_research_pipeline.training.v5.train_encoders

# Example: Trans-rPPG (A4)
python -m morph_research_pipeline.training.a4.train_a4

# Repeat for a1–a13 as needed
```

### Step 6 — Evaluate

```bash
# VAE-Base (V5-B)
python -m morph_research_pipeline.evaluation.v5.evaluate

# Trans-rPPG (A4)
python -m morph_research_pipeline.evaluation.a4.evaluate_a4

# Repeat for each architecture
```

### Step 7 — Generate paper figures

```bash
$env:MPLBACKEND = "Agg"  # PowerShell
# export MPLBACKEND=Agg  # bash

python -m morph_research_pipeline.plotting.fig_gt_diversity        # Fig. 1
python -m morph_research_pipeline.plotting.fig_a13_curves          # Fig. 2
python -m morph_research_pipeline.plotting.fig_harmonic_restoration # Fig. 3
python -m morph_research_pipeline.plotting.fig_template_collapse   # Fig. 4
python -m morph_research_pipeline.plotting.fig_hallucination_gap   # Fig. 5
python -m morph_research_pipeline.plotting.fig_collapse_scatter    # Fig. 6
python -m morph_research_pipeline.plotting.fig_waveform_restoration # Fig. 7
```

### Verify split integrity

```bash
python -m morph_research_pipeline.evaluation.shared.smoke_test
```

---

## Bootstrap CI on GT ceiling (Table 1 footnote)

```bash
python bootstrap_gt_ceiling.py
```

Reproduces: GT cross-subject r = 0.601, 95% CI [0.502, 0.763], N = 27 test subjects, 1000 iterations.

---

## Repository structure

```
morph_research_pipeline/
├── config/          — paths.py (EDIT THIS), hyperparams.py
├── extraction/      — video parsing, rPPG extraction, cycle extraction
├── models/          — VAE, all 16 encoder architectures
├── training/        — training scripts (train_vae.py, v5/, v6/, a1/–a13/)
├── evaluation/      — evaluation scripts (v5/, v6/, a1/–a13/, shared/)
└── plotting/        — 7 paper figures + 2 supplementary figures
bootstrap_gt_ceiling.py  — bootstrap CI on GT ceiling
requirements.txt
```

---

## Citation

If you use this code, please cite:

```
@article{benahmed2026rppg,
  title          = {Template Collapse and Information-Theoretic Limits in Camera rPPG Pulse Morphology Restoration},
  author         = {Ben Ahmed, Achraf},
  year           = {2026},
  eprint         = {2606.03802},
  archivePrefix  = {arXiv},
  primaryClass   = {eess.IV},
  url            = {https://arxiv.org/abs/2606.03802}
}
```
