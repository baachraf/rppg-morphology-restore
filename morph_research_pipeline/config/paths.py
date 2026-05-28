"""
paths.py — Single source of truth for all filesystem paths.
=============================================================
SETUP: Set the two blocks marked "EDIT THIS" below to match your machine.
Everything else is derived automatically.
"""
import os
from pathlib import Path

# ==============================================================================
# EDIT THIS — Results/checkpoints root
# ==============================================================================
# Choose any directory with sufficient disk space (~20 GB for all architectures).
# The pipeline creates all required subdirectories automatically.

ROOT_E = Path(r'C:\path\to\your\results\rPPG_Morphology_Restore')
# Examples:
#   Windows:  Path(r'D:\Results\rPPG_Morphology_Restore')
#   Linux:    Path('/home/user/results/rPPG_Morphology_Restore')

# ==============================================================================
# DERIVED PATHS — do not edit below
# ==============================================================================

# --- Processed data (written by extraction scripts) ---
DATA_DIR       = ROOT_E / 'data'
SPLIT_FILE     = DATA_DIR / 'subject_split_audited.csv'

# Per-dataset dirs (under DATA_DIR)
UBFC_DIR       = DATA_DIR / 'ubfc'
STRESS_DIR     = DATA_DIR / 'stress2023'
FPS2023_DIR    = DATA_DIR / 'fps2023'
CENTAN_DIR     = DATA_DIR / 'centan'

# Parsed RGB patch CSVs (from video parsing)
UBFC_CSV_DIR   = UBFC_DIR   / 'parsed'
STRESS_CSV_DIR = STRESS_DIR / 'parsed'
FPS2023_CSV_DIR = FPS2023_DIR / 'parsed'
CENTAN_CSV_DIR = CENTAN_DIR / 'parsed'

# rPPG extracted signals
UBFC_RPPG_DIR   = UBFC_DIR   / 'rppg'
STRESS_RPPG_DIR = STRESS_DIR / 'rppg'
FPS2023_RPPG_DIR = FPS2023_DIR / 'rppg'
CENTAN_RPPG_DIR = CENTAN_DIR / 'rppg'

# GT contact PPG (NPZ)
UBFC_PPG_DIR   = UBFC_DIR   / 'ppg'
STRESS_PPG_DIR = STRESS_DIR / 'ppg'
FPS2023_PPG_DIR = FPS2023_DIR / 'ppg'
CENTAN_PPG_DIR = CENTAN_DIR / 'ppg'

# Cardiac cycles (NPZ, 256-sample, resampled)
UBFC_CYCLES_DIR   = UBFC_DIR   / 'cycles'
STRESS_CYCLES_DIR = STRESS_DIR / 'cycles'
FPS2023_CYCLES_DIR = FPS2023_DIR / 'cycles'
CENTAN_CYCLES_DIR = CENTAN_DIR / 'cycles'

# Backward-compat aliases
CYCLES_DIR            = DATA_DIR
PARSED_DIR            = DATA_DIR
UBFC_CYCLES_V4_DIR    = UBFC_CYCLES_DIR
STRESS_CYCLES_V4_DIR  = STRESS_CYCLES_DIR
FPS2023_CYCLES_V4_DIR = FPS2023_CYCLES_DIR
CENTAN_CYCLES_V4_DIR  = CENTAN_CYCLES_DIR

# --- Checkpoints ---
CKPT_DIR     = ROOT_E / 'checkpoints'
CKPT_SHARED  = CKPT_DIR / 'shared'
CKPT_V5      = CKPT_DIR / 'v5'
CKPT_V6      = CKPT_DIR / 'v6'
CKPT_A1      = CKPT_DIR / 'a1'
CKPT_A2      = CKPT_DIR / 'a2'
CKPT_A3      = CKPT_DIR / 'a3'
CKPT_A4      = CKPT_DIR / 'a4'
CKPT_A5      = CKPT_DIR / 'a5'
CKPT_A6      = CKPT_DIR / 'a6'
CKPT_A7      = CKPT_DIR / 'a7'
CKPT_A8      = CKPT_DIR / 'a8'
CKPT_A9      = CKPT_DIR / 'a9'
CKPT_A10     = CKPT_DIR / 'a10'
CKPT_A11     = CKPT_DIR / 'a11'
CKPT_A12     = CKPT_DIR / 'a12'
CKPT_A13     = CKPT_DIR / 'a13'
CKPT_V5_ENCODERS = CKPT_V5 / 'encoders'
CKPT_V5_ADAPTERS = CKPT_V5 / 'adapters'  # kept for import compatibility

VAE_CKPT        = CKPT_SHARED / 'stage1_vae_p4.pt'
ENCODER_CKPT    = {e: CKPT_V5_ENCODERS / f'encoder_{e}.pt' for e in ('A', 'B', 'C')}
V6_MACRO_CKPT   = CKPT_V6 / 'macro_encoder.pt'
V6_MICRO_CKPT   = CKPT_V6 / 'micro_encoder.pt'
V6_DECODER_CKPT = CKPT_V6 / 'cond_decoder.pt'

# --- Results ---
RESULTS_DIR    = ROOT_E / 'results'
FIGS           = RESULTS_DIR / 'figures'
RESULTS_SHARED = RESULTS_DIR / 'shared'
RESULTS_V5     = RESULTS_DIR / 'v5'
RESULTS_V6     = RESULTS_DIR / 'v6'
RESULTS_A1     = RESULTS_DIR / 'a1'
RESULTS_A2     = RESULTS_DIR / 'a2'
RESULTS_A3     = RESULTS_DIR / 'a3'
RESULTS_A4     = RESULTS_DIR / 'a4'
RESULTS_A5     = RESULTS_DIR / 'a5'
RESULTS_A6     = RESULTS_DIR / 'a6'
RESULTS_A7     = RESULTS_DIR / 'a7'
RESULTS_A8     = RESULTS_DIR / 'a8'
RESULTS_A9     = RESULTS_DIR / 'a9'
RESULTS_A10    = RESULTS_DIR / 'a10'
RESULTS_A11    = RESULTS_DIR / 'a11'
RESULTS_A12    = RESULTS_DIR / 'a12'
RESULTS_A13    = RESULTS_DIR / 'a13'
BASELINE_CYCLE = RESULTS_SHARED / 'baseline_mean_cycle.npy'

# --- Auto-create output directories ---
for _d in [
    CKPT_SHARED, CKPT_V5_ENCODERS, CKPT_V5_ADAPTERS,
    CKPT_V6, CKPT_A1, CKPT_A2, CKPT_A3, CKPT_A4, CKPT_A5, CKPT_A6, CKPT_A7,
    CKPT_A8, CKPT_A9, CKPT_A10, CKPT_A11, CKPT_A12, CKPT_A13,
    FIGS, FIGS / 'raw_data',
    RESULTS_SHARED, RESULTS_V5, RESULTS_V6,
    RESULTS_A1, RESULTS_A2, RESULTS_A3, RESULTS_A4, RESULTS_A5, RESULTS_A6, RESULTS_A7,
    RESULTS_A8, RESULTS_A9, RESULTS_A10, RESULTS_A11, RESULTS_A12, RESULTS_A13,
    UBFC_CSV_DIR,   UBFC_RPPG_DIR,   UBFC_PPG_DIR,   UBFC_CYCLES_DIR,
    STRESS_CSV_DIR, STRESS_RPPG_DIR, STRESS_PPG_DIR, STRESS_CYCLES_DIR,
    FPS2023_CSV_DIR, FPS2023_RPPG_DIR, FPS2023_PPG_DIR, FPS2023_CYCLES_DIR,
    CENTAN_CSV_DIR, CENTAN_RPPG_DIR, CENTAN_PPG_DIR, CENTAN_CYCLES_DIR,
]:
    _d.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# EDIT THIS — Raw video dataset roots (read-only, never written to)
# ==============================================================================
# Public datasets — download links in README.md:
#   UBFC-rPPG:  https://sites.google.com/view/ybenezeth/ubfcrppg
#   UBFC-PHYS:  https://sites.google.com/view/ybenezeth/ubfc-phys

UBFC_RPPG_RAW_ROOT = r'C:\path\to\raw\UBFC_2'
UBFC_PHYS_RAW_ROOT = r'C:\path\to\raw\UBFC-Phys'

# In-house datasets (not publicly available — results reported in paper use these):
#   DS1/DS2: Internal Polymate 500/1000 Hz recordings (CENTAN/PlesmoSense SARL)
#   Stress 2023 / FPS 2023: Internal Polymate 500 Hz recordings
# Set paths if you have access; leave as-is otherwise.
STRESS2023_ROOT = r'C:\path\to\raw\Stress_2023_DataSet'
FPS2023_ROOT    = r'C:\path\to\raw\FPS_Analysis_April_2023'
FPS2023_VIDEO_DIR = os.path.join(FPS2023_ROOT, 'videos')
FPS2023_DATA_DIR  = os.path.join(FPS2023_ROOT, 'polymate_ext', 'data')
CENTAN_ROOT     = r'C:\path\to\raw\CENTAN_rPPG_video_data'
