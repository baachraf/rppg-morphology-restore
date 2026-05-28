"""
morph_config.py — BACKWARD COMPATIBILITY SHIM
================================================
All config now lives in config/paths.py and config/hyperparams.py.
This file re-exports everything so existing scripts keep working.
New scripts should import from config.paths and config.hyperparams directly.
"""
from config.paths import *
from config.hyperparams import *

# Ensure the new 60fps dir exists at import time
from config.paths import FPS2023_60_CSV_DIR
FPS2023_60_CSV_DIR.mkdir(parents=True, exist_ok=True)

# Aliases that old scripts use
MORPH_STANDALONE_ROOT = str(ROOT_E / 'morph_standalone')
MORPH_ROOT            = str(ROOT_E)
MORPH_RESULTS_DIR     = str(RESULTS_DIR)
CHECKPOINTS_DIR       = str(CKPT_DIR)
MORPH_PARSE_ROOT      = MORPH_STANDALONE_ROOT
MORPH_CSV_DIR         = str(UBFC_CSV_DIR)
MORPH_CYCLES_DIR      = str(UBFC_CYCLES_DIR)

# Old checkpoint name patterns (include subdirectory so scripts doing
# Path(CHECKPOINTS_DIR) / pattern still resolve correctly)
VAE_CKPT_P4           = 'shared/stage1_vae_p4.pt'
ENCODER_CKPT_P5       = 'v5/encoders/encoder_{name}.pt'
CONTRASTIVE_CKPT      = 'v5/pretrain_encoder_{name}.pt'
MACRO_ENCODER_V6_CKPT = 'macro_encoder.pt'
MICRO_ENCODER_V6_CKPT = 'micro_encoder.pt'
COND_DECODER_V6_CKPT  = 'cond_decoder.pt'
ENCODER_CKPT_P4       = ENCODER_CKPT_P5

VAE_CKPT_P5           = VAE_CKPT_P4

# Old aliases
RESULTS_DIR           = str(RESULTS_DIR)
V6_RESULTS_DIR        = str(RESULTS_V6)
V5_RESULTS_DIR        = str(RESULTS_V5)
V6_CKPT_DIR           = str(CKPT_V6)
A1_RESULTS_DIR        = str(RESULTS_A1)
A2_RESULTS_DIR        = str(RESULTS_A2)
A3_RESULTS_DIR        = str(RESULTS_A3)
A4_RESULTS_DIR        = str(RESULTS_A4)
A6_RESULTS_DIR        = str(RESULTS_A6)
A1_CKPT_DIR           = str(CKPT_A1)
A2_CKPT_DIR           = str(CKPT_A2)
A3_CKPT_DIR           = str(CKPT_A3)
A4_CKPT_DIR           = str(CKPT_A4)
A6_CKPT_DIR           = str(CKPT_A6)
CKPT_V5_ADAPTERS      = str(CKPT_V5_ADAPTERS)
CKPT_EVERY_N_EPOCHS   = 10
FORCE_RETRAIN_STAGE2  = True
MAX_EPOCHS_STAGE2     = MAX_EPOCHS_V5
# V5 training should use patience=30, not the VAE's 15
EARLY_STOP_PATIENCE   = EARLY_STOP_V5
CHUNK_SIZE            = 5
