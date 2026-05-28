"""
hyperparams.py — Training hyperparameters and architecture config.
==================================================================
Split into sections: shared, V5, V6.
"""

# ==============================================================================
# SHARED
# ==============================================================================

LATENT_DIM          = 32
CYCLE_SAMPLES       = 256
BATCH_SIZE          = 128
LEARNING_RATE       = 1e-4
SPLIT_SEED          = 42
TRAIN_FRAC          = 0.70
VAL_FRAC            = 0.15
DATALOADER_WORKERS  = 0
DATALOADER_PIN_MEMORY = True

# ==============================================================================
# VAE (Stage 1 — shared backbone)
# ==============================================================================

MAX_EPOCHS_VAE      = 100
EARLY_STOP_PATIENCE = 15
BETA_KL             = 0.5
VAE_HIGH_QUALITY_ONLY = True
VAE_MIN_SID         = 2000

# ==============================================================================
# V5 — CameraEncoder training
# ==============================================================================

MAX_EPOCHS_V5       = 300
EARLY_STOP_V5       = 30

LAMBDA_L1           = 10.0
LAMBDA_NOTCH        = 5.0
LAMBDA_SDTW         = 1.0
LAMBDA_CURV         = 0.5
LAMBDA_LATENT       = 0.05
LAMBDA_VARIANCE     = 2.0
LAMBDA_ADV          = 0.05
ADV_START_EPOCH     = 50
LAMBDA_DIVERSITY    = 1.0
LAMBDA_FREQ         = 0.5
LAMBDA_ASYM         = 0.3
LAMBDA_SPECTRAL     = 1.0
LAMBDA_AUX_MORPHO   = 2.0

MORPHO_AUX_HEADS    = True
CONTRASTIVE_EPOCHS  = 50
CONTRASTIVE_LR      = 1e-4
CONTRASTIVE_TEMP    = 0.07
CONTRASTIVE_BATCH   = 128
PRETRAIN_ENCODER    = 'B'

N_SUBJECTS_PER_BATCH = 16

# ==============================================================================
# V6 — Orthogonal Cascade
# ==============================================================================

MAX_EPOCHS_V6       = 300
EARLY_STOP_V6       = 15

LAMBDA_GRL          = 0.252
LAMBDA_ORTHO        = 1.896
LAMBDA_MORPHO       = 9.856
LAMBDA_ID_CE        = 0.816
LAMBDA_CONTRASTIVE  = 0.5
LAMBDA_MACRO_ID_CE  = 0.3
ID_WARMUP_EPOCHS    = 20

# ==============================================================================
# V5 — Per-subject adapter
# ==============================================================================

ADAPTER_LR          = 1e-3
ADAPTER_EPOCHS      = 100
ADAPTER_PATIENCE    = 10
ADAPTER_MIN_CYCLES  = 10

# ==============================================================================
# A4 — Multi-Cycle Transformer
# ==============================================================================

A4_NUM_CYCLES       = 5
A4_D_MODEL          = 256
A4_NHEAD            = 8
A4_N_LAYERS         = 4
A4_DROPOUT          = 0.1
A4_MAX_EPOCHS       = 300
A4_EARLY_STOP       = 30

# ==============================================================================
# DATASET METADATA
# ==============================================================================

HR_MIN_BPM          = 40
HR_MAX_BPM          = 150
MAX_MISSING_FRAC    = 0.15

UBFC_RPPG_SID_OFFSET  = 0
UBFC_PHYS_SID_OFFSET  = 1000
UBFC_PHYS_BVP_HZ      = 64
STRESS2023_SID_OFFSET  = 2000
STRESS2023_PPG_HZ      = 500
FPS2023_SID_OFFSET     = 3000
FPS2023_PPG_HZ         = 500
CENTAN_SID_OFFSET      = 4000
CENTAN_PPG_HZ          = 1000
CHINA_SID_OFFSET       = 5000

FPS2023_SUB_MAP = {
    '0316': 'sub1_20230316_ext.pkl',
    '0323': 'sub2_20230323_ext.pkl',
    '0330': 'sub3_20230330_ext.pkl',
    '0331': 'sub4_20230331_ext.pkl',
    '0404': 'sub5_20230404_ext.pkl',
}

CENTAN_SUBJECT_MAP = {
    's03': {'folder': 's03(machida)',  'csvs': ['200304_Picture.CSV', '200304_subtraction.CSV']},
    's04': {'folder': 's04(herai)',     'csvs': ['200311_SubtractionPicture.CSV']},
    's05': {'folder': 's05(nagata)',    'csvs': ['200318_SubtractionPicture.CSV']},
    's06': {'folder': 's06(shirasuna)', 'csvs': ['200324_SubtractionPicture.CSV']},
    's07': {'folder': 's07(amemiya)',   'csvs': ['200325_SubtractionPicture.CSV']},
    's08': {'folder': 's08(kimura)',    'csvs': ['200326_SubtractionPicture.CSV']},
    's09': {'folder': 's09(achraf)',    'csvs': ['200331_1_PictureSubtraction.CSV']},
    's10': {'folder': 's10(kin)',       'csvs': ['200331_2_PictureSubtraction.CSV']},
    's11': {'folder': 's11(motoyama)',  'csvs': ['200402_PictureSubtraction.CSV']},
}

# ==============================================================================
# RPPG ALGORITHMS
# ==============================================================================

RPPG_ALGOS = ['pos', 'chrom', 'phybrid', 'graw']
PATCH_NAMES = ['forehead', 'cheeks_top', 'cheeks_bot', 'nose_chin']
PARSE_WORKERS = 8
EXTRACT_WORKERS = 5
