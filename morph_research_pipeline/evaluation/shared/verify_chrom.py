"""
verify_chrom.py — Verify encoder_B.pt reproduces V5-B r=0.711 with CHROM input
================================================================================
Run this AFTER re-extracting rPPG with the updated extract_rppg.py and
re-extracting cycles with the updated extract_cycles.py (FORCE_REEXTRACT=True).

Expected result: per-subject r ≈ 0.711 (±0.01) confirms that:
  - CHROM is now correctly in the extraction pipeline
  - encoder_B.pt was indeed trained on CHROM (not POS)
  - A5 Stage 1 and A13 should use rppg_chrom_cycles as input

Usage:
    python evaluation/shared/verify_chrom.py
"""

import os, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.stats import pearsonr
from tqdm import tqdm

HERE = Path(__file__).parent
PIPELINE_ROOT = HERE.parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from config.paths import (
    SPLIT_FILE, ENCODER_CKPT, VAE_CKPT,
    UBFC_CYCLES_DIR, STRESS_CYCLES_DIR, FPS2023_CYCLES_DIR, CENTAN_CYCLES_DIR,
)
from models.encoder import CameraEncoder
from models.vae import PPGVAE

CYCLES_DIRS = [UBFC_CYCLES_DIR, STRESS_CYCLES_DIR, FPS2023_CYCLES_DIR, CENTAN_CYCLES_DIR]
_ENCODER_CKPT = ENCODER_CKPT['B']

LATENT_DIM = 32
IN_CHANNELS = 1


def shape_r(a, b):
    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return 0.0
    return float(pearsonr(a, b)[0])


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split_df = pd.read_csv(SPLIT_FILE)
    test_sids = set(split_df[split_df['split'] == 'test']['sid'])

    # Load encoder_B
    if not _ENCODER_CKPT.exists():
        print(f'ERROR: encoder_B.pt not found at {_ENCODER_CKPT}')
        return
    enc_state = torch.load(_ENCODER_CKPT, map_location=device, weights_only=False)
    encoder = CameraEncoder(in_channels=IN_CHANNELS, latent_dim=LATENT_DIM).to(device)
    encoder.load_state_dict(enc_state['encoder'], strict=False)
    encoder.eval()

    # Load stage1 VAE from stage1_vae_p4.pt — matches evaluate_v5.py methodology
    if not VAE_CKPT.exists():
        print(f'ERROR: stage1_vae_p4.pt not found at {VAE_CKPT}')
        return
    vae = PPGVAE(latent_dim=LATENT_DIM).to(device)
    vae.load_state_dict(torch.load(VAE_CKPT, map_location=device, weights_only=False))
    vae.eval()

    # Collect test cycles
    rows_by_sid = {}
    n_missing_chrom = 0

    for cycles_dir in CYCLES_DIRS:
        cycles_dir = Path(cycles_dir)
        if not cycles_dir.is_dir():
            continue
        for npz_path in sorted(cycles_dir.glob('*_cycles.npz')):
            if '_vmd' in npz_path.name or '_a12_' in npz_path.name:
                continue
            try:
                data = np.load(npz_path, allow_pickle=True)
                sid = int(data['sid'])
                if sid not in test_sids:
                    continue
                if 'rppg_pos_cycles' not in data:
                    n_missing_chrom += 1
                    continue
                gt = data['gt_cycles']
                rppg = data['rppg_pos_cycles']
                rows_by_sid.setdefault(sid, []).extend(list(zip(rppg, gt)))
            except Exception:
                continue

    if n_missing_chrom > 0:
        print(f'WARNING: {n_missing_chrom} NPZ files missing rppg_pos_cycles key.')

    if not rows_by_sid:
        print('ERROR: No test cycles with POS found.')
        return

    print(f'\nFound POS cycles for {len(rows_by_sid)} test subjects.')

    # Evaluate
    per_subj_rs = []
    with torch.no_grad():
        for sid, pairs in tqdm(sorted(rows_by_sid.items()), desc='Subjects'):
            sid_rs = []
            for chrom_cycle, gt_cycle in pairs:
                inp = torch.from_numpy(chrom_cycle).unsqueeze(0).unsqueeze(0).float().to(device)
                z = encoder(inp)
                recon = vae.decode(z).squeeze().cpu().numpy()
                r = shape_r(recon, gt_cycle)
                sid_rs.append(r)
            per_subj_rs.append(np.mean(sid_rs))

    mean_r = np.mean(per_subj_rs)
    print(f'\nPer-subject r with POS input: {mean_r:.4f}')
    print(f'V5-B baseline target:         0.711')

    if mean_r >= 0.70:
        print(f'\nVERIFICATION PASSED: encoder_B.pt reproduces ~0.711 with POS input.')
        print('  A13 training should use rppg_pos_cycles.')
    elif mean_r >= 0.60:
        print(f'\nPARTIAL: r={mean_r:.4f} — below 0.711 but acceptable.')
    else:
        print(f'\nFAIL: r={mean_r:.4f} — encoder_B.pt not performing as expected with POS.')

    print(f'\nPer-subject breakdown:')
    for sid, r in zip(sorted(rows_by_sid.keys()), per_subj_rs):
        ds = split_df[split_df['sid'] == sid]['dataset'].iloc[0]
        print(f'  SID {sid:4d} ({ds:12s}): r={r:.4f}')


if __name__ == '__main__':
    main()
