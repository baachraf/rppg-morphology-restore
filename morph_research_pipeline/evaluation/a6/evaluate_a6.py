"""
evaluation/a6/evaluate_a6.py — A6 Evaluation
=============================================
Evaluates A6 RGBEncoder on test split. Reports per-subject r, IPA, H2/H1,
cross-subject r, hallucination metrics.

Outputs:
  results/a6/full_eval_a6.csv
  results/a6/summary_a6.txt
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from tqdm import tqdm
from pathlib import Path

HERE = Path(__file__).parent
PIPELINE_ROOT = HERE.parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from morph_config import SPLIT_FILE, BATCH_SIZE
from config.paths import (
    UBFC_DIR, STRESS_DIR, FPS2023_DIR, CENTAN_DIR,
    RESULTS_DIR, VAE_CKPT,
)
from models.vae import PPGVAE
from models.encoder_a6 import RGBEncoderA6
from models.metrics import compute_ipa, notch_index
from training.a6.train_a6 import RGBWindowDataset, find_rgb_window_dirs

LATENT_DIM = 32
IN_CHANNELS = 3

CKPT_A6 = Path(RESULTS_DIR).parent / 'checkpoints' / 'a6'
RESULTS_A6 = RESULTS_DIR / 'a6'


def shape_r(a, b):
    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return 0.0
    return pearsonr(a, b)[0]


def compute_h2_h1(cycle):
    fft = np.abs(np.fft.rfft(cycle))
    if len(fft) < 4 or fft[1] < 1e-8:
        return 0.0
    return fft[2] / fft[1]


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    split_df = pd.read_csv(SPLIT_FILE)
    test_sids = set(split_df[split_df['split'] == 'test']['sid'])

    data_dirs = find_rgb_window_dirs()
    if not data_dirs:
        print('ERROR: No rgb_windows directories found.')
        return

    test_ds = RGBWindowDataset(data_dirs, test_sids)
    if len(test_ds) == 0:
        print('ERROR: No test windows loaded.')
        return

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)

    vae = PPGVAE(latent_dim=LATENT_DIM).to(device)
    if VAE_CKPT.exists():
        vae.load_state_dict(
            torch.load(VAE_CKPT, map_location=device, weights_only=True))
    else:
        print(f'ERROR: VAE checkpoint not found at {VAE_CKPT}')
        return

    ckpt_path = CKPT_A6 / 'encoder_a6_D.pt'
    if not ckpt_path.exists():
        print(f'ERROR: A6 checkpoint not found at {ckpt_path}')
        return

    encoder = RGBEncoderA6(latent_dim=LATENT_DIM,
                           in_channels=IN_CHANNELS).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if 'encoder' in state:
        encoder.load_state_dict(state['encoder'])
        if 'decoder_finetune' in state:
            vae.decoder.load_state_dict(state['decoder_finetune'])
    else:
        encoder.load_state_dict(state)

    encoder.eval()
    vae.eval()

    RESULTS_A6.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_cross = []
    all_recons = []

    print(f'\nEvaluating A6-D on {len(test_ds)} test windows...')

    with torch.no_grad():
        for rgb, gt, sids in tqdm(test_loader):
            rgb = rgb.to(device)
            gt_np = gt.numpy()[:, 0, :]
            sids_np = sids.numpy()

            z, _ = encoder.forward_morpho(rgb)
            recon = vae.decode(z).cpu().numpy()[:, 0, :]

            for i in range(len(gt_np)):
                sid = int(sids_np[i])
                ds = split_df[split_df['sid'] == sid]['dataset'].iloc[0]
                r = shape_r(recon[i], gt_np[i])
                gt_ipa = compute_ipa(gt_np[i])
                pred_ipa = compute_ipa(recon[i])
                gt_h2 = compute_h2_h1(gt_np[i])
                pred_h2 = compute_h2_h1(recon[i])
                gt_notch = notch_index(gt_np[i])
                pred_notch = notch_index(recon[i])

                all_results.append({
                    'architecture': 'A6', 'encoder': 'D', 'sid': sid,
                    'dataset': ds, 'shape_r': r,
                    'gt_ipa': gt_ipa, 'pred_ipa': pred_ipa,
                    'ipa_error': abs(gt_ipa - pred_ipa),
                    'gt_h2h1': gt_h2, 'pred_h2h1': pred_h2,
                    'h2h1_error': abs(gt_h2 - pred_h2),
                    'gt_notch': gt_notch, 'pred_notch': pred_notch,
                })

            z_np = z.cpu().numpy()
            for i in range(len(z_np)):
                all_cross.append({
                    'sid': int(sids_np[i]),
                    'z': z_np[i],
                })
                all_recons.append({
                    'sid': int(sids_np[i]),
                    'recon': recon[i],
                })

    if not all_results:
        print('No results.'); return

    df = pd.DataFrame(all_results)
    df.to_csv(RESULTS_A6 / 'full_eval_a6.csv', index=False)

    print('\n' + '=' * 60)
    print('A6-D (Raw RGB Window Encoder) EVALUATION RESULTS')
    print('=' * 60)

    cycle_r = df['shape_r'].mean()
    subj = df.groupby('sid')['shape_r'].mean()
    per_subj_r = subj.mean()
    ipa_err = df['ipa_error'].mean()
    h2_err = df['h2h1_error'].mean()

    print(f'  Cycle-level r:  {cycle_r:.4f}')
    print(f'  Per-subject r:  {per_subj_r:.4f} ({len(subj)} subjects)')
    print(f'  IPA error:      {ipa_err:.4f}')
    print(f'  H2/H1 error:    {h2_err:.4f}')

    print(f'\n  By dataset:')
    for ds in df['dataset'].unique():
        ds_sub = df[df['dataset'] == ds]
        ds_subj = ds_sub.groupby('sid')['shape_r'].mean()
        print(f'    {ds:12s}: r={ds_subj.mean():.4f} '
              f'({ds_subj.shape[0]} subjects)')

    if len(all_recons) > 10:
        # --- Cross-subject r on OUTPUT PPG waveforms (256-dim) — primary metric ---
        recon_by_sid = {}
        for c in all_recons:
            recon_by_sid.setdefault(c['sid'], []).append(c['recon'])
        recon_means = {s: np.mean(v, axis=0) for s, v in recon_by_sid.items()}
        sid_list = sorted(recon_means.keys())

        if len(sid_list) > 1:
            from itertools import combinations
            rs_out, rs_z = [], []
            z_by_sid = {}
            for c in all_cross:
                z_by_sid.setdefault(c['sid'], []).append(c['z'])
            z_means = {s: np.mean(v, axis=0) for s, v in z_by_sid.items()}

            for s1, s2 in combinations(sid_list, 2):
                rs_out.append(shape_r(recon_means[s1], recon_means[s2]))
                rs_z.append(shape_r(z_means[s1], z_means[s2]))

            cross_r_output = np.mean(rs_out)
            cross_r_z = np.mean(rs_z)
            print(f'\n  Cross-subject r (output PPG, 256-dim): {cross_r_output:.4f}')
            print(f'  Cross-subject r (latent z,   32-dim):  {cross_r_z:.4f}')
            print(f'  (GT ~0.60 test set, lower = more subject-specific)')

    with open(RESULTS_A6 / 'summary_a6.txt', 'w') as f:
        f.write(f'A6-D Evaluation Summary\n')
        f.write(f'========================\n')
        f.write(f'Cycle-level r:  {cycle_r:.4f}\n')
        f.write(f'Per-subject r:  {per_subj_r:.4f} ({len(subj)} subjects)\n')
        f.write(f'IPA error:      {ipa_err:.4f}\n')
        f.write(f'H2/H1 error:    {h2_err:.4f}\n')
        for ds in df['dataset'].unique():
            ds_subj = df[df['dataset'] == ds].groupby('sid')['shape_r'].mean()
            f.write(f'  {ds}: r={ds_subj.mean():.4f} '
                    f'({ds_subj.shape[0]} subjects)\n')
        f.write(f'\nFull results: full_eval_a6.csv\n')

    print(f'\nSaved to {RESULTS_A6}')


if __name__ == '__main__':
    main()
