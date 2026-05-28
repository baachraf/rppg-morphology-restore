"""
evaluation/a8/evaluate_a8.py — A8 Evaluation
=============================================
Reports:
  - Per-subject r (shape fidelity)
  - Cross-subject r on OUTPUT PPG waveforms — correct 256-dim computation
  - Morphological feature prediction r: Pearson r(pred_H2H1, GT_H2H1) and r(pred_IPA, GT_IPA)
  - IPA error and H2/H1 error
  - Per-dataset breakdown
  - Comparison vs V5-B baseline (0.711)

Output:
  results/a8/full_eval_a8.csv
  results/a8/summary_a8.txt
"""

import os, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from itertools import combinations
from tqdm import tqdm
from pathlib import Path

HERE = Path(__file__).parent
PIPELINE_ROOT = HERE.parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from config.paths import RESULTS_DIR, CKPT_DIR, SPLIT_FILE
from config.hyperparams import BATCH_SIZE
from models.encoder_a8 import A8Model
from models.metrics import compute_ipa, notch_index
from training.a8.train_a8 import FPSWindowDataset, find_fps_window_dirs, compute_h2h1_batch, collate_pad

CKPT_A8     = CKPT_DIR / 'a8'
RESULTS_A8  = RESULTS_DIR / 'a8'
LATENT_DIM  = 32
IN_CHANNELS = 6


def shape_r(a, b):
    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return 0.0
    return float(pearsonr(a, b)[0])


def safe_pearsonr(a, b):
    if np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0, 1.0
    return pearsonr(a, b)


def main():
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split_df  = pd.read_csv(SPLIT_FILE)
    test_sids = set(split_df[split_df['split'] == 'test']['sid'])

    data_dirs = find_fps_window_dirs()
    if not data_dirs:
        print('ERROR: No fps_windows directories found.')
        print('  Run extract_rgb_windows_fps.py first.')
        return

    test_ds = FPSWindowDataset(data_dirs, test_sids)
    if len(test_ds) == 0:
        print('ERROR: No test windows loaded.'); return

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_pad)

    ckpt_path = CKPT_A8 / 'a8_model.pt'
    if not ckpt_path.exists():
        print(f'ERROR: A8 checkpoint not found at {ckpt_path}')
        print('  Run training/a8/train_a8.py first.')
        return

    model = A8Model(latent_dim=LATENT_DIM, in_channels=IN_CHANNELS).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state['model'])
    model.eval()

    RESULTS_A8.mkdir(parents=True, exist_ok=True)

    all_rows        = []
    recon_by_sid    = {}
    z_by_sid        = {}
    pred_h2h1_all   = []
    gt_h2h1_all     = []
    pred_ipa_all    = []
    gt_ipa_all      = []

    print(f'\nEvaluating A8 on {len(test_ds)} test windows '
          f'({len(set(test_ds.sid_list))} subjects)...')

    with torch.no_grad():
        for rgb, gt, sids in tqdm(test_loader):
            rgb     = rgb.to(device)
            gt_np   = gt.numpy()[:, 0, :]
            sids_np = sids.numpy() if isinstance(sids, torch.Tensor) else np.array(sids)

            recon, z, morpho_pred = model(rgb)
            recon_np  = recon.cpu().numpy()[:, 0, :]
            z_np      = z.cpu().numpy()
            morpho_np = morpho_pred.cpu().numpy()

            for i in range(len(gt_np)):
                sid = int(sids_np[i])
                ds  = split_df[split_df['sid'] == sid]['dataset'].iloc[0]
                r   = shape_r(recon_np[i], gt_np[i])

                gt_ipa   = compute_ipa(gt_np[i])
                pred_ipa = compute_ipa(recon_np[i])
                fft_gt   = np.abs(np.fft.rfft(gt_np[i]))
                gt_h2    = float(fft_gt[2] / (fft_gt[1] + 1e-8))
                pred_h2  = float(morpho_np[i, 0])  # direct morpho head prediction

                all_rows.append({
                    'architecture': 'A8', 'sid': sid, 'dataset': ds,
                    'shape_r':      r,
                    'gt_ipa':       gt_ipa,    'pred_ipa':  pred_ipa,
                    'ipa_error':    abs(gt_ipa - pred_ipa),
                    'gt_h2h1':      gt_h2,     'pred_h2h1': pred_h2,
                    'h2h1_error':   abs(gt_h2 - pred_h2),
                    'gt_notch_detected': notch_index(gt_np[i]) >= 0,
                })
                recon_by_sid.setdefault(sid, []).append(recon_np[i])
                z_by_sid.setdefault(sid, []).append(z_np[i])
                pred_h2h1_all.append(pred_h2);   gt_h2h1_all.append(gt_h2)
                pred_ipa_all.append(float(morpho_np[i, 1])); gt_ipa_all.append(gt_ipa)

    if not all_rows:
        print('No results.'); return

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_A8 / 'full_eval_a8.csv', index=False)

    per_subj_r = df.groupby('sid')['shape_r'].mean().mean()
    ipa_err    = df['ipa_error'].mean()
    h2_err     = df['h2h1_error'].mean()
    notch_rate = float(df['gt_notch_detected'].mean())

    # Cross-subject r on OUTPUT waveforms (256-dim) — correct computation
    recon_means = {s: np.mean(v, axis=0) for s, v in recon_by_sid.items()}
    sids_sorted = sorted(recon_means.keys())
    rs_out  = [shape_r(recon_means[a], recon_means[b])
               for a, b in combinations(sids_sorted, 2)]
    cross_r = float(np.mean(rs_out)) if rs_out else float('nan')

    # Also report z-vector cross-subject r for reference
    z_means   = {s: np.mean(v, axis=0) for s, v in z_by_sid.items()}
    rs_z      = [shape_r(z_means[a], z_means[b]) for a, b in combinations(sids_sorted, 2)]
    cross_r_z = float(np.mean(rs_z)) if rs_z else float('nan')

    # Morphological prediction Pearson r
    r_h2h1_pred, _ = safe_pearsonr(gt_h2h1_all, pred_h2h1_all)
    r_ipa_pred,  _ = safe_pearsonr(gt_ipa_all,  pred_ipa_all)

    lines = [
        '=' * 65,
        'A8 (FPS-Agnostic Camera-Only) EVALUATION RESULTS',
        '=' * 65,
        f'Per-subject r:                  {per_subj_r:.4f}  (V5-B baseline: 0.711)',
        f'Cross-subject r (output 256-d): {cross_r:.4f}  (lower=better, GT~0.60 test set)',
        f'Cross-subject r (latent  32-d): {cross_r_z:.4f}  (reference)',
        f'IPA error:                      {ipa_err:.4f}  (V5-B: 0.151)',
        f'H2/H1 error:                    {h2_err:.4f}  (V5-B: 0.138)',
        f'GT notch detection rate:        {notch_rate:.1%}',
        '',
        'MORPHOLOGICAL PREDICTION ACCURACY (primary objective):',
        f'  H2/H1 prediction r: {r_h2h1_pred:.4f}  (>0.20 = signal present)',
        f'  IPA   prediction r: {r_ipa_pred:.4f}  (>0.20 = signal present)',
        '',
        'Per dataset:',
    ]
    for ds in df['dataset'].unique():
        ds_r = df[df['dataset'] == ds].groupby('sid')['shape_r'].mean().mean()
        n    = df[df['dataset'] == ds]['sid'].nunique()
        lines.append(f'  {ds:14s}: r={ds_r:.4f}  ({n} subjects)')

    for l in lines:
        print(l)

    (RESULTS_A8 / 'summary_a8.txt').write_text('\n'.join(lines))
    print(f'\nSaved to {RESULTS_A8}')


if __name__ == '__main__':
    main()
