"""
evaluation/a13/evaluate_a13.py — A13 Evaluation
================================================
Reports:
  - Per-subject r (shape fidelity) — compare to V5-B baseline 0.711
  - Cross-subject r — collapse metric (target < 0.70, GT = 0.601 test set)
  - IPA error and H2/H1 error
  - Per-dataset breakdown
  - Comparison vs V5-B (0.711) and A5-v4 (best previous: 0.892 cross-subj r)

Usage:
    python evaluation/a13/evaluate_a13.py
    python evaluation/a13/evaluate_a13.py --suffix lc5.0_t0.3
    python evaluation/a13/evaluate_a13.py --ckpt path/to/encoder_a13_XXX.pt

Output:
    results/a13/full_eval_a13_<suffix>.csv
    results/a13/summary_a13_<suffix>.txt
"""

import os, sys, argparse
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

from config.paths import (
    RESULTS_A13, CKPT_A13, SPLIT_FILE, ENCODER_CKPT, VAE_CKPT,
    UBFC_CYCLES_DIR, STRESS_CYCLES_DIR, FPS2023_CYCLES_DIR, CENTAN_CYCLES_DIR,
)
from config.hyperparams import BATCH_SIZE
from models.encoder import CameraEncoder
from models.vae import PPGVAE
from models.metrics import compute_ipa, notch_index
from training.a13.train_a13 import A13Dataset, find_cycle_dirs

LATENT_DIM  = 32
IN_CHANNELS = 1


def shape_r(a, b):
    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return 0.0
    return float(pearsonr(a, b)[0])


def safe_pearsonr(a, b):
    if len(a) < 3 or np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0, 1.0
    return pearsonr(a, b)


def simple_collate(batch):
    chroms = torch.stack([b[0] for b in batch])
    gts    = torch.stack([b[1] for b in batch])
    sids   = [b[2] for b in batch]
    return chroms, gts, sids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', type=str, default='lc1.0_t0.3',
                        help='Checkpoint suffix, e.g. lc1.0_t0.3')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Override checkpoint path')
    args = parser.parse_args()

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split_df = pd.read_csv(SPLIT_FILE)
    test_sids = set(split_df[split_df['split'] == 'test']['sid'])

    # Checkpoint
    ckpt_path = Path(args.ckpt) if args.ckpt else \
                CKPT_A13 / f'encoder_a13_{args.suffix}.pt'
    if not ckpt_path.exists():
        print(f'ERROR: checkpoint not found at {ckpt_path}')
        print(f'  Run: python training/a13/train_a13.py --lambda_contrast X')
        return

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta_lc   = state.get('lambda_contrast', '?')
    meta_temp = state.get('temperature', '?')
    meta_epoch = state.get('epoch', '?')
    print(f'Checkpoint: {ckpt_path.name}')
    print(f'  λ_contrast={meta_lc}  temp={meta_temp}  epoch={meta_epoch}')
    print(f'  Saved val_r={state.get("val_r", "?"):.4f}  '
          f'cross_r={state.get("cross_r", "?"):.4f}')

    encoder = CameraEncoder(latent_dim=LATENT_DIM, in_channels=IN_CHANNELS,
                            morpho_aux=False).to(device)
    encoder.load_state_dict(state['encoder'])
    encoder.eval()

    vae = PPGVAE(latent_dim=LATENT_DIM).to(device)
    vae.load_state_dict(torch.load(VAE_CKPT, map_location=device, weights_only=False))
    vae.eval()

    dirs = find_cycle_dirs()
    if not dirs:
        print('ERROR: No cycle directories found.'); return

    test_ds = A13Dataset(dirs, test_sids)
    if len(test_ds) == 0:
        print('ERROR: No test cycles.'); return

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, collate_fn=simple_collate)

    RESULTS_A13.mkdir(parents=True, exist_ok=True)

    all_rows      = []
    recon_by_sid  = {}
    gt_ipa_all    = []
    pred_ipa_all  = []
    gt_h2h1_all   = []
    pred_h2h1_all = []

    print(f'\nEvaluating on {len(test_ds)} test cycles '
          f'({len(set(test_ds.sid_list))} subjects)...')

    with torch.no_grad():
        for chrom, gt, sids in tqdm(test_loader):
            chrom  = chrom.to(device)
            gt_np  = gt.numpy()[:, 0, :]
            sids_np = np.array([int(s) for s in sids])

            z       = encoder(chrom)
            ppg_np  = vae.decode(z).cpu().numpy()[:, 0, :]

            for i in range(len(gt_np)):
                sid = int(sids_np[i])
                ds  = split_df[split_df['sid'] == sid]['dataset'].iloc[0]
                r   = shape_r(ppg_np[i], gt_np[i])

                gt_ipa   = compute_ipa(gt_np[i])
                pred_ipa = compute_ipa(ppg_np[i])
                fft_gt   = np.abs(np.fft.rfft(gt_np[i]))
                gt_h2    = float(fft_gt[2] / (fft_gt[1] + 1e-8))
                fft_pred = np.abs(np.fft.rfft(ppg_np[i]))
                pred_h2  = float(fft_pred[2] / (fft_pred[1] + 1e-8))

                all_rows.append({
                    'architecture': 'A13',
                    'sid':          sid,
                    'dataset':      ds,
                    'shape_r':      r,
                    'gt_ipa':       gt_ipa,
                    'pred_ipa':     pred_ipa,
                    'ipa_error':    abs(gt_ipa - pred_ipa),
                    'gt_h2h1':      gt_h2,
                    'pred_h2h1':    pred_h2,
                    'h2h1_error':   abs(gt_h2 - pred_h2),
                    'gt_notch_detected': notch_index(gt_np[i]) >= 0,
                    'lambda_contrast': meta_lc,
                })
                recon_by_sid.setdefault(sid, []).append(ppg_np[i])
                gt_ipa_all.append(gt_ipa);    pred_ipa_all.append(pred_ipa)
                gt_h2h1_all.append(gt_h2);   pred_h2h1_all.append(pred_h2)

    if not all_rows:
        print('No results.'); return

    df = pd.DataFrame(all_rows)
    out_csv = RESULTS_A13 / f'full_eval_a13_{args.suffix}.csv'
    df.to_csv(out_csv, index=False)

    per_subj_r = df.groupby('sid')['shape_r'].mean().mean()
    ipa_err    = df['ipa_error'].mean()
    h2_err     = df['h2h1_error'].mean()
    notch_rate = float(df['gt_notch_detected'].mean())

    means  = {s: np.mean(v, axis=0) for s, v in recon_by_sid.items()}
    sids_s = sorted(means.keys())
    rs_out = [shape_r(means[a], means[b]) for a, b in combinations(sids_s, 2)]
    cross_r = float(np.mean(rs_out)) if rs_out else float('nan')

    r_h2h1, _ = safe_pearsonr(gt_h2h1_all, pred_h2h1_all)
    r_ipa,  _ = safe_pearsonr(gt_ipa_all,  pred_ipa_all)

    lines = [
        '=' * 68,
        f'A13 (Morphological Contrastive, λ={meta_lc}, τ={meta_temp}) EVALUATION',
        '=' * 68,
        '',
        'SHAPE FIDELITY:',
        f'  Per-subject r: {per_subj_r:.4f}  (V5-B baseline: 0.711)',
        '',
        'TEMPLATE COLLAPSE:',
        f'  Cross-subject r: {cross_r:.4f}  '
        f'(GT~0.60 test set; V5-B: 0.808; A5-v4 best: 0.892; goal: <0.70)',
        '',
        'MORPHOLOGICAL ERRORS:',
        f'  IPA error:    {ipa_err:.4f}  (V5-B: 0.151)',
        f'  H2/H1 error:  {h2_err:.4f}  (V5-B: 0.138)',
        f'  GT notch detection rate: {notch_rate:.1%}',
        '',
        'MORPHOLOGICAL FEATURE PREDICTION (direct from waveform):',
        f'  H2/H1 pred r: {r_h2h1:.4f}',
        f'  IPA   pred r: {r_ipa:.4f}',
        '',
        'PER DATASET:',
    ]
    for ds in sorted(df['dataset'].unique()):
        ds_r = df[df['dataset'] == ds].groupby('sid')['shape_r'].mean().mean()
        n    = df[df['dataset'] == ds]['sid'].nunique()
        lines.append(f'  {ds:14s}: r={ds_r:.4f}  ({n} subjects)')

    lines += ['', 'COLLAPSE VERDICT:']
    if cross_r < 0.70:
        lines.append(
            f'  COLLAPSE BROKEN: cross_r={cross_r:.4f} < 0.70 '
            f'(GT=0.60) — A13 SUCCEEDED'
        )
    elif cross_r < 0.808:
        lines.append(
            f'  PARTIAL: cross_r={cross_r:.4f} — improved vs V5-B (0.808) '
            f'but not yet < 0.70 target'
        )
    elif cross_r < 0.892:
        lines.append(
            f'  PARTIAL: cross_r={cross_r:.4f} — improved vs A5-v4 (0.892) '
            f'but not yet < 0.70 target'
        )
    else:
        lines.append(
            f'  FAIL: cross_r={cross_r:.4f} — no improvement vs A5-v4 (0.892)'
        )

    for l in lines:
        print(l)

    out_txt = RESULTS_A13 / f'summary_a13_{args.suffix}.txt'
    out_txt.write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nSaved to {RESULTS_A13}')


if __name__ == '__main__':
    main()
