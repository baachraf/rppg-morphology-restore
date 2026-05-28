"""
evaluation/a5/evaluate_a5.py — A5 Sequential Two-Stage Decoder Evaluation
==========================================================================
Reports:
  - Per-subject r (shape fidelity) — compare to V5-B baseline 0.711
  - Cross-subject r on PPG_refined (collapse metric) — compare to V5-B 0.808
  - IPA error and H2/H1 error
  - Per-dataset breakdown
  - Per-FPS breakdown: 30fps subjects vs 60fps subjects (FPS2023_60 SIDs 3041+)
  - Comparison vs V5-B baseline

Output:
  results/a5/full_eval_a5.csv
  results/a5/summary_a5.txt
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

from config.paths import RESULTS_A5, CKPT_A5, SPLIT_FILE
from config.hyperparams import BATCH_SIZE
from models.refinenet_a5 import RefineNetA5
from models.metrics import compute_ipa, notch_index
from training.a5.train_a5 import (
    A5Dataset, find_cycle_dirs, load_stage1, compute_ppg_base,
)

# FPS2023_60 SID range — used for per-FPS breakdown
FPS2023_60_SID_MIN = 3041
FPS2023_60_SID_MAX = 3306


def shape_r(a, b):
    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return 0.0
    return float(pearsonr(a, b)[0])


def safe_pearsonr(a, b):
    if len(a) < 3 or np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0, 1.0
    return pearsonr(a, b)


def main():
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split_df  = pd.read_csv(SPLIT_FILE)
    test_sids = set(split_df[split_df['split'] == 'test']['sid'])

    ckpt_path = CKPT_A5 / 'refinenet_a5.pt'
    if not ckpt_path.exists():
        print(f'ERROR: A5 checkpoint not found at {ckpt_path}')
        print('  Run training/a5/train_a5.py first.')
        return

    print('Loading Stage 1 (frozen V5-B)...')
    encoder_b, vae = load_stage1(device)

    cycle_dirs = find_cycle_dirs()
    if not cycle_dirs:
        print('ERROR: No cycles directories found.'); return

    print('Building test dataset (pre-computing PPG_base)...')
    test_ds = A5Dataset(cycle_dirs, test_sids, encoder_b, vae, device)
    if len(test_ds) == 0:
        print('ERROR: No test cycles loaded.'); return

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    refine = RefineNetA5().to(device)
    state  = torch.load(ckpt_path, map_location=device, weights_only=False)
    refine.load_state_dict(state['model'])
    refine.eval()

    RESULTS_A5.mkdir(parents=True, exist_ok=True)

    all_rows        = []
    refined_by_sid  = {}
    base_by_sid     = {}
    gt_ipa_all      = []
    pred_ipa_all    = []
    gt_h2h1_all     = []
    pred_h2h1_all   = []

    print(f'\nEvaluating A5 on {len(test_ds)} test cycles '
          f'({len(set(test_ds.sid_list))} subjects)...')

    with torch.no_grad():
        for inp, gt, sids in tqdm(test_loader):
            inp    = inp.to(device)
            gt_np  = gt.numpy()[:, 0, :]
            sids_np = np.array(sids) if not isinstance(sids, np.ndarray) else sids

            ppg_base    = inp[:, :1, :]
            residual    = refine(inp)
            ppg_refined = (ppg_base + residual).cpu().numpy()[:, 0, :]
            ppg_base_np = ppg_base.cpu().numpy()[:, 0, :]

            for i in range(len(gt_np)):
                sid = int(sids_np[i])
                ds  = split_df[split_df['sid'] == sid]['dataset'].iloc[0]
                r_refined = shape_r(ppg_refined[i], gt_np[i])
                r_base    = shape_r(ppg_base_np[i], gt_np[i])

                gt_ipa   = compute_ipa(gt_np[i])
                pred_ipa = compute_ipa(ppg_refined[i])
                fft_gt   = np.abs(np.fft.rfft(gt_np[i]))
                gt_h2    = float(fft_gt[2] / (fft_gt[1] + 1e-8))
                fft_pred = np.abs(np.fft.rfft(ppg_refined[i]))
                pred_h2  = float(fft_pred[2] / (fft_pred[1] + 1e-8))

                all_rows.append({
                    'architecture': 'A5',
                    'sid':          sid,
                    'dataset':      ds,
                    'shape_r':      r_refined,
                    'base_r':       r_base,
                    'gt_ipa':       gt_ipa,
                    'pred_ipa':     pred_ipa,
                    'ipa_error':    abs(gt_ipa - pred_ipa),
                    'gt_h2h1':      gt_h2,
                    'pred_h2h1':    pred_h2,
                    'h2h1_error':   abs(gt_h2 - pred_h2),
                    'gt_notch_detected': notch_index(gt_np[i]) >= 0,
                })
                refined_by_sid.setdefault(sid, []).append(ppg_refined[i])
                base_by_sid.setdefault(sid,    []).append(ppg_base_np[i])
                gt_ipa_all.append(gt_ipa);   pred_ipa_all.append(pred_ipa)
                gt_h2h1_all.append(gt_h2);   pred_h2h1_all.append(pred_h2)

    if not all_rows:
        print('No results.'); return

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_A5 / 'full_eval_a5.csv', index=False)

    # ── aggregate metrics ────────────────────────────────────────────────────
    per_subj_r      = df.groupby('sid')['shape_r'].mean().mean()
    per_subj_r_base = df.groupby('sid')['base_r'].mean().mean()
    ipa_err         = df['ipa_error'].mean()
    h2_err          = df['h2h1_error'].mean()
    notch_rate      = float(df['gt_notch_detected'].mean())

    # Cross-subject r on PPG_refined (collapse metric)
    refined_means = {s: np.mean(v, axis=0) for s, v in refined_by_sid.items()}
    base_means    = {s: np.mean(v, axis=0) for s, v in base_by_sid.items()}
    sids_sorted   = sorted(refined_means.keys())
    rs_ref = [shape_r(refined_means[a], refined_means[b])
              for a, b in combinations(sids_sorted, 2)]
    rs_bas = [shape_r(base_means[a],    base_means[b])
              for a, b in combinations(sids_sorted, 2)]
    cross_r_refined = float(np.mean(rs_ref)) if rs_ref else float('nan')
    cross_r_base    = float(np.mean(rs_bas)) if rs_bas else float('nan')

    # Morpho feature r
    r_h2h1, _ = safe_pearsonr(gt_h2h1_all, pred_h2h1_all)
    r_ipa,  _ = safe_pearsonr(gt_ipa_all,  pred_ipa_all)

    # Per-FPS breakdown (FPS2023_60 SIDs vs rest)
    fps60_mask = df['sid'].between(FPS2023_60_SID_MIN, FPS2023_60_SID_MAX)
    r_fps60 = (df[fps60_mask].groupby('sid')['shape_r'].mean().mean()
               if fps60_mask.any() else float('nan'))
    r_fps30 = (df[~fps60_mask].groupby('sid')['shape_r'].mean().mean()
               if (~fps60_mask).any() else float('nan'))

    lines = [
        '=' * 68,
        'A5 (Sequential Two-Stage: frozen V5-B + RefineNet) EVALUATION',
        '=' * 68,
        '',
        'SHAPE FIDELITY:',
        f'  Per-subject r (PPG_refined): {per_subj_r:.4f}  (V5-B baseline: 0.711)',
        f'  Per-subject r (PPG_base):    {per_subj_r_base:.4f}  (Stage 1 only, should match V5-B)',
        '',
        'TEMPLATE COLLAPSE:',
        f'  Cross-subject r (PPG_refined): {cross_r_refined:.4f}  (V5-B: 0.808; goal: <0.70)',
        f'  Cross-subject r (PPG_base):    {cross_r_base:.4f}  (Stage 1 only, should match V5-B)',
        '',
        'MORPHOLOGICAL ERRORS:',
        f'  IPA error:    {ipa_err:.4f}  (V5-B: 0.151)',
        f'  H2/H1 error:  {h2_err:.4f}  (V5-B: 0.138)',
        f'  GT notch detection rate: {notch_rate:.1%}',
        '',
        'MORPHOLOGICAL FEATURE PREDICTION (direct):',
        f'  H2/H1 pred r: {r_h2h1:.4f}',
        f'  IPA   pred r: {r_ipa:.4f}',
        '',
        'PER-FPS BREAKDOWN:',
        f'  30fps subjects:  r={r_fps30:.4f}',
        f'  60fps subjects:  r={r_fps60:.4f}  (FPS2023_60)',
        '',
        'PER DATASET:',
    ]
    for ds in sorted(df['dataset'].unique()):
        ds_r = df[df['dataset'] == ds].groupby('sid')['shape_r'].mean().mean()
        n    = df[df['dataset'] == ds]['sid'].nunique()
        lines.append(f'  {ds:14s}: r={ds_r:.4f}  ({n} subjects)')

    lines += [
        '',
        'COLLAPSE VERDICT:',
    ]
    if cross_r_refined < 0.70:
        lines.append(f'  ✓ COLLAPSE BROKEN: cross-subj r={cross_r_refined:.4f} < 0.70 target')
    else:
        lines.append(f'  ✗ Collapse persists: cross-subj r={cross_r_refined:.4f} >= 0.70 target')

    for l in lines:
        print(l)

    (RESULTS_A5 / 'summary_a5.txt').write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nSaved to {RESULTS_A5}')


if __name__ == '__main__':
    main()
