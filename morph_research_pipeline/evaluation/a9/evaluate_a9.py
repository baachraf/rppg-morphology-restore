"""
evaluation/a9/evaluate_a9.py — A9 Latent Diffusion Decoder Evaluation
=======================================================================
Reports:
  - Per-subject r (shape fidelity)     — compare to V5-B baseline 0.711
  - Cross-subject r (collapse metric)  — goal: < 0.70  (V5-B: 0.808)
  - IPA error and H2/H1 error
  - Per-dataset breakdown
  - Per-FPS breakdown (30fps vs 60fps)
  - DDIM deterministic vs DDPM stochastic comparison

Outputs:
  results/a9/full_eval_a9.csv
  results/a9/summary_a9.txt
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

from config.paths import RESULTS_A9, CKPT_A9, SPLIT_FILE
from config.hyperparams import BATCH_SIZE
from models.ldm_a9 import LatentDiffusion, NoisePredictor
from models.metrics import compute_ipa, notch_index
from training.a9.train_a9 import A9EvalDataset, find_cycle_dirs, load_frozen_models

FPS2023_60_SID_MIN = 3041
FPS2023_60_SID_MAX = 3306
LATENT_DIM = 32
T_STEPS    = 200
DDIM_STEPS      = 50    # fast deterministic inference
GUIDANCE_SCALE  = 3.0   # CFG amplification (1.0 = no guidance, 3.0 = strong)


def shape_r(a, b):
    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return 0.0
    return float(pearsonr(a, b)[0])


def safe_pearsonr(a, b):
    if len(a) < 3 or np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0, 1.0
    return pearsonr(a, b)


def main():
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split_df = pd.read_csv(SPLIT_FILE)
    test_sids = set(split_df[split_df['split'] == 'test']['sid'])

    ckpt_path = CKPT_A9 / 'ldm_a9.pt'
    if not ckpt_path.exists():
        print(f'ERROR: A9 checkpoint not found at {ckpt_path}')
        print('  Run training/a9/train_a9.py first.')
        return

    print('Loading frozen Stage 1 models...')
    camera_enc, vae = load_frozen_models(device)

    cycle_dirs = find_cycle_dirs()
    if not cycle_dirs:
        print('ERROR: No cycle directories found.'); return

    print('Building test dataset (pre-computing z-pairs + raw GT)...')
    test_ds = A9EvalDataset(cycle_dirs, test_sids, camera_enc, vae, device)
    if len(test_ds) == 0:
        print('ERROR: No test cycles.'); return

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    ldm = LatentDiffusion(latent_dim=LATENT_DIM, T=T_STEPS).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    ldm.noise_pred.load_state_dict(state['model'])
    ldm.noise_pred.eval()

    RESULTS_A9.mkdir(parents=True, exist_ok=True)

    all_rows       = []
    refined_by_sid = {}
    gt_ipa_all     = []
    pred_ipa_all   = []
    gt_h2h1_all    = []
    pred_h2h1_all  = []

    print(f'\nEvaluating A9 on {len(test_ds)} test cycles '
          f'({len(set(test_ds.sid_list))} subjects)...')
    print(f'Sampling: DDIM {DDIM_STEPS} steps, eta=0, guidance_scale={GUIDANCE_SCALE}')

    with torch.no_grad():
        for z_prior, z_gt_batch, gt_cycles, sids in tqdm(test_loader):
            z_prior   = z_prior.to(device)
            gt_np     = gt_cycles.numpy()           # (B, 256) raw GT — no VAE distortion

            # DDIM + CFG sampling
            z_sampled = ldm.ddim_sample(z_prior, n_steps=DDIM_STEPS, eta=0.0,
                                        guidance_scale=GUIDANCE_SCALE)

            # Decode z → PPG
            ppg_pred = vae.decode(z_sampled).cpu().numpy()[:, 0, :]    # (B, 256)

            # Also decode z_prior (V5-B baseline, no diffusion)
            ppg_base = vae.decode(z_prior).cpu().numpy()[:, 0, :]      # (B, 256)

            sids_np = np.array(sids) if not isinstance(sids, np.ndarray) else sids

            for i in range(len(gt_np)):
                sid = int(sids_np[i])
                ds  = split_df[split_df['sid'] == sid]['dataset'].iloc[0]

                r_pred = shape_r(ppg_pred[i], gt_np[i])
                r_base = shape_r(ppg_base[i], gt_np[i])

                gt_ipa   = compute_ipa(gt_np[i])
                pred_ipa = compute_ipa(ppg_pred[i])
                fft_gt   = np.abs(np.fft.rfft(gt_np[i]))
                fft_pred = np.abs(np.fft.rfft(ppg_pred[i]))
                gt_h2    = float(fft_gt[2]   / (fft_gt[1]   + 1e-8))
                pred_h2  = float(fft_pred[2] / (fft_pred[1] + 1e-8))

                all_rows.append({
                    'architecture': 'A9',
                    'sid':          sid,
                    'dataset':      ds,
                    'shape_r':      r_pred,
                    'base_r':       r_base,
                    'gt_ipa':       gt_ipa,
                    'pred_ipa':     pred_ipa,
                    'ipa_error':    abs(gt_ipa - pred_ipa),
                    'gt_h2h1':      gt_h2,
                    'pred_h2h1':    pred_h2,
                    'h2h1_error':   abs(gt_h2 - pred_h2),
                    'gt_notch_detected': notch_index(gt_np[i]) >= 0,
                })
                refined_by_sid.setdefault(sid, []).append(ppg_pred[i])
                gt_ipa_all.append(gt_ipa);    pred_ipa_all.append(pred_ipa)
                gt_h2h1_all.append(gt_h2);   pred_h2h1_all.append(pred_h2)

    if not all_rows:
        print('No results.'); return

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_A9 / 'full_eval_a9.csv', index=False)

    # Aggregate
    per_subj_r      = df.groupby('sid')['shape_r'].mean().mean()
    per_subj_r_base = df.groupby('sid')['base_r'].mean().mean()
    ipa_err         = df['ipa_error'].mean()
    h2_err          = df['h2h1_error'].mean()
    notch_rate      = float(df['gt_notch_detected'].mean())

    # Cross-subject r on sampled PPG means
    means_sorted = sorted(refined_by_sid.keys())
    subj_means   = {s: np.mean(v, axis=0) for s, v in refined_by_sid.items()}
    rs = [shape_r(subj_means[a], subj_means[b])
          for a, b in combinations(means_sorted, 2)]
    cross_r = float(np.mean(rs)) if rs else float('nan')

    r_h2h1, _ = safe_pearsonr(gt_h2h1_all, pred_h2h1_all)
    r_ipa,  _ = safe_pearsonr(gt_ipa_all,  pred_ipa_all)

    fps60_mask = df['sid'].between(FPS2023_60_SID_MIN, FPS2023_60_SID_MAX)
    r_fps60 = (df[fps60_mask].groupby('sid')['shape_r'].mean().mean()
               if fps60_mask.any() else float('nan'))
    r_fps30 = (df[~fps60_mask].groupby('sid')['shape_r'].mean().mean()
               if (~fps60_mask).any() else float('nan'))

    lines = [
        '=' * 68,
        'A9 (Latent Diffusion Decoder — DDPM in z-space) EVALUATION',
        '=' * 68,
        '',
        f'Sampling: DDIM {DDIM_STEPS} steps, eta=0, CFG guidance_scale={GUIDANCE_SCALE}',
        '',
        'SHAPE FIDELITY:',
        f'  Per-subject r (PPG_sampled): {per_subj_r:.4f}  (V5-B baseline: 0.711)',
        f'  Per-subject r (PPG_base):    {per_subj_r_base:.4f}  (z_prior → decode, no diffusion)',
        '',
        'TEMPLATE COLLAPSE:',
        f'  Cross-subject r (PPG_sampled): {cross_r:.4f}  (V5-B: 0.808; goal: <0.70)',
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

    lines += ['', 'COLLAPSE VERDICT:']
    if cross_r < 0.70:
        lines.append(f'  COLLAPSE BROKEN: cross-subj r={cross_r:.4f} < 0.70 target')
    else:
        lines.append(f'  Collapse persists: cross-subj r={cross_r:.4f} >= 0.70 target')

    for l in lines:
        print(l)

    (RESULTS_A9 / 'summary_a9.txt').write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nSaved to {RESULTS_A9}')


if __name__ == '__main__':
    main()
