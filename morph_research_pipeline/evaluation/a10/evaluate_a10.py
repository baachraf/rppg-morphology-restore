"""
evaluation/a10/evaluate_a10.py — A10 DPS Evaluation
====================================================
Reports:
  - Per-subject r (shape fidelity)    — compare to V5-B baseline 0.711
  - Cross-subject r (collapse metric) — goal: < 0.70  (V5-B: 0.808)
  - IPA error and H2/H1 error
  - Baseline (unconditional sampling, no DPS guidance)
  - DPS gradient_scale sweep (0.1, 0.5, 1.0, 3.0, 5.0)
  - Per-dataset and per-FPS breakdowns

Outputs:
  results/a10/full_eval_a10.csv
  results/a10/summary_a10.txt
"""

import os, sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
from itertools import combinations
from tqdm import tqdm
from pathlib import Path

HERE = Path(__file__).parent
PIPELINE_ROOT = HERE.parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from config.paths import (
    UBFC_CYCLES_DIR, STRESS_CYCLES_DIR, FPS2023_CYCLES_DIR,
    FPS2023_60_CYCLES_DIR, CENTAN_CYCLES_DIR,
    CKPT_A10, RESULTS_A10, SPLIT_FILE,
    VAE_CKPT, ENCODER_CKPT,
)
from config.hyperparams import BATCH_SIZE
from models.vae import PPGVAE
from models.forward_model import PPGToRPPG
from models.ddpm_a10 import UnconditionalDDPM
from models.metrics import compute_ipa, notch_index

FPS2023_60_SID_MIN = 3041
FPS2023_60_SID_MAX = 3306
LATENT_DIM  = 32
T_STEPS     = 200
DDIM_STEPS  = 50

# DPS gradient scales to evaluate
GRADIENT_SCALES = [0.0, 0.1, 0.5, 1.0, 3.0, 5.0]


def shape_r(a, b):
    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return 0.0
    return float(pearsonr(a, b)[0])


def safe_pearsonr(a, b):
    if len(a) < 3 or np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0, 1.0
    return pearsonr(a, b)


def find_cycle_dirs():
    return [d for d in [UBFC_CYCLES_DIR, STRESS_CYCLES_DIR, FPS2023_CYCLES_DIR,
                        FPS2023_60_CYCLES_DIR, CENTAN_CYCLES_DIR]
            if Path(d).is_dir()]


class A10EvalDataset(Dataset):
    """Loads (rppg_chrom, gt_cycle, sid) for DPS evaluation."""

    def __init__(self, cycle_dirs, split_sids):
        self.items    = []
        self.sid_list = []

        for d in cycle_dirs:
            for npz_f in sorted(Path(d).glob('*_cycles.npz')):
                try:
                    data = np.load(npz_f, allow_pickle=True)
                    sid  = int(data['sid'])
                    if sid not in split_sids:
                        continue
                    gt = data['gt_cycles']
                    if 'rppg_chrom_cycles' in data:
                        rppg = data['rppg_chrom_cycles']
                    else:
                        rppg = data['rppg_pos_cycles']
                    if len(gt) < 5:
                        continue
                    for i in range(len(gt)):
                        self.items.append((rppg[i].astype(np.float32),
                                           gt[i].astype(np.float32),
                                           sid))
                        self.sid_list.append(sid)
                except Exception:
                    continue

        print(f'EvalDataset: {len(self.items)} cycles / {len(set(self.sid_list))} subjects')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rppg, gt, sid = self.items[idx]
        return (torch.from_numpy(rppg).unsqueeze(0).float(),
                torch.from_numpy(gt).float(),
                sid)


def load_models(device):
    # VAE with fine-tuned decoder
    state = torch.load(ENCODER_CKPT['B'], map_location=device, weights_only=False)
    vae   = PPGVAE(latent_dim=LATENT_DIM)
    vae.decoder.load_state_dict(state['decoder_finetune'])
    vae_full = torch.load(VAE_CKPT, map_location=device, weights_only=False)
    enc_state = {k[len('encoder.'):]: v
                 for k, v in vae_full.items() if k.startswith('encoder.')}
    vae.encoder.load_state_dict(enc_state)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    vae = vae.to(device)

    # Forward model
    fm_ckpt = CKPT_A10 / 'forward_model.pt'
    if not fm_ckpt.exists():
        raise FileNotFoundError(f'Forward model not found: {fm_ckpt}')
    fm = PPGToRPPG().to(device)
    fm.load_state_dict(torch.load(fm_ckpt, map_location=device, weights_only=False)['model'])
    fm.eval()
    for p in fm.parameters():
        p.requires_grad_(False)

    # DDPM
    ddpm_ckpt = CKPT_A10 / 'ddpm_a10.pt'
    if not ddpm_ckpt.exists():
        raise FileNotFoundError(f'DDPM not found: {ddpm_ckpt}')
    ddpm = UnconditionalDDPM(latent_dim=LATENT_DIM, T=T_STEPS).to(device)
    ddpm.noise_pred.load_state_dict(
        torch.load(ddpm_ckpt, map_location=device, weights_only=False)['model']
    )
    ddpm.noise_pred.eval()

    return vae, fm, ddpm


def evaluate_scale(ddpm, vae, fm, test_loader, split_df, gradient_scale, device):
    """Run DPS with a specific gradient_scale and return metrics dict."""
    all_rows       = []
    pred_by_sid    = {}
    gt_ipa_all     = []
    pred_ipa_all   = []
    gt_h2h1_all    = []
    pred_h2h1_all  = []

    for rppg_obs, gt_np_batch, sids in test_loader:
        rppg_obs = rppg_obs.to(device)        # (B, 1, 256)
        gt_np    = gt_np_batch.numpy()         # (B, 256)
        B        = rppg_obs.size(0)

        if gradient_scale == 0.0:
            # Unconditional baseline — no DPS guidance
            z_sampled = ddpm.ddim_sample_unconditional(B, n_steps=DDIM_STEPS, device=device)
        else:
            z_sampled = ddpm.dps_sample(
                rppg_obs, vae.decoder, fm,
                n_steps=DDIM_STEPS, gradient_scale=gradient_scale, device=device
            )

        with torch.no_grad():
            ppg_pred = vae.decode(z_sampled).cpu().numpy()[:, 0, :]   # (B, 256)

        sids_np = np.array(sids) if not isinstance(sids, np.ndarray) else sids

        for i in range(B):
            sid = int(sids_np[i])
            ds  = split_df[split_df['sid'] == sid]['dataset'].iloc[0]

            r_pred = shape_r(ppg_pred[i], gt_np[i])
            gt_ipa   = compute_ipa(gt_np[i])
            pred_ipa = compute_ipa(ppg_pred[i])
            fft_gt   = np.abs(np.fft.rfft(gt_np[i]))
            fft_pr   = np.abs(np.fft.rfft(ppg_pred[i]))
            gt_h2    = float(fft_gt[2]   / (fft_gt[1]   + 1e-8))
            pred_h2  = float(fft_pr[2]   / (fft_pr[1]   + 1e-8))

            all_rows.append({
                'architecture':      f'A10_gs{gradient_scale}',
                'gradient_scale':    gradient_scale,
                'sid':               sid,
                'dataset':           ds,
                'shape_r':           r_pred,
                'gt_ipa':            gt_ipa,
                'pred_ipa':          pred_ipa,
                'ipa_error':         abs(gt_ipa - pred_ipa),
                'gt_h2h1':           gt_h2,
                'pred_h2h1':         pred_h2,
                'h2h1_error':        abs(gt_h2 - pred_h2),
                'gt_notch_detected': notch_index(gt_np[i]) >= 0,
            })
            pred_by_sid.setdefault(sid, []).append(ppg_pred[i])
            gt_ipa_all.append(gt_ipa);   pred_ipa_all.append(pred_ipa)
            gt_h2h1_all.append(gt_h2);  pred_h2h1_all.append(pred_h2)

    df = pd.DataFrame(all_rows)
    per_subj_r = df.groupby('sid')['shape_r'].mean().mean()
    ipa_err    = df['ipa_error'].mean()
    h2_err     = df['h2h1_error'].mean()

    means_sorted = sorted(pred_by_sid.keys())
    subj_means   = {s: np.mean(v, axis=0) for s, v in pred_by_sid.items()}
    rs = [shape_r(subj_means[a], subj_means[b])
          for a, b in combinations(means_sorted, 2)]
    cross_r = float(np.mean(rs)) if rs else float('nan')

    r_h2h1, _ = safe_pearsonr(gt_h2h1_all, pred_h2h1_all)
    r_ipa,  _ = safe_pearsonr(gt_ipa_all,  pred_ipa_all)

    fps60_mask = df['sid'].between(FPS2023_60_SID_MIN, FPS2023_60_SID_MAX)
    r_fps60 = df[fps60_mask].groupby('sid')['shape_r'].mean().mean() if fps60_mask.any() else float('nan')
    r_fps30 = df[~fps60_mask].groupby('sid')['shape_r'].mean().mean() if (~fps60_mask).any() else float('nan')

    per_ds = {}
    for ds_name in sorted(df['dataset'].unique()):
        n  = df[df['dataset'] == ds_name]['sid'].nunique()
        dr = df[df['dataset'] == ds_name].groupby('sid')['shape_r'].mean().mean()
        per_ds[ds_name] = (dr, n)

    return {
        'gs':        gradient_scale,
        'per_subj_r': per_subj_r,
        'cross_r':   cross_r,
        'ipa_err':   ipa_err,
        'h2_err':    h2_err,
        'r_h2h1':    r_h2h1,
        'r_ipa':     r_ipa,
        'r_fps30':   r_fps30,
        'r_fps60':   r_fps60,
        'per_ds':    per_ds,
        'df':        df,
    }


def main():
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split_df = pd.read_csv(SPLIT_FILE)
    test_sids = set(split_df[split_df['split'] == 'test']['sid'])

    print('Loading models...')
    vae, fm, ddpm = load_models(device)

    cycle_dirs = find_cycle_dirs()
    if not cycle_dirs:
        print('ERROR: No cycle directories found.'); return

    print('Building test dataset...')
    test_ds = A10EvalDataset(cycle_dirs, test_sids)
    if len(test_ds) == 0:
        print('ERROR: No test cycles.'); return

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    RESULTS_A10.mkdir(parents=True, exist_ok=True)

    all_dfs  = []
    results  = []

    for gs in GRADIENT_SCALES:
        label = 'UNCONDITIONAL' if gs == 0.0 else f'DPS gs={gs}'
        print(f'\nEvaluating {label} on {len(test_ds)} cycles...')
        m = evaluate_scale(ddpm, vae, fm, test_loader, split_df, gs, device)
        results.append(m)
        all_dfs.append(m['df'])
        print(f'  per-subj r={m["per_subj_r"]:.4f}  cross-subj r={m["cross_r"]:.4f}'
              f'  IPA={m["ipa_err"]:.4f}  H2={m["h2_err"]:.4f}')

    pd.concat(all_dfs, ignore_index=True).to_csv(
        RESULTS_A10 / 'full_eval_a10.csv', index=False
    )

    # Find best DPS scale
    dps_only = [m for m in results if m['gs'] > 0.0]
    best     = max(dps_only, key=lambda m: m['per_subj_r'])

    lines = [
        '=' * 68,
        'A10 (DPS: Unconditional DDPM + rPPG Likelihood Gradient) EVALUATION',
        '=' * 68,
        '',
        f'DDIM steps: {DDIM_STEPS}  |  gradient scales tested: {GRADIENT_SCALES}',
        '',
        'GRADIENT SCALE SWEEP:',
        f'  {"Scale":>8s}  {"per-subj r":>10s}  {"cross-subj r":>13s}  {"IPA err":>8s}  {"H2 err":>8s}',
    ]
    for m in results:
        label = 'uncond' if m['gs'] == 0.0 else f'{m["gs"]}'
        lines.append(
            f'  {label:>8s}  {m["per_subj_r"]:>10.4f}  {m["cross_r"]:>13.4f}'
            f'  {m["ipa_err"]:>8.4f}  {m["h2_err"]:>8.4f}'
        )

    lines += [
        '',
        f'BEST DPS (gs={best["gs"]}):',
        f'  Per-subject r:      {best["per_subj_r"]:.4f}  (V5-B: 0.711)',
        f'  Cross-subject r:    {best["cross_r"]:.4f}  (V5-B: 0.808; goal: <0.70)',
        f'  IPA error:          {best["ipa_err"]:.4f}  (V5-B: 0.151)',
        f'  H2/H1 error:        {best["h2_err"]:.4f}  (V5-B: 0.138)',
        f'  H2/H1 pred r:       {best["r_h2h1"]:.4f}',
        f'  IPA pred r:         {best["r_ipa"]:.4f}',
        '',
        'PER-FPS (best DPS):',
        f'  30fps: r={best["r_fps30"]:.4f}',
        f'  60fps: r={best["r_fps60"]:.4f}',
        '',
        'PER DATASET (best DPS):',
    ]
    for ds_name, (dr, n) in sorted(best['per_ds'].items()):
        lines.append(f'  {ds_name:14s}: r={dr:.4f}  ({n} subjects)')

    lines += ['', 'COLLAPSE VERDICT (best DPS):']
    if best['cross_r'] < 0.70:
        lines.append(f'  COLLAPSE BROKEN: cross-subj r={best["cross_r"]:.4f} < 0.70 !!!')
    else:
        lines.append(f'  Collapse persists: cross-subj r={best["cross_r"]:.4f} >= 0.70')

    for l in lines:
        print(l)

    (RESULTS_A10 / 'summary_a10.txt').write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nSaved to {RESULTS_A10}')


if __name__ == '__main__':
    main()
