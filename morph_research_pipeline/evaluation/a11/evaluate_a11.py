"""
evaluation/a11/evaluate_a11.py — A11 VMD-6ch Encoder Evaluation
================================================================
Reports:
  - Per-subject r (shape fidelity)    — compare to V5-B baseline 0.711
  - Cross-subject r (collapse metric) — goal: < 0.70  (V5-B: 0.808)
  - IPA error and H2/H1 error
  - Per-dataset and per-FPS breakdowns

Outputs:
  results/a11/full_eval_a11.csv
  results/a11/summary_a11.txt
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
    CKPT_A11, RESULTS_A11, SPLIT_FILE,
    VAE_CKPT, ENCODER_CKPT,
)
from config.hyperparams import BATCH_SIZE
from models.encoder import CameraEncoder
from models.vae import PPGVAE
from models.metrics import compute_ipa, notch_index

FPS2023_60_SID_MIN = 3041
FPS2023_60_SID_MAX = 3306
LATENT_DIM = 32

CYCLES_DIRS = [
    UBFC_CYCLES_DIR, STRESS_CYCLES_DIR, FPS2023_CYCLES_DIR,
    FPS2023_60_CYCLES_DIR, CENTAN_CYCLES_DIR,
]


def shape_r(a, b):
    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return 0.0
    return float(pearsonr(a, b)[0])


def safe_pearsonr(a, b):
    if len(a) < 3 or np.std(a) < 1e-10 or np.std(b) < 1e-10:
        return 0.0, 1.0
    return pearsonr(a, b)


class A11EvalDataset(Dataset):
    """Loads (vmd_6ch, gt_cycle, sid, dataset_name) for evaluation."""

    def __init__(self, cycles_dirs, split_sids, split_df):
        self.items    = []
        self.sid_list = []
        sid_to_ds     = dict(zip(split_df['sid'], split_df['dataset']))

        for d in cycles_dirs:
            d = Path(d)
            if not d.is_dir():
                continue
            for vmd_f in sorted(d.glob('*_vmd.npz')):
                stem     = vmd_f.stem.replace('_vmd', '')
                cycles_f = d / (stem + '_cycles.npz')
                if not cycles_f.exists():
                    continue
                try:
                    vmd_data = np.load(vmd_f,    allow_pickle=True)
                    cyc_data = np.load(cycles_f, allow_pickle=True)
                    sid = int(cyc_data['sid'])
                    if sid not in split_sids:
                        continue
                    vmd6 = vmd_data['vmd_6ch_cycles'].astype(np.float32)
                    gt   = cyc_data['gt_cycles'].astype(np.float32)
                    if len(gt) < 5 or len(vmd6) != len(gt):
                        continue
                    ds_name = sid_to_ds.get(sid, 'unknown')
                    for i in range(len(gt)):
                        self.items.append((vmd6[i], gt[i], sid, ds_name))
                        self.sid_list.append(sid)
                except Exception:
                    continue

        print(f'EvalDataset: {len(self.items)} cycles / {len(set(self.sid_list))} subjects')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        vmd6, gt, sid, ds_name = self.items[idx]
        return (torch.from_numpy(vmd6).float(),           # (6, 256)
                torch.from_numpy(gt).float(),              # (256,)
                sid, ds_name)


def load_models(device):
    # Frozen VAE (fine-tuned decoder from V5-B)
    state    = torch.load(ENCODER_CKPT['B'], map_location=device, weights_only=False)
    vae      = PPGVAE(latent_dim=LATENT_DIM)
    vae.decoder.load_state_dict(state['decoder_finetune'])
    vae_full = torch.load(VAE_CKPT, map_location=device, weights_only=False)
    enc_state = {k[len('encoder.'):]: v
                 for k, v in vae_full.items() if k.startswith('encoder.')}
    vae.encoder.load_state_dict(enc_state)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    vae = vae.to(device)

    # A11 encoder
    ckpt_path = CKPT_A11 / 'encoder_a11.pt'
    if not ckpt_path.exists():
        raise FileNotFoundError(f'A11 checkpoint not found: {ckpt_path}\n'
                                f'Run train_a11.py first.')
    encoder = CameraEncoder(latent_dim=LATENT_DIM, in_channels=6, morpho_aux=False)
    ckpt    = torch.load(ckpt_path, map_location=device, weights_only=False)
    encoder.load_state_dict(ckpt['encoder'])
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder = encoder.to(device)

    print(f'Loaded A11 encoder (val_r={ckpt.get("val_r", "?"):.4f} @ epoch {ckpt.get("epoch","?")})')
    return vae, encoder


def main():
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split_df = pd.read_csv(SPLIT_FILE)
    test_sids = set(split_df[split_df['split'] == 'test']['sid'])

    print('Loading models...')
    vae, encoder = load_models(device)

    print('Building test dataset...')
    test_ds = A11EvalDataset(CYCLES_DIRS, test_sids, split_df)
    if len(test_ds) == 0:
        print('ERROR: No test VMD cycles found.'); return

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=0)
    RESULTS_A11.mkdir(parents=True, exist_ok=True)

    all_rows     = []
    pred_by_sid  = {}
    gt_ipa_all   = []
    pred_ipa_all = []
    gt_h2h1_all  = []
    pred_h2h1_all = []

    print(f'\nEvaluating {len(test_ds)} cycles...')
    for vmd6, gt_np_batch, sids, ds_names in tqdm(test_loader):
        vmd6  = vmd6.to(device)
        gt_np = gt_np_batch.numpy()    # (B, 256)

        with torch.no_grad():
            z_pred   = encoder(vmd6)
            ppg_pred = vae.decode(z_pred).cpu().squeeze(1).numpy()  # (B, 256)

        sids_np = np.array(sids) if not isinstance(sids, np.ndarray) else sids

        for i in range(len(gt_np)):
            sid     = int(sids_np[i])
            ds_name = ds_names[i] if isinstance(ds_names[i], str) else str(ds_names[i])

            r_pred   = shape_r(ppg_pred[i], gt_np[i])
            gt_ipa   = compute_ipa(gt_np[i])
            pred_ipa = compute_ipa(ppg_pred[i])
            fft_gt   = np.abs(np.fft.rfft(gt_np[i]))
            fft_pr   = np.abs(np.fft.rfft(ppg_pred[i]))
            gt_h2    = float(fft_gt[2] / (fft_gt[1] + 1e-8))
            pred_h2  = float(fft_pr[2] / (fft_pr[1] + 1e-8))

            all_rows.append({
                'architecture': 'A11_vmd6ch',
                'sid':          sid,
                'dataset':      ds_name,
                'shape_r':      r_pred,
                'gt_ipa':       gt_ipa,
                'pred_ipa':     pred_ipa,
                'ipa_error':    abs(gt_ipa - pred_ipa),
                'gt_h2h1':      gt_h2,
                'pred_h2h1':    pred_h2,
                'h2h1_error':   abs(gt_h2 - pred_h2),
                'gt_notch_detected': notch_index(gt_np[i]) >= 0,
            })
            pred_by_sid.setdefault(sid, []).append(ppg_pred[i])
            gt_ipa_all.append(gt_ipa);    pred_ipa_all.append(pred_ipa)
            gt_h2h1_all.append(gt_h2);   pred_h2h1_all.append(pred_h2)

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_A11 / 'full_eval_a11.csv', index=False)

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
    r_fps60 = df[fps60_mask].groupby('sid')['shape_r'].mean().mean() \
              if fps60_mask.any() else float('nan')
    r_fps30 = df[~fps60_mask].groupby('sid')['shape_r'].mean().mean() \
              if (~fps60_mask).any() else float('nan')

    per_ds = {}
    for ds_name in sorted(df['dataset'].unique()):
        n  = df[df['dataset'] == ds_name]['sid'].nunique()
        dr = df[df['dataset'] == ds_name].groupby('sid')['shape_r'].mean().mean()
        per_ds[ds_name] = (dr, n)

    lines = [
        '=' * 68,
        'A11 (VMD 6-channel Raw RGB Encoder) EVALUATION',
        '=' * 68,
        '',
        f'Per-subject r:    {per_subj_r:.4f}  (V5-B: 0.711)',
        f'Cross-subject r:  {cross_r:.4f}  (V5-B: 0.808; goal: <0.70)',
        f'IPA error:        {ipa_err:.4f}  (V5-B: 0.151)',
        f'H2/H1 error:      {h2_err:.4f}  (V5-B: 0.138)',
        f'H2/H1 pred r:     {r_h2h1:.4f}',
        f'IPA pred r:       {r_ipa:.4f}',
        '',
        'PER-FPS:',
        f'  30fps: r={r_fps30:.4f}',
        f'  60fps: r={r_fps60:.4f}',
        '',
        'PER DATASET:',
    ]
    for ds_name, (dr, n) in sorted(per_ds.items()):
        lines.append(f'  {ds_name:14s}: r={dr:.4f}  ({n} subjects)')

    lines += ['', 'COLLAPSE VERDICT:']
    if cross_r < 0.70:
        lines.append(f'  COLLAPSE BROKEN: cross-subj r={cross_r:.4f} < 0.70 !!!')
    else:
        lines.append(f'  Collapse persists: cross-subj r={cross_r:.4f} >= 0.70')

    for l in lines:
        print(l)

    (RESULTS_A11 / 'summary_a11.txt').write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nSaved to {RESULTS_A11}')


if __name__ == '__main__':
    main()
