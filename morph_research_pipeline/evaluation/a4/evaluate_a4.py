"""
evaluation/a4/evaluate_a4.py — A4: Multi-Cycle Transformer Evaluation
======================================================================
Evaluates all three A4 encoder variants (A/B/C) on the test split.
Reuses V5 evaluation metrics: Pearson r, IPA, H2/H1, notch detection.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from scipy.fft import fft
from tqdm import tqdm
from pathlib import Path

HERE = Path(__file__).parent
PIPELINE_ROOT = HERE.parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from morph_config import (
    CYCLES_DIR, SPLIT_FILE, LATENT_DIM, BATCH_SIZE,
    VAE_CKPT_P4,
)
from config.paths import CKPT_A4, RESULTS_A4
from models.vae import PPGVAE
from models.transformer_a4 import MultiCycleTransformerEncoder
from models.metrics import compute_ipa, notch_index

NUM_CYCLES = 5
D_MODEL = 256
NHEAD = 8
N_LAYERS = 4


def shape_r(a, b):
    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return 0.0
    return pearsonr(a, b)[0]


def harmonic_ratios(c):
    spec = np.abs(fft(c - c.mean()))[:len(c) // 2]
    h1_idx = np.argmax(spec[1:10]) + 1
    h1 = spec[h1_idx]
    h2 = spec[h1_idx * 2] if (h1_idx * 2) < len(spec) else 0
    h3 = spec[h1_idx * 3] if (h1_idx * 3) < len(spec) else 0
    return {'h2h1': h2 / h1 if h1 > 0 else 0, 'h3h1': h3 / h1 if h1 > 0 else 0}


def _check_cycles_healthy(cycles, hr_arr, indices, min_hr=40, max_hr=150,
                          min_amp=0.01):
    for idx in indices:
        if idx >= len(hr_arr):
            return False
        h = hr_arr[idx]
        if h < min_hr or h > max_hr:
            return False
        c = cycles[idx]
        if np.ptp(c) < min_amp:
            return False
    return True


class MultiCycleTestDataset:
    """
    Returns sliding windows of NUM_CYCLES from test sessions.
    Each window maps to the middle cycle's GT.
    Skips windows with unhealthy cycles.
    """
    def __init__(self, root_dir, test_sids):
        self.windows = []
        self.metadata = []

        root = Path(root_dir)
        for npz_p in root.rglob("*_cycles.npz"):
            try:
                data = np.load(npz_p)
                sid = int(data['sid']) if 'sid' in data else 999
                if sid not in test_sids:
                    continue

                gt = data['gt_cycles']
                g = data['g_cycles']
                r = data['rppg_cycles']
                hr = data.get('hr', np.full(len(gt), 70.0))
                n = len(gt)

                if n < NUM_CYCLES:
                    continue

                half = NUM_CYCLES // 2
                for i in range(half, n - half):
                    win_idx = list(range(i - half, i + half + 1))
                    if not _check_cycles_healthy(gt, hr, win_idx):
                        continue
                    self.windows.append((
                        gt[i],
                        g[i - half:i + half + 1],
                        r[i - half:i + half + 1],
                    ))
                    self.metadata.append(sid)
            except Exception:
                continue

        print(f"Built {len(self.windows)} test windows")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        gt, g_win, r_win = self.windows[idx]
        sid = self.metadata[idx]
        return (
            torch.from_numpy(gt).unsqueeze(0).float(),
            torch.from_numpy(g_win).float(),
            torch.from_numpy(r_win).float(),
            sid,
        )


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    split_df = pd.read_csv(SPLIT_FILE)
    test_sids = set(split_df[split_df['split'] == 'test']['sid'])

    test_ds = MultiCycleTestDataset(CYCLES_DIR, test_sids)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    vae = PPGVAE(latent_dim=LATENT_DIM).to(device)
    possible = [
        Path(r'E:\Projects_Results\rPPG_Morphology_Restore\checkpoints') / VAE_CKPT_P4,
    ]
    loaded = False
    for p in possible:
        if p.exists():
            vae.load_state_dict(torch.load(p, map_location=device, weights_only=True))
            vae.eval()
            print(f"Loaded VAE from {p}")
            loaded = True
            break
    if not loaded:
        raise FileNotFoundError(f"VAE not found in {possible}")

    encoders = {}
    for name, ch in [('A', 1), ('B', 1), ('C', 2)]:
        ckpt_p = CKPT_A4 / f'encoder_a4_{name}.pt'
        if not ckpt_p.exists():
            print(f"  Skip A4-{name}: checkpoint not found at {ckpt_p}")
            continue
        enc = MultiCycleTransformerEncoder(
            latent_dim=LATENT_DIM, in_channels=ch, num_cycles=NUM_CYCLES,
            d_model=D_MODEL, nhead=NHEAD, n_layers=N_LAYERS, dropout=0.0,
        ).to(device)
        state = torch.load(ckpt_p, map_location=device, weights_only=True)
        enc.load_state_dict(state['encoder'])
        enc.eval()
        encoders[name] = enc
        print(f"Loaded A4-{name}")

    if not encoders:
        print("No A4 checkpoints found. Exiting.")
        return

    results = []

    with torch.no_grad():
        for gt, g_seq, r_seq, sids in tqdm(test_loader, desc="Eval"):
            gt_np = gt.numpy()[:, 0, :]
            g_seq_np = g_seq.numpy()
            r_seq_np = r_seq.numpy()
            sids_np = sids.numpy()

            for i in range(len(gt_np)):
                sid = int(sids_np[i])
                ds_name = split_df[split_df['sid'] == sid]['dataset'].iloc[0]

                gt_h = harmonic_ratios(gt_np[i])
                gt_ipa_val = compute_ipa(gt_np[i])
                gt_notch = notch_index(gt_np[i]) >= 0

                for name, enc in encoders.items():
                    if name == 'A':
                        x = torch.from_numpy(g_seq_np[i]).unsqueeze(0).unsqueeze(2).to(device)
                    elif name == 'B':
                        x = torch.from_numpy(r_seq_np[i]).unsqueeze(0).unsqueeze(2).to(device)
                    else:
                        g_t = torch.from_numpy(g_seq_np[i]).unsqueeze(0).unsqueeze(2)
                        r_t = torch.from_numpy(r_seq_np[i]).unsqueeze(0).unsqueeze(2)
                        x = torch.cat([g_t, r_t], dim=2).to(device)

                    z = enc(x)
                    recon = vae.decode(z).cpu().numpy()[0, 0, :]

                    corr = shape_r(recon, gt_np[i])
                    pred_h = harmonic_ratios(recon)
                    pred_ipa_val = compute_ipa(recon)
                    pred_notch = notch_index(recon) >= 0

                    results.append({
                        'sid': sid,
                        'dataset': ds_name,
                        'encoder': f'A4-{name}',
                        'shape_r': corr,
                        'gt_h2h1': gt_h['h2h1'],
                        'pred_h2h1': pred_h['h2h1'],
                        'h2h1_error': abs(pred_h['h2h1'] - gt_h['h2h1']),
                        'gt_ipa': gt_ipa_val,
                        'pred_ipa': pred_ipa_val,
                        'ipa_error': abs(pred_ipa_val - gt_ipa_val),
                        'gt_has_notch': gt_notch,
                        'pred_has_notch': pred_notch,
                    })

    res_df = pd.DataFrame(results)
    RESULTS_A4.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(RESULTS_A4 / 'full_eval_a4.csv', index=False)

    print("\n" + "=" * 60)
    print("A4 MULTI-CYCLE TRANSFORMER RESULTS")
    print("=" * 60)

    print("\n[Per-subject Pearson r]")
    per_subj = res_df.groupby(['encoder', 'sid'])['shape_r'].mean().groupby('encoder').mean()
    print(per_subj.round(4).to_string())

    print("\n[Pearson r by Dataset]")
    summary = res_df.groupby(['dataset', 'encoder'])['shape_r'].mean().unstack()
    print(summary.round(4).to_string())

    print("\n[H2/H1 Error]")
    h2 = res_df.groupby('encoder')['h2h1_error'].mean()
    print(h2.round(4).to_string())

    print("\n[IPA Error]")
    ipa = res_df.groupby('encoder')['ipa_error'].mean()
    print(ipa.round(4).to_string())

    print(f"\nResults saved to {RESULTS_A4 / 'full_eval_a4.csv'}")


if __name__ == "__main__":
    main()
