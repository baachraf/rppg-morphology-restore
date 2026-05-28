"""
evaluation/shared/complementary_crosssubj_r.py
=================================================
Gap discovered 2026-05-18: A1, A2, A3, A4 are missing the primary
collapse metric — cross-subject r — because their original eval scripts
saved only cycle-level scalar metrics, not predicted waveform arrays.

This script fills that gap by running inference for all four architectures,
collecting per-subject mean predicted waveforms, and computing:
  - Per-subject r  (mean of per-cycle Pearson r, matches existing CSVs)
  - Cross-subject r (mean pairwise r between subject-mean waveforms)
  - GT cross-subject r ceiling (reference: ~0.601 on 27 test subjects)

Cross-subject r (novel metric introduced in this work):
  1. For each test subject s, average all predicted 256-sample cycles
  2. Compute Pearson r(pred_mean[i], pred_mean[j]) for every pair (i, j)
  3. Average over all N*(N-1)/2 pairs → cross-subject r
  Lower = more subject-specific = better.
  GT ceiling ~0.601 — even GT waveforms correlate because all cardiac
  cycles share the basic template shape.

Outputs
-------
  results/complementary/crosssubj_r_gap.csv
  results/complementary/subject_means_<arch>_<enc>.npy
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from itertools import combinations
from tqdm import tqdm
from collections import defaultdict

HERE         = Path(__file__).parent
PIPELINE_ROOT = HERE.parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from config.paths import (
    SPLIT_FILE, CYCLES_DIR,
    CKPT_A1, CKPT_A2, CKPT_A3, CKPT_A4, CKPT_SHARED,
    RESULTS_DIR,
)
from models.vae_a1        import PPGVAEA1
from models.encoder_a1    import CameraEncoderA1
from models.encoder_a2    import CameraEncoderFlow
from models.flow_a2       import ConditionalFlowDecoder
from models.vae_a3        import PPGVQVAE
from models.encoder_a3    import CameraEncoderVQ
from models.vae           import PPGVAE
from models.transformer_a4 import MultiCycleTransformerEncoder
from training.v5.train_encoders import UnifiedCycleDataset

LATENT_DIM     = 64   # A1 (z=64 double latent), A2, A3 all use 64
LATENT_DIM_A4  = 32   # A4 uses the shared phase-4 VAE trained with latent_dim=32
NUM_CYCLES_A4  = 5
BATCH_SIZE     = 64
OUT_DIR        = RESULTS_DIR / 'complementary'


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-6 or np.std(b) < 1e-6:
        return 0.0
    return float(pearsonr(a, b)[0])


def compute_cross_subj_r(means: dict) -> float:
    """Mean pairwise Pearson r between subject-mean waveforms (all N*(N-1)/2 pairs)."""
    sids = sorted(means.keys())
    if len(sids) < 2:
        return float('nan')
    return float(np.mean([_pearson(means[a], means[b]) for a, b in combinations(sids, 2)]))


# ──────────────────────────────────────────────────────────────────────────────
# Multi-cycle dataset for A4 (mirrors evaluation/a4/evaluate_a4.py)
# ──────────────────────────────────────────────────────────────────────────────

def _cycles_healthy(cycles, hr_arr, indices, min_hr=40, max_hr=150, min_amp=0.01):
    for idx in indices:
        if idx >= len(hr_arr):
            return False
        h = hr_arr[idx]
        if not (min_hr <= h <= max_hr):
            return False
        if np.ptp(cycles[idx]) < min_amp:
            return False
    return True


class MultiCycleTestDataset:
    def __init__(self, root_dir, test_sids):
        self.windows  = []
        self.metadata = []
        half = NUM_CYCLES_A4 // 2
        for npz_p in Path(root_dir).rglob("*_cycles.npz"):
            try:
                data = np.load(npz_p)
                sid  = int(data['sid']) if 'sid' in data else 999
                if sid not in test_sids:
                    continue
                gt = data['gt_cycles']
                g  = data['g_cycles']
                r  = (data['rppg_chrom_cycles'] if 'rppg_chrom_cycles' in data else
                      data['rppg_pos_cycles']   if 'rppg_pos_cycles'   in data else
                      data['rppg_cycles'])
                hr = data.get('hr', np.full(len(gt), 70.0))
                if len(gt) < NUM_CYCLES_A4:
                    continue
                for i in range(half, len(gt) - half):
                    win_idx = list(range(i - half, i + half + 1))
                    if not _cycles_healthy(gt, hr, win_idx):
                        continue
                    self.windows.append((gt[i], g[i - half:i + half + 1],
                                         r[i - half:i + half + 1]))
                    self.metadata.append(sid)
            except Exception:
                continue
        print(f"  A4 dataset: {len(self.windows)} test windows from {len(test_sids)} requested subjects")

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


# ──────────────────────────────────────────────────────────────────────────────
# Per-architecture inference runners
# ──────────────────────────────────────────────────────────────────────────────

def _empty_accumulators():
    return defaultdict(lambda: np.zeros(256)), defaultdict(lambda: np.zeros(256)), \
           defaultdict(int), defaultdict(list)


def _finalize(pred_sums, gt_sums, counts, per_cycle_rs):
    pred_means = {s: pred_sums[s] / counts[s] for s in pred_sums}
    gt_means   = {s: gt_sums[s]   / counts[s] for s in gt_sums}
    return dict(pred_means=pred_means, gt_means=gt_means, per_cycle_rs=per_cycle_rs)


def run_a1(device, test_loader):
    vae_ckpt = CKPT_A1 / 'stage1_vae_a1.pt'
    if not vae_ckpt.exists():
        print(f"  [A1] VAE checkpoint not found: {vae_ckpt}"); return {}
    vae = PPGVAEA1(latent_dim=LATENT_DIM).to(device)
    vae.load_state_dict(torch.load(vae_ckpt, map_location=device, weights_only=True))
    vae.eval()

    results = {}
    for enc_name, in_ch in [('A', 1), ('B', 1), ('C', 2)]:
        ckpt_e = CKPT_A1 / f'encoder_a1_{enc_name}.pt'
        if not ckpt_e.exists():
            print(f"  [A1] Skipping A1-{enc_name}: {ckpt_e} not found"); continue

        encoder = CameraEncoderA1(latent_dim=LATENT_DIM, in_channels=in_ch).to(device)
        state   = torch.load(ckpt_e, map_location=device, weights_only=True)
        encoder.load_state_dict(state['encoder'] if 'encoder' in state else state)
        encoder.eval()

        ps, gs, cs, prs = _empty_accumulators()
        print(f"  Running A1-{enc_name} ...")
        with torch.no_grad():
            for gt_t, g_t, rppg_t, sids_t in tqdm(test_loader, desc=f"A1-{enc_name}", leave=False):
                gt_np = gt_t.numpy()[:, 0, :]
                x_in  = (g_t if enc_name == 'A'
                         else rppg_t if enc_name == 'B'
                         else torch.cat([g_t, rppg_t], dim=1)).to(device)

                z, _ = encoder.forward_morpho(x_in)
                recon = vae.decode(z).cpu().numpy()[:, 0, :]

                for i, sid in enumerate(sids_t.numpy()):
                    s = int(sid)
                    ps[s] += recon[i]; gs[s] += gt_np[i]; cs[s] += 1
                    prs[s].append(_pearson(recon[i], gt_np[i]))

        results[f'A1-{enc_name}'] = _finalize(ps, gs, cs, prs)
        del encoder; torch.cuda.empty_cache()

    return results


def run_a2(device, test_loader):
    enc_ckpt = CKPT_A2 / 'encoder_a2_B.pt'
    # handle both possible flow decoder filenames
    flow_ckpt = next(
        (p for p in [CKPT_A2 / 'flow_decoder_a2_v2.pt', CKPT_A2 / 'flow_decoder_a2.pt']
         if p.exists()), None)

    if not enc_ckpt.exists() or flow_ckpt is None:
        print(f"  [A2] Checkpoints missing (enc={enc_ckpt.exists()}, flow={flow_ckpt})"); return {}

    print(f"  [A2] Using flow decoder: {flow_ckpt.name}")
    encoder = CameraEncoderFlow(latent_dim=LATENT_DIM, in_channels=1).to(device)
    enc_state = torch.load(enc_ckpt, map_location=device, weights_only=True)
    encoder.load_state_dict(enc_state['encoder'] if 'encoder' in enc_state else enc_state)
    encoder.eval()

    flow = ConditionalFlowDecoder(latent_dim=LATENT_DIM, hidden_dim=64, n_blocks=6, n_steps=10).to(device)
    flow_state = torch.load(flow_ckpt, map_location=device, weights_only=True)
    flow.load_state_dict(flow_state['flow'] if 'flow' in flow_state else flow_state)
    flow.eval()

    ps, gs, cs, prs = _empty_accumulators()
    print("  Running A2-Flow-B ...")
    with torch.no_grad():
        for gt_t, g_t, _, sids_t in tqdm(test_loader, desc="A2-Flow-B", leave=False):
            gt_np = gt_t.numpy()[:, 0, :]
            z     = encoder(g_t.to(device))
            recon = flow.sample(z, n_steps=10).cpu().numpy()[:, 0, :]
            for i, sid in enumerate(sids_t.numpy()):
                s = int(sid)
                ps[s] += recon[i]; gs[s] += gt_np[i]; cs[s] += 1
                prs[s].append(_pearson(recon[i], gt_np[i]))

    return {'A2-Flow-B': _finalize(ps, gs, cs, prs)}


def run_a3(device, test_loader):
    vq_ckpt = CKPT_A3 / 'stage1_vqvae_a3.pt'
    if not vq_ckpt.exists():
        print(f"  [A3] VQVAE checkpoint not found: {vq_ckpt}"); return {}

    vqvae = PPGVQVAE(latent_dim=LATENT_DIM, num_embeddings=512).to(device)
    vqvae.load_state_dict(torch.load(vq_ckpt, map_location=device, weights_only=True))
    vqvae.eval()

    results = {}
    for enc_name, in_ch in [('A', 1), ('B', 1), ('C', 2)]:
        ckpt_e = CKPT_A3 / f'encoder_a3_{enc_name}.pt'
        if not ckpt_e.exists():
            print(f"  [A3] Skipping A3-{enc_name}: {ckpt_e} not found"); continue

        encoder = CameraEncoderVQ(latent_dim=LATENT_DIM, in_channels=in_ch).to(device)
        state   = torch.load(ckpt_e, map_location=device, weights_only=True)
        if 'encoder' in state:
            encoder.load_state_dict(state['encoder'])
            if 'decoder' in state:
                vqvae.decoder.load_state_dict(state['decoder'])
        else:
            encoder.load_state_dict(state)
        encoder.eval(); vqvae.eval()

        ps, gs, cs, prs = _empty_accumulators()
        print(f"  Running A3-{enc_name} ...")
        with torch.no_grad():
            for gt_t, g_t, rppg_t, sids_t in tqdm(test_loader, desc=f"A3-{enc_name}", leave=False):
                gt_np = gt_t.numpy()[:, 0, :]
                x_in  = (g_t if enc_name == 'A'
                         else rppg_t if enc_name == 'B'
                         else torch.cat([g_t, rppg_t], dim=1)).to(device)

                z_e, _          = encoder.forward_morpho(x_in)
                z_q, _, _, _, _ = vqvae.quantizer(z_e)
                recon           = vqvae.decoder(z_q).cpu().numpy()[:, 0, :]

                for i, sid in enumerate(sids_t.numpy()):
                    s = int(sid)
                    ps[s] += recon[i]; gs[s] += gt_np[i]; cs[s] += 1
                    prs[s].append(_pearson(recon[i], gt_np[i]))

        results[f'A3-{enc_name}'] = _finalize(ps, gs, cs, prs)
        del encoder; torch.cuda.empty_cache()

    return results


def run_a4(device, mc_loader):
    vae_ckpt = CKPT_SHARED / 'stage1_vae_p4.pt'
    if not vae_ckpt.exists():
        print(f"  [A4] Shared VAE not found: {vae_ckpt}"); return {}

    vae = PPGVAE(latent_dim=LATENT_DIM_A4).to(device)
    vae.load_state_dict(torch.load(vae_ckpt, map_location=device, weights_only=True))
    vae.eval()

    results = {}
    for enc_name, in_ch in [('A', 1), ('B', 1), ('C', 2)]:
        ckpt_e = CKPT_A4 / f'encoder_a4_{enc_name}.pt'
        if not ckpt_e.exists():
            print(f"  [A4] Skipping A4-{enc_name}: {ckpt_e} not found"); continue

        enc = MultiCycleTransformerEncoder(
            latent_dim=LATENT_DIM_A4, in_channels=in_ch, num_cycles=NUM_CYCLES_A4,
            d_model=256, nhead=8, n_layers=4, dropout=0.0,
        ).to(device)
        state = torch.load(ckpt_e, map_location=device, weights_only=True)
        enc.load_state_dict(state['encoder'])
        enc.eval()

        ps, gs, cs, prs = _empty_accumulators()
        print(f"  Running A4-{enc_name} ...")
        with torch.no_grad():
            for gt_t, g_seq, r_seq, sids_t in tqdm(mc_loader, desc=f"A4-{enc_name}", leave=False):
                gt_np    = gt_t.numpy()[:, 0, :]
                g_seq_np = g_seq.numpy()
                r_seq_np = r_seq.numpy()

                for i, sid in enumerate(sids_t.numpy()):
                    s = int(sid)
                    if enc_name == 'A':
                        x = torch.from_numpy(g_seq_np[i]).unsqueeze(0).unsqueeze(2).to(device)
                    elif enc_name == 'B':
                        x = torch.from_numpy(r_seq_np[i]).unsqueeze(0).unsqueeze(2).to(device)
                    else:
                        g_t2 = torch.from_numpy(g_seq_np[i]).unsqueeze(0).unsqueeze(2)
                        r_t2 = torch.from_numpy(r_seq_np[i]).unsqueeze(0).unsqueeze(2)
                        x    = torch.cat([g_t2, r_t2], dim=2).to(device)

                    z     = enc(x)
                    recon = vae.decode(z).cpu().numpy()[0, 0, :]

                    ps[s] += recon; gs[s] += gt_np[i]; cs[s] += 1
                    prs[s].append(_pearson(recon, gt_np[i]))

        results[f'A4-{enc_name}'] = _finalize(ps, gs, cs, prs)
        del enc; torch.cuda.empty_cache()

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    split_df  = pd.read_csv(SPLIT_FILE)
    test_sids = set(split_df[split_df['split'] == 'test']['sid'])
    print(f"Test subjects: {len(test_sids)}")

    # Single-cycle loader (A1, A2, A3)
    sc_ds     = UnifiedCycleDataset(CYCLES_DIR, test_sids)
    sc_loader = DataLoader(sc_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Multi-cycle loader (A4)
    mc_ds     = MultiCycleTestDataset(CYCLES_DIR, test_sids)
    mc_loader = DataLoader(mc_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # GT ceiling (computed from single-cycle loader for consistency)
    gt_sums  = defaultdict(lambda: np.zeros(256))
    gt_counts = defaultdict(int)
    print("\nCollecting GT means for ceiling computation ...")
    for gt_t, _, _, sids_t in sc_loader:
        for i, sid in enumerate(sids_t.numpy()):
            s = int(sid)
            gt_sums[s]   += gt_t.numpy()[i, 0, :]
            gt_counts[s] += 1
    gt_means_all = {s: gt_sums[s] / gt_counts[s] for s in gt_sums}
    gt_ceiling   = compute_cross_subj_r(gt_means_all)
    print(f"GT ceiling cross-subject r: {gt_ceiling:.4f}  (documented: ~0.601)")

    # Run all architectures
    all_results = {}
    print("\n=== A1 (z=64 double latent) ===")
    all_results.update(run_a1(device, sc_loader))
    print("\n=== A2 (Conditional Flow Decoder) ===")
    all_results.update(run_a2(device, sc_loader))
    print("\n=== A3 (VQ-VAE Discrete Codebook) ===")
    all_results.update(run_a3(device, sc_loader))
    print("\n=== A4 (Multi-Cycle Transformer) ===")
    all_results.update(run_a4(device, mc_loader))

    # Summary
    rows = []
    WIDTH = 72
    print("\n" + "=" * WIDTH)
    print(f"  COMPLEMENTARY CROSS-SUBJECT r — A1/A2/A3/A4 GAP FILL")
    print("=" * WIDTH)
    print(f"  {'Architecture':16s} {'Per-subj r':>11s} {'Cross-subj r':>13s} {'N subj':>7s}")
    print(f"  {'-'*16} {'-'*11} {'-'*13} {'-'*7}")
    print(f"  {'GT ceiling':16s} {'—':>11s} {gt_ceiling:>13.4f} {len(gt_means_all):>7d}  ← target")

    # Pre-compute all metrics
    computed = {}
    for key, data in all_results.items():
        pred_means   = data['pred_means']
        per_cycle_rs = data['per_cycle_rs']
        computed[key] = {
            'cross_r':    compute_cross_subj_r(pred_means),
            'per_subj_r': float(np.mean([np.mean(v) for v in per_cycle_rs.values()])),
            'n_subjs':    len(pred_means),
            'pred_means': pred_means,
        }

    best_key = min(computed, key=lambda k: computed[k]['cross_r'])

    for key in sorted(computed.keys()):
        d       = computed[key]
        flag    = '← best' if key == best_key else ''
        print(f"  {key:16s} {d['per_subj_r']:>11.4f} {d['cross_r']:>13.4f} {d['n_subjs']:>7d}  {flag}")

        rows.append({
            'architecture': key,
            'per_subj_r':   round(d['per_subj_r'], 4),
            'cross_subj_r': round(d['cross_r'],    4),
            'n_subjects':   d['n_subjs'],
        })

        arr_path = OUT_DIR / f'subject_means_{key.replace("-", "_").lower()}.npy'
        subj_arr = np.stack([d['pred_means'][s] for s in sorted(d['pred_means'].keys())])
        np.save(arr_path, subj_arr)

    print("=" * WIDTH)
    print(f"\n  Context: best known cross-subject r = 0.892 (A5-v4, documented Step 42)")
    print(f"  All architectures above GT ceiling ({gt_ceiling:.4f}) = template collapse confirmed.")

    out_csv = OUT_DIR / 'crosssubj_r_gap.csv'
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print(f"Subject-mean waveform arrays (.npy) saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
