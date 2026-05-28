"""
evaluation/a1/smoke_test_a1.py — A1 Smoke Test (Synthetic)
==========================================================
Runs A1 end-to-end on synthetic data to verify architecture, training loop,
and evaluation pipeline work without needing real data.

Outputs:
  results/a1/smoke_test_results.csv
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

HERE = Path(__file__).parent
PIPELINE_ROOT = HERE.parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from config.paths import RESULTS_A1
from models.vae_a1 import PPGVAEA1, vae_loss_a1
from models.encoder_a1 import CameraEncoderA1
from models.metrics import compute_ipa, notch_index
from scipy.stats import pearsonr

LATENT_DIM = 64
N_CYCLES = 200
N_SUBJECTS = 10
EPOCHS_VAE = 20
EPOCHS_ENC = 20
BS = 32


def synthetic_ppg(n, seed=42):
    rng = np.random.RandomState(seed)
    x = np.linspace(0, 2 * np.pi, 256)
    cycles = []
    for _ in range(n):
        a1 = rng.uniform(0.5, 1.0)
        a2 = rng.uniform(0.1, 0.5)
        a3 = rng.uniform(0.02, 0.15)
        phase = rng.uniform(-0.3, 0.3)
        y = a1 * np.sin(x + phase) + a2 * np.sin(2 * x + phase) + a3 * np.sin(3 * x)
        y = (y - y.min()) / (y.max() - y.min() + 1e-8)
        cycles.append(y)
    return np.array(cycles, dtype=np.float32)


def synthetic_camera(ppg, noise_std=0.05, seed=42):
    rng = np.random.RandomState(seed)
    noisy = ppg + rng.randn(*ppg.shape) * noise_std
    return np.clip((noisy - noisy.min(axis=1, keepdims=True)) /
                   (noisy.max(axis=1, keepdims=True) - noisy.min(axis=1, keepdims=True) + 1e-8), 0, 1)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results_dir = Path(RESULTS_A1)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    print("Generating synthetic data...")
    gt = synthetic_ppg(N_CYCLES)
    cam = synthetic_camera(gt, noise_std=0.08)

    gt_t = torch.tensor(gt[:, np.newaxis, :], dtype=torch.float32)
    cam_t = torch.tensor(cam[:, np.newaxis, :], dtype=torch.float32)
    loader = DataLoader(TensorDataset(gt_t, cam_t), batch_size=BS, shuffle=True)

    # --- Stage 1: VAE ---
    print("\n--- A1 Stage 1: VAE z=64 (20 epochs) ---")
    vae = PPGVAEA1(latent_dim=LATENT_DIM).to(device)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    for epoch in range(1, EPOCHS_VAE + 1):
        vae.train()
        losses = []
        for gt_batch, cam_batch in loader:
            gt_batch = gt_batch.to(device)
            recon, mu, logvar = vae(gt_batch)
            loss, _, _ = vae_loss_a1(recon, gt_batch, mu, logvar)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if epoch % 5 == 0:
            print(f"  Epoch {epoch} | Loss: {np.mean(losses):.4f}")

    # --- Stage 2: Encoder ---
    print("\n--- A1 Stage 2: Encoder (20 epochs) ---")
    encoder = CameraEncoderA1(latent_dim=LATENT_DIM, in_channels=1).to(device)
    for p in vae.encoder.parameters():
        p.requires_grad = False
    opt = torch.optim.Adam(list(encoder.parameters()) + list(vae.decoder.parameters()), lr=1e-3)

    for epoch in range(1, EPOCHS_ENC + 1):
        encoder.train(); vae.decoder.train()
        losses = []
        for gt_batch, cam_batch in loader:
            gt_batch, cam_batch = gt_batch.to(device), cam_batch.to(device)
            z = encoder(cam_batch)
            recon = vae.decode(z)
            with torch.no_grad():
                z_gt = vae.encode(gt_batch)
            loss = F.l1_loss(recon, gt_batch) + 0.1 * F.mse_loss(z, z_gt)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if epoch % 5 == 0:
            print(f"  Epoch {epoch} | Loss: {np.mean(losses):.4f}")

    # --- Evaluate ---
    print("\n--- A1 Evaluation ---")
    encoder.eval(); vae.eval()
    with torch.no_grad():
        z = encoder(cam_t.to(device))
        recon = vae.decode(z).cpu().numpy()[:, 0, :]

    correlations = []
    ipa_errors = []
    for i in range(N_CYCLES):
        if np.std(recon[i]) > 1e-6 and np.std(gt[i]) > 1e-6:
            correlations.append(pearsonr(recon[i], gt[i])[0])
        ipa_errors.append(abs(compute_ipa(gt[i]) - compute_ipa(recon[i])))

    avg_r = np.mean(correlations) if correlations else 0
    avg_ipa_err = np.mean(ipa_errors)
    gt_h2 = np.mean([np.abs(np.fft.rfft(gt[i])[2]) / (np.abs(np.fft.rfft(gt[i])[1]) + 1e-8) for i in range(N_CYCLES)])
    pred_h2 = np.mean([np.abs(np.fft.rfft(recon[i])[2]) / (np.abs(np.fft.rfft(recon[i])[1]) + 1e-8) for i in range(N_CYCLES)])

    print(f"  Avg Pearson r:  {avg_r:.4f}")
    print(f"  Avg IPA error:  {avg_ipa_err:.4f}")
    print(f"  GT H2/H1:       {gt_h2:.4f}")
    print(f"  Pred H2/H1:     {pred_h2:.4f}")

    result = {'architecture': 'A1', 'latent_dim': LATENT_DIM, 'avg_r': avg_r,
              'avg_ipa_error': avg_ipa_err, 'gt_h2h1': gt_h2, 'pred_h2h1': pred_h2}
    print(f"\n  SMOKE TEST: {'PASS' if avg_r > 0.3 else 'FAIL'} (r={avg_r:.4f})")

    import pandas as pd
    pd.DataFrame([result]).to_csv(results_dir / 'smoke_test_results.csv', index=False)
    print(f"  Saved to {results_dir / 'smoke_test_results.csv'}")


if __name__ == "__main__":
    main()
