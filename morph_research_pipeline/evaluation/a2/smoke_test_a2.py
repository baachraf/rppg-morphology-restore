"""
evaluation/a2/smoke_test_a2.py — A2 Smoke Test (Synthetic)
==========================================================
Runs A2 end-to-end on synthetic data.

Outputs:
  results/a2/smoke_test_results.csv
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

from config.paths import RESULTS_A2
from models.encoder_a2 import CameraEncoderFlow
from models.flow_a2 import ConditionalFlowDecoder, flow_loss
from models.metrics import compute_ipa
from scipy.stats import pearsonr

LATENT_DIM = 64
N_CYCLES = 200
EPOCHS_ENC = 30
EPOCHS_FLOW = 60
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
    results_dir = Path(RESULTS_A2)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    print("Generating synthetic data...")
    gt = synthetic_ppg(N_CYCLES)
    cam = synthetic_camera(gt, noise_std=0.08)

    gt_t = torch.tensor(gt[:, np.newaxis, :], dtype=torch.float32)
    cam_t = torch.tensor(cam[:, np.newaxis, :], dtype=torch.float32)
    loader = DataLoader(TensorDataset(gt_t, cam_t), batch_size=BS, shuffle=True)

    # --- Stage 1: Encoder ---
    print("\n--- A2 Stage 1: Encoder (20 epochs) ---")
    encoder = CameraEncoderFlow(latent_dim=LATENT_DIM, in_channels=1).to(device)
    dec_proj = nn.Sequential(nn.Linear(LATENT_DIM, 256 * 16), nn.LeakyReLU(0.2)).to(device)
    dec_conv = nn.Sequential(
        nn.ConvTranspose1d(256, 128, 4, 2, 1), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
        nn.ConvTranspose1d(128, 64, 4, 2, 1), nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
        nn.ConvTranspose1d(64, 32, 4, 2, 1), nn.BatchNorm1d(32), nn.LeakyReLU(0.2),
        nn.ConvTranspose1d(32, 1, 4, 2, 1), nn.Sigmoid(),
    ).to(device)
    opt = torch.optim.Adam(list(encoder.parameters()) + list(dec_proj.parameters()) + list(dec_conv.parameters()), lr=1e-3)

    for epoch in range(1, EPOCHS_ENC + 1):
        encoder.train(); dec_proj.train(); dec_conv.train()
        losses = []
        for gt_batch, cam_batch in loader:
            gt_batch, cam_batch = gt_batch.to(device), cam_batch.to(device)
            z = encoder(cam_batch)
            h = dec_proj(z).view(-1, 256, 16)
            recon = dec_conv(h)
            loss = F.l1_loss(recon, gt_batch)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if epoch % 5 == 0:
            print(f"  Epoch {epoch} | Loss: {np.mean(losses):.4f}")

    # --- Stage 2: Flow Decoder ---
    print("\n--- A2 Stage 2: Flow Decoder (30 epochs) ---")
    flow = ConditionalFlowDecoder(latent_dim=LATENT_DIM, hidden_dim=64, n_blocks=6, n_steps=5).to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=2e-4)
    encoder.eval()

    for epoch in range(1, EPOCHS_FLOW + 1):
        flow.train()
        losses = []
        for gt_batch, cam_batch in loader:
            gt_batch, cam_batch = gt_batch.to(device), cam_batch.to(device)
            with torch.no_grad():
                z = encoder(cam_batch)
            loss = flow_loss(flow, z, gt_batch)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if epoch % 5 == 0:
            print(f"  Epoch {epoch} | Loss: {np.mean(losses):.4f}")

    # --- Evaluate ---
    print("\n--- A2 Evaluation ---")
    encoder.eval(); flow.eval()
    with torch.no_grad():
        z = encoder(cam_t.to(device))
        recon = flow.sample(z, n_steps=20).cpu().numpy()[:, 0, :]

    correlations = []
    ipa_errors = []
    for i in range(N_CYCLES):
        if np.std(recon[i]) > 1e-6 and np.std(gt[i]) > 1e-6:
            correlations.append(pearsonr(recon[i], gt[i])[0])
        ipa_errors.append(abs(compute_ipa(gt[i]) - compute_ipa(recon[i])))

    avg_r = np.mean(correlations) if correlations else 0
    avg_ipa_err = np.mean(ipa_errors)

    print(f"  Avg Pearson r:  {avg_r:.4f}")
    print(f"  Avg IPA error:  {avg_ipa_err:.4f}")
    print(f"\n  SMOKE TEST: {'PASS' if avg_r > 0.3 else 'FAIL'} (r={avg_r:.4f})")

    import pandas as pd
    pd.DataFrame([{'architecture': 'A2', 'latent_dim': LATENT_DIM, 'avg_r': avg_r,
                    'avg_ipa_error': avg_ipa_err}]).to_csv(results_dir / 'smoke_test_results.csv', index=False)
    print(f"  Saved to {results_dir / 'smoke_test_results.csv'}")


if __name__ == "__main__":
    main()
