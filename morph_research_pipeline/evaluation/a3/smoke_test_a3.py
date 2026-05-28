"""
evaluation/a3/smoke_test_a3.py — A3 Smoke Test (Synthetic)
==========================================================
Runs A3 VQ-VAE end-to-end on synthetic data.

Outputs:
  results/a3/smoke_test_results.csv
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

from config.paths import RESULTS_A3
from models.vae_a3 import PPGVQVAE, vqvae_loss
from models.encoder_a3 import CameraEncoderVQ
from models.metrics import compute_ipa
from scipy.stats import pearsonr

LATENT_DIM = 64
NUM_EMBEDDINGS = 512
N_CYCLES = 200
EPOCHS_VQ = 30
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
    results_dir = Path(RESULTS_A3)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    print("Generating synthetic data...")
    gt = synthetic_ppg(N_CYCLES)
    cam = synthetic_camera(gt, noise_std=0.08)

    gt_t = torch.tensor(gt[:, np.newaxis, :], dtype=torch.float32)
    cam_t = torch.tensor(cam[:, np.newaxis, :], dtype=torch.float32)
    loader = DataLoader(TensorDataset(gt_t, cam_t), batch_size=BS, shuffle=True)

    # --- Stage 1: VQ-VAE ---
    print("\n--- A3 Stage 1: VQ-VAE K=512, dim=64 (30 epochs) ---")
    vqvae = PPGVQVAE(latent_dim=LATENT_DIM, num_embeddings=NUM_EMBEDDINGS).to(device)
    opt = torch.optim.Adam(vqvae.parameters(), lr=1e-3)

    for epoch in range(1, EPOCHS_VQ + 1):
        vqvae.train()
        losses = []; pplx = []
        for gt_batch, cam_batch in loader:
            gt_batch = gt_batch.to(device)
            recon, cl, cbl, perplexity, _ = vqvae(gt_batch)
            loss, _, _, _ = vqvae_loss(recon, gt_batch, cl, cbl)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
            pplx.append(perplexity.item())
        if epoch % 5 == 0:
            print(f"  Epoch {epoch} | Loss: {np.mean(losses):.4f} | Perplexity: {np.mean(pplx):.1f}")

    # --- Stage 2: Encoder ---
    print("\n--- A3 Stage 2: Encoder (20 epochs) ---")
    encoder = CameraEncoderVQ(latent_dim=LATENT_DIM, in_channels=1).to(device)
    for p in vqvae.encoder.parameters():
        p.requires_grad = False
    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(vqvae.decoder.parameters()) + list(vqvae.quantizer.parameters()),
        lr=1e-3
    )

    for epoch in range(1, EPOCHS_ENC + 1):
        encoder.train(); vqvae.decoder.train(); vqvae.quantizer.train()
        losses = []
        for gt_batch, cam_batch in loader:
            gt_batch, cam_batch = gt_batch.to(device), cam_batch.to(device)
            z_e = encoder(cam_batch)
            z_q, cl, cbl, _, _ = vqvae.quantizer(z_e)
            recon = vqvae.decoder(z_q)
            loss = F.l1_loss(recon, gt_batch) + cl * 0.25
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        if epoch % 5 == 0:
            print(f"  Epoch {epoch} | Loss: {np.mean(losses):.4f}")

    # --- Evaluate ---
    print("\n--- A3 Evaluation ---")
    encoder.eval(); vqvae.eval()
    with torch.no_grad():
        z_e = encoder(cam_t.to(device))
        z_q, _, _, _, indices = vqvae.quantizer(z_e)
        recon = vqvae.decoder(z_q).cpu().numpy()[:, 0, :]

    correlations = []
    ipa_errors = []
    for i in range(N_CYCLES):
        if np.std(recon[i]) > 1e-6 and np.std(gt[i]) > 1e-6:
            correlations.append(pearsonr(recon[i], gt[i])[0])
        ipa_errors.append(abs(compute_ipa(gt[i]) - compute_ipa(recon[i])))

    avg_r = np.mean(correlations) if correlations else 0
    avg_ipa_err = np.mean(ipa_errors)

    unique_codes = len(np.unique(indices.cpu().numpy()))
    print(f"  Avg Pearson r:      {avg_r:.4f}")
    print(f"  Avg IPA error:      {avg_ipa_err:.4f}")
    print(f"  Codebook usage:     {unique_codes}/{NUM_EMBEDDINGS} entries used")

    print(f"\n  SMOKE TEST: {'PASS' if avg_r > 0.3 else 'FAIL'} (r={avg_r:.4f})")

    import pandas as pd
    pd.DataFrame([{'architecture': 'A3', 'latent_dim': LATENT_DIM, 'avg_r': avg_r,
                    'avg_ipa_error': avg_ipa_err, 'codebook_usage': unique_codes,
                    'codebook_total': NUM_EMBEDDINGS}]).to_csv(
        results_dir / 'smoke_test_results.csv', index=False)
    print(f"  Saved to {results_dir / 'smoke_test_results.csv'}")


if __name__ == "__main__":
    main()
