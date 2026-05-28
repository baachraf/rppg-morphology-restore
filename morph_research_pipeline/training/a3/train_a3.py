"""
training/a3/train_a3.py — A3: VQ-VAE (Vector-Quantized)
======================================================
Stage 1: Train VQ-VAE with codebook K=512, dim=64 on Polymate-only GT
Stage 2: Train CameraEncoder -> z_e -> quantize -> VQ-VAE Decoder

Outputs:
  checkpoints/a3/stage1_vqvae_a3.pt
  checkpoints/a3/encoder_a3_B.pt
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

HERE = Path(__file__).parent
PIPELINE_ROOT = HERE.parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from morph_config import CYCLES_DIR, SPLIT_FILE, BATCH_SIZE, LEARNING_RATE
from config.paths import CKPT_A3, RESULTS_A3
from models.vae_a3 import PPGVQVAE, vqvae_loss
from models.encoder_a3 import CameraEncoderVQ
from models.encoder import spectral_l1_loss, subject_contrastive_loss
from models.metrics import batch_morpho_labels
from training.v5.train_encoders import UnifiedCycleDataset, SubjectStratifiedSampler

LATENT_DIM = 64
NUM_EMBEDDINGS = 512
COMMITMENT_COST = 0.25
MAX_EPOCHS_VQ = 150
MAX_EPOCHS_ENC = 300
EARLY_STOP_VQ = 20
EARLY_STOP_ENC = 30
VAE_MIN_SID = 2000

LAMBDAS = {
    'l1': 10.0, 'notch': 5.0, 'sdtw': 1.0, 'curv': 0.5,
    'latent': 0.05, 'variance': 2.0, 'adv': 0.05,
    'adv_start': 200, 'diversity': 1.0,
    'freq': 0.0, 'spectral': 1.0, 'asym': 0.3,
    'contrastive': 0.5, 'aux_morpho': 2.0,
}


def train_vqvae(device):
    print("\n" + "=" * 60)
    print("A3 Stage 1: Training VQ-VAE (K=512, dim=64)")
    print("=" * 60)

    all_cycles = []
    root = Path(CYCLES_DIR)
    for npz_p in tqdm(list(root.rglob("*_cycles.npz")), desc="Loading GT"):
        try:
            data = np.load(npz_p)
            sid = int(data['sid']) if 'sid' in data else 999
            if sid < VAE_MIN_SID:
                for c in data['gt_cycles']:
                    all_cycles.append(c)
        except Exception:
            continue

    print(f"Loaded {len(all_cycles)} Polymate GT cycles")
    tensors = torch.tensor(np.stack(all_cycles)[:, np.newaxis, :], dtype=torch.float32)
    loader = DataLoader(torch.utils.data.TensorDataset(tensors), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = PPGVQVAE(latent_dim=LATENT_DIM, num_embeddings=NUM_EMBEDDINGS, commitment_cost=COMMITMENT_COST).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_loss = float('inf'); patience = 0

    for epoch in range(1, MAX_EPOCHS_VQ + 1):
        model.train()
        losses = []; perplexities = []
        for (batch,) in loader:
            batch = batch.to(device)
            recon, commit_loss, cb_loss, perplexity, _ = model(batch)
            loss, _, _, _ = vqvae_loss(recon, batch, commit_loss, cb_loss, COMMITMENT_COST)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(loss.item())
            perplexities.append(perplexity.item())

        avg = np.mean(losses)
        avg_pplx = np.mean(perplexities)
        if avg < best_loss:
            best_loss = avg; patience = 0
            CKPT_A3.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), CKPT_A3 / 'stage1_vqvae_a3.pt')
        else:
            patience += 1
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:03d} | Loss: {avg:.4f} | Perplexity: {avg_pplx:.1f} (best: {best_loss:.4f})")
        if patience >= EARLY_STOP_VQ:
            print(f"  Early stop epoch {epoch}. Best: {best_loss:.4f}"); break

    print(f"VQ-VAE done. Best: {best_loss:.4f}")
    return model


def train_encoder(vqvae, device):
    print("\n" + "=" * 60)
    print("A3 Stage 2: Training CameraEncoder -> VQ-VAE")
    print("=" * 60)

    split_df = pd.read_csv(SPLIT_FILE)
    train_sids = set(split_df[split_df['split'] == 'train']['sid'])
    val_sids = set(split_df[split_df['split'] == 'val']['sid'])

    train_ds = UnifiedCycleDataset(CYCLES_DIR, train_sids)
    val_ds = UnifiedCycleDataset(CYCLES_DIR, val_sids)
    sampler = SubjectStratifiedSampler(train_ds, BATCH_SIZE, 16)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    for p in vqvae.encoder.parameters():
        p.requires_grad = False

    configs = [('A', 1, lambda g, r: g), ('B', 1, lambda g, r: r), ('C', 2, lambda g, r: torch.cat([g, r], dim=1))]

    for enc_name, in_ch, get_input in configs:
        ckpt_p = CKPT_A3 / f'encoder_a3_{enc_name}.pt'
        if ckpt_p.exists():
            print(f"  Skipping A3-{enc_name} — exists"); continue

        print(f"\n  Training A3-{enc_name} (in_ch={in_ch})")
        encoder = CameraEncoderVQ(latent_dim=LATENT_DIM, in_channels=in_ch).to(device)
        opt = torch.optim.Adam(
            list(encoder.parameters()) + list(vqvae.decoder.parameters()) + list(vqvae.quantizer.parameters()),
            lr=LEARNING_RATE
        )
        best_loss = float('inf'); patience = 0

        for epoch in range(1, MAX_EPOCHS_ENC + 1):
            encoder.train(); vqvae.decoder.train(); vqvae.quantizer.train()
            tr_losses = []
            for gt, g, rppg, sids in train_loader:
                gt, g, rppg = gt.to(device), g.to(device), rppg.to(device)
                x_in = get_input(g, rppg)
                morpho_t = torch.from_numpy(batch_morpho_labels(gt.cpu().numpy()[:, 0, :])).to(device)

                opt.zero_grad()
                z_e, morpho_pred = encoder.forward_morpho(x_in)
                z_q, commit_loss, cb_loss, _, _ = vqvae.quantizer(z_e)
                recon = vqvae.decoder(z_q)

                l_recon = F.l1_loss(recon, gt) * LAMBDAS['l1']
                l_spec = spectral_l1_loss(recon, gt) * LAMBDAS['spectral']
                l_commit = commit_loss * COMMITMENT_COST
                l_cb = cb_loss * 0.25
                l_aux = F.mse_loss(morpho_pred, morpho_t) * LAMBDAS['aux_morpho'] if morpho_pred is not None else 0

                loss = l_recon + l_spec + l_commit + l_cb + l_aux
                loss.backward(); opt.step()
                tr_losses.append(loss.item())

            encoder.eval(); vqvae.decoder.eval(); vqvae.quantizer.eval()
            vl_losses = []
            with torch.no_grad():
                for gt, g, rppg, _ in val_loader:
                    gt, g, rppg = gt.to(device), g.to(device), rppg.to(device)
                    z_e, _ = encoder.forward_morpho(get_input(g, rppg))
                    z_q, cl, cbl, _, _ = vqvae.quantizer(z_e)
                    recon = vqvae.decoder(z_q)
                    vl_losses.append((F.l1_loss(recon, gt) + cl * COMMITMENT_COST).item())

            avg_vl = np.mean(vl_losses)
            if epoch % 20 == 0:
                print(f"    Epoch {epoch:03d} | Tr: {np.mean(tr_losses):.4f} | Val: {avg_vl:.4f}")
            if avg_vl < best_loss:
                best_loss = avg_vl; patience = 0
                torch.save({'encoder': encoder.state_dict(), 'decoder': vqvae.decoder.state_dict()}, ckpt_p)
            else:
                patience += 1
                if patience >= EARLY_STOP_ENC:
                    print(f"    Early stop epoch {epoch}. Best: {best_loss:.4f}"); break
        print(f"  A3-{enc_name} done. Best Val: {best_loss:.4f}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    CKPT_A3.mkdir(parents=True, exist_ok=True)

    vq_ckpt = CKPT_A3 / 'stage1_vqvae_a3.pt'
    vqvae = PPGVQVAE(latent_dim=LATENT_DIM, num_embeddings=NUM_EMBEDDINGS).to(device)
    if vq_ckpt.exists():
        vqvae.load_state_dict(torch.load(vq_ckpt, map_location=device, weights_only=True))
        print(f"Loaded VQ-VAE from {vq_ckpt}")
    else:
        vqvae = train_vqvae(device)

    train_encoder(vqvae, device)
    print("\nA3 Training Complete.")


if __name__ == "__main__":
    main()
