"""
training/a1/train_a1.py — A1: z=64 VAE + CameraEncoder
======================================================
Stage 1: Train VAE with latent_dim=64 on Polymate-only GT
Stage 2: Train 3 CameraEncoders (A/B/C) targeting 64-dim latent

Outputs:
  checkpoints/a1/stage1_vae_a1.pt
  checkpoints/a1/encoder_a1_A.pt, encoder_a1_B.pt, encoder_a1_C.pt
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

HERE = Path(__file__).parent
PIPELINE_ROOT = HERE.parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from morph_config import (
    CYCLES_DIR, SPLIT_FILE, BATCH_SIZE, LEARNING_RATE
)
from config.paths import CKPT_A1
from models.vae_a1 import PPGVAEA1, vae_loss_a1
from models.encoder_a1 import CameraEncoderA1
from models.encoder import (
    Discriminator, gradient_penalty, stage2_loss
)
from models.metrics import batch_morpho_labels
from training.v5.train_encoders import UnifiedCycleDataset, SubjectStratifiedSampler

LATENT_DIM = 64
MAX_EPOCHS_VAE = 100
MAX_EPOCHS_ENC = 300
EARLY_STOP_VAE = 15
EARLY_STOP_ENC = 30
BETA_KL = 0.5
VAE_MIN_SID = 2000

LAMBDAS = {
    'l1': 10.0, 'notch': 5.0, 'sdtw': 1.0, 'curv': 0.5,
    'latent': 0.05, 'variance': 2.0, 'adv': 0.05,
    'adv_start': 50, 'diversity': 1.0,
    'freq': 0.0, 'spectral': 1.0, 'asym': 0.3,
    'contrastive': 0.5, 'aux_morpho': 2.0,
}


def train_vae(device):
    print("\n" + "=" * 60)
    print("A1 Stage 1: Training VAE with z=64")
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

    print(f"Loaded {len(all_cycles)} Polymate GT cycles (sid>={VAE_MIN_SID})")
    tensors = torch.tensor(np.stack(all_cycles)[:, np.newaxis, :], dtype=torch.float32)
    loader = DataLoader(torch.utils.data.TensorDataset(tensors), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = PPGVAEA1(latent_dim=LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_loss = float('inf')
    patience = 0

    for epoch in range(1, MAX_EPOCHS_VAE + 1):
        model.train()
        losses = []
        for (batch,) in loader:
            batch = batch.to(device)
            recon, mu, logvar = model(batch)
            loss, _, _ = vae_loss_a1(recon, batch, mu, logvar, beta=BETA_KL)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(loss.item())

        avg = np.mean(losses)
        if avg < best_loss:
            best_loss = avg; patience = 0
            CKPT_A1.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), CKPT_A1 / 'stage1_vae_a1.pt')
        else:
            patience += 1
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:03d} | Loss: {avg:.4f} (best: {best_loss:.4f})")
        if patience >= EARLY_STOP_VAE:
            print(f"  Early stop epoch {epoch}. Best: {best_loss:.4f}")
            break

    print(f"VAE done. Best: {best_loss:.4f}")
    return model


def train_encoders(vae, device):
    print("\n" + "=" * 60)
    print("A1 Stage 2: Training CameraEncoders (z=64)")
    print("=" * 60)

    split_df = pd.read_csv(SPLIT_FILE)
    train_sids = set(split_df[split_df['split'] == 'train']['sid'])
    val_sids = set(split_df[split_df['split'] == 'val']['sid'])

    train_ds = UnifiedCycleDataset(CYCLES_DIR, train_sids)
    val_ds = UnifiedCycleDataset(CYCLES_DIR, val_sids)
    sampler = SubjectStratifiedSampler(train_ds, BATCH_SIZE, 16)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    for p in vae.encoder.parameters():
        p.requires_grad = False

    configs = [
        ('A', 1, lambda g, r: g),
        ('B', 1, lambda g, r: r),
        ('C', 2, lambda g, r: torch.cat([g, r], dim=1)),
    ]

    for enc_name, in_ch, get_input in configs:
        ckpt_p = CKPT_A1 / f'encoder_a1_{enc_name}.pt'
        if ckpt_p.exists():
            print(f"  Skipping A1-{enc_name} — exists"); continue

        print(f"\n  Training A1-{enc_name} (in_ch={in_ch})")
        encoder = CameraEncoderA1(latent_dim=LATENT_DIM, in_channels=in_ch).to(device)
        disc = Discriminator().to(device)
        opt = torch.optim.Adam(list(encoder.parameters()) + list(vae.decoder.parameters()), lr=LEARNING_RATE)
        opt_d = torch.optim.Adam(disc.parameters(), lr=LEARNING_RATE * 0.5)

        best_loss = float('inf'); patience = 0

        for epoch in range(1, MAX_EPOCHS_ENC + 1):
            encoder.train(); vae.decoder.train(); disc.train()
            tr_losses = []
            for gt, g, rppg, sids in train_loader:
                gt, g, rppg = gt.to(device), g.to(device), rppg.to(device)
                x_in = get_input(g, rppg)
                morpho_t = torch.from_numpy(batch_morpho_labels(gt.cpu().numpy()[:, 0, :])).to(device)

                if epoch >= 50:
                    with torch.no_grad():
                        z_p, _ = encoder.forward_morpho(x_in)
                        fake = vae.decode(z_p)
                    d_real, d_fake = disc(gt), disc(fake.detach())
                    gp = gradient_penalty(disc, gt, fake.detach(), device)
                    d_loss = d_fake.mean() - d_real.mean() + gp
                    opt_d.zero_grad(); d_loss.backward(); opt_d.step()

                opt.zero_grad()
                z_p, morpho_pred = encoder.forward_morpho(x_in)
                recon = vae.decode(z_p)
                with torch.no_grad(): z_gt = vae.encode(gt)
                adv_l = -disc(recon).mean() if epoch >= 50 else None
                loss, _ = stage2_loss(
                    recon, gt, z_p, z_gt,
                    torch.full((gt.size(0),), -1, device=device),
                    LAMBDAS, epoch, adv_l, sids=sids.to(device),
                    morpho_pred=morpho_pred, morpho_labels=morpho_t,
                )
                loss.backward(); opt.step()
                tr_losses.append(loss.item())

            encoder.eval(); vae.decoder.eval()
            vl_losses = []
            with torch.no_grad():
                for gt, g, rppg, _ in val_loader:
                    gt, g, rppg = gt.to(device), g.to(device), rppg.to(device)
                    z_p, _ = encoder.forward_morpho(get_input(g, rppg))
                    recon = vae.decode(z_p); z_gt = vae.encode(gt)
                    loss, _ = stage2_loss(recon, gt, z_p, z_gt,
                        torch.full((gt.size(0),), -1, device=device), LAMBDAS, epoch, None)
                    vl_losses.append(loss.item())

            avg_vl = np.mean(vl_losses)
            if epoch % 20 == 0:
                print(f"    Epoch {epoch:03d} | Tr: {np.mean(tr_losses):.4f} | Val: {avg_vl:.4f}")
            if avg_vl < best_loss:
                best_loss = avg_vl; patience = 0
                torch.save({'encoder': encoder.state_dict(), 'decoder_finetune': vae.decoder.state_dict()}, ckpt_p)
            else:
                patience += 1
                if patience >= EARLY_STOP_ENC:
                    print(f"    Early stop epoch {epoch}. Best: {best_loss:.4f}"); break
        print(f"  A1-{enc_name} done. Best Val: {best_loss:.4f}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    CKPT_A1.mkdir(parents=True, exist_ok=True)

    vae_ckpt = CKPT_A1 / 'stage1_vae_a1.pt'
    vae = PPGVAEA1(latent_dim=LATENT_DIM).to(device)
    if vae_ckpt.exists():
        vae.load_state_dict(torch.load(vae_ckpt, map_location=device, weights_only=True))
        print(f"Loaded VAE from {vae_ckpt}")
    else:
        vae = train_vae(device)

    train_encoders(vae, device)
    print("\nA1 Training Complete.")


if __name__ == "__main__":
    main()
