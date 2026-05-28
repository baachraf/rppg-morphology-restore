"""
training/a2/train_a2.py — A2: Conditional Flow Decoder
======================================================
Stage 1: Train CameraEncoder (z=64) using existing VAE as feature space
Stage 2: Train Conditional Flow Decoder conditioned on encoder z'

Outputs:
  checkpoints/a2/encoder_a2_B.pt
  checkpoints/a2/flow_decoder_a2.pt
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
from config.paths import CKPT_A2, CKPT_SHARED, RESULTS_A2
from models.encoder_a2 import CameraEncoderFlow
from models.flow_a2 import ConditionalFlowDecoder, flow_loss
from models.encoder import spectral_l1_loss, subject_contrastive_loss
from models.metrics import batch_morpho_labels
from training.v5.train_encoders import UnifiedCycleDataset, SubjectStratifiedSampler

LATENT_DIM = 64
MAX_EPOCHS_ENC = 200
MAX_EPOCHS_FLOW = 200
EARLY_STOP = 30
FLOW_STEPS = 10
FLOW_LR = 2e-4

LAMBDAS = {
    'l1': 10.0, 'notch': 5.0, 'sdtw': 1.0, 'curv': 0.5,
    'latent': 0.05, 'variance': 2.0, 'adv': 0.05,
    'adv_start': 200, 'diversity': 1.0,
    'freq': 0.0, 'spectral': 1.0, 'asym': 0.3,
    'contrastive': 0.5, 'aux_morpho': 2.0,
}


def train_encoder(device):
    print("\n" + "=" * 60)
    print("A2 Stage 1: Training CameraEncoder (z=64, reconstruct via L1)")
    print("=" * 60)

    split_df = pd.read_csv(SPLIT_FILE)
    train_sids = set(split_df[split_df['split'] == 'train']['sid'])
    val_sids = set(split_df[split_df['split'] == 'val']['sid'])

    train_ds = UnifiedCycleDataset(CYCLES_DIR, train_sids)
    val_ds = UnifiedCycleDataset(CYCLES_DIR, val_sids)
    sampler = SubjectStratifiedSampler(train_ds, BATCH_SIZE, 16)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    encoder = CameraEncoderFlow(latent_dim=LATENT_DIM, in_channels=1).to(device)
    decoder_proj = nn.Sequential(
        nn.Linear(LATENT_DIM, 256 * 16), nn.LeakyReLU(0.2),
    ).to(device)
    decoder_conv = nn.Sequential(
        nn.ConvTranspose1d(256, 128, 4, 2, 1), nn.BatchNorm1d(128), nn.LeakyReLU(0.2),
        nn.ConvTranspose1d(128, 64, 4, 2, 1), nn.BatchNorm1d(64), nn.LeakyReLU(0.2),
        nn.ConvTranspose1d(64, 32, 4, 2, 1), nn.BatchNorm1d(32), nn.LeakyReLU(0.2),
        nn.ConvTranspose1d(32, 1, 4, 2, 1), nn.Sigmoid(),
    ).to(device)

    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder_proj.parameters()) + list(decoder_conv.parameters()),
        lr=LEARNING_RATE
    )
    best_loss = float('inf'); patience = 0

    for epoch in range(1, MAX_EPOCHS_ENC + 1):
        encoder.train(); decoder_proj.train(); decoder_conv.train()
        tr_losses = []
        for gt, g, rppg, sids in train_loader:
            gt, g = gt.to(device), g.to(device)
            morpho_t = torch.from_numpy(batch_morpho_labels(gt.cpu().numpy()[:, 0, :])).to(device)
            opt.zero_grad()
            z, morpho_pred = encoder.forward_morpho(g)
            h = decoder_proj(z).view(-1, 256, 16)
            recon = decoder_conv(h)
            l_recon = F.l1_loss(recon, gt) * LAMBDAS['l1']
            l_spec = spectral_l1_loss(recon, gt) * LAMBDAS['spectral']
            l_aux = F.mse_loss(morpho_pred, morpho_t) * LAMBDAS['aux_morpho'] if morpho_pred is not None else 0
            loss = l_recon + l_spec + l_aux
            loss.backward(); opt.step()
            tr_losses.append(loss.item())

        encoder.eval(); decoder_proj.eval(); decoder_conv.eval()
        vl_losses = []
        with torch.no_grad():
            for gt, g, rppg, _ in val_loader:
                gt, g = gt.to(device), g.to(device)
                z, _ = encoder.forward_morpho(g)
                h = decoder_proj(z).view(-1, 256, 16)
                recon = decoder_conv(h)
                vl_losses.append(F.l1_loss(recon, gt).item())

        avg_vl = np.mean(vl_losses)
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:03d} | Tr: {np.mean(tr_losses):.4f} | Val: {avg_vl:.4f}")
        if avg_vl < best_loss:
            best_loss = avg_vl; patience = 0
            CKPT_A2.mkdir(parents=True, exist_ok=True)
            torch.save(encoder.state_dict(), CKPT_A2 / 'encoder_a2_B.pt')
            torch.save({'proj': decoder_proj.state_dict(), 'conv': decoder_conv.state_dict()}, CKPT_A2 / 'decoder_init_a2.pt')
        else:
            patience += 1
            if patience >= EARLY_STOP:
                print(f"  Early stop epoch {epoch}. Best: {best_loss:.4f}"); break

    print(f"Encoder done. Best Val: {best_loss:.4f}")
    return encoder


def train_flow(encoder, device):
    print("\n" + "=" * 60)
    print("A2 Stage 2: Training Conditional Flow Decoder")
    print("=" * 60)

    split_df = pd.read_csv(SPLIT_FILE)
    train_sids = set(split_df[split_df['split'] == 'train']['sid'])
    val_sids = set(split_df[split_df['split'] == 'val']['sid'])

    train_ds = UnifiedCycleDataset(CYCLES_DIR, train_sids)
    val_ds = UnifiedCycleDataset(CYCLES_DIR, val_sids)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    encoder.eval()
    flow = ConditionalFlowDecoder(latent_dim=LATENT_DIM, hidden_dim=64, n_blocks=6, n_steps=FLOW_STEPS).to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=FLOW_LR)

    best_loss = float('inf'); patience = 0

    for epoch in range(1, MAX_EPOCHS_FLOW + 1):
        flow.train()
        tr_losses = []
        for gt, g, rppg, sids in train_loader:
            gt, g = gt.to(device), g.to(device)
            with torch.no_grad():
                z = encoder(g)
            loss = flow_loss(flow, z, gt)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_losses.append(loss.item())

        flow.eval()
        vl_losses = []
        with torch.no_grad():
            for gt, g, rppg, _ in val_loader:
                gt, g = gt.to(device), g.to(device)
                z = encoder(g)
                loss = flow_loss(flow, z, gt)
                vl_losses.append(loss.item())

        avg_vl = np.mean(vl_losses)
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:03d} | Tr: {np.mean(tr_losses):.4f} | Val: {avg_vl:.4f}")
        if avg_vl < best_loss:
            best_loss = avg_vl; patience = 0
            torch.save(flow.state_dict(), CKPT_A2 / 'flow_decoder_a2.pt')
        else:
            patience += 1
            if patience >= EARLY_STOP:
                print(f"  Early stop epoch {epoch}. Best: {best_loss:.4f}"); break

    print(f"Flow decoder done. Best Val: {best_loss:.4f}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    CKPT_A2.mkdir(parents=True, exist_ok=True)

    enc_ckpt = CKPT_A2 / 'encoder_a2_B.pt'
    encoder = CameraEncoderFlow(latent_dim=LATENT_DIM, in_channels=1).to(device)
    if enc_ckpt.exists():
        encoder.load_state_dict(torch.load(enc_ckpt, map_location=device, weights_only=True))
        print(f"Loaded encoder from {enc_ckpt}")
    else:
        encoder = train_encoder(device)

    train_flow(encoder, device)
    print("\nA2 Training Complete.")


if __name__ == "__main__":
    main()
