"""
step5_stage2_encoders.py — Stage 2: The Restorative Mapping (The "Translators")
================================================================================
Trains three parallel encoders to map camera signals into the VAE latent space.
Proves exactly which signal source (Raw G vs. Processed rPPG) carries the notch.

Battle Versions:
  Encoder A: Raw G-channel only (Tests H_A)
  Encoder B: windowed rPPG POS only (Tests H_B)
  Encoder C: Fusion [Raw G || rPPG] (The Master Model)
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
PIPELINE_ROOT = HERE.parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from morph_config import (
    CYCLES_DIR, CHECKPOINTS_DIR, RESULTS_DIR, SPLIT_FILE,
    LATENT_DIM, BATCH_SIZE, LEARNING_RATE,
    MAX_EPOCHS_STAGE2, EARLY_STOP_PATIENCE,
    LAMBDA_L1, LAMBDA_NOTCH, LAMBDA_SDTW, LAMBDA_CURV,
    LAMBDA_LATENT, LAMBDA_VARIANCE, LAMBDA_ADV, ADV_START_EPOCH,
    LAMBDA_DIVERSITY, LAMBDA_FREQ, LAMBDA_ASYM,
    N_SUBJECTS_PER_BATCH, FORCE_RETRAIN_STAGE2,
    SPLIT_SEED, CKPT_EVERY_N_EPOCHS,
    # Phase 4 additions
    MORPHO_AUX_HEADS, LAMBDA_AUX_MORPHO, LAMBDA_SPECTRAL,
    VAE_CKPT_P4, ENCODER_CKPT_P4, CONTRASTIVE_CKPT,
    PRETRAIN_ENCODER, VAE_CKPT_P5, ENCODER_CKPT_P5,
)
from models.vae import PPGVAE
from models.encoder import (
    CameraEncoder, Discriminator, gradient_penalty, stage2_loss
)
from models.metrics import batch_morpho_labels

# Shared Research Weights
LAMBDAS = {
    'l1': LAMBDA_L1, 'notch': LAMBDA_NOTCH, 'sdtw': LAMBDA_SDTW, 'curv': LAMBDA_CURV,
    'latent': LAMBDA_LATENT, 'variance': LAMBDA_VARIANCE, 'adv': LAMBDA_ADV,
    'adv_start': ADV_START_EPOCH, 'diversity': LAMBDA_DIVERSITY,
    'freq': 0.0,                  # disabled — replaced by spectral
    'spectral': LAMBDA_SPECTRAL,  # Phase 4: full FFT amplitude L1
    'asym': LAMBDA_ASYM, 'contrastive': 0.5,
    'aux_morpho': LAMBDA_AUX_MORPHO if MORPHO_AUX_HEADS else 0.0,  # Phase 4
}

# ==============================================================================
# DATASET
# ==============================================================================

class UnifiedCycleDataset(Dataset):
    """Loads cycle triplets [GT, G-Raw, rPPG] from the new v3 folder structure."""
    def __init__(self, root_dir, split_sids):
        self.items = [] # (gt, g, rppg, sid)
        self.sid_list = []
        
        root = Path(root_dir)
        print(f"Loading matched cycles for split subjects...")
        
        all_npz = list(root.rglob("*_cycles.npz"))
        for npz_p in tqdm(all_npz, desc="Loading Triplets"):
            try:
                data = np.load(npz_p)
                sid = int(data['sid']) if 'sid' in data else 999
                if sid not in split_sids: continue
                
                gt = data['gt_cycles']
                g  = data['g_cycles']
                # Prefer CHROM (de Haan 2013) — historically gave r=0.711; fall back to POS
                r = (data['rppg_chrom_cycles'] if 'rppg_chrom_cycles' in data else
                     data['rppg_pos_cycles']   if 'rppg_pos_cycles'   in data else
                     data['rppg_cycles'])
                
                for i in range(len(gt)):
                    self.items.append((gt[i], g[i], r[i], sid))
                    self.sid_list.append(sid)
            except: continue
            
        print(f"Loaded {len(self.items)} triplets from {len(set(self.sid_list))} subjects.")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        gt, g, rppg, sid = self.items[idx]
        return (torch.from_numpy(gt).unsqueeze(0).float(),
                torch.from_numpy(g).unsqueeze(0).float(),
                torch.from_numpy(rppg).unsqueeze(0).float(),
                sid)

class SubjectStratifiedSampler(Sampler):
    """Forces each batch to contain cycles from multiple subjects."""
    def __init__(self, dataset, batch_size, n_subs_per_batch, seed=42):
        self.batch_size = batch_size
        self.n_subs_per_batch = n_subs_per_batch
        self.rng = random.Random(seed)
        self.sid_to_idxs = {}
        for idx, sid in enumerate(dataset.sid_list):
            self.sid_to_idxs.setdefault(sid, []).append(idx)
        self.all_sids = list(self.sid_to_idxs.keys())
        self.n_total = len(dataset)

    def __iter__(self):
        sids = self.all_sids[:]
        self.rng.shuffle(sids)
        n_per_sub = max(1, self.batch_size // min(self.n_subs_per_batch, len(sids)))
        indices = []
        for sid in sids:
            pool = self.sid_to_idxs[sid][:]
            self.rng.shuffle(pool)
            indices.extend(pool[:n_per_sub])
        self.rng.shuffle(indices)
        return iter(indices)

    def __len__(self): return self.n_total

# ==============================================================================
# TRAINING ENGINE
# ==============================================================================

def train_one_encoder(enc_name, in_ch, get_input_fn, train_loader, val_loader, stage1, device):
    print(f"\nTraining ENCODER {enc_name} (in_channels={in_ch})")

    ckpt_name = ENCODER_CKPT_P5.format(name=enc_name)
    ckpt_p = Path(CHECKPOINTS_DIR) / ckpt_name
    if ckpt_p.exists() and not FORCE_RETRAIN_STAGE2:
        print(f"  Skipping {enc_name} - Checkpoint exists.")
        return

    encoder = CameraEncoder(latent_dim=LATENT_DIM, in_channels=in_ch,
                            morpho_aux=MORPHO_AUX_HEADS).to(device)

    # Phase 4: load contrastive pre-trained weights if available
    if enc_name == PRETRAIN_ENCODER:
        pretrain_ckpt = Path(CHECKPOINTS_DIR) / CONTRASTIVE_CKPT.format(name=enc_name)
        if pretrain_ckpt.exists():
            state = torch.load(pretrain_ckpt, map_location=device, weights_only=True)
            missing, unexpected = encoder.load_state_dict(state, strict=False)
            print(f'    Loaded pre-trained weights for Encoder {enc_name} '
                  f'(missing={len(missing)}, unexpected={len(unexpected)})')
        else:
            print(f'    No pre-trained checkpoint found for Encoder {enc_name} — training from scratch')

    discriminator = Discriminator().to(device)

    # Freeze VAE encoder, but keep decoder partially trainable for fine-tuning
    for p in stage1.encoder.parameters(): p.requires_grad = False

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(stage1.decoder.parameters()), lr=LEARNING_RATE)
    opt_disc = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE * 0.5)

    best_loss = float('inf')
    patience = 0

    for epoch in range(1, MAX_EPOCHS_STAGE2 + 1):
        encoder.train(); stage1.decoder.train(); discriminator.train()
        tr_losses = []
        tr_components = []

        for gt, g, rppg, sids in train_loader:
            gt, g, rppg = gt.to(device), g.to(device), rppg.to(device)
            x_in = get_input_fn(g, rppg)

            # Phase 4: morphological labels for aux head supervision
            morpho_labels_np = batch_morpho_labels(gt.cpu().numpy()[:, 0, :])
            morpho_labels_t  = torch.from_numpy(morpho_labels_np).to(device)

            # WGAN-GP Step
            if epoch >= ADV_START_EPOCH:
                with torch.no_grad():
                    z_p, _ = encoder.forward_morpho(x_in)
                    fake = stage1.decode(z_p)
                d_real = discriminator(gt)
                d_fake = discriminator(fake.detach())
                gp = gradient_penalty(discriminator, gt, fake.detach(), device)
                d_loss = d_fake.mean() - d_real.mean() + gp
                opt_disc.zero_grad(); d_loss.backward(); opt_disc.step()

            # Mapping Step
            optimizer.zero_grad()
            z_p, morpho_pred = encoder.forward_morpho(x_in)
            recon = stage1.decode(z_p)
            with torch.no_grad(): z_gt = stage1.encode(gt)

            adv_l = -discriminator(recon).mean() if epoch >= ADV_START_EPOCH else None
            loss, components = stage2_loss(
                recon, gt, z_p, z_gt,
                torch.full((gt.size(0),), -1, device=device),
                LAMBDAS, epoch, adv_l, sids=sids.to(device),
                morpho_pred=morpho_pred, morpho_labels=morpho_labels_t,
            )
            loss.backward(); optimizer.step()
            tr_losses.append(loss.item())
            tr_components.append(components)

        # Validation
        encoder.eval(); stage1.decoder.eval()
        vl_losses = []
        with torch.no_grad():
            for gt, g, rppg, _ in val_loader:
                gt, g, rppg = gt.to(device), g.to(device), rppg.to(device)
                z_p, _ = encoder.forward_morpho(get_input_fn(g, rppg))
                recon = stage1.decode(z_p); z_gt = stage1.encode(gt)
                loss, _ = stage2_loss(recon, gt, z_p, z_gt,
                                      torch.full((gt.size(0),), -1, device=device),
                                      LAMBDAS, epoch, None)
                vl_losses.append(loss.item())

        avg_tr, avg_vl = np.mean(tr_losses), np.mean(vl_losses)
        if epoch % 5 == 0:
            print(f"  Epoch {epoch:03d} | Train: {avg_tr:.4f} | Val: {avg_vl:.4f} | "
                  f"[l1={np.mean([c['l1'] for c in tr_components]):.3f} "
                  f"contr={np.mean([c['contrastive'] for c in tr_components]):.3f} "
                  f"adv={np.mean([c['adv'] for c in tr_components]):.3f}]")

        if avg_vl < best_loss:
            best_loss = avg_vl; patience = 0
            torch.save({
                'encoder':          encoder.state_dict(),
                'decoder_finetune': stage1.decoder.state_dict(),
            }, ckpt_p)
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE: break

    print(f"  Encoder {enc_name} Complete. Best Val: {best_loss:.4f}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Load split and data
    split_df = pd.read_csv(SPLIT_FILE)
    train_sids = set(split_df[split_df['split']=='train']['sid'])
    val_sids   = set(split_df[split_df['split']=='val']['sid'])
    
    train_ds = UnifiedCycleDataset(CYCLES_DIR, train_sids)
    val_ds   = UnifiedCycleDataset(CYCLES_DIR, val_sids)
    
    sampler = SubjectStratifiedSampler(train_ds, BATCH_SIZE, N_SUBJECTS_PER_BATCH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. Load Phase 5 VAE (reused Phase 4 Polymate-only VAE)
    stage1 = PPGVAE(latent_dim=LATENT_DIM).to(device)
    stage1.load_state_dict(torch.load(Path(CHECKPOINTS_DIR) / VAE_CKPT_P5, map_location=device, weights_only=True))
    print(f"Loaded Phase 5 VAE from: {VAE_CKPT_P5}")

    # 3. The 3-Encoder Battle
    # A: G-Raw
    train_one_encoder('A', 1, lambda g, r: g, train_loader, val_loader, stage1, device)
    # B: rPPG
    train_one_encoder('B', 1, lambda g, r: r, train_loader, val_loader, stage1, device)
    # C: Fusion
    train_one_encoder('C', 2, lambda g, r: torch.cat([g, r], dim=1), train_loader, val_loader, stage1, device)

    print("\nPhase 2 Complete. All restorative translators trained.")
    print("Next Step: python training/step6_evaluate.py")

if __name__ == "__main__":
    main()
