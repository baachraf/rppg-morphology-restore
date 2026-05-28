"""
training/a6/train_a6.py — A6: Raw RGB Window Encoder Training
==============================================================
Trains RGBEncoderA6 on fixed-time raw RGB windows targeting the
Stage 1 VAE latent space (z=32).

Data: *_rgb_windows.npz from extraction/extract_rgb_windows.py
  - rgb_windows: (N, 3, 256) — R, G, B detrended + z-scored + PCHIP resampled
  - gt_targets:  (N, 256)    — average GT PPG cycle template

Training uses the same loss as V5-B for fair comparison:
  L1 + Soft-DTW + Curvature + Frequency + Spectral + Asymmetry
  + Diversity + Subject Contrastive + WGAN-GP + Aux Morpho

Outputs:
  checkpoints/a6/encoder_a6_D.pt  (A6-D: raw RGB detrended)
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm
from pathlib import Path

HERE = Path(__file__).parent
PIPELINE_ROOT = HERE.parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from morph_config import SPLIT_FILE, BATCH_SIZE, LEARNING_RATE
from config.paths import (
    UBFC_DIR, STRESS_DIR, FPS2023_DIR, CENTAN_DIR,
    CKPT_SHARED, RESULTS_DIR, VAE_CKPT,
)
from models.vae import PPGVAE, vae_loss
from models.encoder_a6 import RGBEncoderA6
from models.encoder import (
    Discriminator, gradient_penalty, stage2_loss,
)
from models.metrics import batch_morpho_labels

LATENT_DIM = 32
MAX_EPOCHS = 300
EARLY_STOP = 30
IN_CHANNELS = 3

CKPT_A6 = Path(RESULTS_DIR).parent / 'checkpoints' / 'a6'
RESULTS_A6 = RESULTS_DIR / 'a6'

LAMBDAS = {
    'l1': 10.0, 'notch': 5.0, 'sdtw': 1.0, 'curv': 0.5,
    'latent': 0.05, 'variance': 2.0, 'adv': 0.05,
    'adv_start': 50, 'diversity': 1.0,
    'freq': 0.0, 'spectral': 1.0, 'asym': 0.3,
    'contrastive': 0.5, 'aux_morpho': 2.0,
}


class RGBWindowDataset(Dataset):
    def __init__(self, data_dirs, split_sids):
        self.items = []
        self.sid_list = []

        for data_dir in data_dirs:
            data_dir = Path(data_dir)
            if not data_dir.is_dir():
                continue
            for npz_p in tqdm(list(data_dir.glob('*_rgb_windows.npz')),
                              desc=f'Loading {data_dir.parent.name}'):
                try:
                    data = np.load(npz_p)
                    sid = int(data['sid'])
                    if sid not in split_sids:
                        continue
                    rgb = data['rgb_windows']
                    gt = data['gt_targets']
                    for i in range(len(rgb)):
                        self.items.append((rgb[i], gt[i], sid))
                        self.sid_list.append(sid)
                except Exception:
                    continue

        print(f'Loaded {len(self.items)} windows from '
              f'{len(set(self.sid_list))} subjects.')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rgb, gt, sid = self.items[idx]
        return (torch.from_numpy(rgb).float(),
                torch.from_numpy(gt).unsqueeze(0).float(),
                sid)


class SubjectStratifiedSampler(Sampler):
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
        return iter(indices[:self.n_total])

    def __len__(self):
        return self.n_total


def find_rgb_window_dirs():
    dirs = []
    for base in [UBFC_DIR, STRESS_DIR, FPS2023_DIR, CENTAN_DIR]:
        d = base / 'rgb_windows'
        if d.is_dir():
            dirs.append(d)
    return dirs


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    CKPT_A6.mkdir(parents=True, exist_ok=True)
    RESULTS_A6.mkdir(parents=True, exist_ok=True)

    vae = PPGVAE(latent_dim=LATENT_DIM).to(device)
    if VAE_CKPT.exists():
        vae.load_state_dict(
            torch.load(VAE_CKPT, map_location=device, weights_only=True))
        print(f'Loaded VAE from {VAE_CKPT}')
    else:
        print(f'ERROR: VAE checkpoint not found at {VAE_CKPT}')
        return

    split_df = pd.read_csv(SPLIT_FILE)
    train_sids = set(split_df[split_df['split'] == 'train']['sid'])
    val_sids = set(split_df[split_df['split'] == 'val']['sid'])

    data_dirs = find_rgb_window_dirs()
    if not data_dirs:
        print('ERROR: No rgb_windows directories found.')
        print('Run extraction/extract_rgb_windows.py first.')
        return

    print(f'Found rgb_window dirs: {[str(d) for d in data_dirs]}')

    train_ds = RGBWindowDataset(data_dirs, train_sids)
    val_ds = RGBWindowDataset(data_dirs, val_sids)

    if len(train_ds) == 0:
        print('ERROR: No training windows loaded.')
        return

    sampler = SubjectStratifiedSampler(train_ds, BATCH_SIZE, 16)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)

    for p in vae.encoder.parameters():
        p.requires_grad = False

    encoder = RGBEncoderA6(latent_dim=LATENT_DIM,
                           in_channels=IN_CHANNELS,
                           morpho_aux=True).to(device)
    disc = Discriminator().to(device)

    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(vae.decoder.parameters()),
        lr=LEARNING_RATE)
    opt_d = torch.optim.Adam(disc.parameters(), lr=LEARNING_RATE * 0.5)

    ckpt_path = CKPT_A6 / 'encoder_a6_D.pt'
    log_path = RESULTS_A6 / 'training_log_a6.csv'

    best_loss = float('inf')
    patience = 0
    log_rows = []

    print(f'\nTraining A6-D (RGB detrended, in_ch={IN_CHANNELS})')
    print(f'  Train: {len(train_ds)} windows, Val: {len(val_ds)} windows')
    print(f'  Max epochs: {MAX_EPOCHS}, Early stop: {EARLY_STOP}')
    print()

    for epoch in range(1, MAX_EPOCHS + 1):
        encoder.train()
        vae.decoder.train()
        disc.train()
        tr_losses = []

        for rgb, gt, sids in train_loader:
            rgb = rgb.to(device)
            gt = gt.to(device)
            morpho_t = torch.from_numpy(
                batch_morpho_labels(gt.cpu().numpy()[:, 0, :])
            ).to(device)

            if epoch >= LAMBDAS['adv_start']:
                with torch.no_grad():
                    z_p, _ = encoder.forward_morpho(rgb)
                    fake = vae.decode(z_p)
                d_real = disc(gt)
                d_fake = disc(fake.detach())
                gp = gradient_penalty(disc, gt, fake.detach(), device)
                d_loss = d_fake.mean() - d_real.mean() + gp
                opt_d.zero_grad()
                d_loss.backward()
                opt_d.step()

            opt.zero_grad()
            z_p, morpho_pred = encoder.forward_morpho(rgb)
            recon = vae.decode(z_p)
            with torch.no_grad():
                z_gt = vae.encode(gt)
            adv_l = -disc(recon).mean() if epoch >= LAMBDAS['adv_start'] else None
            loss, components = stage2_loss(
                recon, gt, z_p, z_gt,
                torch.full((gt.size(0),), -1, device=device),
                LAMBDAS, epoch, adv_l, sids=sids.to(device),
                morpho_pred=morpho_pred, morpho_labels=morpho_t,
            )
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())

        encoder.eval()
        vae.decoder.eval()
        vl_losses = []
        with torch.no_grad():
            for rgb, gt, _ in val_loader:
                rgb, gt = rgb.to(device), gt.to(device)
                z_p, _ = encoder.forward_morpho(rgb)
                recon = vae.decode(z_p)
                z_gt = vae.encode(gt)
                vl_loss, _ = stage2_loss(
                    recon, gt, z_p, z_gt,
                    torch.full((gt.size(0),), -1, device=device),
                    LAMBDAS, epoch, None,
                )
                vl_losses.append(vl_loss.item())

        avg_tr = np.mean(tr_losses)
        avg_vl = np.mean(vl_losses)

        log_rows.append({
            'epoch': epoch, 'train_loss': avg_tr, 'val_loss': avg_vl,
            **{k: v for k, v in components.items() if k != 'total'},
        })

        if epoch % 10 == 0:
            print(f'  Epoch {epoch:03d} | Tr: {avg_tr:.4f} | Val: {avg_vl:.4f}')

        if avg_vl < best_loss:
            best_loss = avg_vl
            patience = 0
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder_finetune': vae.decoder.state_dict(),
                'epoch': epoch,
                'val_loss': avg_vl,
            }, ckpt_path)
        else:
            patience += 1
            if patience >= EARLY_STOP:
                print(f'  Early stop epoch {epoch}. Best: {best_loss:.4f}')
                break

    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(log_path, index=False)
    print(f'\nA6-D training complete. Best Val: {best_loss:.4f}')
    print(f'Checkpoint: {ckpt_path}')
    print(f'Training log: {log_path}')


if __name__ == '__main__':
    main()
