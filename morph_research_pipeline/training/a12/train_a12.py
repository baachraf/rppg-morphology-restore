"""
training/a12/train_a12.py — A12 VMD-Peak-Aligned Encoder Training
==================================================================
Hypothesis: VMD-peak-detected cycle boundaries produce phase-consistent
cycles (per-cycle r=+0.374 vs rPPG per-cycle r=-0.048), giving the encoder
cleaner training signal and potentially less template collapse.

Architecture: identical to V5-B
  CameraEncoder(in_channels=1, latent_dim=32) → z' → frozen VAE Decoder → PPG

Input: vmd_g_cycles (1, 256) — VMD-G cardiac mode, VMD-peak-aligned
       from {stem}_a12_cycles.npz written by extract_vmd_rppg.py

Loss:  Pearson (1.0) + spectral_l1 (1.0) + frequency_loss (1.0)
       (same as A11 / V5-B)

Usage:
  python morph_research_pipeline/training/a12/train_a12.py
"""

import os, sys, random
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
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

from config.paths import (
    UBFC_CYCLES_DIR, STRESS_CYCLES_DIR, FPS2023_CYCLES_DIR,
    FPS2023_60_CYCLES_DIR, CENTAN_CYCLES_DIR,
    CKPT_A12, RESULTS_A12, SPLIT_FILE,
    VAE_CKPT, ENCODER_CKPT,
)
from config.hyperparams import BATCH_SIZE, N_SUBJECTS_PER_BATCH
from models.encoder import CameraEncoder, spectral_l1_loss, frequency_loss
from models.vae import PPGVAE

LATENT_DIM      = 32
MAX_EPOCHS      = 300
EARLY_STOP      = 30
LR              = 1e-4
LAMBDA_PEARSON  = 1.0
LAMBDA_SPECTRAL = 1.0
LAMBDA_FREQ     = 1.0

CYCLES_DIRS = [
    UBFC_CYCLES_DIR, STRESS_CYCLES_DIR, FPS2023_CYCLES_DIR,
    FPS2023_60_CYCLES_DIR, CENTAN_CYCLES_DIR,
]


class A12Dataset(Dataset):
    """Loads (vmd_g_cycle, gt_cycle, sid) from *_a12_cycles.npz files."""

    def __init__(self, cycles_dirs, split_sids):
        self.items    = []
        self.sid_list = []

        for d in cycles_dirs:
            d = Path(d)
            if not d.is_dir():
                continue
            for npz_f in sorted(d.glob('*_a12_cycles.npz')):
                try:
                    data  = np.load(npz_f, allow_pickle=True)
                    sid   = int(data['sid'])
                    if sid not in split_sids:
                        continue
                    vmd_g = data['vmd_g_cycles'].astype(np.float32)  # (N, 256)
                    gt    = data['gt_cycles'].astype(np.float32)      # (N, 256)
                    if len(gt) < 5 or len(vmd_g) != len(gt):
                        continue
                    for i in range(len(gt)):
                        self.items.append((vmd_g[i], gt[i], sid))
                        self.sid_list.append(sid)
                except Exception:
                    continue

        n_subs = len(set(self.sid_list))
        print(f'A12Dataset: {len(self.items)} cycles / {n_subs} subjects')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        vmd_g, gt, sid = self.items[idx]
        return (torch.from_numpy(vmd_g).unsqueeze(0).float(),   # (1, 256)
                torch.from_numpy(gt).unsqueeze(0).float(),        # (1, 256)
                sid)


class SubjectStratifiedSampler(Sampler):
    def __init__(self, dataset, batch_size, n_subs_per_batch, seed=42):
        self.batch_size       = batch_size
        self.n_subs_per_batch = n_subs_per_batch
        self.rng              = random.Random(seed)
        self.sid_to_idxs      = {}
        for idx, sid in enumerate(dataset.sid_list):
            self.sid_to_idxs.setdefault(sid, []).append(idx)
        self.all_sids = list(self.sid_to_idxs.keys())
        self.n_total  = len(dataset)

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
        n_full = (len(indices) // self.batch_size) * self.batch_size
        return iter(indices[:n_full])

    def __len__(self):
        return self.n_total


def pearson_loss(pred, target):
    p  = pred.squeeze(1)
    t  = target.squeeze(1)
    pm = p - p.mean(dim=1, keepdim=True)
    tm = t - t.mean(dim=1, keepdim=True)
    r  = (pm * tm).sum(dim=1) / (pm.norm(dim=1) * tm.norm(dim=1) + 1e-8)
    return (1.0 - r).mean()


def load_frozen_vae(device):
    state = torch.load(ENCODER_CKPT['B'], map_location=device, weights_only=False)
    vae   = PPGVAE(latent_dim=LATENT_DIM)
    vae.decoder.load_state_dict(state['decoder_finetune'])
    vae_full  = torch.load(VAE_CKPT, map_location=device, weights_only=False)
    enc_state = {k[len('encoder.'):]: v
                 for k, v in vae_full.items() if k.startswith('encoder.')}
    vae.encoder.load_state_dict(enc_state)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae.to(device)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\ntrain_a12 — VMD-peak-aligned 1ch encoder  |  device={device}')

    split_df   = pd.read_csv(SPLIT_FILE)
    train_sids = set(split_df[split_df['split'] == 'train']['sid'])
    val_sids   = set(split_df[split_df['split'] == 'val']['sid'])

    print('\nBuilding datasets...')
    train_ds = A12Dataset(CYCLES_DIRS, train_sids)
    val_ds   = A12Dataset(CYCLES_DIRS, val_sids)

    if len(train_ds) == 0:
        print('ERROR: No A12 cycles found. Run extract_vmd_rppg.py first.')
        sys.exit(1)

    sampler      = SubjectStratifiedSampler(train_ds, BATCH_SIZE, N_SUBJECTS_PER_BATCH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    print('\nLoading frozen VAE decoder (V5-B fine-tuned)...')
    vae     = load_frozen_vae(device)
    encoder = CameraEncoder(latent_dim=LATENT_DIM, in_channels=1,
                            morpho_aux=False).to(device)
    optimiser = torch.optim.Adam(encoder.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=MAX_EPOCHS)

    CKPT_A12.mkdir(parents=True, exist_ok=True)
    RESULTS_A12.mkdir(parents=True, exist_ok=True)

    best_val_r = -np.inf
    patience   = 0
    ckpt_path  = CKPT_A12 / 'encoder_a12.pt'

    print(f'\nTraining up to {MAX_EPOCHS} epochs (early stop patience={EARLY_STOP})...\n')

    for epoch in range(1, MAX_EPOCHS + 1):
        encoder.train()
        train_losses = []

        for vmd_g, gt, _ in train_loader:
            vmd_g = vmd_g.to(device)   # (B, 1, 256)
            gt    = gt.to(device)      # (B, 1, 256)

            z_pred   = encoder(vmd_g)
            ppg_pred = vae.decode(z_pred)

            l_pearson  = LAMBDA_PEARSON  * pearson_loss(ppg_pred, gt)
            l_spectral = LAMBDA_SPECTRAL * spectral_l1_loss(ppg_pred, gt)
            l_freq     = LAMBDA_FREQ     * frequency_loss(ppg_pred, gt)
            loss       = l_pearson + l_spectral + l_freq

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            optimiser.step()
            train_losses.append(loss.item())

        scheduler.step()

        encoder.eval()
        val_rs = []
        with torch.no_grad():
            for vmd_g, gt, _ in val_loader:
                vmd_g = vmd_g.to(device)
                gt_np = gt.squeeze(1).numpy()
                z_pred = encoder(vmd_g)
                ppg_np = vae.decode(z_pred).cpu().squeeze(1).numpy()
                for i in range(len(gt_np)):
                    if ppg_np[i].std() > 1e-8 and gt_np[i].std() > 1e-8:
                        from scipy.stats import pearsonr
                        val_rs.append(float(pearsonr(ppg_np[i], gt_np[i])[0]))

        val_r  = float(np.mean(val_rs)) if val_rs else 0.0
        t_loss = float(np.mean(train_losses))

        if epoch % 10 == 0 or epoch <= 5:
            print(f'Epoch {epoch:3d}  train_loss={t_loss:.4f}  val_r={val_r:.4f}')

        if val_r > best_val_r:
            best_val_r = val_r
            patience   = 0
            torch.save({'encoder': encoder.state_dict(),
                        'epoch':   epoch,
                        'val_r':   val_r}, ckpt_path)
        else:
            patience += 1
            if patience >= EARLY_STOP:
                print(f'\nEarly stop at epoch {epoch}  best_val_r={best_val_r:.4f}')
                break

    print(f'\nTraining complete. Best val_r={best_val_r:.4f}')
    print(f'Checkpoint: {ckpt_path}')
    print('\nNext:')
    print('  python morph_research_pipeline/evaluation/a12/evaluate_a12.py')


if __name__ == '__main__':
    main()
