"""
training/a10/train_a10.py — A10 DPS: Train Forward Model + Unconditional DDPM
==============================================================================
Two sequential training phases:

Phase 1 — Forward Model (PPGToRPPG):
  Learns f: GT_PPG → CHROM_rPPG on aligned pairs from train split.
  Loss: Pearson correlation (sign-sensitive, scale-invariant).
  Saves: checkpoints/a10/forward_model.pt

Phase 2 — Unconditional DDPM:
  Learns P(z_gt) from GT PPG cycles encoded by the frozen VAE encoder.
  NO conditioning — pure unconditional prior.
  Loss: standard DDPM MSE on z_gt.
  Saves: checkpoints/a10/ddpm_a10.pt

Both phases use the clean 153-subject train split.
VAE encoder and decoder are frozen in both phases.
"""

import os, sys, random
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path

HERE = Path(__file__).parent
PIPELINE_ROOT = HERE.parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from config.paths import (
    UBFC_CYCLES_DIR, STRESS_CYCLES_DIR, FPS2023_CYCLES_DIR,
    FPS2023_60_CYCLES_DIR, CENTAN_CYCLES_DIR,
    CKPT_A10, RESULTS_A10, SPLIT_FILE,
    VAE_CKPT, ENCODER_CKPT,
)
from config.hyperparams import BATCH_SIZE, LEARNING_RATE
from models.vae import PPGVAE
from models.forward_model import PPGToRPPG
from models.ddpm_a10 import UnconditionalDDPM

LATENT_DIM    = 32
T_STEPS       = 200
MAX_EPOCHS_FM = 100    # forward model epochs
MAX_EPOCHS_DM = 300    # DDPM epochs
EARLY_STOP    = 30


# ── Frozen model loading ───────────────────────────────────────────────────────

def load_frozen_vae(device):
    """Returns frozen PPGVAE with fine-tuned decoder from encoder_B.pt."""
    state = torch.load(ENCODER_CKPT['B'], map_location=device, weights_only=False)
    vae   = PPGVAE(latent_dim=LATENT_DIM)
    vae.decoder.load_state_dict(state['decoder_finetune'])
    # Load only encoder weights from stage1_vae_p4.pt (do NOT overwrite decoder)
    vae_full = torch.load(VAE_CKPT, map_location=device, weights_only=False)
    enc_state = {k[len('encoder.'):]: v
                 for k, v in vae_full.items() if k.startswith('encoder.')}
    vae.encoder.load_state_dict(enc_state)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae.to(device)


def find_cycle_dirs():
    return [d for d in [UBFC_CYCLES_DIR, STRESS_CYCLES_DIR, FPS2023_CYCLES_DIR,
                        FPS2023_60_CYCLES_DIR, CENTAN_CYCLES_DIR]
            if Path(d).is_dir()]


# ── Dataset ────────────────────────────────────────────────────────────────────

class A10ForwardDataset(Dataset):
    """
    Loads (gt_ppg, rppg_chrom) pairs for forward model training.
    Falls back to rppg_pos_cycles if chrom is absent.
    """
    def __init__(self, cycle_dirs, split_sids):
        self.items    = []   # (gt: float32 (256,), rppg: float32 (256,), sid)
        self.sid_list = []

        for d in cycle_dirs:
            for npz_f in sorted(Path(d).glob('*_cycles.npz')):
                try:
                    data = np.load(npz_f, allow_pickle=True)
                    sid  = int(data['sid'])
                    if sid not in split_sids:
                        continue
                    gt = data['gt_cycles']    # (N, 256)
                    if 'rppg_chrom_cycles' in data:
                        rppg = data['rppg_chrom_cycles']
                    else:
                        rppg = data['rppg_pos_cycles']
                    if len(gt) < 5:
                        continue
                    for i in range(len(gt)):
                        self.items.append((gt[i].astype(np.float32),
                                           rppg[i].astype(np.float32),
                                           sid))
                        self.sid_list.append(sid)
                except Exception:
                    continue

        print(f'ForwardDataset: {len(self.items)} cycles / {len(set(self.sid_list))} subjects')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        gt, rppg, sid = self.items[idx]
        return (torch.from_numpy(gt).unsqueeze(0).float(),
                torch.from_numpy(rppg).unsqueeze(0).float(),
                sid)


class A10DDPMDataset(Dataset):
    """
    Pre-computes z_gt = VAE.encoder.mu(GT_PPG) for DDPM training.
    No rPPG needed — unconditional.
    """
    def __init__(self, cycle_dirs, split_sids, vae, device):
        self.z_list   = []
        self.sid_list = []

        for d in cycle_dirs:
            for npz_f in tqdm(sorted(Path(d).glob('*_cycles.npz')),
                              desc=Path(d).parent.name, leave=False):
                try:
                    data = np.load(npz_f, allow_pickle=True)
                    sid  = int(data['sid'])
                    if sid not in split_sids:
                        continue
                    gt = data['gt_cycles']    # (N, 256)
                    if len(gt) < 5:
                        continue
                    BATCH = 256
                    z_list = []
                    for i in range(0, len(gt), BATCH):
                        x = torch.from_numpy(gt[i:i+BATCH]).unsqueeze(1).float().to(device)
                        with torch.no_grad():
                            mu, _ = vae.encoder(x)
                        z_list.append(mu.cpu().numpy())
                    z_all = np.concatenate(z_list, axis=0)
                    for i in range(len(z_all)):
                        self.z_list.append(z_all[i].astype(np.float32))
                        self.sid_list.append(sid)
                except Exception:
                    continue

        print(f'DDPMDataset: {len(self.z_list)} z-vectors / {len(set(self.sid_list))} subjects')

    def __len__(self):
        return len(self.z_list)

    def __getitem__(self, idx):
        return torch.from_numpy(self.z_list[idx]).float()


# ── Loss ───────────────────────────────────────────────────────────────────────

def pearson_loss(pred, target):
    """1 - mean Pearson r across batch. pred, target: (B, 1, 256)."""
    p = pred.view(pred.size(0), -1)
    t = target.view(target.size(0), -1)
    p_z = (p - p.mean(1, keepdim=True)) / (p.std(1, keepdim=True) + 1e-8)
    t_z = (t - t.mean(1, keepdim=True)) / (t.std(1, keepdim=True) + 1e-8)
    return 1.0 - (p_z * t_z).mean(1).mean()


# ── Phase 1: Forward Model ────────────────────────────────────────────────────

def train_forward_model(device, split_df):
    print('\n' + '=' * 60)
    print('PHASE 1: Training PPGToRPPG forward model')
    print('=' * 60)

    ckpt_path = CKPT_A10 / 'forward_model.pt'
    if ckpt_path.exists():
        print(f'  Checkpoint exists: {ckpt_path}  — skipping.')
        return

    train_sids = set(split_df[split_df['split'] == 'train']['sid'])
    val_sids   = set(split_df[split_df['split'] == 'val']['sid'])
    cycle_dirs = find_cycle_dirs()

    train_ds = A10ForwardDataset(cycle_dirs, train_sids)
    val_ds   = A10ForwardDataset(cycle_dirs, val_sids)
    if len(train_ds) == 0:
        print('ERROR: No training cycles.'); return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model   = PPGToRPPG().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  PPGToRPPG: {n_params:,} parameters')

    opt      = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val = float('inf')
    patience = 0

    for epoch in range(1, MAX_EPOCHS_FM + 1):
        model.train()
        tr_loss = []
        for gt, rppg, _ in train_loader:
            gt   = gt.to(device)
            rppg = rppg.to(device)
            loss = pearson_loss(model(gt), rppg)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss.append(loss.item())

        model.eval()
        vl_loss = []
        with torch.no_grad():
            for gt, rppg, _ in val_loader:
                vl_loss.append(pearson_loss(model(gt.to(device)), rppg.to(device)).item())

        avg_tr = float(np.mean(tr_loss))
        avg_vl = float(np.mean(vl_loss))

        if epoch % 10 == 0:
            print(f'  FM Ep {epoch:03d} | Tr {avg_tr:.4f} | Val {avg_vl:.4f}')

        if avg_vl < best_val:
            best_val, patience = avg_vl, 0
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'val_loss': avg_vl},
                       ckpt_path)
        else:
            patience += 1
            if patience >= EARLY_STOP:
                print(f'  FM Early stop epoch {epoch}. Best val: {best_val:.4f}')
                break

    print(f'  Forward model saved → {ckpt_path}  (val loss: {best_val:.4f})')


# ── Phase 2: Unconditional DDPM ───────────────────────────────────────────────

def train_ddpm(device, split_df):
    print('\n' + '=' * 60)
    print('PHASE 2: Training Unconditional DDPM on z_gt')
    print('=' * 60)

    ckpt_path = CKPT_A10 / 'ddpm_a10.pt'
    if ckpt_path.exists():
        print(f'  Checkpoint exists: {ckpt_path}  — skipping.')
        return

    vae = load_frozen_vae(device)

    train_sids = set(split_df[split_df['split'] == 'train']['sid'])
    val_sids   = set(split_df[split_df['split'] == 'val']['sid'])
    cycle_dirs = find_cycle_dirs()

    print('  Pre-computing z_gt for train split...')
    train_ds = A10DDPMDataset(cycle_dirs, train_sids, vae, device)
    print('  Pre-computing z_gt for val split...')
    val_ds   = A10DDPMDataset(cycle_dirs, val_sids,   vae, device)
    if len(train_ds) == 0:
        print('ERROR: No training cycles.'); return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    ddpm = UnconditionalDDPM(latent_dim=LATENT_DIM, T=T_STEPS).to(device)
    n_params = sum(p.numel() for p in ddpm.noise_pred.parameters())
    print(f'  UnconditionalNoisePredictor: {n_params:,} parameters  (T={T_STEPS})')

    opt      = torch.optim.Adam(ddpm.noise_pred.parameters(), lr=LEARNING_RATE)
    log_rows = []
    best_val = float('inf')
    patience = 0

    for epoch in range(1, MAX_EPOCHS_DM + 1):
        ddpm.noise_pred.train()
        tr_loss = []
        for z0 in train_loader:
            z0   = z0.to(device)
            loss = ddpm(z0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(ddpm.noise_pred.parameters(), 1.0)
            opt.step()
            tr_loss.append(loss.item())

        ddpm.noise_pred.eval()
        vl_loss = []
        with torch.no_grad():
            for z0 in val_loader:
                vl_loss.append(ddpm(z0.to(device)).item())

        avg_tr = float(np.mean(tr_loss))
        avg_vl = float(np.mean(vl_loss))
        log_rows.append({'epoch': epoch, 'train_loss': avg_tr, 'val_loss': avg_vl})

        if epoch % 10 == 0:
            print(f'  DDPM Ep {epoch:03d} | Tr {avg_tr:.4f} | Val {avg_vl:.4f}')

        if avg_vl < best_val:
            best_val, patience = avg_vl, 0
            torch.save({'model': ddpm.noise_pred.state_dict(),
                        'epoch': epoch, 'val_loss': avg_vl}, ckpt_path)
        else:
            patience += 1
            if patience >= EARLY_STOP:
                print(f'  DDPM Early stop epoch {epoch}. Best val: {best_val:.4f}')
                break

    pd.DataFrame(log_rows).to_csv(RESULTS_A10 / 'training_log_ddpm_a10.csv', index=False)
    print(f'  DDPM saved → {ckpt_path}  (val loss: {best_val:.4f})')


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    CKPT_A10.mkdir(parents=True, exist_ok=True)
    RESULTS_A10.mkdir(parents=True, exist_ok=True)

    split_df = pd.read_csv(SPLIT_FILE)

    train_forward_model(device, split_df)
    train_ddpm(device, split_df)

    print('\nA10 training complete.')
    print(f'  Forward model: {CKPT_A10 / "forward_model.pt"}')
    print(f'  DDPM:          {CKPT_A10 / "ddpm_a10.pt"}')


if __name__ == '__main__':
    main()
