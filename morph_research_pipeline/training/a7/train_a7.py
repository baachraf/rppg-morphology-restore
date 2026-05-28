"""
training/a7/train_a7.py — A7: Physics-Informed RGB Encoder Training
====================================================================
Trains A7Model (PhysicsEncoderA7 + DirectDecoder) on native-resolution
6-channel RGB windows (R, G, B, R/G, G/B, R/B, mean-centered).

Key differences from A6 training:
  - No VAE — direct encoder→decoder
  - Spectral loss (harmonic amplitudes H1-H4) instead of pure L1
  - Anti-collapse loss (maximize pairwise output distance across subjects)
  - Input at native 60-sample resolution

Outputs:
  checkpoints/a7/a7_model.pt
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from tqdm import tqdm
from pathlib import Path

HERE = Path(__file__).parent
PIPELINE_ROOT = HERE.parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from morph_config import SPLIT_FILE, BATCH_SIZE, LEARNING_RATE
from config.paths import (
    UBFC_DIR, STRESS_DIR, FPS2023_DIR, CENTAN_DIR,
    RESULTS_DIR,
)
from models.encoder_a7 import A7Model
from models.metrics import batch_morpho_labels

LATENT_DIM = 32
IN_CHANNELS = 6
INPUT_LEN = 60
MAX_EPOCHS = 300
EARLY_STOP = 30

CKPT_A7 = Path(RESULTS_DIR).parent / 'checkpoints' / 'a7'
RESULTS_A7 = RESULTS_DIR / 'a7'

LAMBDAS = {
    'l1': 5.0,
    'spectral': 3.0,
    'curv': 0.5,
    'asym': 0.3,
    'sdtw': 1.0,
    'diversity': 2.0,
    'anti_collapse': 1.0,
    'aux_morpho': 2.0,
    'contrastive': 0.5,
}


def spectral_loss(pred, target, n_harmonics=6):
    pred_fft = torch.abs(torch.fft.rfft(pred, dim=-1))
    target_fft = torch.abs(torch.fft.rfft(target, dim=-1))
    n = min(n_harmonics + 1, pred_fft.shape[-1])
    return F.l1_loss(pred_fft[:, :, :n], target_fft[:, :, :n])


def anti_collapse_loss(z, sids, margin=1.0):
    unique_sids = torch.unique(sids)
    if len(unique_sids) < 2:
        return torch.tensor(0.0, device=z.device)

    means = {}
    for sid in unique_sids:
        mask = sids == sid
        means[sid.item()] = z[mask].mean(dim=0)

    means_list = list(means.values())
    if len(means_list) < 2:
        return torch.tensor(0.0, device=z.device)

    centroids = torch.stack(means_list)
    n = len(centroids)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = F.pairwise_distance(centroids[i].unsqueeze(0), centroids[j].unsqueeze(0))
            total += F.relu(margin - d)
            count += 1
    if count == 0:
        return torch.tensor(0.0, device=z.device)
    return total / count


def curvature_loss(pred, target):
    def curv(x):
        d2 = x[:, :, 2:] - 2 * x[:, :, 1:-1] + x[:, :, :-2]
        return d2
    return F.l1_loss(curv(pred), curv(target))


def soft_dtw_loss(pred, target, gamma=1.0):
    diff = pred - target
    return torch.mean(torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-8))


class A7WindowDataset(Dataset):
    def __init__(self, data_dirs, split_sids):
        self.items = []
        self.sid_list = []

        for data_dir in data_dirs:
            data_dir = Path(data_dir)
            if not data_dir.is_dir():
                continue
            for npz_p in tqdm(list(data_dir.glob('*_a7_windows.npz')),
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


def find_a7_window_dirs():
    dirs = []
    for base in [UBFC_DIR, STRESS_DIR, FPS2023_DIR, CENTAN_DIR]:
        d = base / 'a7_windows'
        if d.is_dir():
            dirs.append(d)
    return dirs


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    CKPT_A7.mkdir(parents=True, exist_ok=True)
    RESULTS_A7.mkdir(parents=True, exist_ok=True)

    split_df = pd.read_csv(SPLIT_FILE)
    train_sids = set(split_df[split_df['split'] == 'train']['sid'])
    val_sids = set(split_df[split_df['split'] == 'val']['sid'])

    data_dirs = find_a7_window_dirs()
    if not data_dirs:
        print('ERROR: No a7_windows directories found.')
        print('Run extraction/extract_rgb_windows_a7.py first.')
        return

    print(f'Found a7_window dirs: {[str(d) for d in data_dirs]}')

    train_ds = A7WindowDataset(data_dirs, train_sids)
    val_ds = A7WindowDataset(data_dirs, val_sids)

    if len(train_ds) == 0:
        print('ERROR: No training windows loaded.')
        return

    sampler = SubjectStratifiedSampler(train_ds, BATCH_SIZE, 16)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)

    model = A7Model(latent_dim=LATENT_DIM, in_channels=IN_CHANNELS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'A7 model: {n_params:,} parameters')

    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    ckpt_path = CKPT_A7 / 'a7_model.pt'
    log_path = RESULTS_A7 / 'training_log_a7.csv'

    best_loss = float('inf')
    patience = 0
    log_rows = []

    print(f'\nTraining A7 (Physics-Informed, {IN_CHANNELS}ch x {INPUT_LEN} samples)')
    print(f'  Train: {len(train_ds)} windows ({len(set(train_ds.sid_list))} subjects)')
    print(f'  Val:   {len(val_ds)} windows ({len(set(val_ds.sid_list))} subjects)')
    print(f'  Max epochs: {MAX_EPOCHS}, Early stop: {EARLY_STOP}')
    print(f'  Losses: L1={LAMBDAS["l1"]}, Spectral={LAMBDAS["spectral"]}, '
          f'AntiCollapse={LAMBDAS["anti_collapse"]}')
    print()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tr_losses = []
        tr_components = {}

        for rgb, gt, sids in train_loader:
            rgb = rgb.to(device)
            gt = gt.to(device)
            sids_t = sids.to(device)

            morpho_t = torch.from_numpy(
                batch_morpho_labels(gt.cpu().numpy()[:, 0, :])
            ).to(device)

            recon, z, morpho_pred = model(rgb)

            l_l1 = F.l1_loss(recon, gt) * LAMBDAS['l1']
            l_spec = spectral_loss(recon, gt) * LAMBDAS['spectral']
            l_curv = curvature_loss(recon, gt) * LAMBDAS['curv']
            l_asym = torch.tensor(0.0, device=device)
            l_sdtw = soft_dtw_loss(recon, gt) * LAMBDAS['sdtw']

            l_div = F.relu(1.0 - torch.mean(torch.var(z, dim=0))) * LAMBDAS['diversity']

            l_anti = anti_collapse_loss(z, sids_t) * LAMBDAS['anti_collapse']

            l_morpho = torch.tensor(0.0, device=device)
            if morpho_pred is not None:
                l_morpho = F.mse_loss(morpho_pred, morpho_t) * LAMBDAS['aux_morpho']

            loss = l_l1 + l_spec + l_curv + l_sdtw + l_div + l_anti + l_morpho

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_losses.append(loss.item())
            for name, val in [('l1', l_l1), ('spectral', l_spec),
                              ('anti_collapse', l_anti), ('diversity', l_div),
                              ('morpho', l_morpho)]:
                tr_components.setdefault(name, []).append(val.item())

        model.eval()
        vl_losses = []
        with torch.no_grad():
            for rgb, gt, _ in val_loader:
                rgb, gt = rgb.to(device), gt.to(device)
                recon, z, _ = model(rgb)
                vl_l1 = F.l1_loss(recon, gt)
                vl_spec = spectral_loss(recon, gt)
                vl_loss = vl_l1 * LAMBDAS['l1'] + vl_spec * LAMBDAS['spectral']
                vl_losses.append(vl_loss.item())

        avg_tr = np.mean(tr_losses)
        avg_vl = np.mean(vl_losses)

        log_rows.append({
            'epoch': epoch, 'train_loss': avg_tr, 'val_loss': avg_vl,
            **{k: np.mean(v) for k, v in tr_components.items()},
        })

        if epoch % 10 == 0:
            comp_str = ' | '.join(f'{k}={np.mean(v):.3f}'
                                  for k, v in tr_components.items())
            print(f'  Epoch {epoch:03d} | Tr: {avg_tr:.4f} | Val: {avg_vl:.4f} | {comp_str}')

        if avg_vl < best_loss:
            best_loss = avg_vl
            patience = 0
            torch.save({
                'model': model.state_dict(),
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
    print(f'\nA7 training complete. Best Val: {best_loss:.4f}')
    print(f'Checkpoint: {ckpt_path}')
    print(f'Training log: {log_path}')


if __name__ == '__main__':
    main()
