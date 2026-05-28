"""
training/a4/train_a4.py — A4: Multi-Cycle Transformer Training
===============================================================
Stage 1 VAE (frozen) provides the PPG decoder.
A4 encoder maps a sequence of 5 consecutive rPPG cycles → VAE latent.
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

from morph_config import (
    CYCLES_DIR, SPLIT_FILE, BATCH_SIZE, LEARNING_RATE,
    LATENT_DIM, MAX_EPOCHS_V5, EARLY_STOP_V5,
    VAE_CKPT_P4, CKPT_A4 as _CKPT_A4_STR, RESULTS_A4 as _RESULTS_A4_STR,
)
from config.paths import CKPT_A4 as CKPT_A4_PATH, RESULTS_A4 as RESULTS_A4_PATH
from models.vae import PPGVAE
from models.transformer_a4 import MultiCycleTransformerEncoder, MultiCycleTransformerA4
from models.encoder import stage2_loss
from models.metrics import batch_morpho_labels

NUM_CYCLES = 5
D_MODEL = 256
NHEAD = 8
N_LAYERS = 4
DROPOUT = 0.1

LAMBDAS = {
    'l1': 10.0, 'notch': 5.0, 'sdtw': 1.0, 'curv': 0.5,
    'latent': 0.05, 'variance': 2.0, 'adv': 0.05,
    'adv_start': 50, 'diversity': 1.0,
    'freq': 0.0, 'spectral': 1.0, 'asym': 0.3,
    'contrastive': 0.5, 'aux_morpho': 2.0,
}


def _check_cycles_healthy(cycles, hr_arr, indices, min_hr=40, max_hr=150,
                          min_amp=0.01):
    for idx in indices:
        if idx >= len(hr_arr):
            return False
        h = hr_arr[idx]
        if h < min_hr or h > max_hr:
            return False
        c = cycles[idx]
        if np.ptp(c) < min_amp:
            return False
    return True


class MultiCycleDataset(Dataset):
    """
    Returns sequences of NUM_CYCLES consecutive cycles from each session.
    Each item maps to the middle cycle's GT for reconstruction.
    Windows containing any cycle with invalid HR or near-zero amplitude
    are skipped.
    """
    def __init__(self, root_dir, split_sids):
        self.windows = []
        self.sids_for_windows = []

        root = Path(root_dir)
        all_npz = list(root.rglob("*_cycles.npz"))

        skipped_hr = 0
        skipped_amp = 0

        for npz_p in tqdm(all_npz, desc="Building windows"):
            try:
                data = np.load(npz_p)
                sid = int(data['sid']) if 'sid' in data else 999
                if sid not in split_sids:
                    continue

                gt = data['gt_cycles']
                g = data['g_cycles']
                r = data['rppg_cycles']
                hr = data.get('hr', np.full(len(gt), 70.0))
                n = len(gt)

                if n < NUM_CYCLES:
                    continue

                half = NUM_CYCLES // 2
                for i in range(half, n - half):
                    win_idx = list(range(i - half, i + half + 1))
                    if not _check_cycles_healthy(gt, hr, win_idx):
                        skipped_hr += 1
                        continue
                    self.windows.append((
                        gt[i],
                        g[i - half:i + half + 1],
                        r[i - half:i + half + 1],
                    ))
                    self.sids_for_windows.append(sid)
            except Exception:
                continue

        print(f"Built {len(self.windows)} windows from {len(set(self.sids_for_windows))} subjects"
              f" (skipped {skipped_hr} unhealthy)")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        gt, g_win, r_win = self.windows[idx]
        sid = self.sids_for_windows[idx]
        return (
            torch.from_numpy(gt).unsqueeze(0).float(),
            torch.from_numpy(g_win).float(),
            torch.from_numpy(r_win).float(),
            sid,
        )


class SubjectStratifiedSampler(Sampler):
    def __init__(self, dataset, batch_size, n_subs_per_batch, seed=42):
        self.batch_size = batch_size
        self.n_subs_per_batch = n_subs_per_batch
        self.rng = random.Random(seed)
        self.sid_to_idxs = {}
        for idx, sid in enumerate(dataset.sids_for_windows):
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

    def __len__(self):
        return self.n_total


def train_one_encoder(enc_name, in_ch, get_input_fn, train_loader, val_loader,
                      vae, vae_decoder, device):
    print(f"\n{'=' * 60}")
    print(f"Training A4-{enc_name} (in_channels={in_ch}, num_cycles={NUM_CYCLES})")
    print(f"{'=' * 60}")

    ckpt_p = CKPT_A4_PATH / f'encoder_a4_{enc_name}.pt'

    encoder = MultiCycleTransformerEncoder(
        latent_dim=LATENT_DIM, in_channels=in_ch, num_cycles=NUM_CYCLES,
        d_model=D_MODEL, nhead=NHEAD, n_layers=N_LAYERS, dropout=DROPOUT,
    ).to(device)

    params = list(encoder.parameters()) + list(vae_decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
    best_loss = float('inf')
    patience = 0

    for epoch in range(1, MAX_EPOCHS_V5 + 1):
        encoder.train()
        vae_decoder.train()
        tr_losses = []

        for gt, g_seq, r_seq, sids in train_loader:
            gt = gt.to(device)
            sids = sids.to(device)

            x_in = get_input_fn(g_seq.to(device), r_seq.to(device))
            morpho_t = torch.from_numpy(
                batch_morpho_labels(gt.cpu().numpy()[:, 0, :])
            ).to(device)

            optimizer.zero_grad()
            z_p = encoder(x_in)
            recon = vae_decoder(z_p)
            with torch.no_grad():
                z_gt = vae.encode(gt)

            loss, _ = stage2_loss(
                recon, gt, z_p, z_gt,
                torch.full((gt.size(0),), -1, device=device),
                LAMBDAS, epoch, None, sids=sids,
                morpho_pred=None, morpho_labels=morpho_t,
            )
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())

        encoder.eval()
        vae_decoder.eval()
        vl_losses = []
        with torch.no_grad():
            for gt, g_seq, r_seq, _ in val_loader:
                gt = gt.to(device)
                x_in = get_input_fn(g_seq.to(device), r_seq.to(device))
                z_p = encoder(x_in)
                recon = vae_decoder(z_p)
                z_gt = vae.encode(gt)
                loss, _ = stage2_loss(
                    recon, gt, z_p, z_gt,
                    torch.full((gt.size(0),), -1, device=device),
                    LAMBDAS, epoch, None,
                )
                vl_losses.append(loss.item())

        avg_vl = np.mean(vl_losses)
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:03d} | Tr: {np.mean(tr_losses):.4f} | Val: {avg_vl:.4f}")

        if avg_vl < best_loss:
            best_loss = avg_vl
            patience = 0
            CKPT_A4_PATH.mkdir(parents=True, exist_ok=True)
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder_finetune': vae_decoder.state_dict(),
            }, ckpt_p)
        else:
            patience += 1
            if patience >= EARLY_STOP_V5:
                print(f"  Early stop epoch {epoch}. Best: {best_loss:.4f}")
                break

    print(f"  A4-{enc_name} done. Best Val: {best_loss:.4f}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    CKPT_A4_PATH.mkdir(parents=True, exist_ok=True)
    RESULTS_A4_PATH.mkdir(parents=True, exist_ok=True)

    vae = PPGVAE(latent_dim=LATENT_DIM).to(device)
    vae_ckpt = Path(str(CYCLES_DIR).replace('data', 'checkpoints')) / VAE_CKPT_P4
    possible = [
        vae_ckpt,
        Path(r'E:\Projects_Results\rPPG_Morphology_Restore\checkpoints') / VAE_CKPT_P4,
    ]
    loaded = False
    for p in possible:
        if p.exists():
            state = torch.load(p, map_location=device, weights_only=True)
            vae.load_state_dict(state)
            print(f"Loaded VAE from {p}")
            loaded = True
            break
    if not loaded:
        raise FileNotFoundError(f"VAE checkpoint not found. Looked in: {possible}")

    for p in vae.parameters():
        p.requires_grad = False
    for p in vae.decoder.parameters():
        p.requires_grad = True

    split_df = pd.read_csv(SPLIT_FILE)
    train_sids = set(split_df[split_df['split'] == 'train']['sid'])
    val_sids = set(split_df[split_df['split'] == 'val']['sid'])

    train_ds = MultiCycleDataset(CYCLES_DIR, train_sids)
    val_ds = MultiCycleDataset(CYCLES_DIR, val_sids)

    sampler = SubjectStratifiedSampler(train_ds, BATCH_SIZE, 16)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    configs = [
        ('A', 1, lambda g, r: g.unsqueeze(2)),
        ('B', 1, lambda g, r: r.unsqueeze(2)),
        ('C', 2, lambda g, r: torch.cat([g.unsqueeze(2), r.unsqueeze(2)], dim=2)),
    ]

    for enc_name, in_ch, get_input in configs:
        ckpt_p = CKPT_A4_PATH / f'encoder_a4_{enc_name}.pt'
        if ckpt_p.exists():
            print(f"  Skipping A4-{enc_name} — checkpoint exists")
            continue
        train_one_encoder(enc_name, in_ch, get_input, train_loader, val_loader,
                          vae, vae.decoder, device)

    print("\nA4 Training Complete.")


if __name__ == "__main__":
    main()
